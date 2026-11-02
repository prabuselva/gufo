// Raw v_wmma throughput probe on gfx1151: wave32-instruction form (the
// shipped attention kernel uses v_wmma_f32_16x16x16_f16_w32 with a duplicated
// half-wave per operand) versus the full-wave wave64 form
// (v_wmma_f32_16x16x16_f16). Reports achieved TFLOP/s per form for the same
// total FLOPs so the attention kernel's wave width can be chosen from data.
//
// Build (wave32):
//   PATH=/opt/rocm/lib/llvm/bin:$PATH clang++ --offload-arch=gfx1151 \
//     --driver-mode=g++ -L/opt/rocm/lib --hip-link -O3 -std=c++20 -x hip \
//     -I . tools/bench/flint_wmma_rate.hip -o /tmp/wmma_rate32
// Build (wave64): add -mwavefrontsize64 and -DWMMA_W64=1, output wmma_rate64.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef WMMA_W64
#define WMMA_W64 0
#endif

#define CHECK(cmd)                                                     \
  do {                                                                 \
    const hipError_t err__ = (cmd);                                    \
    if (err__ != hipSuccess) {                                         \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                  \
                   hipGetErrorString(err__), __FILE__, __LINE__);      \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

#if WMMA_W64
using ah = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using ac = __attribute__((__vector_size__(4 * sizeof(float)))) float;
__device__ __forceinline__ ac One(ah a, ah b, ac c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16(a, b, c);
}
constexpr std::uint32_t kLanes = 64;
#else
using ah = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using ac = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__device__ __forceinline__ ac One(ah a, ah b, ac c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
constexpr std::uint32_t kLanes = 32;
#endif

// 64-bit LCG-filled operands so folds cannot elide the chain.
__global__ void RateKernel(float* sink, std::uint32_t iters) {
  const std::uint32_t lane = threadIdx.x & (kLanes - 1U);
  ah a;
  ah b;
  for (std::uint32_t i = 0; i < 8; ++i) {
    a[i] = static_cast<_Float16>(lane + i);
    b[i] = static_cast<_Float16>(i - lane);
  }
#if !WMMA_W64
  for (std::uint32_t i = 8; i < 16; ++i) {  // wave32 form is 16 halves/lane
    a[i] = static_cast<_Float16>(lane ^ i);
    b[i] = static_cast<_Float16>(i + lane);
  }
#endif
  ac c0 = {};
  ac c1 = {};
  ac c2 = {};
  ac c3 = {};
  for (std::uint32_t it = 0; it < iters; ++it) {
    c0 = One(a, b, c0);
    c1 = One(a, b, c1);
    c2 = One(a, b, c2);
    c3 = One(a, b, c3);
  }
  const float t = c0[0] + c1[1] + c2[2] + c3[3];
  if (t == 12345.678F) sink[0] = t;
}

int main() {
  constexpr std::uint32_t kBlocks = 40;  // 2 per CU on 40 CUs
  constexpr std::uint32_t kThreads = kLanes * 8;
  constexpr std::uint32_t kIters = 100000;
  // Tiles per call: 4 independent WMMA per chain step; a 16x16x16 tile is
  // 8192 FLOP. wave32 counts one tile per instruction; wave64 too.
  const double flops = static_cast<double>(kBlocks) *
                       (kThreads / kLanes) * 4.0 * kIters * (16.0 * 16.0 * 16.0 *
                                                             2.0) /
                       2.0;
  float* sink = nullptr;
  CHECK(hipMalloc(&sink, sizeof(float)));
  RateKernel<<<kBlocks, kThreads>>>(sink, 3);
  CHECK(hipDeviceSynchronize());
  hipEvent_t beg, end;
  CHECK(hipEventCreate(&beg));
  CHECK(hipEventCreate(&end));
  CHECK(hipEventRecord(beg));
  for (int i = 0; i < 5; ++i) RateKernel<<<kBlocks, kThreads>>>(sink, kIters);
  CHECK(hipEventRecord(end));
  CHECK(hipEventSynchronize(end));
  CHECK(hipFree(sink));
  float ms = 0.0F;
  CHECK(hipEventElapsedTime(&ms, beg, end));
  const double tflops = flops * 5.0 / (ms * 1.0e-3) / 1.0e12;
#if WMMA_W64
  std::printf("wave64 %.2f TF/s\n", tflops);
#else
  std::printf("wave32 %.2f TF/s\n", tflops);
#endif
  return 0;
}

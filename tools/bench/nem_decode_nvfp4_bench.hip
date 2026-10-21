// Nemotron Bolt NVFP4 ssm_in decode GEMV bench (gfx1151). Decodes run
// GemvNvfp4F32Kernel with a 256-thread block per output row, thread s owning
// sub-block s (16 values): one 8 B qs load, one ue4m3 scale byte and 16
// scalar x loads plus 32 __constant__ kE2M1 lookups with divergent indices.
// Variants keep the exact thread->sub ownership and fold order and only
// change load granularity (u64 qs, float4 x) and the nibble decode
// (arithmetical E2M1 instead of the constant table), so outputs stay
// bitwise identical to the production kernel.
//
//   tools/bench/build.sh nem_decode_nvfp4_bench && /tmp/nem_decode_nvfp4_bench

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cmd)                                                    \
  do {                                                                \
    hipError_t err__ = (cmd);                                         \
    if (err__ != hipSuccess) {                                        \
      fprintf(stderr, "HIP error %s at %s:%d\n",                      \
              hipGetErrorString(err__), __FILE__, __LINE__);          \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

namespace {

constexpr int kBlock = 256;
constexpr int kNvfp4K = 64;
constexpr int kNvfp4Sub = 16;
constexpr int kWarp = 32;

__device__ __constant__ float kE2M1[16] = {
    0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

__device__ __forceinline__ float Ue4m3ToF32(uint8_t v) {
  const int e = (v >> 3) & 0xF;
  const int m = v & 7;
  if (e == 0) {
    return static_cast<float>(m) * (1.0F / 512.0F);
  }
  const uint32_t bits =
      static_cast<uint32_t>(e + 120) << 23 | static_cast<uint32_t>(m) << 20;
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

// Arithmetical E2M1 nibble decode: bit-identical to the kE2M1 table, which
// holds only exactly representable magnitudes (incl. -0.0 at nibble 8).
__device__ __forceinline__ float E2M1Alu(uint8_t v) {
  const uint32_t s = (v >> 3) & 1U;
  const uint32_t e = (v >> 1) & 3U;
  const uint32_t m = v & 1U;
  const uint32_t mag = e == 0U ? m * 0x3F000000U
                               : ((e + 126U) << 23) | (m << 22);
  const uint32_t bits = mag | (s << 31);
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

__device__ __forceinline__ void ProdFoldStore(float acc, float* warp_part) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  const int s = static_cast<int>(threadIdx.x);
  const int lane = s % kWarp;
  if (lane == 0) {
    warp_part[s / kWarp] = acc;
  }
  __syncthreads();
}

// Production kernel (nb_ops.hip.cpp GemvNvfp4F32Kernel).
__global__ void SsmBase(const float* __restrict__ x, const uint8_t* __restrict__ w,
                        float* __restrict__ out, int n_out, int k, int tokens) {
  const int row = blockIdx.x;
  const int token = static_cast<int>(blockIdx.y);
  const int s = static_cast<int>(threadIdx.x);
  const int n_sub = k / kNvfp4Sub;
  const float* xr = x + static_cast<int64_t>(token) * k;
  out += static_cast<int64_t>(token) * n_out;
  const uint8_t* wr = w + static_cast<int64_t>(row) * (k / kNvfp4K) * 36;
  float acc = 0.0F;
  for (int b = s; b < n_sub; b += kBlock) {
    const int blk = b / (kNvfp4K / kNvfp4Sub);
    const int sub = b % (kNvfp4K / kNvfp4Sub);
    const uint8_t* wp = wr + static_cast<int64_t>(blk) * 36;
    const float d = Ue4m3ToF32(wp[sub]);
    const uint8_t* qs = wp + 4 + sub * (kNvfp4Sub / 2);
    const float* xv = xr + b * kNvfp4Sub;
    for (int i = 0; i < kNvfp4Sub / 2; ++i) {
      const uint8_t byte = qs[i];
      acc += kE2M1[byte & 0xF] * d * xv[i];
      acc += kE2M1[byte >> 4] * d * xv[i + kNvfp4Sub / 2];
    }
  }
  __shared__ float warp[8];
  ProdFoldStore(acc, warp);
  if (s == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kBlock / kWarp; ++i) {
      sum += warp[i];
    }
    out[row] = sum;
  }
}

// Fast variants: identical arithmetic and fold, wider loads (+/- ALU nibble
// decode). qs is 8 contiguous bytes and xv 16 contiguous floats (16 B
// aligned), so the vector loads fetch exactly the same values in the same
// FMA order.
template<bool kAlu>
__global__ void SsmFast(const float* __restrict__ x, const uint8_t* __restrict__ w,
                        float* __restrict__ out, int n_out, int k,
                        int tokens) {
  const int row = blockIdx.x;
  const int token = static_cast<int>(blockIdx.y);
  const int s = static_cast<int>(threadIdx.x);
  const int n_sub = k / kNvfp4Sub;
  const float* xr = x + static_cast<int64_t>(token) * k;
  out += static_cast<int64_t>(token) * n_out;
  const uint8_t* wr = w + static_cast<int64_t>(row) * (k / kNvfp4K) * 36;
  float acc = 0.0F;
  for (int b = s; b < n_sub; b += kBlock) {
    const int blk = b / (kNvfp4K / kNvfp4Sub);
    const int sub = b % (kNvfp4K / kNvfp4Sub);
    const uint8_t* wp = wr + static_cast<int64_t>(blk) * 36;
    const float d = Ue4m3ToF32(wp[sub]);
    uint64_t q;
    __builtin_memcpy(&q, wp + 4 + sub * (kNvfp4Sub / 2), sizeof(q));
    const float* xv = xr + b * kNvfp4Sub;
    float4 x4[4];
    x4[0] = *reinterpret_cast<const float4*>(xv);
    x4[1] = *reinterpret_cast<const float4*>(xv + 4);
    x4[2] = *reinterpret_cast<const float4*>(xv + 8);
    x4[3] = *reinterpret_cast<const float4*>(xv + 12);
    const float* xf = reinterpret_cast<const float*>(x4);
    for (int i = 0; i < kNvfp4Sub / 2; ++i) {
      const uint8_t byte = static_cast<uint8_t>((q >> (8 * i)) & 0xFFU);
      const float lo = kAlu ? E2M1Alu(byte & 0xF) : kE2M1[byte & 0xF];
      const float hi = kAlu ? E2M1Alu(byte >> 4) : kE2M1[byte >> 4];
      acc += lo * d * xf[i];
      acc += hi * d * xf[i + kNvfp4Sub / 2];
    }
  }
  __shared__ float warp[8];
  ProdFoldStore(acc, warp);
  if (s == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kBlock / kWarp; ++i) {
      sum += warp[i];
    }
    out[row] = sum;
  }
}

struct Run {
  uint8_t* w = nullptr;
  float* x = nullptr;
  float* out_ref = nullptr;
  float* out_new = nullptr;
  int64_t set_bytes = 0;
  int n_sets = 8;
};

uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

void run_case(const char* name, int n_out, int k, int tokens, int reps) {
  Run r;
  r.set_bytes = static_cast<int64_t>(n_out) * (k / kNvfp4K) * 36;
  const int64_t out_elems = static_cast<int64_t>(n_out) * tokens;
  const int64_t x_elems = static_cast<int64_t>(k) * tokens;
  CHECK(hipMalloc(&r.w, static_cast<size_t>(r.set_bytes) * r.n_sets));
  CHECK(hipMalloc(&r.x, static_cast<size_t>(x_elems) * 4));
  CHECK(hipMalloc(&r.out_ref, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMalloc(&r.out_new, static_cast<size_t>(out_elems) * 4));

  std::vector<uint8_t> hw(static_cast<size_t>(r.set_bytes) * r.n_sets);
  uint32_t seed = 0x1234u;
  for (size_t i = 0; i < hw.size(); ++i) {
    hw[i] = static_cast<uint8_t>(lcg(seed) & 0xFF);
  }
  std::vector<float> hx(static_cast<size_t>(x_elems));
  for (size_t i = 0; i < hx.size(); ++i) {
    hx[i] = (static_cast<float>(lcg(seed)) / 16777216.0F) - 1.0F;
  }
  CHECK(hipMemcpy(r.w, hw.data(), hw.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(r.x, hx.data(), hx.size(), hipMemcpyHostToDevice));

  const dim3 grid(static_cast<unsigned>(n_out), static_cast<unsigned>(tokens));
  const dim3 block(kBlock, 1);
  auto launch_base = [&](int set, float* out) {
    SsmBase<<<grid, block>>>(r.x, r.w + static_cast<int64_t>(set) * r.set_bytes,
                             out, n_out, k, tokens);
  };
  auto launch_v64 = [&](int set, float* out) {
    SsmFast<false><<<grid, block>>>(r.x,
                                    r.w + static_cast<int64_t>(set) * r.set_bytes,
                                    out, n_out, k, tokens);
  };
  auto launch_alu = [&](int set, float* out) {
    SsmFast<true><<<grid, block>>>(r.x,
                                   r.w + static_cast<int64_t>(set) * r.set_bytes,
                                   out, n_out, k, tokens);
  };

  CHECK(hipMemset(r.out_ref, 0xFF, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMemset(r.out_new, 0xFF, static_cast<size_t>(out_elems) * 4));
  launch_base(0, r.out_ref);
  launch_v64(0, r.out_new);
  CHECK(hipDeviceSynchronize());
  std::vector<float> href(static_cast<size_t>(out_elems)),
      hnew(static_cast<size_t>(out_elems));
  CHECK(hipMemcpy(href.data(), r.out_ref, href.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(hnew.data(), r.out_new, hnew.size() * 4, hipMemcpyDeviceToHost));
  const bool eq_v64 =
      std::memcmp(href.data(), hnew.data(), href.size() * 4) == 0;

  CHECK(hipMemset(r.out_new, 0xFF, static_cast<size_t>(out_elems) * 4));
  launch_alu(0, r.out_new);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hnew.data(), r.out_new, hnew.size() * 4, hipMemcpyDeviceToHost));
  const bool eq_alu =
      std::memcmp(href.data(), hnew.data(), href.size() * 4) == 0;

  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto bench = [&](auto launch) {
    for (int i = 0; i < 3; ++i) launch(0, r.out_new);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
      launch(i % r.n_sets, (i % 2 == 0) ? r.out_new : r.out_ref);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) * 1000.0 / reps;
  };
  const double us_base = bench(launch_base);
  const double us_v64 = bench(launch_v64);
  const double us_alu = bench(launch_alu);
  const double mb = static_cast<double>(r.set_bytes) * tokens / 1e6;
  printf("%-12s n=%d k=%d t=%d set=%.1fMB | base %8.2f us (%6.1f GB/s) | v64 %8.2f us (%6.1f GB/s) %.2fx | alu %8.2f us (%6.1f GB/s) %.2fx | bitwise %d %d\n",
         name, n_out, k, tokens, mb, us_base, mb / us_base, us_v64,
         mb / us_v64, us_base / us_v64, us_alu, mb / us_alu, us_base / us_alu,
         eq_v64 ? 1 : 0, eq_alu ? 1 : 0);
  CHECK(hipGetLastError());
  CHECK(hipFree(r.w));
  CHECK(hipFree(r.x));
  CHECK(hipFree(r.out_ref));
  CHECK(hipFree(r.out_new));
}

}  // namespace

int main() {
  int warp = 0;
  CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, 0));
  if (warp != 32) {
    fprintf(stderr, "expected warpSize 32, got %d\n", warp);
    return 1;
  }
  // ssm_in decode shape: proj = 2*8192 + 2*8*128 + 128 = 18560, k = hidden.
  run_case("ssm_t1", 18560, 4096, 1, 300);
  run_case("ssm_t4", 18560, 4096, 4, 100);
  return 0;
}

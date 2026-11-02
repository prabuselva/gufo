// Nemotron Bolt decoder router GEMM bench (gfx1151): out[1,n_exp] = x[4096] w^T.
// Decode calls the production 64x64 LDS-tiled F32 GEMM once per MoE layer with
// n_tokens=1, streaming 4-6 MB of F32 router weights at ~24 GB/s. Probe
// single-thread-per-output variants that keep the exact ascending-k fold so the
// result stays bitwise identical to the tiled kernel, then time them.
//
// tools/bench/build.sh nem_decode_router_bench && /tmp/nem_decode_router_bench

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

constexpr int kK = 4096;
constexpr int kKStepP = 32;
constexpr int kThreadsP = 256;
constexpr int kGridRP = 64;
constexpr int kThreadTokensNP = kK * 0 + kK;  // unused placeholder
constexpr int kTileTokensP = 64;
constexpr int kTileExpertsP = 64;
constexpr int kThreadGridTP = kThreadsP / kGridRP;
constexpr int kThreadTokensNP2 = kTileTokensP / kThreadGridTP;
constexpr int kThreadExpertsNP = kTileExpertsP / kGridRP;

// Production GemmF32TiledKernel (nb_ops.hip.cpp) verbatim.
__global__ void GemmTiled(const float* __restrict__ x,
                          const float* __restrict__ w, float* __restrict__ out,
                          int n_out, int k, int n_tokens) {
  const int t0 = blockIdx.x * kTileTokensP;
  const int r0 = blockIdx.y * kTileExpertsP;
  const int tid = threadIdx.x;
  const int ty = tid / kGridRP;
  const int tx = tid % kGridRP;
  __shared__ float xs[kTileTokensP][kKStepP + 1];
  __shared__ float ws[kTileExpertsP][kKStepP + 1];
  const float4* xp = reinterpret_cast<const float4*>(x + (int64_t)t0 * k);
  const float4* wp = reinterpret_cast<const float4*>(w + (int64_t)r0 * k);
  float acc[kThreadTokensNP2][kThreadExpertsNP] = {};
  for (int kc = 0; kc < k; kc += kKStepP) {
    for (int i = tid; i < kTileTokensP * (kKStepP / 4); i += kThreadsP) {
      const int row = i / (kKStepP / 4);
      const int c4 = i % (kKStepP / 4);
      float4 v = make_float4(0, 0, 0, 0);
      if (t0 + row < n_tokens) {
        v = *reinterpret_cast<const float4*>(xp + (int64_t)row * (k / 4) +
                                             (kc / 4) + c4);
      }
      xs[row][c4 * 4 + 0] = v.x;
      xs[row][c4 * 4 + 1] = v.y;
      xs[row][c4 * 4 + 2] = v.z;
      xs[row][c4 * 4 + 3] = v.w;
    }
    for (int i = tid; i < kTileExpertsP * (kKStepP / 4); i += kThreadsP) {
      const int row = i / (kKStepP / 4);
      const int c4 = i % (kKStepP / 4);
      float4 v = make_float4(0, 0, 0, 0);
      if (r0 + row < n_out) {
        v = *reinterpret_cast<const float4*>(wp + (int64_t)row * (k / 4) +
                                             (kc / 4) + c4);
      }
      ws[row][c4 * 4 + 0] = v.x;
      ws[row][c4 * 4 + 1] = v.y;
      ws[row][c4 * 4 + 2] = v.z;
      ws[row][c4 * 4 + 3] = v.w;
    }
    __syncthreads();
#pragma unroll 1
    for (int e = 0; e < kKStepP; ++e) {
      float ra[kThreadTokensNP2], rb[kThreadExpertsNP];
#pragma unroll
      for (int i = 0; i < kThreadTokensNP2; ++i)
        ra[i] = xs[ty * kThreadTokensNP2 + i][e];
#pragma unroll
      for (int j = 0; j < kThreadExpertsNP; ++j)
        rb[j] = ws[tx * kThreadExpertsNP + j][e];
#pragma unroll
      for (int i = 0; i < kThreadTokensNP2; ++i)
#pragma unroll
        for (int j = 0; j < kThreadExpertsNP; ++j)
          acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll 1
  for (int i = 0; i < kThreadTokensNP2; ++i) {
    const int t = t0 + ty * kThreadTokensNP2 + i;
    if (t >= n_tokens) break;
#pragma unroll
    for (int j = 0; j < kThreadExpertsNP; ++j) {
      const int r = r0 + tx * kThreadExpertsNP + j;
      if (r < n_out) out[(int64_t)t * n_out + r] = acc[i][j];
    }
  }
}

// V1: one thread per expert row; ascending-k scalar fold == tiled fold order.
__global__ void RouterRow(const float* __restrict__ x,
                          const float* __restrict__ w, float* __restrict__ out,
                          int n_out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = kK / 4;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* w4 =
      reinterpret_cast<const float4*>(w + (int64_t)r * kK);  // NOLINT
  if (r < n_out) {
    float acc = 0.0F;
    for (int i = 0; i < n4; ++i) {
      const float4 xv = x4[i];
      const float4 wv = w4[i];
      acc += xv.x * wv.x;
      acc += xv.y * wv.y;
      acc += xv.z * wv.z;
      acc += xv.w * wv.w;
    }
    out[r] = acc;
  }
}

// V2: same fold, x cached in LDS so only weights stream from memory.
__global__ void RouterRowLds(const float* __restrict__ x,
                             const float* __restrict__ w,
                             float* __restrict__ out, int n_out) {
  __shared__ float4 xs[kK / 4];
  for (int i = threadIdx.x; i < kK / 4; i += blockDim.x)
    xs[i] = reinterpret_cast<const float4*>(x)[i];
  __syncthreads();
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < n_out) {
    const float4* w4 =
        reinterpret_cast<const float4*>(w + (int64_t)r * kK);  // NOLINT
    float acc = 0.0F;
    for (int i = 0; i < kK / 4; ++i) {
      const float4 xv = xs[i];
      const float4 wv = w4[i];
      acc += xv.x * wv.x;
      acc += xv.y * wv.y;
      acc += xv.z * wv.z;
      acc += xv.w * wv.w;
    }
    out[r] = acc;
  }
}

// V3: V1 unrolled by 8 float4 chunks to lift memory-level parallelism.
__global__ void RouterRowU8(const float* __restrict__ x,
                            const float* __restrict__ w,
                            float* __restrict__ out, int n_out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_out) return;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* w4 =
      reinterpret_cast<const float4*>(w + (int64_t)r * kK);  // NOLINT
  float acc = 0.0F;
  for (int i = 0; i < kK / 4; i += 16) {
    float4 xv[16], wv[16];
#pragma unroll
    for (int u = 0; u < 16; ++u) xv[u] = x4[i + u];
#pragma unroll
    for (int u = 0; u < 16; ++u) wv[u] = w4[i + u];
#pragma unroll
    for (int u = 0; u < 16; ++u) {
      acc += xv[u].x * wv[u].x;
      acc += xv[u].y * wv[u].y;
      acc += xv[u].z * wv[u].z;
      acc += xv[u].w * wv[u].w;
    }
  }
  out[r] = acc;
}

struct V {
  const char* name;
  void (*run)(const float*, const float*, float*, int, hipStream_t);
};

void LaunchTiled(const float* x, const float* w, float* out, int n,
                 hipStream_t s) {
  GemmTiled<<<dim3((1 + 63) / 64, (n + 63) / 64), 256, 0, s>>>(x, w, out, n,
                                                               kK, 1);
}
void LaunchRow(const float* x, const float* w, float* out, int n,
               hipStream_t s) {
  RouterRow<<<(n + 127) / 128, 128, 0, s>>>(x, w, out, n);
}
void LaunchRowBig(const float* x, const float* w, float* out, int n,
                  hipStream_t s) {
  RouterRow<<<1, n, 0, s>>>(x, w, out, n);
}
void LaunchRowLds(const float* x, const float* w, float* out, int n,
                  hipStream_t s) {
  RouterRowLds<<<(n + 255) / 256, 256, 0, s>>>(x, w, out, n);
}
void LaunchRowU8(const float* x, const float* w, float* out, int n,
                 hipStream_t s) {
  RouterRowU8<<<(n + 255) / 256, 256, 0, s>>>(x, w, out, n);
}

}  // namespace

int main() {
  const int experts[] = {240, 368};
  for (int n : experts) {
    std::vector<float> hx(kK), hw((size_t)n * kK);
    uint32_t st = 12345u;
    auto rnd = [&st]() {
      st = st * 1664525U + 1013904223U;
      return static_cast<float>(static_cast<int>(st >> 8) % 20000 - 10000) /
             10000.0F;
    };
    for (auto& v : hx) v = rnd() * 0.001F;
    for (auto& v : hw) v = rnd() * 0.001F;
    float *dx, *dw, *dref, *dout;
    CHECK(hipMalloc(&dx, hx.size() * 4));
    CHECK(hipMalloc(&dw, hw.size() * 4));
    CHECK(hipMalloc(&dref, n * 4));
    CHECK(hipMalloc(&dout, n * 4));
    CHECK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    LaunchTiled(dx, dw, dref, n, nullptr);
    CHECK(hipDeviceSynchronize());

    const V variants[] = {
        {"tiled(prod)  ", LaunchTiled},     {"row128         ", LaunchRow},
        {"row-block1   ", LaunchRowBig},    {"row-lds256     ", LaunchRowLds},
        {"row-u8x16-256", LaunchRowU8},
    };
    std::printf("== n_experts=%d k=%d t=1 ==\n", n, kK);
    std::vector<float> ref(n), cand(n);
    CHECK(hipMemcpy(ref.data(), dref, n * 4, hipMemcpyDeviceToHost));
    for (const auto& v : variants) {
      CHECK(hipMemset(dout, 0, n * 4));
      v.run(dx, dw, dout, n, nullptr);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(cand.data(), dout, n * 4, hipMemcpyDeviceToHost));
      int diff = 0;
      for (int i = 0; i < n; ++i)
        if (memcmp(&cand[i], &ref[i], 4) != 0) ++diff;
      double mb = (double)n * kK * 4 / 1e6;
      float ms = 0.0F;
      hipEvent_t e0, e1;
      CHECK(hipEventCreate(&e0));
      CHECK(hipEventCreate(&e1));
      // Match decode spacing: 41 launches back-to-back, one sync per token.
      for (int warm = 0; warm < 20; ++warm) v.run(dx, dw, dout, n, nullptr);
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(e0));
      for (int it = 0; it < 410; ++it) v.run(dx, dw, dout, n, nullptr);
      CHECK(hipEventRecord(e1));
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double us = ms * 1000.0 / 410.0;
      std::printf("  %s bitwise_diff=%d  %7.1f us/launch  %5.1f GB/s\n",
                  v.name, diff, us, mb / us);
      CHECK(hipEventDestroy(e0));
      CHECK(hipEventDestroy(e1));
    }
    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dref));
    CHECK(hipFree(dout));
  }
  return 0;
}

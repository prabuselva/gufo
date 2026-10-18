// Standalone A/B: ling3_0 RmsNorm decode shapes, bit-exact variants.
// Baseline = production RmsNormKernel verbatim. Cached = stage x in LDS in
// pass 1 and reread from LDS in pass 2 (identical values, identical double
// accumulation order). CachedW = Cached plus w prefetched into registers
// during pass 1 (pure data movement; arithmetic sequence unchanged).
// Build: hipcc -O3 --offload-arch=gfx1151 ling3_rms_norm_bench.hip
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint32_t kBlock = 256U;
constexpr std::uint32_t kMaxN = 2560U;
constexpr int kIters = 96;  // launches per timed call, like one decode step

__device__ __forceinline__ double BlockSumDouble(double value, double* smem) {
  const unsigned lane = threadIdx.x & 31U;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned warps = (blockDim.x + 31U) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_down_sync(~0ULL, value, off);
  }
  if (lane == 0U) {
    smem[warp] = value;
  }
  __syncthreads();
  if (warp == 0U) {
    value = (lane < warps) ? smem[lane] : 0.0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      value += __shfl_down_sync(~0ULL, value, off);
    }
    if (lane == 0U) {
      smem[0] = value;
    }
  }
  __syncthreads();
  return smem[0];
}

__global__ void RmsNormBaseline(const float* __restrict__ x,
                                const float* __restrict__ w, float eps,
                                float* __restrict__ out,
                                std::uint32_t n) {
  __shared__ double smem[8];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const double v = static_cast<double>(x[i]);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = x[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

__global__ void RmsNormCached(const float* __restrict__ x,
                              const float* __restrict__ w, float eps,
                              float* __restrict__ out, std::uint32_t n) {
  __shared__ double smem[8];
  __shared__ float xs[kMaxN];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float xv = x[i];
    xs[i] = xv;
    const double v = static_cast<double>(xv);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = xs[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

__global__ void RmsNormCachedW(const float* __restrict__ x,
                               const float* __restrict__ w, float eps,
                               float* __restrict__ out, std::uint32_t n) {
  __shared__ double smem[8];
  __shared__ float xs[kMaxN];
  float wv[kMaxN / kBlock];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x, k = 0; i < n; i += blockDim.x, ++k) {
    const float xv = x[i];
    xs[i] = xv;
    if (w != nullptr) wv[k] = w[i];
    const double v = static_cast<double>(xv);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x, k = 0; i < n; i += blockDim.x, ++k) {
    out[i] = xs[i] * scale * (w != nullptr ? wv[k] : 1.0F);
  }
}

void Check(const char* what, hipError_t e) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "hip error %s @ %s\n", hipGetErrorString(e), what);
    std::exit(1);
  }
}

typedef void (*LaunchFn)(const float*, const float*, float, float*,
                         std::uint32_t, hipStream_t);
void LaunchBaseline(const float* x, const float* w, float e, float* o,
                    std::uint32_t n, hipStream_t s) {
  RmsNormBaseline<<<1, kBlock, 0, s>>>(x, w, e, o, n);
}
void LaunchCached(const float* x, const float* w, float e, float* o,
                  std::uint32_t n, hipStream_t s) {
  RmsNormCached<<<1, kBlock, 0, s>>>(x, w, e, o, n);
}
void LaunchCachedW(const float* x, const float* w, float e, float* o,
                   std::uint32_t n, hipStream_t s) {
  RmsNormCachedW<<<1, kBlock, 0, s>>>(x, w, e, o, n);
}

}  // namespace

int main(int argc, char** argv) {
  const bool time_mode = argc > 1 && std::strcmp(argv[1], "-t") == 0;
  std::vector<std::uint32_t> shapes = {kMaxN, 2048U, 2551U};
  if (time_mode) shapes = {kMaxN};
  const int rounds = time_mode ? 12 : 300;

  std::vector<const float*> pool;
  std::vector<const float*> wpool;
  std::vector<float*> out;
  for (int p = 0; p < kIters; ++p) {
    float *x, *w, *o;
    Check("malloc x", hipMalloc(&x, kMaxN * sizeof(float)));
    Check("malloc w", hipMalloc(&w, kMaxN * sizeof(float)));
    Check("malloc o", hipMalloc(&o, kMaxN * sizeof(float)));
    pool.push_back(x);
    wpool.push_back(w);
    out.push_back(o);
  }
  // Host-side random data per pool slot, one shared w per shape round.
  std::vector<std::vector<float>> hx(kIters), hw(kIters);
  for (int p = 0; p < kIters; ++p) {
    hx[p].resize(kMaxN);
    hw[p].resize(kMaxN);
    for (std::uint32_t i = 0; i < kMaxN; ++i) {
      hx[p][i] = static_cast<float>(rand()) / RAND_MAX * 4.0F - 2.0F;
      hw[p][i] = static_cast<float>(rand()) / RAND_MAX * 2.0F + 0.5F;
    }
    Check("cpy x", hipMemcpy(const_cast<float*>(pool[p]), hx[p].data(),
                             kMaxN * sizeof(float), hipMemcpyHostToDevice));
    Check("cpy w", hipMemcpy(const_cast<float*>(wpool[p]), hw[p].data(),
                             kMaxN * sizeof(float), hipMemcpyHostToDevice));
  }

  LaunchFn fn[3] = {&LaunchBaseline, &LaunchCached, &LaunchCachedW};
  const char* name[3] = {"baseline", "cached", "cachedW"};

  if (!time_mode) {
    int fails = 0;
    for (std::uint32_t n : shapes) {
      for (int round = 0; round < rounds; ++round) {
        // Fresh random x per pool slot for this round.
        for (int p = 0; p < kIters; ++p) {
          for (std::uint32_t i = 0; i < n; ++i) {
            hx[p][i] = static_cast<float>(rand()) / RAND_MAX * 4.0F - 2.0F;
          }
          Check("cpy x", hipMemcpy(const_cast<float*>(pool[p]), hx[p].data(),
                                   n * sizeof(float), hipMemcpyHostToDevice));
        }
        // Baseline outputs to host.
        std::vector<std::vector<float>> ref(kIters, std::vector<float>(n));
        for (int p = 0; p < kIters; ++p) {
          LaunchBaseline(pool[p], wpool[p], 1e-6F, out[p], n, 0);
        }
        Check("sync", hipDeviceSynchronize());
        for (int p = 0; p < kIters; ++p) {
          Check("get ref", hipMemcpy(ref[p].data(), out[p],
                                     n * sizeof(float),
                                     hipMemcpyDeviceToHost));
        }
        for (int r = 1; r < 3; ++r) {
          for (int p = 0; p < kIters; ++p) {
            fn[r](pool[p], wpool[p], 1e-6F, out[p], n, 0);
          }
          Check("sync", hipDeviceSynchronize());
          std::vector<float> got(n);
          for (int p = 0; p < kIters; ++p) {
            Check("get got", hipMemcpy(got.data(), out[p],
                                       n * sizeof(float),
                                       hipMemcpyDeviceToHost));
            if (std::memcmp(got.data(), ref[p].data(), n * sizeof(float)) !=
                0) {
              std::printf("n=%u round=%d variant %s: MISMATCH slot %d\n", n,
                          round, name[r], p);
              ++fails;
            }
          }
        }
      }
      std::printf("n=%u: %d rounds x %d slots, %s\n", n, rounds, kIters,
                  fails == 0 ? "bit-exact yes" : "BIT-EXACT NO");
    }
    return fails == 0 ? 0 : 2;
  }

  // Timing mode: ABAB phases, per-launch us from events, rotating variants.
  for (int r = 0; r < 3; ++r) {
    for (int p = 0; p < kIters; ++p) fn[r](pool[p], wpool[p], 1e-6F, out[p],
                                           kMaxN, 0);
    Check("warm", hipDeviceSynchronize());
  }
  std::vector<double> best[3];
  for (int round = 0; round < rounds; ++round) {
    for (int pass = 0; pass < 4; ++pass) {
      const int r = pass % 3;
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      hipEventRecord(e0);
      for (int p = 0; p < kIters; ++p) {
        fn[r](pool[p], wpool[p], 1e-6F, out[p], kMaxN, 0);
      }
      hipEventRecord(e1);
      Check("synchronize", hipDeviceSynchronize());
      float ms = 0.0F;
      hipEventElapsedTime(&ms, e0, e1);
      best[r].push_back(static_cast<double>(ms) * 1000.0 / kIters);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }
  }
  for (int r = 0; r < 3; ++r) {
    std::sort(best[r].begin(), best[r].end());
    std::printf("%-8s median %.3f us/launch  min %.3f\n", name[r],
                best[r][best[r].size() / 2], best[r].front());
  }
  return 0;
}

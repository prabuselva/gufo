// A/B: Add+RmsNorm pair vs fused AddRmsNorm (ling3 lever 5.5-B1).
// Kernels copied verbatim from
// src/models/ling3_flash/kernels/rocm/kernels.hip.cpp at this revision.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

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

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w, float eps,
                              float* __restrict__ out, std::uint32_t n) {
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

__global__ void AddKernel(const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ out,
                          std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void AddRmsNormKernel(const float* __restrict__ res,
                                 const float* __restrict__ x,
                                 const float* __restrict__ w, float eps,
                                 float* __restrict__ res_out,
                                 float* __restrict__ y, std::uint32_t n) {
  __shared__ double smem[8];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float s = res[i] + x[i];
    res_out[i] = s;
    const double v = static_cast<double>(s);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    y[i] = res_out[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

#define CHECK(cmd)                                                         \
  do {                                                                     \
    hipError_t err__ = (cmd);                                              \
    if (err__ != hipSuccess) {                                             \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                   hipGetErrorString(err__), __FILE__, __LINE__);           \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

}  // namespace

int main() {
  constexpr std::uint32_t kShapes[3] = {2560U, 2048U, 2551U};  // incl. ragged
  constexpr int kRounds = 300;
  std::mt19937 rng(12345U);
  std::uniform_real_distribution<float> uni(-4.0F, 4.0F);

  for (const std::uint32_t n : kShapes) {
    std::vector<float> h_res(n), h_x(n), h_w(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      h_res[i] = uni(rng);
      h_x[i] = uni(rng);
      h_w[i] = 0.5F + uni(rng) * 0.1F;
    }
    float *d_res, *d_x, *d_w, *d_r1, *d_y1, *d_r2, *d_y2;
    CHECK(hipMalloc(&d_res, n * sizeof(float)));
    CHECK(hipMalloc(&d_x, n * sizeof(float)));
    CHECK(hipMalloc(&d_w, n * sizeof(float)));
    CHECK(hipMalloc(&d_r1, n * sizeof(float)));
    CHECK(hipMalloc(&d_y1, n * sizeof(float)));
    CHECK(hipMalloc(&d_r2, n * sizeof(float)));
    CHECK(hipMalloc(&d_y2, n * sizeof(float)));
    CHECK(hipMemcpy(d_res, h_res.data(), n * sizeof(float),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_w, h_w.data(), n * sizeof(float), hipMemcpyHostToDevice));

    int bad = 0;
    std::vector<float> a_res(n), a_y(n), b_res(n), b_y(n);
    for (int round = 0; round < kRounds; ++round) {
      if (round > 0) {
        for (std::uint32_t i = 0; i < n; ++i) {
          h_res[i] = uni(rng);
          h_x[i] = uni(rng);
        }
        CHECK(hipMemcpy(d_res, h_res.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));
      }
      const float eps = 1e-6F;
      AddKernel<<<(n + 255U) / 256U, 256>>>(d_res, d_x, d_r1, n);
      RmsNormKernel<<<1, 256>>>(d_r1, d_w, eps, d_y1, n);
      AddRmsNormKernel<<<1, 256>>>(d_res, d_x, d_w, eps, d_r2, d_y2, n);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(a_res.data(), d_r1, n * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(a_y.data(), d_y1, n * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(b_res.data(), d_r2, n * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(b_y.data(), d_y2, n * sizeof(float),
                      hipMemcpyDeviceToHost));
      if (std::memcmp(a_res.data(), b_res.data(), n * sizeof(float)) != 0 ||
          std::memcmp(a_y.data(), b_y.data(), n * sizeof(float)) != 0) {
        ++bad;
      }
    }
    std::printf("n=%u rounds=%d bit-exact %s\n", n, kRounds,
                bad == 0 ? "YES" : "NO");

    // Launch-pair wall comparison (contended-tolerant: relative only).
    const int kIters = 2000;
    for (int pass = 0; pass < 3; ++pass) {
      hipEvent_t e0, e1;
      CHECK(hipEventCreate(&e0));
      CHECK(hipEventCreate(&e1));
      for (int i = 0; i < 100; ++i) {
        AddKernel<<<(n + 255U) / 256U, 256>>>(d_res, d_x, d_r1, n);
        RmsNormKernel<<<1, 256>>>(d_r1, d_w, 1e-6F, d_y1, n);
        AddRmsNormKernel<<<1, 256>>>(d_res, d_x, d_w, 1e-6F, d_r2, d_y2, n);
      }
      CHECK(hipDeviceSynchronize());
      float pair_ms = 0.0F, fused_ms = 0.0F;
      CHECK(hipEventRecord(e0));
      for (int i = 0; i < kIters; ++i) {
        AddKernel<<<(n + 255U) / 256U, 256>>>(d_res, d_x, d_r1, n);
        RmsNormKernel<<<1, 256>>>(d_r1, d_w, 1e-6F, d_y1, n);
      }
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      CHECK(hipEventElapsedTime(&pair_ms, e0, e1));
      CHECK(hipEventRecord(e0));
      for (int i = 0; i < kIters; ++i) {
        AddRmsNormKernel<<<1, 256>>>(d_res, d_x, d_w, 1e-6F, d_r2, d_y2, n);
      }
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      CHECK(hipEventElapsedTime(&fused_ms, e0, e1));
      std::printf(
          "n=%u pass=%d pair %6.2f us/fused %6.2f us/launch-pair delta %+6.2f "
          "us\n",
          n, pass, pair_ms * 1000.0F / kIters, fused_ms * 1000.0F / kIters,
          (fused_ms - pair_ms) * 1000.0F / kIters);
      CHECK(hipEventDestroy(e0));
      CHECK(hipEventDestroy(e1));
    }
    CHECK(hipFree(d_res));
    CHECK(hipFree(d_x));
    CHECK(hipFree(d_w));
    CHECK(hipFree(d_r1));
    CHECK(hipFree(d_y1));
    CHECK(hipFree(d_r2));
    CHECK(hipFree(d_y2));
  }
  return 0;
}

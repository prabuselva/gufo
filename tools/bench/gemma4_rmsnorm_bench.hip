// Standalone decode-shape RMSNorm bench for gemma4 (gfx1151).
// Decode calls RmsNormRows with rows=1, dim=2816 -> grid=1, block=256, and the
// production kernel accumulates the sum-of-squares in double (weak on RDNA3).
// This measures whether float accumulation / vectorized loads / a larger block
// cut the single-block kernel time, and whether the result stays inside the
// forward tolerance. Build: tools/bench/build.sh gemma4_rmsnorm_bench
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error %s at %s\n", hipGetErrorString(e), what);
    std::exit(1);
  }
}

__device__ double BlockReduceSumDouble(double v, double* reduce) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  if (lane == 0) {
    reduce[warp] = v;
  }
  __syncthreads();
  const std::uint32_t nw = blockDim.x >> 5;
  if (warp == 0) {
    v = lane < nw ? reduce[lane] : 0.0;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down(v, off);
    }
    if (lane == 0) {
      reduce[0] = v;
    }
  }
  __syncthreads();
  return reduce[0];
}

__device__ float BlockReduceSumFloat(float v, float* reduce) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  if (lane == 0) {
    reduce[warp] = v;
  }
  __syncthreads();
  const std::uint32_t nw = blockDim.x >> 5;
  if (warp == 0) {
    v = lane < nw ? reduce[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down(v, off);
    }
    if (lane == 0) {
      reduce[0] = v;
    }
  }
  __syncthreads();
  return reduce[0];
}

// Production: double accumulation, scalar strided loads.
__global__ void RmsNormDouble(const float* x, const float* gamma, float* out,
                              std::uint32_t dim, float eps) {
  __shared__ double reduce[32];
  const std::size_t row = blockIdx.x;
  const float* xr = x + row * dim;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const double v = xr[i];
    ss += v * v;
  }
  const float scale = 1.0F / sqrtf(static_cast<float>(
                          BlockReduceSumDouble(ss, reduce) / dim) + eps);
  float* orow = out + row * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = xr[i] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

// Float accumulation, scalar strided loads.
__global__ void RmsNormFloat(const float* x, const float* gamma, float* out,
                             std::uint32_t dim, float eps) {
  __shared__ float reduce[32];
  const std::size_t row = blockIdx.x;
  const float* xr = x + row * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xr[i];
    ss += v * v;
  }
  const float scale =
      1.0F / sqrtf(BlockReduceSumFloat(ss, reduce) / dim + eps);
  float* orow = out + row * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = xr[i] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

// Float accumulation, float4 loads/stores (dim % 4 == 0).
__global__ void RmsNormFloat4(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              float* __restrict__ out, std::uint32_t dim,
                              float eps) {
  __shared__ float reduce[32];
  const std::size_t row = blockIdx.x;
  const float4* xr = reinterpret_cast<const float4*>(x + row * dim);
  const std::uint32_t n4 = dim / 4U;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
    const float4 v = xr[i];
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  const float scale =
      1.0F / sqrtf(BlockReduceSumFloat(ss, reduce) / dim + eps);
  float4* orow = reinterpret_cast<float4*>(out + row * dim);
  const float4* gr = reinterpret_cast<const float4*>(gamma);
  for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 v = xr[i];
    const float4 g = gamma != nullptr ? gr[i] : make_float4(1, 1, 1, 1);
    v.x *= scale * g.x;
    v.y *= scale * g.y;
    v.z *= scale * g.z;
    v.w *= scale * g.w;
    orow[i] = v;
  }
}

template <typename F>
double Time(F&& fn, int iters = 500) {
  for (int i = 0; i < 50; ++i) fn();
  Check(hipDeviceSynchronize(), "sync");
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) fn();
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, a, b);
  hipEventDestroy(a);
  hipEventDestroy(b);
  return ms / iters;
}

double WorstRelative(const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = std::abs(static_cast<double>(ref[i]));
    const double d = std::abs(static_cast<double>(ref[i]) - got[i]);
    const double rel = d / (r > 1e-3 ? r : 1e-3);
    if (rel > worst) {
      worst = rel;
    }
  }
  return worst;
}

std::vector<float> Download(void* p, std::size_t n) {
  std::vector<float> h(n);
  Check(hipMemcpy(h.data(), p, n * sizeof(float), hipMemcpyDeviceToHost),
        "download");
  return h;
}

}  // namespace

int main() {
  constexpr std::uint32_t kDim = 2816;
  constexpr float kEps = 1e-6F;
  // Decode: rows=1. Prefill: rows=512 (both are exercised in the model).
  for (std::uint32_t rows : {1U, 512U}) {
    std::printf("=== rmsnorm rows=%u dim=%u ===\n", rows, kDim);
    std::vector<float> hx(rows * kDim), hg(kDim);
    std::uint32_t seed = 12345;
    auto rnd = [&] {
      seed = seed * 1664525U + 1013904223U;
      return (static_cast<int>((seed >> 16) & 0xFFFF) - 32768) / 32768.0F;
    };
    for (auto& v : hx) v = rnd();
    for (auto& v : hg) v = 0.5F + 0.5F * rnd();

    float *dx, *dg, *dout;
    Check(hipMalloc(&dx, hx.size() * 4), "malloc x");
    Check(hipMalloc(&dg, hg.size() * 4), "malloc g");
    Check(hipMalloc(&dout, hx.size() * 4), "malloc out");
    Check(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "cp x");
    Check(hipMemcpy(dg, hg.data(), hg.size() * 4, hipMemcpyHostToDevice), "cp g");

    // Double-oracle reference.
    std::vector<float> ref(rows * kDim);
    for (std::uint32_t r = 0; r < rows; ++r) {
      double ss = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        const double v = hx[static_cast<std::size_t>(r) * kDim + i];
        ss += v * v;
      }
      const float scale = 1.0F / sqrtf(static_cast<float>(ss / kDim) + kEps);
      for (std::uint32_t i = 0; i < kDim; ++i) {
        ref[static_cast<std::size_t>(r) * kDim + i] =
            hx[static_cast<std::size_t>(r) * kDim + i] * scale * hg[i];
      }
    }

    const dim3 b256(256);
    const dim3 b1024(1024);
    auto run = [&](auto launch) {
      launch();
      Check(hipDeviceSynchronize(), "run");
    };
    run([&] { RmsNormDouble<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double w_d = WorstRelative(ref, Download(dout, ref.size()));
    run([&] { RmsNormFloat<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double w_f = WorstRelative(ref, Download(dout, ref.size()));
    run([&] { RmsNormFloat4<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double w_f4 = WorstRelative(ref, Download(dout, ref.size()));
    run([&] { RmsNormFloat<<<rows, b1024>>>(dx, dg, dout, kDim, kEps); });
    double w_f1024 = WorstRelative(ref, Download(dout, ref.size()));

    std::printf("  worst-rel vs double: double %.2e  float %.2e  float4 %.2e  "
                "float@1024 %.2e\n",
                w_d, w_f, w_f4, w_f1024);
    double t_d = Time([&] { RmsNormDouble<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double t_f = Time([&] { RmsNormFloat<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double t_f4 = Time([&] { RmsNormFloat4<<<rows, b256>>>(dx, dg, dout, kDim, kEps); });
    double t_f1024 = Time([&] { RmsNormFloat<<<rows, b1024>>>(dx, dg, dout, kDim, kEps); });
    std::printf("  us/launch: double %.3f  float %.3f  float4 %.3f  float@1024 %.3f\n",
                t_d * 1000, t_f * 1000, t_f4 * 1000, t_f1024 * 1000);
    hipFree(dx);
    hipFree(dg);
    hipFree(dout);
  }
  return 0;
}
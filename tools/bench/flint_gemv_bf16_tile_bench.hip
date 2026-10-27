// A/B for a token-tiled GemvBatchBf16Kernel (flint router shape): the tiled
// kernel must be BITWISE identical to the warp-per-token kernel and faster.
//
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip \
//     tools/bench/flint_gemv_bf16_tile_bench.hip -o /tmp/flint_gemv_bf16
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                   \
  do {                                                                \
    const hipError_t err__ = (call);                                  \
    if (err__ != hipSuccess) {                                        \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                 \
                   hipGetErrorString(err__), __FILE__, __LINE__);     \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kTileTokens = 32;

[[nodiscard]] __device__ __forceinline__ float Bf16(std::uint16_t v) {
  union {
    std::uint32_t u;
    float f;
  } bits;
  bits.u = static_cast<std::uint32_t>(v) << 16;
  return bits.f;
}

__device__ __forceinline__ float WarpSum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v += __shfl_down_sync(__activemask(), v, off);
  }
  return v;
}

__global__ void Old(const std::uint16_t* __restrict__ w, std::uint32_t rows,
                    std::uint32_t cols, const float* __restrict__ x,
                    float* __restrict__ out) {
  const std::uint32_t token = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  x += static_cast<std::size_t>(token) * cols;
  out += static_cast<std::size_t>(token) * rows;
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * cols;
  float acc = 0.0F;
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    acc = fmaf(Bf16(wr[k]), x[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

__global__ void Tile(const std::uint16_t* __restrict__ w, std::uint32_t rows,
                     std::uint32_t cols, const float* __restrict__ x,
                     float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kTileTokens;
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * cols;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(row) + static_cast<std::size_t>(t0) * rows;
  float acc[kTileTokens];
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    acc[t] = 0.0F;
  }
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    const float wf = Bf16(wr[k]);
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      acc[t] = fmaf(wf, xt[static_cast<std::size_t>(t) * cols + k], acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<std::size_t>(t) * rows] = a;
    }
  }
}

template <typename K>
double TimeK(K kernel, int gy, const std::uint16_t* w, std::uint32_t rows,
             std::uint32_t cols, const float* x, float* out, int reps) {
  const dim3 grid(rows / (kThreads / kWarp), gy);
  kernel<<<grid, kThreads>>>(w, rows, cols, x, out);
  CHECK(hipDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    kernel<<<grid, kThreads>>>(w, rows, cols, x, out);
  }
  CHECK(hipDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

}  // namespace

int main() {
  const struct {
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t batch;
  } shapes[] = {{256, 2048, 2048}, {256, 2048, 1024}, {256, 2048, 256}};

  std::vector<std::uint16_t> hw;
  std::vector<float> hx;
  for (const auto& s : shapes) {
    hw.resize(static_cast<std::size_t>(s.rows) * s.cols);
    hx.resize(static_cast<std::size_t>(s.batch) * s.cols);
    std::uint32_t seed = 12345U;
    for (auto& v : hw) {
      seed = seed * 1664525U + 1013904223U;
      v = static_cast<std::uint16_t>(0x3F80U | ((seed >> 16) & 0x7FU));
    }
    for (auto& v : hx) {
      seed = seed * 1664525U + 1013904223U;
      v = ((seed >> 8) & 0xFFFF) / 32768.0F - 1.0F;
    }
    std::uint16_t* w = nullptr;
    float *x = nullptr, *o1 = nullptr, *o2 = nullptr;
    CHECK(hipMalloc(&w, hw.size() * 2));
    CHECK(hipMalloc(&x, hx.size() * 4));
    CHECK(hipMalloc(&o1, static_cast<std::size_t>(s.batch) * s.rows * 4));
    CHECK(hipMalloc(&o2, static_cast<std::size_t>(s.batch) * s.rows * 4));
    CHECK(hipMemcpy(w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));

    Old<<<dim3(s.rows / 8, s.batch), kThreads>>>(w, s.rows, s.cols, x, o1);
    Tile<<<dim3(s.rows / 8, s.batch / kTileTokens), kThreads>>>(
        w, s.rows, s.cols, x, o2);
    CHECK(hipDeviceSynchronize());
    std::vector<float> h1(static_cast<std::size_t>(s.batch) * s.rows),
        h2(h1.size());
    CHECK(hipMemcpy(h1.data(), o1, h1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h2.data(), o2, h2.size() * 4, hipMemcpyDeviceToHost));
    int mism = 0;
    for (std::size_t i = 0; i < h1.size(); ++i) {
      if (std::memcmp(&h1[i], &h2[i], 4) != 0) {
        ++mism;
      }
    }
    const double o =
        TimeK(Old, static_cast<int>(s.batch), w, s.rows, s.cols, x, o1, 20);
    const double t = TimeK(Tile, static_cast<int>(s.batch / kTileTokens), w,
                           s.rows, s.cols, x, o2, 20);
    std::printf(
        "rows=%u cols=%u B=%u bitwise_mismatches=%d old=%8.3f ms "
        "tile=%8.3f ms speedup=%.2fx\n",
        s.rows, s.cols, s.batch, mism, o, t, o / t);
    (void)hipFree(w);
    (void)hipFree(x);
    (void)hipFree(o1);
    (void)hipFree(o2);
  }
  return 0;
}

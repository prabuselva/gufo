// A/B for multi-row token tiles on the flint router shape (256 rows x 2048
// cols). Baseline Pipe = the 1-row x 32-token warp (C7n pipeline); variants
// Tile2<kRows, kTok> give one warp kRows rows x kTok tokens so a loaded x
// k-step feeds kRows accumulator chains. The pair-pipelined k loop keeps
// each (row, token) k-ascending fmaf chain and WarpSum tree, so every
// variant must be BITWISE identical to the warp-per-token kernel.
// Result (exclusive, gfx1151): t8x8 2.4x / t4x16 2.3x over Pipe at B=2048
// (0.41 vs 0.96 ms), flat win at B=256..8192, 0 mismatches; t8x8 shipped
// as GemvBatchBf16TileRKernel<8, 8>.
//
//   export PATH=/opt/rocm/lib/llvm/bin:$PATH   # provides ld.lld
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip \
//     tools/bench/flint_gemv_tile2_bench.hip -o /tmp/flint_gemv_tile2
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

// Reference: warp per (row, token), k-ascending chain.
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
  out += static_cast<std::uint32_t>(0) + static_cast<std::size_t>(token) * rows;
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

// Production baseline: 1 row x 32 tokens per warp, pair-pipelined k.
__global__ void Pipe(const std::uint16_t* __restrict__ w, std::uint32_t rows,
                     std::uint32_t cols, const float* __restrict__ x,
                     float* __restrict__ out) {
  constexpr int kTileTokens = 32;
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
  std::uint32_t k = lane;
  for (; k + kWarp < cols; k += 2 * kWarp) {
    float xv0[kTileTokens];
    float xv1[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      xv0[t] = xt[static_cast<std::size_t>(t) * cols + k];
      xv1[t] = xt[static_cast<size_t>(t) * cols + k + kWarp];
    }
    const float w0 = Bf16(wr[k]);
    const float w1 = Bf16(wr[k + kWarp]);
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      acc[t] = fmaf(w0, xv0[t], acc[t]);
      acc[t] = fmaf(w1, xv1[t], acc[t]);
    }
  }
  if (k < cols) {
    const float wf = Bf16(wr[k]);
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      acc[t] = fmaf(wf, xt[static_cast<size_t>(t) * cols + k], acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<size_t>(t) * rows] = a;
    }
  }
}

// Multi-row tile: warp owns RPW consecutive rows x TK consecutive tokens.
// rows/block = (kThreads / kWarp) * RPW; grid = (rows / that, B / TK).
template <int RPW, int TK>
__global__ void Tile2(const std::uint16_t* __restrict__ w, std::uint32_t rows,
                      std::uint32_t cols, const float* __restrict__ x,
                      float* __restrict__ out) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const std::uint32_t r0 = blockIdx.x * ((kThreads / kWarp) * RPW) +
                           warp * static_cast<std::uint32_t>(RPW);
  const std::uint32_t t0 = blockIdx.y * TK;
  if (r0 + RPW > rows) {
    return;
  }
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(r0) * cols;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(r0) + static_cast<std::size_t>(t0) * rows;
  float acc[RPW][TK];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
#pragma unroll
    for (int t = 0; t < TK; ++t) {
      acc[r][t] = 0.0F;
    }
  }
  std::uint32_t k = lane;
  for (; k + kWarp < cols; k += 2 * kWarp) {
    float xv0[TK];
    float xv1[TK];
#pragma unroll
    for (int t = 0; t < TK; ++t) {
      xv0[t] = xt[static_cast<size_t>(t) * cols + k];
      xv1[t] = xt[static_cast<size_t>(t) * cols + k + kWarp];
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float w0 = Bf16(wr[static_cast<size_t>(r) * cols + k]);
      const float w1 = Bf16(wr[static_cast<size_t>(r) * cols + k + kWarp]);
#pragma unroll
      for (int t = 0; t < TK; ++t) {
        acc[r][t] = fmaf(w0, xv0[t], acc[r][t]);
        acc[r][t] = fmaf(w1, xv1[t], acc[r][t]);
      }
    }
  }
  if (k < cols) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float wf = Bf16(wr[static_cast<size_t>(r) * cols + k]);
#pragma unroll
      for (int t = 0; t < TK; ++t) {
        acc[r][t] = fmaf(wf, xt[static_cast<size_t>(t) * cols + k], acc[r][t]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
#pragma unroll
    for (int t = 0; t < TK; ++t) {
      const float a = WarpSum(acc[r][t]);
      if (lane == 0) {
        ot[static_cast<size_t>(r) + static_cast<size_t>(t) * rows] = a;
      }
    }
  }
}

template <typename K>
double TimeK(K kernel, const dim3 grid, const std::uint16_t* w,
             std::uint32_t rows, std::uint32_t cols, const float* x,
             float* out, int reps) {
  kernel<<<grid, kThreads>>>(w, rows, cols, x, out);
  CHECK(hipGetLastError());
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
  CHECK(hipFree(nullptr));
  constexpr std::uint32_t kRows = 256;
  constexpr std::uint32_t kCols = 2048;
  for (const std::uint32_t batch : {256U, 1024U, 2048U, 8192U}) {
    std::vector<std::uint16_t> hw(kRows * kCols);
    std::vector<float> hx(static_cast<std::size_t>(batch) * kCols);
    for (std::size_t i = 0; i < hw.size(); ++i) {
      const float v = std::sin(0.0031F * static_cast<float>(i));
      const std::uint32_t u =
          *reinterpret_cast<const std::uint32_t*>(&v) & 0xFFFF0000U;
      hw[i] = static_cast<std::uint16_t>(u >> 16);
    }
    for (std::size_t i = 0; i < hx.size(); ++i) {
      hx[i] = std::cos(0.0113F * static_cast<float>(i)) * 3.0F;
    }
    std::uint16_t* dw;
    float *dx, *d_out, *d_ref;
    CHECK(hipMalloc(&dw, sizeof(std::uint16_t) * hw.size()));
    CHECK(hipMalloc(&dx, sizeof(float) * hx.size()));
    CHECK(hipMalloc(&d_out, sizeof(float) * batch * kRows));
    CHECK(hipMalloc(&d_ref, sizeof(float) * batch * kRows));
    CHECK(hipMemcpy(dw, hw.data(), sizeof(std::uint16_t) * hw.size(),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * hx.size(),
                    hipMemcpyHostToDevice));
    // Reference logits.
    Old<<<dim3(kRows / (kThreads / kWarp), batch), kThreads>>>(dw, kRows, kCols,
                                                               dx, d_ref);
    CHECK(hipDeviceSynchronize());
    // Launch each variant with its own grid; re-time and verify.
    auto run = [&](const char* name, auto kernel, unsigned gx, unsigned gy) {
      const dim3 grid(gx, gy);
      const double ms = TimeK(kernel, grid, dw, kRows, kCols, dx, d_out, 20);
      CHECK(hipMemset(d_out, 0, sizeof(float) * batch * kRows));
      kernel<<<grid, kThreads>>>(dw, kRows, kCols, dx, d_out);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      std::vector<float> ref(static_cast<std::size_t>(batch) * kRows);
      std::vector<float> got(ref.size());
      CHECK(hipMemcpy(ref.data(), d_ref, sizeof(float) * ref.size(),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(got.data(), d_out, sizeof(float) * got.size(),
                      hipMemcpyDeviceToHost));
      long mism = 0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
          ++mism;
        }
      }
      std::printf("B=%5u %-12s %8.3f ms  mismatches=%ld\n", batch, name, ms,
                  mism);
    };
    run("pipe32x1", Pipe, kRows / 8, batch / 32);
    run("t4x8", Tile2<4, 8>, kRows / 32, batch / 8);
    run("t4x16", Tile2<4, 16>, kRows / 32, batch / 16);
    run("t2x16", Tile2<2, 16>, kRows / 16, batch / 16);
    run("t8x8", Tile2<8, 8>, kRows / 64, batch / 8);
    run("t2x8", Tile2<2, 8>, kRows / 16, batch / 8);
    run("t4x4", Tile2<4, 4>, kRows / 32, batch / 4);
    hipFree(dw);
    hipFree(dx);
    hipFree(d_out);
    hipFree(d_ref);
  }
  return 0;
}

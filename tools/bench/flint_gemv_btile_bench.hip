// Flint prefill linear tier: production GemvBatchQ8 (one block per row x
// token, so every token re-reads the whole weight matrix) versus warp-level
// token tiling that loads each Q8 weight fragment once and feeds T
// independent per-token accumulator chains. The tiling must stay bitwise
// identical to the production kernel: same ascending-strided block order per
// token, same float4 x loads with j-ascending dot order, same WarpSum tree.
//
//   tools/bench/build.sh flint_gemv_btile_bench && /tmp/flint_gemv_btile_bench
//
// Run through tools/bench/gpu_exclusive.sh: it still occupies real bandwidth.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr std::uint32_t kQ8Block = 32;
constexpr std::size_t kQ8BlockBytes = 34;
constexpr int kRot = 3;  // weight rotation sets to defeat the 32 MiB MALL

#define CK(cmd)                                                     \
  do {                                                              \
    hipError_t e_ = (cmd);                                          \
    if (e_ != hipSuccess) {                                         \
      std::fprintf(stderr, "HIP %s: %s\n", #cmd,                    \
                   hipGetErrorString(e_));                          \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

[[nodiscard]] __device__ __forceinline__ float Q8Scale(const std::uint8_t* bp) {
  union {
    std::uint16_t u;
    __half h;
  } bits;
  bits.u = static_cast<std::uint16_t>(bp[0]) |
           (static_cast<std::uint16_t>(bp[1]) << 8);
  return __half2float(bits.h);
}

__device__ __forceinline__ float WarpSum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v += __shfl_down_sync(__activemask(), v, off);
  }
  return v;
}

__device__ __forceinline__ float Q8DotBlock(const std::int8_t* __restrict__ qs,
                                            const float* __restrict__ xk) {
  float part = 0.0F;
#pragma unroll
  for (std::uint32_t q = 0; q < kQ8Block / 4U; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(xk + q * 4U);
    part += static_cast<float>(qs[q * 4U + 0U]) * v.x;
    part += static_cast<float>(qs[q * 4U + 1U]) * v.y;
    part += static_cast<float>(qs[q * 4U + 2U]) * v.z;
    part += static_cast<float>(qs[q * 4U + 3U]) * v.w;
  }
  return part;
}

// ---- production reference (identical to GemvBatchQ8Kernel) ----------------
__global__ void Baseline(const std::uint8_t* __restrict__ w, std::uint32_t rows,
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
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc = 0.0F;
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
    const float* __restrict__ xk = x + static_cast<std::size_t>(b) * kQ8Block;
    acc += d * Q8DotBlock(qs, xk);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

// ---- tiled variant: full tiles only, T tokens per warp --------------------
// Per token the accumulation sequence is exactly the baseline's: b strides
// lane, lane+32, ...; the dot reuses the hoisted exact float(qs) values in
// j-ascending order; WarpSum is the same tree, computed last per token.
template <int T>
__global__ void Tiled(const std::uint8_t* __restrict__ w, std::uint32_t rows,
                      std::uint32_t cols, const float* __restrict__ x,
                      float* __restrict__ out, std::uint32_t tokens) {
  const std::uint32_t tile = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint32_t t0 = tile * static_cast<std::uint32_t>(T);
  if (t0 + T > tokens) {
    return;  // tails run through the baseline kernel
  }
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(row) + static_cast<std::size_t>(t0) * rows;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    acc[t] = 0.0F;
  }
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      acc[t] += d * Q8DotBlock(qs, xt + static_cast<std::size_t>(t) * cols +
                                  static_cast<std::size_t>(b) * kQ8Block);
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<std::size_t>(t) * rows] = a;
    }
  }
}

// ---- tiled + hoisted-float(qs) variant ------------------------------------
// float(qs[j]) is exact and identical to the value the baseline converts per
// token, but here each conversion happens once per weight byte instead of
// once per (byte, token). Per token the b-order, j-order, d*x products, FMA
// contraction shape and WarpSum tree stay identical.
template <int T>
__global__ void Hoist(const std::uint8_t* __restrict__ w, std::uint32_t rows,
                      std::uint32_t cols, const float* __restrict__ x,
                      float* __restrict__ out, std::uint32_t tokens) {
  const std::uint32_t tile = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint32_t t0 = tile * static_cast<std::uint32_t>(T);
  if (t0 + T > tokens) {
    return;
  }
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(row) + static_cast<std::size_t>(t0) * rows;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    acc[t] = 0.0F;
  }
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
    float qf[kQ8Block];
#pragma unroll
    for (std::uint32_t q = 0; q < kQ8Block; ++q) {
      qf[q] = static_cast<float>(qs[q]);
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const float* __restrict__ xk =
          xt + static_cast<std::size_t>(t) * cols +
          static_cast<std::size_t>(b) * kQ8Block;
      float part = 0.0F;
#pragma unroll
      for (std::uint32_t q = 0; q < kQ8Block / 4U; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(xk + q * 4U);
        part += qf[q * 4U + 0U] * v.x;
        part += qf[q * 4U + 1U] * v.y;
        part += qf[q * 4U + 2U] * v.z;
        part += qf[q * 4U + 3U] * v.w;
      }
      acc[t] += d * part;
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<std::size_t>(t) * rows] = a;
    }
  }
}

struct Diff {
  int count;
  int idx[4];
  float a[4];
  float b[4];
};

__global__ void Compare(const float* a, const float* b, std::size_t n,
                        Diff* diff) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a[i], 4);
    std::memcpy(&ub, &b[i], 4);
    if (ua != ub) {
      const int slot = atomicAdd(&diff->count, 1);
      if (slot < 4) {
        diff->idx[slot] = static_cast<int>(i);
        diff->a[slot] = a[i];
        diff->b[slot] = b[i];
      }
    }
  }
}

struct Shape {
  std::uint32_t rows;
  std::uint32_t cols;
  std::uint32_t batch;
};

float BenchKernel(void (*launch)(const std::uint8_t*, std::uint32_t,
                                 std::uint32_t, const float*, float*,
                                 std::uint32_t, hipStream_t),
                  const std::vector<const std::uint8_t*>& ws, const float* x,
                  float* out, const Shape& s) {
  const int reps = 5;
  hipEvent_t ev0, ev1;
  CK(hipEventCreate(&ev0));
  CK(hipEventCreate(&ev1));
  launch(ws[0], s.rows, s.cols, x, out, s.batch, nullptr);  // warm
  CK(hipDeviceSynchronize());
  float best = 1e30F;
  for (int r = 0; r < reps; ++r) {
    const std::uint8_t* w = ws[r % kRot];
    CK(hipEventRecord(ev0));
    launch(w, s.rows, s.cols, x, out, s.batch, nullptr);
    CK(hipEventRecord(ev1));
    CK(hipEventSynchronize(ev1));
    float ms;
    CK(hipEventElapsedTime(&ms, ev0, ev1));
    if (ms < best) {
      best = ms;
    }
  }
  CK(hipEventDestroy(ev0));
  CK(hipEventDestroy(ev1));
  return best;
}

template <int T>
void LaunchTiled(const std::uint8_t* w, std::uint32_t rows, std::uint32_t cols,
                 const float* x, float* out, std::uint32_t batch,
                 hipStream_t s) {
  const dim3 grid((rows + kThreads / kWarp - 1) / (kThreads / kWarp),
                  batch / T);
  Tiled<T><<<grid, kThreads, 0, s>>>(w, rows, cols, x, out, batch);
}

template <int T>
void LaunchHoist(const std::uint8_t* w, std::uint32_t rows, std::uint32_t cols,
                 const float* x, float* out, std::uint32_t batch,
                 hipStream_t s) {
  const dim3 grid((rows + kThreads / kWarp - 1) / (kThreads / kWarp),
                  batch / T);
  Hoist<T><<<grid, kThreads, 0, s>>>(w, rows, cols, x, out, batch);
}

void LaunchBaseline(const std::uint8_t* w, std::uint32_t rows,
                    std::uint32_t cols, const float* x, float* out,
                    std::uint32_t batch, hipStream_t s) {
  const dim3 grid((rows + kThreads / kWarp - 1) / (kThreads / kWarp), batch);
  Baseline<<<grid, kThreads, 0, s>>>(w, rows, cols, x, out);
}

}  // namespace

int main() {
  const Shape shapes[] = {
      {4096, 2048, 2048},  // q_proj
      {2048, 4096, 2048},  // o_proj
      {2048, 2048, 2048},  // dense out / shared mid
      {512, 2048, 2048},   // k_proj / v_proj
  };
  int dev = 0;
  CK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  CK(hipGetDeviceProperties(&prop, dev));
  const double peak_fmas = static_cast<double>(prop.multiProcessorCount) *
                           128.0 * prop.clockRate * 1e3;  // f32 FMAs/s
  std::printf("device %s  CUs=%d  clock=%.0f MHz  f32 peak ~%.1f TFMA/s\n\n",
              prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0,
              peak_fmas * 1e-12);

  Diff* diff = nullptr;
  CK(hipMalloc(&diff, sizeof(Diff)));
  std::uint32_t rng = 0x9e3779b9U;
  int bad = 0;
  std::printf("%-16s %-12s %10s %10s %8s  %s\n", "shape", "variant", "us",
              "floor us", "x-speed", "bitwise");
  for (const Shape& s : shapes) {
    const std::size_t wset = static_cast<std::size_t>(s.rows) *
                             (s.cols / kQ8Block) * kQ8BlockBytes;
    std::vector<const std::uint8_t*> ws(kRot);
    std::vector<std::uint8_t> hw(wset);
    for (int r = 0; r < kRot; ++r) {
      for (std::size_t i = 0; i < wset; ++i) {
        rng = rng * 1664525U + 1013904223U;
        hw[i] = static_cast<std::uint8_t>((rng >> 24) ^ (r * 13U));
      }
      // Fix the f16 scale heads to normal magnitudes (bit pattern is random
      // enough for the contract; only NaN/denormal heads would be odd).
      for (std::size_t b = 0; b < wset / kQ8BlockBytes; ++b) {
        hw[b * kQ8BlockBytes] = 0x00;
        hw[b * kQ8BlockBytes + 1] =
          static_cast<std::uint8_t>(0x38 + ((rng >> (b % 24)) & 0x07));
      }
      void* p = nullptr;
      CK(hipMalloc(&p, wset));
      CK(hipMemcpy(p, hw.data(), wset, hipMemcpyHostToDevice));
      ws[r] = static_cast<const std::uint8_t*>(p);
    }
    const std::size_t xsz = static_cast<std::size_t>(s.batch) * s.cols * 4;
    const std::size_t osz = static_cast<std::size_t>(s.batch) * s.rows * 4;
    float* x = nullptr;
    float *oref = nullptr, *ocand = nullptr;
    CK(hipMalloc(&x, xsz));
    CK(hipMalloc(&oref, osz));
    CK(hipMalloc(&ocand, osz));
    std::vector<float> hx(xsz / 4);
    for (std::size_t i = 0; i < hx.size(); ++i) {
      rng = rng * 1664525U + 1013904223U;
      hx[i] = ((rng >> 8) & 0xFFFF) * 0.00025F - 8.0F;
    }
    CK(hipMemcpy(x, hx.data(), xsz, hipMemcpyHostToDevice));
    CK(hipMemset(oref, 0, osz));

    const double fmas = static_cast<double>(s.rows) * s.cols * s.batch;
    const double floor_us = fmas / peak_fmas * 1e6;

    LaunchBaseline(ws[0], s.rows, s.cols, x, oref, s.batch, nullptr);
    CK(hipDeviceSynchronize());
    const float base_us =
        BenchKernel(LaunchBaseline, ws, x, oref, s) * 1000.0F;
    // BenchKernel rotates weights; regenerate the reference with ws[0].
    LaunchBaseline(ws[0], s.rows, s.cols, x, oref, s.batch, nullptr);
    CK(hipDeviceSynchronize());
    char name[32];

#define VARIANT(LABEL, LAUNCH, TV)                                             \
  {                                                                         \
    std::snprintf(name, sizeof name, "%s", LABEL);                          \
    CK(hipMemset(ocand, 0, osz));                                           \
    LAUNCH(ws[0], s.rows, s.cols, x, ocand, s.batch, nullptr);              \
    CK(hipGetLastError());                                                  \
    CK(hipDeviceSynchronize());                                             \
    CK(hipMemset(diff, 0, sizeof(Diff)));                                   \
    Compare<<<static_cast<unsigned>((osz / 4 + 255) / 256), 256>>>(oref,    \
                                                           ocand, osz / 4,  \
                                                           diff);           \
    Diff hd;                                                                \
    hd.count = 0;                                                           \
    CK(hipMemcpy(&hd, diff, sizeof(Diff), hipMemcpyDeviceToHost));          \
    const int hflag = hd.count != 0 ? 1 : 0;                                \
    const float us = BenchKernel(LAUNCH, ws, x, ocand, s) * 1000.0F;        \
    std::printf("%-16s %-12s %10.1f %10.1f %7.2fx  %s (n=%d/%zu)\n", buf_,  \
                name, us, floor_us, base_us / us,                            \
                hflag ? "MISMATCH" : "bitwise", hd.count, osz / 4);          \
    for (int z = 0; z < 4 && z < hd.count; ++z) {                           \
      std::printf("    diff idx=%d token=%d row=%d base=%.9g cand=%.9g\n",  \
                  hd.idx[z], hd.idx[z] / static_cast<int>(s.rows),          \
                  hd.idx[z] % static_cast<int>(s.rows), hd.a[z], hd.b[z]);  \
    }                                                                       \
    if (hflag) {                                                            \
      std::vector<float> hc(std::min<std::size_t>(osz, 64 * 4), 0.0F);      \
      CK(hipMemcpy(hc.data(), ocand, hc.size() * 4, hipMemcpyDeviceToHost));\
      std::printf("    ocand[0..7]:");                                      \
      for (int z = 0; z < 8; ++z) std::printf(" %.6g", hc[z]);              \
      CK(hipMemcpy(hc.data(), x, hc.size() * 4, hipMemcpyDeviceToHost));    \
      std::printf("  x[0..7]:");                                            \
      for (int z = 0; z < 8; ++z) std::printf(" %.6g", hc[z]);              \
      std::printf("  grid=(%u,%u)\n", (s.rows + 7) / 8, s.batch / TV);      \
    }                                                                       \
    bad += hflag;                                                           \
  }
    char buf_[32];
    std::snprintf(buf_, sizeof buf_, "%ux%u B%u", s.rows, s.cols, s.batch);
    VARIANT("control base", LaunchBaseline, 1)
    VARIANT("tiled T4", LaunchTiled<4>, 4)
    VARIANT("tiled T32", LaunchTiled<32>, 32)
    VARIANT("hoist T8", LaunchHoist<8>, 8)
    VARIANT("hoist T16", LaunchHoist<16>, 16)
    VARIANT("hoist T32", LaunchHoist<32>, 32)
    CK(hipFree(x));
    CK(hipFree(oref));
    CK(hipFree(ocand));
    for (int r = 0; r < kRot; ++r) {
      CK(hipFree(const_cast<std::uint8_t*>(ws[r])));
    }
  }
  CK(hipFree(diff));
  std::printf("\n%s\n", bad == 0 ? "ALL BITWISE OK" : "BITWISE FAILURE");
  return bad == 0 ? 0 : 1;
}

// Nemotron Bolt Q8_0 decode GEMV bench (gfx1151). Decode runs the dense
// Q8_0 roles (attn q/out 4096x4096, latent down/up, kv 256x4096, lm_head
// 131072x4096) through GemvF32PackedKernel: thread-owned chunk
// c = lane + 32*warp_in_row, wpr = ceil(k/32/32), ONE chunk per thread, a
// 2B f16 scale built from two byte loads plus 32 scalar int8 loads and 32
// scalar float x loads. k=4096 gives n_chunk=128 -> 4 warps/row, zero ILP per
// thread - the same latency structure the K-quant tiers had before the
// dual-chain win d29d0e8a/c74ec948. This bench ports that lever:
//
//   v0     production replica: byte qs loads, chunk-per-thread (baseline).
//   vread  pure weight-read sum at the production addresses (roof).
//   v1     dual chain with the byte loader: warps_per_row halved, each thread
//          walks two independent chunk chains 32*wpr apart per iteration.
//   v2     aligned u64 funnel loader (5 u64 + funnel re-align + 8 float4),
//          v0 geometry: same j accumulation order as v0 (must be bitwise).
//          Q8_0 blocks are 34 B (2B scale + 32B qs), so both the scale and
//          the qs window unaligned-load through 16 B-page-resident funnels.
//   v3     v2 loader + dual chain (the combined candidate).
//   v5     v2 loader + 4 chunk chains (warps_per_row/4, deep-ILP probe).
//
// The dual chain changes the cross-chunk accumulation order (ULP-scale, same
// contract as the K-quant rows); v2 is checked bitwise against v0.
//
//   tools/bench/build.sh nem_decode_q8_bench && ./nem_decode_q8_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cmd)                                                     \
  do {                                                                 \
    hipError_t err__ = (cmd);                                          \
    if (err__ != hipSuccess) {                                         \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err__), \
              __FILE__, __LINE__);                                     \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;
constexpr int kQ8Elems = 32;
constexpr int kQ8Block = 34;

__device__ __forceinline__ uint64_t LoadU64(const uint8_t* p) {
  return *reinterpret_cast<const uint64_t*>(p);
}

__device__ __forceinline__ float Fp16(uint16_t h) {
  __half_raw r;
  r.x = h;
  return __half2float(*reinterpret_cast<const __half*>(&r));
}

// v0/v1 chunk: exact GemvChunkDot<kNblGemvQ8_0> from nb_ops.hip.cpp.
__device__ __forceinline__ float ChunkDotBytes(const uint8_t* __restrict__ w,
                                               const float* __restrict__ x,
                                               int chunk) {
  const uint8_t* blk = w + static_cast<int64_t>(chunk) * kQ8Block;
  const uint16_t dh =
      static_cast<uint16_t>(blk[0]) | static_cast<uint16_t>(blk[1] << 8);
  const float d = Fp16(dh);
  const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
  const float* xv = x + chunk * kQ8Elems;
  float acc = 0.0F;
#pragma unroll
  for (int j = 0; j < kQ8Elems; ++j) {
    acc += static_cast<float>(qs[j]) * d * xv[j];
  }
  return acc;
}

// v2/v3/v5 chunk: 5 aligned u64 loads re-aligned by funnel for the 32 qs
// bytes, 2 more for the 16 B scale window, x as 8 float4. The accumulation
// walks j in the same order as ChunkDotBytes (bitwise).
__device__ __forceinline__ float ChunkDotFunnel(const uint8_t* __restrict__ w,
                                                const float* __restrict__ x,
                                                int chunk) {
  const uint8_t* blk = w + static_cast<int64_t>(chunk) * kQ8Block;
  const uintptr_t sa = reinterpret_cast<uintptr_t>(blk);
  const uint64_t* sw = reinterpret_cast<const uint64_t*>(sa & ~uintptr_t(7));
  const uint32_t ssh = static_cast<uint32_t>(sa & 7U) * 8U;
  const uint64_t s0 = LoadU64(reinterpret_cast<const uint8_t*>(sw));
  const uint64_t s1 = LoadU64(reinterpret_cast<const uint8_t*>(sw) + 8);
  const uint64_t spair =
      (ssh == 0U) ? s0 : ((s0 >> ssh) | (s1 << (64 - ssh)));
  const float d = Fp16(static_cast<uint16_t>(spair & 0xFFFFU));

  const uint8_t* p = blk + 2;
  const uintptr_t qa = reinterpret_cast<uintptr_t>(p);
  const uint64_t* qw = reinterpret_cast<const uint64_t*>(qa & ~uintptr_t(7));
  const uint32_t qsh = static_cast<uint32_t>(qa & 7U) * 8U;
  uint64_t e[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    e[i] = LoadU64(reinterpret_cast<const uint8_t*>(qw) + i * 8);
  }
  uint64_t q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = (qsh == 0U) ? e[i] : ((e[i] >> qsh) | (e[i + 1] << (64 - qsh)));
  }
  const float4* xv = reinterpret_cast<const float4*>(x + chunk * kQ8Elems);
  float acc = 0.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = i * 8 + b;
      const int8_t v = static_cast<int8_t>((q[i] >> (8 * b)) & 0xFFU);
      const float4 x4 = xv[j / 4];
      const float xl = (&x4.x)[j & 3];
      acc += static_cast<float>(v) * d * xl;
    }
  }
  return acc;
}

// Pure read roof: touch exactly the v0 bytes, no math.
__device__ __forceinline__ uint32_t ChunkRead(const uint8_t* __restrict__ w,
                                              int chunk) {
  const uint8_t* blk = w + static_cast<int64_t>(chunk) * kQ8Block;
  uint32_t s = 0;
#pragma unroll
  for (int j = 0; j < kQ8Block; ++j) {
    s += blk[j];
  }
  return s;
}

template<int VAR>
__global__ void Q8GemvKernel(const float* __restrict__ x,
                             const uint8_t* __restrict__ w,
                             float* __restrict__ out, int n_out, int k,
                             int stride, int warps_per_row,
                             int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[64];
  if (static_cast<int>(threadIdx.x) < 64) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int row =
      static_cast<int>(blockIdx.x) * (kBlock / warpSize / warps_per_row) +
      row_local;
  const bool active = row < n_out;
  float acc = 0.0F;
  if (active) {
    const uint8_t* wr = w + static_cast<int64_t>(row) * stride;
    const float* xr = x;
    const int n_chunk = k / kQ8Elems;
    if constexpr (VAR == 0 || VAR == 2) {
      const int c = lane + warpSize * warp_in_row;
      if (c < n_chunk) {
        acc = (VAR == 0) ? ChunkDotBytes(wr, xr, c)
                         : ChunkDotFunnel(wr, xr, c);
      }
    } else if constexpr (VAR == 99) {
      uint32_t s = 0;
      for (int c = lane + warpSize * warp_in_row; c < n_chunk;
           c += warps_per_row * warpSize) {
        s += ChunkRead(wr, c);
      }
      acc = static_cast<float>(s);
    } else {
      // VAR 1: byte loader, 2 chains per iteration; VAR 3: funnel + 2;
      // VAR 5: funnel + 4 chains per iteration.
      constexpr int kChains = (VAR == 5) ? 4 : 2;
      using DotFn = float (*)(const uint8_t*, const float*, int);
      int c = lane + warpSize * warp_in_row;
      const int step = warps_per_row * warpSize;
      for (; c + (kChains - 1) * step < n_chunk; c += kChains * step) {
        float a[kChains];
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
          a[i] = (VAR == 1) ? ChunkDotBytes(wr, xr, c + i * step)
                            : ChunkDotFunnel(wr, xr, c + i * step);
        }
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
          acc += a[i];
        }
      }
      for (; c < n_chunk; c += step) {
        acc += (VAR == 1) ? ChunkDotBytes(wr, xr, c) : ChunkDotFunnel(wr, xr, c);
      }
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kBlock / warpSize; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[row] = sum;
  }
}

// ------------------------------------------------------------- launcher

constexpr int BaseWpr(int k) { return (k / kQ8Elems + kWarp - 1) / kWarp; }
constexpr int RowsPerBlock(int wpr) { return kBlock / (kWarp * wpr); }

// VAR 1/3: warps_per_row/2 (>=1); VAR 5: /4 (>=1), i.e. 2/4 chunks per
// thread per iteration on k=4096 rows.
template<int VAR>
constexpr int VarWpr(int k) {
  const int base = BaseWpr(k);
  if constexpr (VAR == 1 || VAR == 3) {
    return base > 1 ? base / 2 : 1;
  } else if constexpr (VAR == 5) {
    return base > 3 ? base / 4 : 1;
  } else {
    return base;
  }
}

template<int VAR>
void LaunchOne(const float* x, const uint8_t* w, float* out, int n_out, int k,
               hipStream_t stream) {
  const int stride = (k / kQ8Elems) * kQ8Block;
  const int wpr = VarWpr<VAR>(k);
  const int rpb = RowsPerBlock(wpr);
  const int blocks = (n_out + rpb - 1) / rpb;
  Q8GemvKernel<VAR><<<blocks, kBlock, 0, stream>>>(x, w, out, n_out, k, stride,
                                                   wpr, blocks);
}

// ---------------------------------------------------------------- harness

float RandUnit() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5F;
}

struct Shape {
  const char* name;
  int n_out;
  int k;
};

constexpr Shape kShapes[] = {
    {"lm_head   131072x4096", 131072, 4096},
    {"attn      4096x4096", 4096, 4096},
    {"lat_down  1024x4096", 1024, 4096},
    {"lat_up    4096x1024", 4096, 1024},
    {"kv         256x4096", 256, 4096},
};

void RunShape(const Shape& sh) {
  if (sh.k % kQ8Elems != 0) return;
  const int stride = (sh.k / kQ8Elems) * kQ8Block;
  const size_t wbytes = static_cast<size_t>(sh.n_out) * stride;
  // Slack so the funnel windows at the last block stay inside the
  // allocation (mirrors page-rounded GGUF storage).
  const size_t slack = 128;
  uint8_t* w = nullptr;
  float* x = nullptr;
  float* out0 = nullptr;
  float* outv = nullptr;
  CHECK(hipMalloc(&w, wbytes + slack));
  CHECK(hipMalloc(&x, static_cast<size_t>(sh.k) * sizeof(float)));
  CHECK(hipMalloc(&out0, static_cast<size_t>(sh.n_out) * sizeof(float)));
  CHECK(hipMalloc(&outv, static_cast<size_t>(sh.n_out) * sizeof(float)));
  std::vector<uint8_t> hw(wbytes);
  for (size_t i = 0; i < wbytes; ++i) {
    hw[i] = static_cast<uint8_t>(rand() & 0xFF);
  }
  for (size_t blk = 0; blk < wbytes; blk += kQ8Block) {
    hw[blk + 0] = 0x77;
    hw[blk + 1] = 0x3D;  // d ~ 0.06
  }
  std::vector<float> hx(sh.k);
  for (auto& v : hx) v = RandUnit();
  CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(x, hx.data(), hx.size() * sizeof(float),
                  hipMemcpyHostToDevice));

  hipEvent_t ev0;
  hipEvent_t ev1;
  CHECK(hipEventCreate(&ev0));
  CHECK(hipEventCreate(&ev1));
  const int reps = 11;
  auto timed = [&](auto launch) {
    for (int i = 0; i < 3; ++i) launch();
    CHECK(hipDeviceSynchronize());
    float tot = 0.0F;
    for (int i = 0; i < reps; ++i) {
      CHECK(hipEventRecord(ev0));
      launch();
      CHECK(hipEventRecord(ev1));
      CHECK(hipEventSynchronize(ev1));
      float ms = 0.0F;
      CHECK(hipEventElapsedTime(&ms, ev0, ev1));
      tot += ms;
    }
    return tot / reps;
  };
  const float base_ms =
      timed([&] { LaunchOne<0>(x, w, out0, sh.n_out, sh.k, 0); });
  printf("  %-24s v0 %8.3f ms %7.1f GB/s\n", sh.name, base_ms,
         wbytes / (base_ms * 1e6F));

  // DRAM regime: rotate across >512 MB of weight parks; SPREAD: scatter the
  // parks over a faulted 20 GB span (see nem_decode_kq_bench.hip).
  uint8_t* wp = nullptr;
  size_t npark = (std::max<size_t>(2, (size_t)544 << 20) + wbytes - 1) / wbytes;
  if (npark > 64) npark = 64;
  CHECK(hipMalloc(&wp, npark * (wbytes + slack)));
  for (size_t p = 0; p < npark; ++p) {
    CHECK(hipMemcpy(wp + p * (wbytes + slack), hw.data(), wbytes,
                    hipMemcpyHostToDevice));
  }
  auto timed_rot = [&](auto launch) {
    for (int i = 0; i < 2; ++i) launch(i % npark);
    CHECK(hipDeviceSynchronize());
    float tot = 0.0F;
    for (int i = 0; i < reps; ++i) {
      CHECK(hipEventRecord(ev0));
      launch(i % npark);
      CHECK(hipEventRecord(ev1));
      CHECK(hipEventSynchronize(ev1));
      float ms = 0.0F;
      CHECK(hipEventElapsedTime(&ms, ev0, ev1));
      tot += ms;
    }
    return tot / reps;
  };
  const float v0d = timed_rot([&](size_t p) {
    LaunchOne<0>(x, wp + p * (wbytes + slack), out0, sh.n_out, sh.k, 0);
  });
  printf("  %-24s v0DRAM %8.3f ms %7.1f GB/s\n", sh.name, v0d,
         wbytes / (v0d * 1e6F));
  uint8_t* ws = nullptr;
  const size_t span = (size_t)20 << 30;
  if (hipMalloc(&ws, span) == hipSuccess) {
    CHECK(hipMemset(ws, 0, span));
    auto off = [&](size_t p, size_t seed) {
      size_t h = (p + 1) * (seed * 2654435761UL);
      return (h % (((size_t)18 << 30) / ((size_t)32 << 20))) *
             ((size_t)32 << 20);
    };
    float v0s = timed_rot([&](size_t p) {
      LaunchOne<0>(x, ws + off(p, 7), out0, sh.n_out, sh.k, 0);
    });
    float v1s = timed_rot([&](size_t p) {
      LaunchOne<1>(x, ws + off(p, 13), outv, sh.n_out, sh.k, 0);
    });
    float v3s = timed_rot([&](size_t p) {
      LaunchOne<3>(x, ws + off(p, 17), outv, sh.n_out, sh.k, 0);
    });
    printf("  %-24s v0SPREAD %7.3f ms %6.1f GB/s  v1SPREAD %7.3f %6.1f  "
           "v3SPREAD %7.3f %6.1f GB/s\n",
           sh.name, v0s, wbytes / (v0s * 1e6F), v1s, wbytes / (v1s * 1e6F),
           v3s, wbytes / (v3s * 1e6F));
    CHECK(hipFree(ws));
  }
  float v1d = timed_rot([&](size_t p) {
    LaunchOne<1>(x, wp + p * (wbytes + slack), outv, sh.n_out, sh.k, 0);
  });
  float v2d = timed_rot([&](size_t p) {
    LaunchOne<2>(x, wp + p * (wbytes + slack), outv, sh.n_out, sh.k, 0);
  });
  float v3d = timed_rot([&](size_t p) {
    LaunchOne<3>(x, wp + p * (wbytes + slack), outv, sh.n_out, sh.k, 0);
  });
  float v5d = timed_rot([&](size_t p) {
    LaunchOne<5>(x, wp + p * (wbytes + slack), outv, sh.n_out, sh.k, 0);
  });
  printf("  %-24s v1DRAM %7.3f %6.1f GB/s (%.3fx)  v2DRAM %7.3f %6.1f (%.3fx)"
         "  v3DRAM %7.3f %6.1f (%.3fx)  v5DRAM %7.3f %6.1f (%.3fx)\n",
         sh.name, v1d, wbytes / (v1d * 1e6F), v0d / v1d, v2d,
         wbytes / (v2d * 1e6F), v0d / v2d, v3d, wbytes / (v3d * 1e6F),
         v0d / v3d, v5d, wbytes / (v5d * 1e6F), v0d / v5d);

  auto F16 = [](uint16_t h) {
    if (((h >> 10) & 0x1FU) == 0) return static_cast<float>(h & 0x3FFU) * (1.0f / 1048576.0f);
    const uint32_t bits = (static_cast<uint32_t>(h & 0x8000U) << 16) |
                          ((((h >> 10) & 0x1FU) + 112U) << 23) |
                          (static_cast<uint32_t>(h & 0x3FFU) << 13);
    float v;
    __builtin_memcpy(&v, &bits, 4);
    return v;
  };
  auto RefRow = [&](int r) {
    const uint8_t* wr = hw.data() + static_cast<size_t>(r) * stride;
    double acc = 0.0;
    for (int c = 0; c < sh.k / kQ8Elems; ++c) {
      const uint16_t dh = static_cast<uint16_t>(wr[c * kQ8Block]) |
                          (static_cast<uint16_t>(wr[c * kQ8Block + 1]) << 8);
      const float d = F16(dh);
      const int8_t* qs = reinterpret_cast<const int8_t*>(wr + c * kQ8Block + 2);
      for (int j = 0; j < kQ8Elems; ++j) {
        acc += static_cast<double>(qs[j]) * d * hx[c * kQ8Elems + j];
      }
    }
    return acc;
  };
  auto check_v = [&](auto launch, const char* tag, bool bitwise) {
    // Regenerate the v0 baseline from the hot weight buffer: the timed
    // rotations above left out0 holding the zeroed SPREAD probe's output.
    LaunchOne<0>(x, w, out0, sh.n_out, sh.k, 0);
    CHECK(hipMemsetAsync(outv, 0, sh.n_out * sizeof(float)));
    launch();
    CHECK(hipDeviceSynchronize());
    std::vector<float> a(sh.n_out);
    std::vector<float> b(sh.n_out);
    CHECK(hipMemcpy(a.data(), out0, a.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(b.data(), outv, b.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    double worst = 0.0;
    bool exact = true;
    int bad_a = 0;
    int bad_b = 0;
    int first_bad = -1;
    for (size_t i = 0; i < a.size(); ++i) {
      const double ref = RefRow(static_cast<int>(i));
      if (std::fabs(a[i] - ref) > 2e-2 * (std::fabs(ref) + 1.0)) {
        ++bad_a;
        if (first_bad < 0) first_bad = static_cast<int>(i);
      }
      if (std::fabs(b[i] - ref) > 2e-2 * (std::fabs(ref) + 1.0)) ++bad_b;
      exact = exact && (a.data()[i] == b[i]);
      const double denom = std::fabs(a[i]) + 1e-2;
      worst = std::max(worst, std::fabs(a[i] - b[i]) / denom);
    }
    printf("  %-24s correctness %s rel=%.2e bad_a=%d bad_b=%d first_bad=%d "
           "a=%g b=%g ref=%g%s%s\n",
           "", tag, worst, bad_a, bad_b, first_bad, a[first_bad < 0 ? 0 : first_bad],
           b[first_bad < 0 ? 0 : first_bad],
           RefRow(first_bad < 0 ? 0 : first_bad),
           bitwise ? (exact ? " bitwise" : "  <-- NOT BITWISE") : "",
           (!bitwise && worst >= 1e-3) ? "  <-- TOO LARGE" : "");
  };

#define RUNVAR(V)                                                      \
  {                                                                    \
    const float ms =                                                   \
        timed([&] { LaunchOne<V>(x, w, outv, sh.n_out, sh.k, 0); });   \
    printf("  %-24s v%-2d %7.3f ms %7.1f GB/s (%.3fx)\n", sh.name, (V),       \
           ms, wbytes / (ms * 1e6F), base_ms / ms);                    \
  }
  RUNVAR(99)
  RUNVAR(1)
  RUNVAR(2)
  RUNVAR(3)
  RUNVAR(5)
  check_v(
      [&] { LaunchOne<1>(x, w, outv, sh.n_out, sh.k, 0); },
      "v1", false);
  check_v(
      [&] { LaunchOne<2>(x, w, outv, sh.n_out, sh.k, 0); },
      "v2", true);
  check_v(
      [&] { LaunchOne<3>(x, w, outv, sh.n_out, sh.k, 0); },
      "v3", false);
  check_v(
      [&] { LaunchOne<5>(x, w, outv, sh.n_out, sh.k, 0); },
      "v5", false);

  CHECK(hipEventDestroy(ev0));
  CHECK(hipEventDestroy(ev1));
  CHECK(hipFree(w));
  CHECK(hipFree(x));
  CHECK(hipFree(out0));
  CHECK(hipFree(outv));
  CHECK(hipFree(wp));
}

}  // namespace

int main() {
  srand(1234);
  int dev = 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device: %s\n", prop.name);
  for (const Shape& sh : kShapes) RunShape(sh);
  return 0;
}

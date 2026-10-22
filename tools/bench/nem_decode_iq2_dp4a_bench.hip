// Nemotron Bolt IQ2_XXS decode GEMV: production packed-float kernel vs a
// llama.cpp b11721-style integer path (gfx1151). The candidate quantizes the
// f32 activation once into q8_1 sub-blocks (d = amax/127, q = round(x/d),
// ds = (__half2){d, sum} per 32 values) and replaces the 32-float select/cvt
// dot with 4x(vcmpne4/vsub4 sign expansion + SUDOT4 dp4a), matching
// vec_dot_iq2_xxs_q8_1. Weight addressing, the lane->chunk ownership and the
// fold tree are copied from GemvF32PackedKernel<2>, so the only deltas are
// (a) 1 B/sub-block activation reads instead of 4 B floats and (b) integer
// inner products. Outputs are NOT bitwise: correctness compares max relative
// error against the float kernel (quantization error is the mechanism).
//
//   tools/bench/build.sh nem_decode_iq2_dp4a_bench && \
//   /tmp/nem_decode_iq2_dp4a_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/nemotron_h_moe/cpu_tables.hpp"

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

using gufo::models::nemotron_h_moe::cpu::kIq2XxsGrid;
using gufo::models::nemotron_h_moe::cpu::kSignsIq2Xs;

constexpr int kWarp = 32;
constexpr int kIq2Sub = 32;
constexpr int kMaxRowWarps = 8;

__device__ __constant__ uint64_t cIq2Grid[256];
__device__ __constant__ uint8_t cIq2Signs[128];
// [aux] -> XOR of the grid word with its byte-negated word. With the
// 0xFF-byte sign-mask table, the signed grid word is R = G ^ (m & X):
// byte-lane select of g and -g with no per-lane subtraction or carry.
__device__ __constant__ uint64_t cIq2Xor[256];
// [7-bit sign index] -> {byte i = 0xFF iff element i (i=0..3) flips,
//                        byte i = 0xFF iff element i+4 flips}: spreads the
// packed uint8 sign bits of kSignsIq2Xs for the XOR-negate grid trick,
// replacing llama's __vcmpne4 on a per-byte sign word.
__device__ __constant__ uint2 cIq2SignMasks[128];

// 36-byte q8_1 sub-block, same layout as ggml block_q8_1.
struct Q8Block {
  __half2 ds;
  int8_t qs[kIq2Sub];
};
static_assert(sizeof(Q8Block) == 36, "q8_1 must stay 36 bytes");

// ---------------------------------------------------------------------------
// Signed int8x4 dot via the native unsigned byte dot; LLVM inserts the sign
// fix-ups for the signed grid bytes (same lowering as the vendored
// __vsubss4/vcmpeq4 path used by the mmq iq2 kernels).
__device__ __forceinline__ int Dp4a(int a, int b, int c) {
#if defined(__GFX11__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c += static_cast<int>(pa[i]) * static_cast<int>(pb[i]);
  }
  return c;
#endif
}

// ---------------------------------------------------------------------------
// Baseline: production chunk dot + packed kernel (nb_ops.hip.cpp copy).
__device__ __forceinline__ float ChunkDotF32(const uint8_t* __restrict__ w,
                                             const float* __restrict__ x,
                                             int chunk) {
  float acc = 0.0F;
  const uint8_t* blk = w + static_cast<int64_t>(chunk / 8) * 66;
  uint16_t dh = 0;
  uint32_t aux32[2] = {0, 0};
  __builtin_memcpy(&dh, blk, sizeof(dh));
  __builtin_memcpy(aux32, blk + 2 + (chunk % 8) * 8, sizeof(aux32));
  const float d = __half2float(*reinterpret_cast<__half*>(&dh));
  const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
  const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
  const float* xv = x + chunk * kIq2Sub;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint64_t grid = cIq2Grid[aux8[l]];
    const uint8_t signs = cIq2Signs[(aux32[1] >> (7 * l)) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int8_t v = static_cast<int8_t>((grid >> (8 * j)) & 0xFF);
      acc += db * static_cast<float>(v) *
             ((signs & (1u << j)) != 0 ? -xv[l * 8 + j] : xv[l * 8 + j]);
    }
  }
  return acc;
}

// Fold shared by both kernels: 5-step shfl_down tree + seeded ascending
// 8-entry sum, exactly like GemvF32PackedKernel.
template <typename DotFn>
__global__ void PackedKernel(const void* __restrict__ x,
                             const uint8_t* __restrict__ w,
                             float* __restrict__ out, int n_out, int k,
                             int expert_stride,
                             const int32_t* __restrict__ ids, int slots,
                             int64_t x_stride, int warps_per_row,
                             int blocks_per_slot, DotFn dot) {
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[kMaxRowWarps * 8];
  if (static_cast<int>(threadIdx.x) < kMaxRowWarps * 8) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int gid = static_cast<int>(blockIdx.x);
  const int slot = gid / blocks_per_slot;
  const int row = (gid % blocks_per_slot) * (static_cast<int>(blockDim.x) /
                                             kWarp / warps_per_row) +
                  row_local;
  const bool active = (row < n_out);
  const int token = static_cast<int>(blockIdx.y);
  float acc = 0.0F;
  int64_t out_row = 0;
  if (active) {
    const char* xr = reinterpret_cast<const char*>(x);
    int64_t w_row = row;
    out_row = static_cast<int64_t>(token) * n_out + row;
    if (slots > 1) {
      const int64_t s = static_cast<int64_t>(token) * slots + slot;
      w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
      out_row = s * n_out + row;
      xr += static_cast<int64_t>(slot) * x_stride;
    }
    xr += static_cast<int64_t>(token) *
          ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                       : 0);
    const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
    const int c = lane + kWarp * warp_in_row;
    if (c < k / kIq2Sub) {
      acc = dot(wr, xr, c);
    }
  }
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kMaxRowWarps; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[out_row] = sum;
  }
}

struct DotF32 {
  __device__ float operator()(const uint8_t* w, const char* x, int c) const {
    return ChunkDotF32(w, reinterpret_cast<const float*>(x), c);
  }
};

// Candidate: llama b11721 vec_dot_iq2_xxs_q8_1 on the production word layout.
struct DotDp4a {
  __device__ float operator()(const uint8_t* w, const char* xq, int chunk) const {
    const uint8_t* blk = w + static_cast<int64_t>(chunk / 8) * 66;
    uint16_t dh = 0;
    uint32_t aux32[2] = {0, 0};
    __builtin_memcpy(&dh, blk, sizeof(dh));
    __builtin_memcpy(aux32, blk + 2 + (chunk % 8) * 8, sizeof(aux32));
    const float d = __half2float(*reinterpret_cast<__half*>(&dh));
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
    const Q8Block* xb = reinterpret_cast<const Q8Block*>(xq) + chunk;
    const float da = __low2float(xb->ds);
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint64_t grid = cIq2Grid[aux8[l]];
      const uint64_t xw = cIq2Xor[aux8[l]];
      const uint2 msk = cIq2SignMasks[(aux32[1] >> (7 * l)) & 127];
      const int lo = static_cast<int>(static_cast<uint32_t>(grid) ^
                                      (msk.x & static_cast<uint32_t>(xw)));
      const int hi = static_cast<int>(static_cast<uint32_t>(grid >> 32) ^
                                      (msk.y & static_cast<uint32_t>(xw >> 32)));
      const int8_t* q = xb->qs + l * 8;
      sumi = Dp4a(lo, *reinterpret_cast<const int*>(q), sumi);
      sumi = Dp4a(hi, *reinterpret_cast<const int*>(q + 4), sumi);
    }
    const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
    return da * db * static_cast<float>(sumi);
  }
};

// ---------------------------------------------------------------------------
// Quantize rows x k f32 into packed q8_1 sub-blocks (one warp per sub-block).
__global__ void QuantQ8_1(const float* __restrict__ x, Q8Block* __restrict__ q,
                          int64_t sub_blocks) {
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t sub = gtid / kWarp;
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  if (sub >= sub_blocks) {
    return;
  }
  const float* xb = x + sub * kIq2Sub;
  float v = xb[lane];
  float a = fabsf(v);
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    a = fmaxf(a, __shfl_xor_sync(~0ULL, a, off));
  }
  const float d = a / 127.0F;
  const float id = a != 0.0F ? 127.0F / a : 0.0F;
  const int8_t qv =
      a == 0.0F ? 0 : static_cast<int8_t>(roundf(v * id));  // llama parity
  int sum = static_cast<int>(qv);
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    sum += __shfl_xor_sync(~0ULL, sum, off);
  }
  q[sub].qs[lane] = qv;
  if (lane == 0) {
    q[sub].ds = __floats2half2_rn(d, static_cast<float>(sum));
  }
}

// ---------------------------------------------------------------------------
// Host harness (same shape list / id rotation as nem_decode_iq2_bench).
struct Shape {
  const char* name;
  int n_out;
  int k;
  int n_experts;
  int slots;
  int64_t x_stride;  // 0 => latent shared across slots
};

uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

void run_shape(const Shape& sh, int reps) {
  const int row_bytes = sh.k / 256 * 66;
  const int64_t w_bytes =
      static_cast<int64_t>(sh.n_experts) * sh.n_out * row_bytes;
  const int64_t x_rows = sh.x_stride > 0 ? sh.slots : 1;
  const int64_t x_len = sh.x_stride > 0 ? sh.x_stride : sh.k;
  const int64_t x_elems = x_rows * x_len;
  const int64_t out_elems = static_cast<int64_t>(sh.slots) * sh.n_out;
  constexpr int n_sets = 64;

  uint8_t* w;
  float* x;
  Q8Block* xq;
  float* out_ref;
  float* out_new;
  int32_t* ids;
  CHECK(hipMalloc(&w, static_cast<size_t>(w_bytes)));
  CHECK(hipMalloc(&x, static_cast<size_t>(x_elems) * 4));
  CHECK(hipMalloc(&xq, static_cast<size_t>(x_elems + x_elems / 8)));
  CHECK(hipMalloc(&out_ref, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMalloc(&out_new, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMalloc(&ids, static_cast<size_t>(n_sets) * sh.slots * 4));

  std::vector<uint8_t> hw(static_cast<size_t>(w_bytes));
  uint32_t seed = 0x1234u;
  for (size_t i = 0; i < hw.size(); ++i) {
    hw[i] = static_cast<uint8_t>(lcg(seed) & 0xFF);
  }
  // Keep the f16 scale of every 256-value block a normal ±(1..2) value so
  // neither engine sees NaN/Inf in random weight bytes.
  for (int64_t blk = 0; blk < w_bytes; blk += 66) {
    hw[static_cast<size_t>(blk)] = static_cast<uint8_t>(0x00 | (lcg(seed) & 0x3Fu) << 2);
    hw[static_cast<size_t>(blk) + 1] =
        static_cast<uint8_t>(0x3C | ((lcg(seed) & 1u) << 7));
  }
  std::vector<float> hx(static_cast<size_t>(x_elems));
  for (size_t i = 0; i < hx.size(); ++i) {
    hx[i] = (static_cast<float>(lcg(seed)) / 16777216.0F) - 1.0F;
  }
  std::vector<int32_t> hids(static_cast<size_t>(n_sets) * sh.slots);
  for (int s = 0; s < n_sets; ++s) {
    for (int j = 0; j < sh.slots; ++j) {
      hids[static_cast<size_t>(s) * sh.slots + j] =
          static_cast<int32_t>(lcg(seed) % static_cast<uint32_t>(sh.n_experts));
    }
  }
  CHECK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(x, hx.data(), hx.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ids, hids.data(), hids.size() * 4, hipMemcpyHostToDevice));

  const int n_chunk = sh.k / kIq2Sub;
  const int R = (n_chunk + 31) / 32;
  const int rpb = (R >= kMaxRowWarps) ? 1 : (kMaxRowWarps / R);
  const int threads = kWarp * R * rpb;
  const int blocks_per_slot = (sh.n_out + rpb - 1) / rpb;
  // The templated kernel walks x as raw bytes, so both engines pass the slot
  // row stride in bytes: f32 rows are 4 B/value, q8_1 rows 1 B/value + a
  // 4-byte ds per 32 values. token stays 0 (grid.y == 1), so the token term
  // never fires, exactly like the production t == 1 decode.
  const int64_t f32_stride = sh.x_stride * 4;
  const int64_t q8_stride =
      sh.x_stride > 0 ? sh.x_stride + sh.x_stride / 8 : 0;

  auto launch_packed = [&](int ids_off, float* out) {
    const dim3 grid(static_cast<unsigned>(sh.slots) * blocks_per_slot, 1);
    const dim3 block(static_cast<unsigned>(threads), 1);
    PackedKernel<<<grid, block>>>(x, w, out, sh.n_out, sh.k, row_bytes,
                                  ids + ids_off, sh.slots, f32_stride, R,
                                  blocks_per_slot, DotF32{});
  };
  auto quantize = [&]() {
    const int64_t subs = x_rows * (x_len / kIq2Sub);
    const int64_t blocks = (subs * kWarp + 255) / 256;
    QuantQ8_1<<<static_cast<unsigned>(blocks), 256>>>(x, xq, subs);
  };
  auto launch_dp4a = [&](int ids_off, float* out) {
    const dim3 grid(static_cast<unsigned>(sh.slots) * blocks_per_slot, 1);
    const dim3 block(static_cast<unsigned>(threads), 1);
    PackedKernel<<<grid, block>>>(xq, w, out, sh.n_out, sh.k, row_bytes,
                                  ids + ids_off, sh.slots, q8_stride, R,
                                  blocks_per_slot, DotDp4a{});
  };

  // Correctness: candidate vs production-float reference.
  launch_packed(0, out_ref);
  quantize();
  launch_dp4a(0, out_new);
  CHECK(hipDeviceSynchronize());
  std::vector<float> href(static_cast<size_t>(out_elems)),
      hnew(static_cast<size_t>(out_elems));
  CHECK(hipMemcpy(href.data(), out_ref, href.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(hnew.data(), out_new, hnew.size() * 4, hipMemcpyDeviceToHost));
  float max_ref = 0.0F, max_abs = 0.0F;
  for (size_t i = 0; i < href.size(); ++i) {
    max_ref = fmaxf(max_ref, fabsf(href[i]));
    max_abs = fmaxf(max_abs, fabsf(hnew[i] - href[i]));
  }
  const float rel = max_abs / fmaxf(max_ref, 1e-6F);

  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto bench = [&](auto launch) {
    for (int i = 0; i < 3; ++i) {
      launch(0, out_new);
    }
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
      launch((i % n_sets) * sh.slots, (i % 2 == 0) ? out_new : out_ref);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) * 1000.0 / reps;
  };
  auto quant_then_launch = [&](int ids_off, float* out) {
    quantize();
    launch_dp4a(ids_off, out);
  };
  const double us_f32 = bench(launch_packed);
  const double us_q = bench([&](int, float*) { quantize(); });
  const double us_dp = bench(quant_then_launch);
  const double mb =
      static_cast<double>(sh.slots) * sh.n_out * row_bytes / 1e6;
  printf("%-16s slots=%d n=%d k=%d | f32 %8.2f us (%5.1f GB/s) | q+dp4a %8.2f us "
         "(q %6.2f) | %5.2fx | rel %.4g %s\n",
         sh.name, sh.slots, sh.n_out, sh.k, us_f32, mb / us_f32 * 1000.0,
         us_dp, us_q, us_f32 / us_dp, rel, rel <= 0.03f ? "OK" : "FAIL");

  CHECK(hipEventDestroy(e0));
  CHECK(hipEventDestroy(e1));
  CHECK(hipFree(w));
  CHECK(hipFree(x));
  CHECK(hipFree(xq));
  CHECK(hipFree(out_ref));
  CHECK(hipFree(out_new));
  CHECK(hipFree(ids));
}

}  // namespace

int main() {
  int warp = 0;
  CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, 0));
  if (warp != kWarp) {
    fprintf(stderr, "bench requires wave32 (device warp %d)\n", warp);
    return 1;
  }
  CHECK(hipMemcpyToSymbol(cIq2Grid, kIq2XxsGrid, sizeof(kIq2XxsGrid)));
  uint64_t xors[256];
  for (int i = 0; i < 256; ++i) {
    uint64_t g = kIq2XxsGrid[i], n = 0;
    for (int j = 0; j < 8; ++j) {
      const uint8_t v = static_cast<uint8_t>(kIq2XxsGrid[i] >> (8 * j));
      n |= static_cast<uint64_t>(static_cast<uint8_t>(0u - v)) << (8 * j);
    }
    xors[i] = g ^ n;
  }
  CHECK(hipMemcpyToSymbol(cIq2Xor, xors, sizeof(xors)));
  CHECK(hipMemcpyToSymbol(cIq2Signs, kSignsIq2Xs, sizeof(kSignsIq2Xs)));
  uint2 masks[128];
  for (int i = 0; i < 128; ++i) {
    const uint32_t s = kSignsIq2Xs[i];
    uint32_t lo = 0, hi = 0;
    for (int j = 0; j < 4; ++j) {
      lo |= (((s >> j) & 1u) ? 0xFFu : 0u) << (8 * j);
      hi |= (((s >> (j + 4)) & 1u) ? 0xFFu : 0u) << (8 * j);
    }
    masks[i] = make_uint2(lo, hi);
  }
  CHECK(hipMemcpyToSymbol(cIq2SignMasks, masks, sizeof(masks)));
  const int reps = 30;
  const Shape shapes[] = {
      {"up-ff1536-e240", 1536, 1024, 240, 8, 0},
      {"up-ff2048-e240", 2048, 1024, 240, 8, 0},
      {"up-ff2560-e368", 2560, 1024, 368, 8, 0},
      {"down-ff1536-e240", 1024, 1536, 240, 8, 1536},
      {"down-ff2048-e240", 1024, 2048, 240, 8, 2048},
      {"down-ff2560-e368", 1024, 2560, 368, 8, 2560},
      {"shup-5376", 5376, 1024, 1, 1, 0},
      {"shdown-5376", 1024, 5376, 1, 1, 5376},
  };
  for (const Shape& s : shapes) {
    run_shape(s, reps);
  }
  return 0;
}

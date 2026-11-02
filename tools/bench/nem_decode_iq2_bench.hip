// Nemotron Bolt IQ2_XXS MoE GEMV bench (gfx1151). Decodes run
// GemvF32Kernel<kNblGemvIq2Xxs> with a 256-thread block per output row and one
// 32-value chunk per thread, so k=1024 (latent) activates only 32 of 256
// threads and the up-projection launches n_out*slots tiny blocks. Probe
// row-packed variants that keep the exact lane->chunk ownership and the exact
// fold order (5-step shfl_down tree per 32-lane group, then the seeded
// ascending warp sum, padding lanes contribute +0.0F), so every output stays
// bitwise identical to the production kernel.
//
//   tools/bench/build.sh nem_decode_iq2_bench && /tmp/nem_decode_iq2_bench

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

constexpr int kBlock = 256;
constexpr int kIq2Sub = 32;     // 8 B of codes per 32-value sub-block
constexpr int kWarp = 32;
constexpr int kMaxRowWarps = 8;

__device__ __constant__ uint64_t cIq2Grid[256];
__device__ __constant__ uint8_t cIq2Signs[128];
__device__ uint64_t* g_grid64 = nullptr;  // V2 global table alternatives
__device__ uint8_t* g_signs_global = nullptr;

__device__ __forceinline__ float BlockReduceSum(float acc) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  __shared__ float warp[8];
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  if (lane == 0) {
    warp[threadIdx.x / kWarp] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kBlock / kWarp; ++i) {
      sum += warp[i];
    }
    return sum;
  }
  return 0.0F;
}

// Production IQ2 chunk dot (nb_ops.hip.cpp GemvChunkDot<kNblGemvIq2Xxs>).
__device__ __forceinline__ float ChunkDot(const uint8_t* __restrict__ w,
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

// Production kernel, IQ2 branch, verbatim structure.
__global__ void GemvIq2(const float* __restrict__ x,
                        const uint8_t* __restrict__ w, float* __restrict__ out,
                        int n_out, int k, int expert_stride,
                        const int32_t* __restrict__ ids, int slots,
                        int64_t x_stride, int tokens) {
  const int row = static_cast<int>(blockIdx.x % static_cast<unsigned>(n_out));
  const int token = static_cast<int>(blockIdx.y);
  int64_t w_row = row;
  int64_t out_row = static_cast<int64_t>(token) * n_out + row;
  if (slots > 1) {
    const int slot = static_cast<int>(blockIdx.x / n_out);
    const int64_t s = static_cast<int64_t>(token) * slots + slot;
    w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
    out_row = s * n_out + row;
    x += static_cast<int64_t>(slot) * x_stride;
  }
  x += static_cast<int64_t>(token) *
       ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                    : k);
  const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
  const int n_chunk = k / kIq2Sub;
  float acc = 0.0F;
  for (int c = static_cast<int>(threadIdx.x); c < n_chunk;
       c += static_cast<int>(blockDim.x)) {
    acc += ChunkDot(wr, x, c);
  }
  acc = BlockReduceSum(acc);
  if (threadIdx.x == 0) {
    out[out_row] = acc;
  }
}

// Aux/decode + dot with a single u64 aux load. Same arithmetic and operand
// order as ChunkDot; only the weight-load instructions change.
__device__ __forceinline__ float ChunkDot64(const uint8_t* __restrict__ w,
                                            const float* __restrict__ x,
                                            int chunk, bool tbl_const) {
  float acc = 0.0F;
  const uint8_t* blk = w + static_cast<int64_t>(chunk / 8) * 66;
  uint16_t dh = 0;
  uint64_t aux64 = 0;
  __builtin_memcpy(&dh, blk, sizeof(dh));
  __builtin_memcpy(&aux64, blk + 2 + (chunk % 8) * 8, sizeof(aux64));
  const float d = __half2float(*reinterpret_cast<__half*>(&dh));
  const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(&aux64);
  // Little-endian: aux64 bytes 0..3 are aux32[0], bytes 4..7 aux32[1].
  const uint32_t hi = static_cast<uint32_t>(aux64 >> 32);
  const float db = d * (0.5F + static_cast<float>(hi >> 28)) * 0.25F;
  const float* xv = x + chunk * kIq2Sub;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint64_t grid =
        tbl_const ? cIq2Grid[aux8[l]] : g_grid64[aux8[l]];
    const uint8_t signs =
        tbl_const ? cIq2Signs[(hi >> (7 * l)) & 127]
                  : g_signs_global[(hi >> (7 * l)) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int8_t v = static_cast<int8_t>((grid >> (8 * j)) & 0xFF);
      acc += db * static_cast<float>(v) *
             ((signs & (1u << j)) != 0 ? -xv[l * 8 + j] : xv[l * 8 + j]);
    }
  }
  return acc;
}

// Row-packed kernel. A row owns R = ceil(n_chunk/32) consecutive warps of the
// block and a block owns rpb rows: blockDim = 32*R*rpb (<= 256 for decode
// shapes). Warp i lane l of a row owns chunk 32*i + l -- the same chunk the
// production thread 32*i + l owned -- so the per-warp shfl_down tree, the
// seeded ascending 8-entry sum and the +0.0F padding reproduce
// BlockReduceSum bitwise. Valid for n_chunk <= 256.
template<bool kTableConst>
__global__ void GemvIq2Packed(const float* __restrict__ x,
                              const uint8_t* __restrict__ w,
                              float* __restrict__ out, int n_out, int k,
                              int expert_stride,
                              const int32_t* __restrict__ ids, int slots,
                              int64_t x_stride, int tokens, int warps_per_row,
                              int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[kMaxRowWarps * 8];
  if (threadIdx.x < kMaxRowWarps * 8) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();

  const int gid = static_cast<int>(blockIdx.x);
  const int slot = gid / blocks_per_slot;
  const int row =
      (gid % blocks_per_slot) * (static_cast<int>(blockDim.x) / kWarp /
                                 warps_per_row) +
      row_local;
  const bool active = (row < n_out);
  const int token = static_cast<int>(blockIdx.y);

  float acc = 0.0F;
  int64_t out_row = 0;
  if (active) {
    const float* xr = x;
    int64_t w_row = row;
    out_row = static_cast<int64_t>(token) * n_out + row;
    if (slots > 1) {
      const int64_t s = static_cast<int64_t>(token) * slots + slot;
      w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
      out_row = s * n_out + row;
      xr += static_cast<int64_t>(slot) * x_stride;
    }
    xr += static_cast<int64_t>(token) * ((slots > 1 && x_stride > 0)
                                             ? static_cast<int64_t>(slots) *
                                                   x_stride
                                             : k);
    const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
    const int n_chunk = k / kIq2Sub;
    const int c = lane + kWarp * warp_in_row;
    if (c < n_chunk) {
      acc = ChunkDot64(wr, xr, c, kTableConst);
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
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[out_row] = sum;
  }
}

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

struct Run {
  uint8_t* w = nullptr;
  float* x = nullptr;
  float* out_ref = nullptr;
  float* out_new = nullptr;
  int32_t* ids = nullptr;  // sets * slots
  int64_t w_bytes = 0;
  int row_bytes = 0;
  int n_sets = 64;
};

void run_shape(const Shape& sh, bool tbl_const, int reps) {
  Run r;
  r.row_bytes = sh.k / 256 * 66;
  r.w_bytes = static_cast<int64_t>(sh.n_experts) * sh.n_out * r.row_bytes;
  const int64_t n_out64 = sh.n_out;
  const int64_t out_elems =
      static_cast<int64_t>(sh.slots > 1 ? sh.slots : 1) * n_out64;
  const int64_t x_elems =
      sh.x_stride > 0 ? sh.slots * sh.x_stride : static_cast<int64_t>(sh.k);
  CHECK(hipMalloc(&r.w, static_cast<size_t>(r.w_bytes)));
  CHECK(hipMalloc(&r.x, static_cast<size_t>(x_elems) * 4));
  CHECK(hipMalloc(&r.out_ref, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMalloc(&r.out_new, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMalloc(&r.ids, static_cast<size_t>(r.n_sets) * sh.slots * 4));

  // Host-side random fill.
  std::vector<uint8_t> hw(static_cast<size_t>(r.w_bytes));
  uint32_t seed = 0x1234u;
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = static_cast<uint8_t>(lcg(seed) & 0xFF);
  std::vector<float> hx(static_cast<size_t>(x_elems));
  for (size_t i = 0; i < hx.size(); ++i) {
    hx[i] = (static_cast<float>(lcg(seed)) / 16777216.0F) - 1.0F;
  }
  std::vector<int32_t> hids(static_cast<size_t>(r.n_sets) * sh.slots);
  for (int s = 0; s < r.n_sets; ++s) {
    for (int j = 0; j < sh.slots; ++j) {
      hids[static_cast<size_t>(s) * sh.slots + j] =
          static_cast<int32_t>(lcg(seed) % static_cast<uint32_t>(sh.n_experts));
    }
  }
  CHECK(hipMemcpy(r.w, hw.data(), hw.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(r.x, hx.data(), hx.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(r.ids, hids.data(), hids.size() * 4, hipMemcpyHostToDevice));

  const int n_chunk = sh.k / kIq2Sub;
  const int R = (n_chunk + 31) / 32;
  const int rpb_rounded = (R >= 8) ? 1 : (8 / R);
  const int threads = 32 * R * rpb_rounded;
  const int blocks_per_slot = (sh.n_out + rpb_rounded - 1) / rpb_rounded;

  auto launch_base = [&](int ids_off, float* out) {
    const dim3 grid(static_cast<unsigned>(sh.slots) * static_cast<unsigned>(sh.n_out), 1);
    const dim3 block(kBlock, 1);
    GemvIq2<<<grid, block>>>(r.x, r.w, out, sh.n_out, sh.k, r.row_bytes,
                             r.ids + ids_off, sh.slots, sh.x_stride, 1);
  };
  auto launch_packed = [&](int ids_off, float* out) {
    const dim3 grid(static_cast<unsigned>(sh.slots) *
                        static_cast<unsigned>(blocks_per_slot),
                    1);
    const dim3 block(static_cast<unsigned>(threads), 1);
    if (tbl_const) {
      GemvIq2Packed<true><<<grid, block>>>(r.x, r.w, out, sh.n_out, sh.k,
                                           r.row_bytes, r.ids + ids_off,
                                           sh.slots, sh.x_stride, 1, R,
                                           blocks_per_slot);
    } else {
      GemvIq2Packed<false><<<grid, block>>>(r.x, r.w, out, sh.n_out, sh.k,
                                            r.row_bytes, r.ids + ids_off,
                                            sh.slots, sh.x_stride, 1, R,
                                            blocks_per_slot);
    }
  };

  // Bitwise check with a fixed id set.
  CHECK(hipMemset(r.out_ref, 0xFF, static_cast<size_t>(out_elems) * 4));
  CHECK(hipMemset(r.out_new, 0xFF, static_cast<size_t>(out_elems) * 4));
  launch_base(0, r.out_ref);
  launch_packed(0, r.out_new);
  CHECK(hipDeviceSynchronize());
  std::vector<float> href(static_cast<size_t>(out_elems)),
      hnew(static_cast<size_t>(out_elems));
  CHECK(hipMemcpy(href.data(), r.out_ref, href.size() * 4,
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(hnew.data(), r.out_new, hnew.size() * 4,
                  hipMemcpyDeviceToHost));
  bool equal = std::memcmp(href.data(), hnew.data(), href.size() * 4) == 0;

  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto bench = [&](auto launch) {
    for (int i = 0; i < 3; ++i) launch(0, r.out_new);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
      launch((i % r.n_sets) * sh.slots, (i % 2 == 0) ? r.out_new : r.out_ref);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) * 1000.0 / reps;
  };
  const double us_base = bench(launch_base);
  const double us_pack = bench(launch_packed);
  const double mb = static_cast<double>(sh.slots) * sh.n_out * r.row_bytes /
                    1e6;
  printf("%-16s slots=%d k=%d n=%d | base %8.2f us (%6.1f GB/s) | packed %8.2f us (%6.1f GB/s) | %5.2fx | bitwise %s\n",
         sh.name, sh.slots, sh.k, sh.n_out, us_base,
         mb / us_base * 1000.0, us_pack, mb / us_pack * 1000.0,
         us_base / us_pack, equal ? "OK" : "FAIL");
  if (!equal) {
    for (size_t i = 0; i < href.size(); ++i) {
      if (href[i] != hnew[i]) {
        printf("  first diff at %zu: ref=%.9g new=%.9g\n", i, href[i],
               hnew[i]);
        break;
      }
    }
  }

  CHECK(hipEventDestroy(e0));
  CHECK(hipEventDestroy(e1));
  CHECK(hipFree(r.w));
  CHECK(hipFree(r.x));
  CHECK(hipFree(r.out_ref));
  CHECK(hipFree(r.out_new));
  CHECK(hipFree(r.ids));
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
  CHECK(hipMemcpyToSymbol(cIq2Signs, kSignsIq2Xs, sizeof(kSignsIq2Xs)));
  uint64_t* grid_buf = nullptr;
  uint8_t* signs_buf = nullptr;
  CHECK(hipMalloc(&grid_buf, sizeof(kIq2XxsGrid)));
  CHECK(hipMalloc(&signs_buf, sizeof(kSignsIq2Xs)));
  CHECK(hipMemcpy(grid_buf, kIq2XxsGrid, sizeof(kIq2XxsGrid),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(signs_buf, kSignsIq2Xs, sizeof(kSignsIq2Xs),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpyToSymbol(g_grid64, &grid_buf, sizeof(grid_buf)));
  CHECK(hipMemcpyToSymbol(g_signs_global, &signs_buf, sizeof(signs_buf)));

  const int reps = 30;
  // decode MoE shapes: up (k = latent 1024), down (n = latent 1024),
  // shared-expert up/down (5376), expert counts early/deep.
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
  for (const Shape& s : shapes) run_shape(s, true, reps);
  return 0;
}

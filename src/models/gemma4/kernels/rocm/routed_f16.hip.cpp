#include "src/models/gemma4/kernels/rocm/routed_f16.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Routed F16 WMMA expert GEMM, ported from the Qwen 27B route
// (src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp). The dp4a MMQ
// grouped path is ~55x slower per MAC than the dense MMQ on gfx1151; the
// matrix-core F16 route with the weights dequantized after the LDS read is
// the production prefill path. The source's fused gate+up (SwiGLU) variant
// and swiglu_gate epilogue were stripped: Gemma-4 uses gelu-tanh, applied by
// GegluF16/GegluF32 after the gate_up GEMM.

namespace gufo::models::gemma4::rocm {
namespace {

constexpr unsigned kThreads = 256;

// Tile index of a map entry past the last tile: beyond every bucket, so the
// routed GEMM's block for it returns without touching the weight or row maps.
constexpr std::int32_t kDeadTile = 0x7FFF;

// gelu_tanh(gate) * up, matching fused.hip.cpp's DGelu/GegluPair so the fused
// gate/up epilogue rounds identically to the separate GegluF16 pass it
// replaces (the gate and up accumulators are narrowed to binary16 first, as
// GegluF16 reads them from the binary16 gu buffer).
__device__ __forceinline__ float DGelu(float x) {
  constexpr float kAlpha = 0.7978845608028654F;  // sqrt(2/pi)
  const float inner = kAlpha * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

__device__ __forceinline__ __half GeGluHalf(float gate, float up) {
  return __float2half(DGelu(__half2float(__float2half(gate))) *
                      __half2float(__float2half(up)));
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or narrowed to F16 when `out_half` is given:
// the gate_up GEMM writes the geglu kernel's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
         : kType == WeightType::kBF16 ? k * 2
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

/// Two BF16 lanes packed in `w` widened to an F16 pair. A BF16 value is the
/// top 16 bits of an F32, so shifting it into place and converting is exact
/// for the exponent range of expert weights (F16 keeps 10 mantissa bits,
/// BF16 carries 7).
__device__ __forceinline__ __half2 Bf16x2ToF16x2(std::uint32_t w) {
  const float lo = __uint_as_float((w & 0xFFFFU) << 16);
  const float hi = __uint_as_float(w & 0xFFFF0000U);
  return __floats2half2_rn(lo, hi);
}

template<WeightType kType, int BM, int BN, int BK, bool kGeGlu = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  static_assert(!kGeGlu || (BN >= 48 && BM % 2 == 0),
                "the fused gate/up epilogue uses the wide-tile store");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kBF16 = kType == WeightType::kBF16;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four, and BF16's
  // 64-byte K block is four.
  constexpr int kChunks = kBF16 ? 4 * BK : (kQ8 ? 2 * BK : BK);

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  if ((tile >> 16) == kDeadTile) {
    return;
  }
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const int half_m = m_i / 2;  // fused gate/up: gate rows [0, half_m), up rows
                               // [half_m, m); a block covers BM/2 of each.
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  // A fused block spans BM/2 gate rows and the matching BM/2 up rows, so its
  // grid.x is half_m / (BM/2) == m / BM (the launcher grid is unchanged).
  const int r_block = static_cast<int>(blockIdx.x) * (kGeGlu ? BM / 2 : BM);

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int rr = (tid >> 1) + (u * 128);  // block row-slot [0, BM)
    // Fused: the first BM/2 slots read gate rows, the rest read the matching
    // up rows (offset by half_m). Non-fused: slot == matrix row.
    int mrow = r_block + rr;
    if (kGeGlu && rr >= BM / 2)
      mrow = half_m + r_block + (rr - BM / 2);
    f_live[u] = mrow < m_i;
    const std::size_t f_row =
        static_cast<std::size_t>(f_live[u] ? mrow : (m_i - 1));
    f_ptr[u] = w_expert + f_row * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_bf16[kWaveRowTiles][4];   ///< BF16: the block's 64 bytes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else if constexpr (kBF16) {
        // A BF16 K block is 32 lanes = 64 bytes = four 16-byte chunks.
        const auto* u4 =
            reinterpret_cast<const uint4*>(f_ptr[u]) + ((kb0 + f_c) * 4);
        f_bf16[u][0] = u4[0];
        f_bf16[u][1] = u4[1];
        f_bf16[u][2] = u4[2];
        f_bf16[u][3] = u4[3];
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      if constexpr (kBF16) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          s_codes[swizzle(row, (4 * f_c) + q)] = f_bf16[u][q];
        }
      } else {
        std::uint32_t scale_bias = 0;
        if constexpr (kQ8) {
          s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
          s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
          scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
        } else {
          s_codes[swizzle(row, f_c)] = f_codes[u];
        }
        if constexpr (kQ5K) {
          s_high[(f_c * BM) + row] = f_high[u];
        }
        if constexpr (kQ8) {
        } else if constexpr (kQ5) {
          s_high[(f_c * BM) + row] = f_high[u];
          const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
          const float d = f_live[u] ? __low2float(dm) : 0.0F;
          const float mn = f_live[u] ? __high2float(dm) : 0.0F;
          scale_bias =
              __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
        } else {
          const int sb32 = f_sb32[u];
          std::uint32_t sc = 0;
          std::uint32_t mn = 0;
          if (sb32 < 4) {
            sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
            mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
          } else {
            sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
                 ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
            mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
                 ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
          }
          const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
          const float scale =
              f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
          const float offset =
              f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
          scale_bias = __builtin_bit_cast(std::uint32_t,
                                          __floats2half2_rn(scale, -offset));
        }
        s_scale[(f_c * BM) + row] = scale_bias;
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8 && !kBF16) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        if constexpr (kBF16) {
          // BF16: K block kb is chunks 4 kb .. 4 kb + 3 (32 lanes). The
          // first 16 elements feed a_lo, the next 16 a_hi, matching the
          // element order the quantized paths hand to the WMMA fragments.
          const uint4 c0 = s_codes[swizzle(row, (4 * kb) + 0)];
          const uint4 c1 = s_codes[swizzle(row, (4 * kb) + 1)];
          const uint4 c2 = s_codes[swizzle(row, (4 * kb) + 2)];
          const uint4 c3 = s_codes[swizzle(row, (4 * kb) + 3)];
          const std::uint32_t words[16] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y,
                                           c1.z, c1.w, c2.x, c2.y, c2.z, c2.w,
                                           c3.x, c3.y, c3.z, c3.w};
          __half2 h[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            h[i] = Bf16x2ToF16x2(words[i]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            h[i] = Bf16x2ToF16x2(words[8 + i]);
          }
          __builtin_memcpy(&a_hi[u], &h[0], 32);
        } else {
          const __half2 sb =
              __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
          const __half2 scale2 = __low2half2(sb);
          const __half2 bias2 = __high2half2(sb);
          std::uint32_t nib[8];
          if constexpr (kQ8) {
            // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
            // flipping the sign bit carries q + 128, which the 1152 magic
            // takes back out.
            const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
            const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
            const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                            c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = words[i] ^ 0x80808080U;
            }
          } else if constexpr (kQ5) {
            // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
            // the low nibbles, 16-31 the high, plus bit j of the high-bit
            // word.
            const uint4 r = raw[u][kb];
            const std::uint32_t high = s_high[(kb * BM) + row];
            const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              nib[i] = (words[i] & 0x0F0F0F0FU) |
                       SpreadHighBits((high >> (4 * i)) & 0xFU);
              nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                           SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
            }
          } else {
            // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
            // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
            // of chunk 1; Q5_K adds bit j of the staged high-bit word.
            const unsigned shift = 4U * static_cast<unsigned>(kb);
            const std::uint32_t words[8] = {
                raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w,
                raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
            const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
              if constexpr (kQ5K) {
                nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
              }
            }
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        }
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr ((kQ5 || kQ8) && BN >= 48) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr (((kQ5 || kQ8) && BN >= 48) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
        if constexpr (kGeGlu) {
          // Fused gate/up: scratch rows [0, BM/2) hold the gate projection and
          // rows [BM/2, BM) the matching up projection. act = gelu_tanh(gate) *
          // up is written to a half_m-wide row, replacing the gu buffer and the
          // separate GegluF16 pass. rr is always even, so the float2 reads and
          // the half2 store stay aligned.
#pragma unroll
          for (int round = 0; round < 16 * (BM / 2) / (256 * 2); ++round) {
            const unsigned flat = (round * 256 + tid) * 2;
            const unsigned tr = flat / (BM / 2), rr = flat % (BM / 2);
            const unsigned t = t_local + j * 16 + tr, r = r_block + rr;
            if (t < unsigned(bucket_rows) && r < unsigned(half_m)) {
              const int dst = rows_out[bucket_begin + t];
              if (dst >= 0) {
                const float2 g = *reinterpret_cast<const float2*>(
                    scratch + tr * stride + rr);
                const float2 u = *reinterpret_cast<const float2*>(
                    scratch + tr * stride + rr + BM / 2);
                const size_t o = size_t(dst) * half_m + r;
                if (r + 1 < unsigned(half_m))
                  *reinterpret_cast<__half2*>(out_half + o) = __floats2half2_rn(
                      GeGluHalf(g.x, u.x), GeGluHalf(g.y, u.y));
                else
                  out_half[o] = GeGluHalf(g.x, u.x);
              }
            }
          }
        } else {
#pragma unroll
          for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
            const unsigned flat = (round * 256 + tid) * 2;
            const unsigned tr = flat / BM, row = flat % BM;
            const unsigned t = t_local + j * 16 + tr, r = r_block + row;
            if (t < unsigned(bucket_rows) && r < m) {
              const int dst = rows_out[bucket_begin + t];
              if (dst >= 0) {
                const float* srow = scratch + tr * stride + row;
                float2 v = *reinterpret_cast<const float2*>(srow);
                const size_t o = size_t(dst) * m + r;
                if (m % 2 == 0 && r + 1 < m)
                  *reinterpret_cast<__half2*>(out_half + o) =
                      __floats2half2_rn(v.x, v.y);
                else {
                  out_half[o] = __float2half(v.x);
                  if (r + 1 < m)
                    out_half[o + 1] = __float2half(v.y);
                }
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// One block: thread e counts expert e's `rows`-row tiles, an ordered scan
// places them, and the remaining capacity entries become dead tiles.
__global__ void __launch_bounds__(256)
    BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                           std::uint32_t experts, std::uint32_t rows,
                           std::uint32_t capacity,
                           std::int32_t* __restrict__ tiles) {
  __shared__ std::uint32_t scan[256];
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t padded = e < experts ? (counts[e] + 15U) / 16U * 16U : 0;
  const std::uint32_t n = (padded + rows - 1) / rows;
  scan[e] = n;
  __syncthreads();
  // Inclusive Hillis-Steele scan; integer sums, so order-free.
  for (std::uint32_t step = 1; step < 256; step <<= 1) {
    const std::uint32_t add = e >= step ? scan[e - step] : 0;
    __syncthreads();
    scan[e] += add;
    __syncthreads();
  }
  const std::uint32_t first = scan[e] - n;
  for (std::uint32_t j = 0; j < n; ++j) {
    tiles[first + j] = static_cast<std::int32_t>(e | (j << 16));
  }
  constexpr std::int32_t kDead = kDeadTile << 16;
  for (std::uint32_t i = scan[255] + e; i < capacity; i += 256) {
    tiles[i] = kDead;
  }
}

}  // namespace

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  const std::size_t rows = RoutedCompactRows(slots, n_experts);
  (void)hipMemsetAsync(rows_token, 0xFF, rows * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(rows_slot, 0xFF, rows * sizeof(std::int32_t), stream);
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows, std::uint32_t capacity,
                      std::int32_t* tiles, hipStream_t stream) {
  BuildRoutedTilesKernel<<<1, 256, 0, stream>>>(counts, experts, rows, capacity,
                                                tiles);
}

template<int BN, bool kGeGlu>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     float* out, __half* out_half, std::size_t m, std::size_t k,
                     hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  // grid.x spans m / BM blocks; a fused block covers BM/2 gate and BM/2 up
  // rows, so half_m / (BM/2) == m / BM leaves the grid unchanged.
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK, kGeGlu>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in, rows_out,
          out, out_half, m, k);
      return true;
    case WeightType::kQ5_1:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK, kGeGlu>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in, rows_out,
          out, out_half, m, k);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK, kGeGlu>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in, rows_out,
          out, out_half, m, k);
      return true;
    case WeightType::kBF16:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kBF16, kBM, BN, kBK, kGeGlu>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in, rows_out,
          out, out_half, m, k);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK, kGeGlu>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in, rows_out,
          out, out_half, m, k);
      return true;
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream, bool geglu) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K)
          ? 256
          : (type == WeightType::kBF16 ? 32 : 64);
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  // The fused gate/up epilogue lives in the wide-tile store only.
  if (geglu && (tile_rows == 16 || m % 2 != 0)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16, false>(w, type, x, tiles, n_tiles, pad_bounds,
                                        rows_in, rows_out, out, out_half, m, k,
                                        stream);
    case 48:
      return geglu ? LaunchRoutedF16<48, true>(w, type, x, tiles, n_tiles,
                                               pad_bounds, rows_in, rows_out,
                                               out, out_half, m, k, stream)
                   : LaunchRoutedF16<48, false>(w, type, x, tiles, n_tiles,
                                                pad_bounds, rows_in, rows_out,
                                                out, out_half, m, k, stream);
    case 64:
      return geglu ? LaunchRoutedF16<64, true>(w, type, x, tiles, n_tiles,
                                               pad_bounds, rows_in, rows_out,
                                               out, out_half, m, k, stream)
                   : LaunchRoutedF16<64, false>(w, type, x, tiles, n_tiles,
                                                pad_bounds, rows_in, rows_out,
                                                out, out_half, m, k, stream);
    default:
      return false;
  }
}

}  // namespace gufo::models::gemma4::rocm
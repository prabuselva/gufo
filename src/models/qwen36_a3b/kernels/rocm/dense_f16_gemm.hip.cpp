#include "src/models/qwen36_a3b/kernels/rocm/dense_f16_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen36_a3b::rocm {
namespace {

constexpr unsigned kThreads = 256;

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32 accumulators
// per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

// Builds two __half2 from eight Q8_0 codes packed in a word: each code is
// widened into an F16 mantissa through the 0x6464 magic, offset by -1152 (the
// caller pre-flips the sign bit so q + 128 carries out), then scaled.
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

// Dense F16 WMMA GEMM over Q8_0 weights: block = BM rows x BN tokens, BK
// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
// The codes are dequantized to F16 once per stage as they are committed to LDS
// (magic-number F16 construction, exact for a Q8_0 code), and the activations
// are F16 rows [batch][k], so the matrix cores accumulate in F32 with no
// per-block scaling. y is [batch][m]. Generic core: the only epilogue is the
// transposed store, so every global store covers 32 consecutive rows of one
// token (a full 128-byte line).
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1>
__launch_bounds__(256) __global__
    void DenseF16GEMMKernel(const void* __restrict__ w,
                            const __half* __restrict__ x, float* __restrict__ y,
                            std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the stage's
  // block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles so a narrow projection reuses each weight fragment across
  // the token tiles. The tile grid's row count is divisible by kRowGroup; every
  // dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with a
  // zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    a_ptr[p] = w_bytes + static_cast<std::size_t>(a_live[p] ? r : (m_i - 1)) *
                             static_cast<std::size_t>(num_kb) * 34;
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      const std::uint8_t* blk =
          a_ptr[p] + (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
      a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
      __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
      __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
      // the 1152 magic takes back out.
      const std::uint32_t words[8] = {
          a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
          a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
      const __half2 scale2 = __half2half2(
          __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
      __half2 h[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                      h[2 * i + 1]);
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 v;
        __builtin_memcpy(&v, &h[4 * c], 16);
        s_a[kk][row][swizzle(row, c)] = v;
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled freely,
    // the compiler sinks them below it and the commit waits on the full memory
    // latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every global
  // store covers 32 consecutive rows of one token: a full 128-byte line (half
  // lines cost a read-modify-write on the fabric). Four padding floats reduce
  // scatter bank conflicts and retain float4 alignment for the output stores.
  constexpr unsigned kOutputStride = 36;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row], with 32 output rows per token.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>(((wave_row * kWaveRowTiles) + i) * 16);
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>(((wave_tok * kWaveTokTiles) + j) * 16);
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = (lane_id >> 1);
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
        auto* dst = reinterpret_cast<float4*>(y + (tok * m) + r0 +
                                              static_cast<std::size_t>(row_l));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          dst[q] = src[q];
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            y[(tok * m) + r] =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // Tile plan: a 128-token macro tile for wide batches, 64-row tiles for the
  // narrow projections, and 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product of
    // the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

}  // namespace gufo::models::qwen36_a3b::rocm
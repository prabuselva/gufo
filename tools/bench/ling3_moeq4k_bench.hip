// Lever 6.0 probe: decode grouped expert GEMV (Q4_K gate/up stacks).
//
// Production decode reads ~8.85 MB of Q4_K weights per gate-or-up launch
// (8 slots x 768 rows x 2560 cols). The shipped GemvGroupedQ4K<2u> walks
// two rows per wave with per-lane single-byte quant loads off global
// memory. This probe measures the production launcher against bit-exact
// variants:
//   V0  production GemvGroupedQ4K<2U> via the real GemvGrouped launcher
//   V1  four rows per wave (more independent weight streams)
//   V2  whole supers-blocks staged into warp-owned LDS with coalesced
//       uint4 loads, consumed in the original ascending order
//   V3  V2 software-pipelined: staging for block b+1 issues between the
//       LDS reads of block b (single buffer, never reordered)
// Every variant must stay bit-exact against V0: each row keeps the exact
// super-block/j accumulation chain.
#include "src/models/ling3_flash/kernels/rocm/gemv.hip.cpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace gufo::models::ling3_flash::rocm;  // NOLINT

namespace {

#define CU(x)                                                        \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    if (e_ != hipSuccess) {                                          \
      std::fprintf(stderr, "hip error %s at %s:%d\n",                \
                   hipGetErrorString(e_), __FILE__, __LINE__);       \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

constexpr std::uint32_t kUsed = 8U;     // routed slots per token
constexpr std::uint32_t kRows = 768U;   // expert_ff rows per matrix
constexpr std::uint32_t kCols = 2560U;  // hidden_size
constexpr std::uint32_t kNB = kCols / 256U;
constexpr std::uint32_t kRing = 64U;   // expert ring, >70 MB > L2
constexpr int kLayerRing = 16;         // "layers" per round
constexpr int kIters = 64;             // rounds per timing event

// Variant kernel mirroring the production per-row chain. Rows = 2 or 4.
// kSmem stages the 128 qs bytes of each row into warp-private LDS with
// coalesced uint4 loads; kPf pipelines the staging of b+1 after the LDS
// reads of b are issued.
template <std::uint32_t Rows, bool kSmem, bool kPf, bool kFull = false,
          bool kNoUnpack = false, bool kFold = false>
__global__ void MoeQ4KV(const Q4KBlock* __restrict__ w,
                        const std::int32_t* __restrict__ ids,
                        std::size_t expert_stride_blocks,
                        const float* __restrict__ x, std::uint32_t x_stride,
                        float* __restrict__ out, std::uint32_t rows) {
  __shared__ std::uint8_t qs_stage[4][Rows][128];  // warp-private
  __shared__ Q4KBlock blk_stage[4][2][Rows];       // full-block ping-pong
  const std::uint32_t g = blockIdx.x * 4U + (threadIdx.x >> 5);
  const std::uint32_t rows_per_slot = rows / Rows;
  if (g >= kUsed * rows_per_slot) {
    return;
  }
  const std::uint32_t s = g / rows_per_slot;
  const std::uint32_t r = (g - s * rows_per_slot) * Rows;
  const Q4KBlock* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * kNB;
  // Production decode has one token row: xs = x + (s/slots_per_row) *
  // x_stride with slots_per_row == kUsed, i.e. always x itself.
  const float* __restrict__ xs = x;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wid = threadIdx.x >> 5;
  float acc[Rows] = {};

  auto compute = [&](std::uint32_t b, int bufsel) {
    const float* __restrict__ xb = xs + b * 256 + lane;
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      xv[j] = xb[32 * j];
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const Q4KBlock& blk =
          kFull && bufsel >= 0 ? blk_stage[wid][bufsel][o] : row[o * kNB + b];
      const float d = __half2float(blk.d);
      const float dmin = __half2float(blk.dmin);
      if constexpr (kFold) {
        // Bit-safe reassociation: half*(<=6-bit sc) and *(<=4-bit q) are
        // exact in fp32, so integer sc*quant then one cvt equals the
        // production double product; dmin*mn stays the same single product.
        const std::uint32_t* sp =
            reinterpret_cast<const std::uint32_t*>(blk.scales);
        const std::uint32_t sa = sp[0];
        const std::uint32_t sb = sp[1];
        const std::uint32_t sh = sp[2];
        int scv[8];
        float dmn[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int s0 = 0;
          int m0 = 0;
          if (j < 4) {
            s0 = static_cast<int>((sa >> (8 * j)) & 0x3FU);
            m0 = static_cast<int>((sb >> (8 * j)) & 0x3FU);
          } else {
            const std::uint32_t pj4 = (sh >> (8 * (j - 4))) & 0xFFU;
            s0 = static_cast<int>(pj4 & 0xFU) |
                 (static_cast<int>(((sa >> (8 * (j - 4))) & 0xFFU) >> 6) << 4);
            m0 = static_cast<int>(pj4 >> 4) |
                 (static_cast<int>(((sb >> (8 * (j - 4))) & 0xFFU) >> 6) << 4);
          }
          scv[j] = s0;
          dmn[j] = dmin * static_cast<float>(m0);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const std::uint8_t packed = kSmem ? qs_stage[wid][o][(j >> 1) * 32 + lane]
                                            : blk.qs[(j >> 1) * 32 + lane];
          const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
          const float t = static_cast<float>(scv[j] * quant);
          acc[o] += ((d * t) - dmn[j]) * xv[j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          std::uint8_t packed;
          if constexpr (kSmem) {
            packed = qs_stage[wid][o][(j >> 1) * 32 + lane];
          } else {
            packed = blk.qs[(j >> 1) * 32 + lane];
          }
          const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
          int sc = 0;
          int mn = 0;
          if constexpr (!kNoUnpack) {
            Q4ScaleMin(j, blk.scales, sc, mn);
          } else {
            sc = 15;
            mn = 0;
          }
          acc[o] += (d * static_cast<float>(sc) * static_cast<float>(quant) -
                     dmin * static_cast<float>(mn)) *
                    xv[j];
        }
      }
    }
  };

  if constexpr (kFull) {
    auto stage_full = [&](std::uint32_t b, int slot) {
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        if (lane < 9U) {
          reinterpret_cast<uint4*>(&blk_stage[wid][slot][o])[lane] =
              reinterpret_cast<const uint4*>(&row[o * kNB + b])[lane];
        }
      }
    };
    stage_full(0, 0);
    for (std::uint32_t b = 0; b < kNB; ++b) {
      __syncwarp();  // reads of the previous block retired; stage(b) done
      if (b + 1U < kNB) {
        stage_full(b + 1U, static_cast<int>((b + 1U) & 1U));
      }
      compute(b, static_cast<int>(b & 1U));
    }
  } else if constexpr (!kSmem) {
    for (std::uint32_t b = 0; b < kNB; ++b) {
      compute(b, -1);
    }
  } else if constexpr (!kPf) {
    for (std::uint32_t b = 0; b < kNB; ++b) {
      // Coalesced: lanes 0..7 fetch uint4 of qs per row.
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        if (lane < 8U) {
          reinterpret_cast<uint4*>(&qs_stage[wid][o][0])[lane] =
              reinterpret_cast<const uint4*>(row[o * kNB + b].qs)[lane];
        }
      }
      __syncwarp();
      compute(b, -1);
      __syncwarp();
    }
  } else {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      if (lane < 8U) {
        reinterpret_cast<uint4*>(&qs_stage[wid][o][0])[lane] =
            reinterpret_cast<const uint4*>(row[o * kNB].qs)[lane];
      }
    }
    __syncwarp();
    for (std::uint32_t b = 0; b < kNB; ++b) {
      compute(b, -1);  // reads staged block b
      __syncwarp();
      if (b + 1U < kNB) {
#pragma unroll
        for (std::uint32_t o = 0; o < Rows; ++o) {
          if (lane < 8U) {
            reinterpret_cast<uint4*>(&qs_stage[wid][o][0])[lane] =
                reinterpret_cast<const uint4*>(row[o * kNB + b + 1U].qs)[lane];
          }
        }
        __syncwarp();
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[s * rows + r + o] = acc[o];
    }
  }
}

std::uint32_t* mismatches = nullptr;

// Diagnostic ceiling: linearly read the exact same footprint the GEMV
// covers (the 8 selected expert matrices) with perfectly coalesced
// uint4 loads. If this also tops out near the GEMV rate, no kernel
// restructuring can recover the missing bandwidth at this pattern.
__global__ void StreamKernel(const Q4KBlock* __restrict__ w,
                             const std::int32_t* __restrict__ ids,
                             float* __restrict__ sink) {
  const std::size_t chunks_per_expert = sizeof(Q4KBlock) * kRows * kNB / 16U;
  const std::size_t c =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t total = chunks_per_expert * kUsed;
  float acc = 0.0F;
  for (std::size_t i = c; i < total; i += stride) {
    const std::uint32_t q = static_cast<std::uint32_t>(i / chunks_per_expert);
    const std::size_t off = i - static_cast<std::size_t>(q) * chunks_per_expert;
    const uint4 v = reinterpret_cast<const uint4*>(
        w + static_cast<std::size_t>(ids[q]) * kRows * kNB)[off];
    acc += __uint_as_float(v.x) + __uint_as_float(v.y) +
           __uint_as_float(v.z) + __uint_as_float(v.w);
  }
  if (acc == 12345.678F) {  // keep loads alive without atomics
    sink[0] = acc;
  }
}

__global__ void DiffKernel(const float* a, const float* b, std::size_t n,
                           std::uint32_t* bad) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (__float_as_uint(a[i]) != __float_as_uint(b[i])) {
    atomicAdd(bad, 1U);
  }
}

__device__ __forceinline__ void FoldExtract(std::uint32_t sa, std::uint32_t sb,
                                           std::uint32_t sh, int j, int& s0,
                                           int& m0) {
  if (j < 4) {
    s0 = static_cast<int>((sa >> (8 * j)) & 0x3FU);
    m0 = static_cast<int>((sb >> (8 * j)) & 0x3FU);
  } else {
    const std::uint32_t pj4 = (sh >> (8 * (j - 4))) & 0xFFU;
    s0 = static_cast<int>(pj4 & 0xFU) |
         (static_cast<int>(((sa >> (8 * (j - 4))) & 0xFFU) >> 6) << 4);
    m0 = static_cast<int>(pj4 >> 4) |
         (static_cast<int>(((sb >> (8 * (j - 4))) & 0xFFU) >> 6) << 4);
  }
}

__global__ void FoldCheck(const Q4KBlock* __restrict__ w,
                          std::size_t total_blocks, std::uint32_t* bad) {
  const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
  if (i >= total_blocks) {
    return;
  }
  const Q4KBlock& blk = w[i];
  const std::uint32_t* sp = reinterpret_cast<const std::uint32_t*>(blk.scales);
  for (int j = 0; j < 8; ++j) {
    int s = 0;
    int m = 0;
    Q4ScaleMin(j, blk.scales, s, m);
    int e0 = 0;
    int e1 = 0;
    FoldExtract(sp[0], sp[1], sp[2], j, e0, e1);
    if (e0 != s || e1 != m) {
      atomicAdd(bad, 1U);
    }
  }
}

__global__ void NaNFixKernel(std::uint16_t* p, std::size_t n_half) {
  const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
  if (i >= n_half) {
    return;
  }
  const std::uint16_t h = p[i];
  if ((h & 0x7C00U) == 0x7C00U && (h & 0x03FFU) != 0U) {
    p[i] = 0x3C00U;  // replace NaN half with 1.0
  }
}

__global__ void FillKernel(float* p, std::size_t n, unsigned seed) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U + seed;
  h ^= h >> 13;
  h *= 2246822519U;
  h ^= h >> 16;
  p[i] = static_cast<float>(h & 0xFFFFU) / 32768.0F - 1.0F;
}

}  // namespace

int main() {
  CU(hipSetDevice(0));
  CU(hipMalloc(&mismatches, sizeof(std::uint32_t)));

  const std::size_t expert_blocks = static_cast<std::size_t>(kRows) * kNB;
  const std::size_t expert_bytes = expert_blocks * sizeof(Q4KBlock);
  const double mb_per_launch =
      static_cast<double>(expert_bytes) * kUsed / 1e6;
  std::printf("expert %zu B, weights per launch %.3f MB\n", expert_bytes,
              mb_per_launch);

  // One contiguous expert stack, like the real stacked tensor: ids index
  // into it directly. The ring covers > L2 so routing misses persist.
  Q4KBlock* w0 = nullptr;
  CU(hipMalloc(&w0, expert_bytes * kRing));
  {
    const std::size_t n_f = expert_bytes * kRing / sizeof(float);
    FillKernel<<<(n_f + 255U) / 256U, 256>>>(reinterpret_cast<float*>(w0),
                                             n_f, 1000U);
    const std::size_t n_h = expert_bytes * kRing / sizeof(std::uint16_t);
    NaNFixKernel<<<(n_h + 255U) / 256U, 256>>>(
        reinterpret_cast<std::uint16_t*>(w0), n_h);
  }
  {
    CU(hipMemset(mismatches, 0, sizeof(std::uint32_t)));
    const std::size_t tot_sup = expert_blocks * kRing;
    FoldCheck<<<static_cast<std::uint32_t>((tot_sup + 255U) / 256U), 256>>>(
        w0, tot_sup, mismatches);
    CU(hipDeviceSynchronize());
    std::uint32_t bad = 0;
    CU(hipMemcpy(&bad, mismatches, sizeof(bad), hipMemcpyDeviceToHost));
    std::printf("FoldCheck: %u bad (scale,min) extractions\n", bad);
  }
  float* x = nullptr;
  CU(hipMalloc(&x, kCols * sizeof(float)));
  FillKernel<<<(kCols + 255U) / 256U, 256>>>(x, kCols, 7U);
  std::int32_t* ids = nullptr;
  CU(hipMalloc(&ids, kUsed * sizeof(std::int32_t)));

  const std::size_t out_n = static_cast<std::size_t>(kUsed) * kRows;
  std::vector<float*> out(7);
  for (int v = 0; v < 7; ++v) {
    CU(hipMalloc(&out[v], out_n * sizeof(float)));
    CU(hipMemset(out[v], 0, out_n * sizeof(float)));
  }
  float* golden = nullptr;
  CU(hipMalloc(&golden, out_n * sizeof(float)));

  const std::size_t pairs2 = static_cast<std::size_t>(kUsed) * kRows / 2U;
  const std::size_t grid2 = (pairs2 + 3U) / 4U;
  const std::size_t grid4 = (pairs2 / 2U + 3U) / 4U;

  auto one_launch = [&](int v, std::int32_t* use_ids, float* dst) {
    switch (v) {
      case 0:
        GemvGrouped(w0, GemvType::kQ4_K, expert_bytes, use_ids, kUsed, kUsed,
                    kRows, kCols, x, 0U, dst, nullptr);
        break;
      case 1:
        MoeQ4KV<4U, false, false><<<grid4, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
      case 2:
        MoeQ4KV<2U, true, false><<<grid2, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
      case 3:
        MoeQ4KV<2U, true, true><<<grid2, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
      case 4:
        MoeQ4KV<2U, false, false, true><<<grid2, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
      case 5:
        MoeQ4KV<2U, false, false, false, true><<<grid2, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
      default:
        MoeQ4KV<2U, false, false, false, false, true><<<grid2, dim3(128)>>>(
            w0, use_ids, expert_blocks, x, kCols, dst, kRows);
        break;
    }
  };

  auto run_ring = [&](int v) {  // kLayerRing layers x (gate,up)
    for (std::uint32_t layer = 0; layer < static_cast<std::uint32_t>(kLayerRing);
         ++layer) {
      std::int32_t rot[kUsed];
      for (std::uint32_t s = 0; s < kUsed; ++s) {
        rot[s] = static_cast<std::int32_t>((layer * kUsed + s * 8U + 3U) %
                                           kRing);
      }
      CU(hipMemcpy(ids, rot, sizeof(rot), hipMemcpyHostToDevice));
      one_launch(v, ids, out[v]);
      one_launch(v, ids, out[v]);
    }
  };

  run_ring(0);
  CU(hipDeviceSynchronize());
  CU(hipMemcpy(golden, out[0], out_n * sizeof(float),
               hipMemcpyDeviceToDevice));

  const char* names[7] = {"V0 production r2", "V1 rows4        ",
                          "V2 smem-qs r2   ", "V3 pipe qs r2   ",
                          "V4 full-stg r2  ", "D5 no-unpack(dg)",
                          "V6 fold sc*mn   "};
  for (int v = 1; v < 7; ++v) {
    CU(hipMemset(mismatches, 0, sizeof(std::uint32_t)));
    CU(hipMemset(out[v], 0, out_n * sizeof(float)));
    run_ring(v);
    CU(hipDeviceSynchronize());
    DiffKernel<<<(out_n + 255U) / 256U, 256>>>(out[v], golden, out_n,
                                               mismatches);
    std::uint32_t bad = 0;
    CU(hipMemcpy(&bad, mismatches, sizeof(bad), hipMemcpyDeviceToHost));
    std::printf("%s : %s (%u mismatch words)\n", names[v],
                bad == 0U ? "BIT-EXACT vs V0" : "DIFFERS", bad);
  }

  hipEvent_t t0, t1;
  CU(hipEventCreate(&t0));
  CU(hipEventCreate(&t1));
  double best[7] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
  const double launches =
      static_cast<double>(kIters) * kLayerRing * 2.0;  // gate+up
  for (int round = 0; round < 5; ++round) {
    for (int v = 0; v < 7; ++v) {
      CU(hipDeviceSynchronize());
      CU(hipEventRecord(t0));
      for (int r = 0; r < kIters; ++r) {
        for (std::uint32_t layer = 0;
             layer < static_cast<std::uint32_t>(kLayerRing); ++layer) {
          std::int32_t rot[kUsed];
          for (std::uint32_t s = 0; s < kUsed; ++s) {
            rot[s] = static_cast<std::int32_t>((r * kUsed + layer * 3U +
                                                s * 8U + 3U) %
                                               kRing);
          }
          CU(hipMemcpy(ids, rot, sizeof(rot), hipMemcpyHostToDevice));
          one_launch(v, ids, out[v]);
          one_launch(v, ids, out[v]);
        }
      }
      CU(hipEventRecord(t1));
      CU(hipEventSynchronize(t1));
      float ms = 0.0F;
      CU(hipEventElapsedTime(&ms, t0, t1));
      best[v] = std::min(best[v], ms * 1e3 / launches);
    }
  }
  std::printf("\nbest us/launch (one gate-or-up over 8 slots):\n");
  for (int v = 0; v < 7; ++v) {
    std::printf("%s : %8.3f us  %6.1f GB/s  (%+.1f%% vs V0)\n", names[v],
                best[v], mb_per_launch * 1e3 / best[v],
                (best[v] / best[0] - 1.0) * 100.0);
  }

  // Stream diagnostic over the identical expert footprint.
  float* sink = nullptr;
  CU(hipMalloc(&sink, sizeof(float)));
  double sbest = 1e30;
  for (int round = 0; round < 5; ++round) {
    CU(hipDeviceSynchronize());
    CU(hipEventRecord(t0));
    for (int r = 0; r < kIters; ++r) {
      for (std::uint32_t layer = 0;
           layer < static_cast<std::uint32_t>(kLayerRing); ++layer) {
        std::int32_t rot[kUsed];
        for (std::uint32_t s = 0; s < kUsed; ++s) {
          rot[s] = static_cast<std::int32_t>((r * kUsed + layer * 3U +
                                              s * 8U + 3U) %
                                             kRing);
        }
        CU(hipMemcpy(ids, rot, sizeof(rot), hipMemcpyHostToDevice));
        StreamKernel<<<768, 128>>>(w0, ids, sink);
        StreamKernel<<<768, 128>>>(w0, ids, sink);
      }
    }
    CU(hipEventRecord(t1));
    CU(hipEventSynchronize(t1));
    float ms = 0.0F;
    CU(hipEventElapsedTime(&ms, t0, t1));
    sbest = std::min(sbest, ms * 1e3 / launches);
  }
  std::printf("STREAM same footprint   : %8.3f us  %6.1f GB/s\n", sbest,
              mb_per_launch * 1e3 / sbest);
  return 0;
}

// Standalone lever-5.5 benchmark: decode router as one fused launch
// (GemvF16 logit chain + last-block top-k selection) versus the current
// Gemv + RouterTopK pair. Verifies bit-identical logits/ids/weights, then
// times both launch sequences back to back on the decode shapes.
//
// Build:  tools/bench/build.sh ling3_router_fused_bench
// Run:    /tmp/ling3_router_fused_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#define CHECK(expr)                                                     \
  do {                                                                  \
    hipError_t err__ = (expr);                                          \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                   hipGetErrorString(err__), __FILE__, __LINE__);       \
      return 1;                                                         \
    }                                                                   \
  } while (0)

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

__device__ __forceinline__ float SigmoidD(float x) {
  return 1.0F / (1.0F + ::expf(-x));
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ std::uint32_t BlockArgMax(float v,
                                                     std::uint32_t idx,
                                                     float* vsm,
                                                     std::uint32_t* ism) {
  const unsigned lane = threadIdx.x & 31U;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned warps = (blockDim.x + 31U) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(~0ULL, v, off);
    const std::uint32_t oi = __shfl_down_sync(~0ULL, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
  if (lane == 0U) {
    vsm[warp] = v;
    ism[warp] = idx;
  }
  __syncthreads();
  if (warp == 0U) {
    const bool owned = lane < warps;
    v = owned ? vsm[lane] : kNegInf;
    idx = owned ? ism[lane] : 0xFFFFFFFFU;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(~0ULL, v, off);
      const std::uint32_t oi = __shfl_down_sync(~0ULL, idx, off);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0U) {
      ism[warps] = idx;
    }
  }
  __syncthreads();
  return ism[warps];
}

/// Exact copy of the production GemvF16 walk (gemv.hip.cpp).
__global__ void GemvF16(const __half* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ out, std::uint32_t rows,
                        std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const __half* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += static_cast<float>(row[i]) * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

/// Exact copy of the production RouterSelectRow (kernels.hip.cpp).
constexpr float kWeightsSumEps = 6.103515625e-5F;

__device__ __forceinline__ void RouterSelectRow(
    const float* __restrict__ logits, const float* __restrict__ bias,
    std::uint32_t n_exp, std::uint32_t groups, std::uint32_t groups_used,
    std::uint32_t used, float weights_scale, std::int32_t* __restrict__ ids,
    float* __restrict__ weights, bool l2_reads = false) {
  extern __shared__ float sm[];
  float* selection = sm;
  float* probs = sm + n_exp;
  float* group_scores = sm + 2 * n_exp;
  const std::uint32_t per_group = n_exp / groups;
  __shared__ bool keep[64];
  __shared__ float red_v[20];
  __shared__ std::uint32_t red_i[20];
  for (std::uint32_t e = threadIdx.x; e < n_exp; e += blockDim.x) {
    probs[e] = SigmoidD(logits[e]);
    selection[e] = probs[e] + bias[e];
  }
  __syncthreads();
  if (groups > 1U) {
    if (threadIdx.x < groups) {
      const std::uint32_t g = threadIdx.x;
      float best = kNegInf;
      float second = kNegInf;
      for (std::uint32_t i = 0; i < per_group; ++i) {
        const float value =
            selection[static_cast<std::size_t>(g) * per_group + i];
        if (value >= best) {
          second = best;
          best = value;
        } else if (value > second) {
          second = value;
        }
      }
      group_scores[g] = best + second;
    }
    __syncthreads();
    if (threadIdx.x < groups) {
      keep[threadIdx.x] = false;
    }
    __syncthreads();
    for (std::uint32_t s = 0; s < groups_used; ++s) {
      const std::uint32_t g = threadIdx.x;
      const std::uint32_t best_g =
          BlockArgMax((g < groups && !keep[g]) ? group_scores[g] : kNegInf, g,
                      red_v, red_i);
      if (threadIdx.x == best_g) {
        keep[best_g] = true;
      }
      __syncthreads();
    }
    for (std::uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
      if (!keep[g]) {
        for (std::uint32_t i = 0; i < per_group; ++i) {
          selection[static_cast<std::size_t>(g) * per_group + i] = kNegInf;
        }
      }
    }
    __syncthreads();
  }
  double sum = 0.0;
  for (std::uint32_t s = 0; s < used; ++s) {
    float v = kNegInf;
    std::uint32_t idx = 0xFFFFFFFFU;
    for (std::uint32_t e = threadIdx.x; e < n_exp; e += blockDim.x) {
      const float value = selection[e];
      if (value > v) {
        v = value;
        idx = e;
      }
    }
    const std::uint32_t best_e = BlockArgMax(v, idx, red_v, red_i);
    if (threadIdx.x == best_e % blockDim.x) {
      selection[best_e] = kNegInf;
    }
    if (threadIdx.x == 0U) {
      ids[s] = static_cast<std::int32_t>(best_e);
      weights[s] = probs[best_e];
      sum += static_cast<double>(probs[best_e]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    const float norm = 1.0F / fmaxf(static_cast<float>(sum), kWeightsSumEps);
    for (std::uint32_t s = 0; s < used; ++s) {
      weights[s] *= norm * weights_scale;
    }
  }
}

__global__ void RouterTopKKernel(const float* __restrict__ logits,
                                 const float* __restrict__ bias,
                                 std::uint32_t n_exp, std::uint32_t groups,
                                 std::uint32_t groups_used, std::uint32_t used,
                                 float weights_scale,
                                 std::int32_t* __restrict__ ids,
                                 float* __restrict__ weights) {
  RouterSelectRow(logits, bias, n_exp, groups, groups_used, used, weights_scale,
                  ids, weights);
}

__global__ void RouterTopKFusedF16Kernel(
    const __half* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ logits, const float* __restrict__ bias,
    std::uint32_t n_exp, std::uint32_t cols, std::uint32_t groups,
    std::uint32_t groups_used, std::uint32_t used, float weights_scale,
    std::int32_t* __restrict__ ids, float* __restrict__ weights,
    std::int32_t* __restrict__ ticket) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r < n_exp) {
    const __half* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
    const std::uint32_t lane = threadIdx.x & 31U;
    float acc = 0.0f;
#pragma unroll 4
    for (std::uint32_t i = lane; i < cols; i += 32) {
      acc += static_cast<float>(row[i]) * x[i];
    }
    acc = WarpReduceSum(acc);
    if (lane == 0U) {
      logits[r] = acc;
    }
  }
  __shared__ bool last_block;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0U) {
    last_block =
        atomicAdd(ticket, 1) == static_cast<std::int32_t>(gridDim.x) - 1;
  }
  __syncthreads();
  if (!last_block) {
    return;
  }
  __threadfence();
  if (threadIdx.x == 0U) {
    *ticket = 0;
  }
  RouterSelectRow(logits, bias, n_exp, groups, groups_used, used, weights_scale,
                  ids, weights);
}

// CPU oracle replicate of the selection on GPU-produced logits (float ops
// mirrored with expf/exp2f-free chain; ids must match exactly).
std::vector<int> OracleTopK(const std::vector<float>& logits,
                            const std::vector<float>& bias,
                            std::vector<float>* probs_out, std::uint32_t n_exp,
                            std::uint32_t groups, std::uint32_t groups_used,
                            std::uint32_t used) {
  std::vector<float> probs(n_exp), selection(n_exp), working(n_exp);
  for (std::uint32_t e = 0; e < n_exp; ++e) {
    probs[e] = 1.0f / (1.0f + ::expf(-logits[e]));
    selection[e] = probs[e] + bias[e];
    working[e] = selection[e];
  }
  const std::uint32_t per_group = n_exp / groups;
  std::vector<char> keep(groups, 0);
  if (groups > 1U) {
    std::vector<float> gs(groups);
    for (std::uint32_t g = 0; g < groups; ++g) {
      float best = kNegInf, second = kNegInf;
      for (std::uint32_t i = 0; i < per_group; ++i) {
        const float value = working[static_cast<std::size_t>(g) * per_group + i];
        if (value >= best) {
          second = best;
          best = value;
        } else if (value > second) {
          second = value;
        }
      }
      gs[g] = best + second;
    }
    for (std::uint32_t s = 0; s < groups_used; ++s) {
      std::uint32_t best_g = 0xFFFFFFFFU;
      float best = kNegInf;
      for (std::uint32_t g = 0; g < groups; ++g) {
        if (keep[g]) continue;
        if (gs[g] > best) {
          best = gs[g];
          best_g = g;
        }
      }
      keep[best_g] = 1;
    }
    for (std::uint32_t g = 0; g < groups; ++g) {
      if (!keep[g]) {
        for (std::uint32_t i = 0; i < per_group; ++i) {
          working[static_cast<std::size_t>(g) * per_group + i] = kNegInf;
        }
      }
    }
  }
  std::vector<int> ids(used);
  for (std::uint32_t s = 0; s < used; ++s) {
    std::uint32_t best_e = 0xFFFFFFFFU;
    float best = kNegInf;
    for (std::uint32_t e = 0; e < n_exp; ++e) {
      if (working[e] > best) {
        best = working[e];
        best_e = e;
      }
    }
    ids[s] = static_cast<int>(best_e);
    working[best_e] = kNegInf;
  }
  *probs_out = probs;
  return ids;
}

struct Shape {
  const char* name;
  std::uint32_t n_exp;
  std::uint32_t cols;
};

}  // namespace

int main() {
  constexpr std::uint32_t kGroups = 8U, kGroupsUsed = 4U, kUsed = 8U;
  constexpr float kScale = 2.5F;
  const Shape shapes[] = {
      {"ling3  512x2560", 512U, 2560U},
      {"odd-cols 512x2568", 512U, 2568U},
      {"half   512x1024", 512U, 1024U},
  };
  std::mt19937 rng(12345U);
  std::normal_distribution<float> norm(0.0F, 1.0F);

  for (const Shape& s : shapes) {
    // ---- bit-exactness over rounds of fresh inputs ----
    const std::size_t w_n = static_cast<std::size_t>(s.n_exp) * s.cols;
    std::vector<float> hx(s.cols);
    std::vector<__half> hw(w_n);
    std::vector<float> hbias(s.n_exp);
    float* dx;
    __half* dw;
    float* dbias;
    float* dlogits;
    float* dlogits2;
    std::int32_t* dids;
    std::int32_t* dids2;
    float* dwts;
    float* dwts2;
    std::int32_t* dticket;
    CHECK(hipMalloc(&dx, hx.size() * sizeof(float)));
    CHECK(hipMalloc(&dw, w_n * sizeof(__half)));
    CHECK(hipMalloc(&dbias, hbias.size() * sizeof(float)));
    CHECK(hipMalloc(&dlogits, s.n_exp * sizeof(float)));
    CHECK(hipMalloc(&dlogits2, s.n_exp * sizeof(float)));
    CHECK(hipMalloc(&dids, kUsed * sizeof(std::int32_t)));
    CHECK(hipMalloc(&dids2, kUsed * sizeof(std::int32_t)));
    CHECK(hipMalloc(&dwts, kUsed * sizeof(float)));
    CHECK(hipMalloc(&dwts2, kUsed * sizeof(float)));
    CHECK(hipMalloc(&dticket, sizeof(std::int32_t)));
    CHECK(hipMemset(dticket, 0, sizeof(std::int32_t)));
    const std::uint32_t smem = (2U * s.n_exp + kGroups) * sizeof(float);

    int bad_exact = 0;
    int bad_oracle = 0;
    constexpr int kRounds = 300;
    std::vector<float> host_logits(s.n_exp), oracle_probs, prev_pair_logits;
    std::vector<std::int32_t> hids(kUsed), hids2(kUsed);
    std::vector<float> hwts(kUsed), hwts2(kUsed);
    for (int round = 0; round < kRounds; ++round) {
      for (auto& v : hx) v = norm(rng);
      for (auto& v : hw) v = __float2half_rn(norm(rng) * 0.02F);
      for (auto& v : hbias) v = norm(rng) * 0.1F;
      CHECK(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dw, hw.data(), w_n * sizeof(__half),
                      hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dbias, hbias.data(), hbias.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      GemvF16<<<(s.n_exp + 3U) / 4U, 128>>>(dw, dx, dlogits, s.n_exp, s.cols);
      RouterTopKKernel<<<1, 512, smem>>>(dlogits, dbias, s.n_exp, kGroups,
                                         kGroupsUsed, kUsed, kScale, dids,
                                         dwts);
      RouterTopKFusedF16Kernel<<<(s.n_exp + 3U) / 4U, 128, smem>>>(
          dw, dx, dlogits2, dbias, s.n_exp, s.cols, kGroups, kGroupsUsed, kUsed,
          kScale, dids2, dwts2, dticket);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(host_logits.data(), dlogits, s.n_exp * sizeof(float),
                      hipMemcpyDeviceToHost));
      std::vector<float> host_logits2(s.n_exp);
      CHECK(hipMemcpy(host_logits2.data(), dlogits2, s.n_exp * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(hids.data(), dids, kUsed * sizeof(std::int32_t),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(hids2.data(), dids2, kUsed * sizeof(std::int32_t),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(hwts.data(), dwts, kUsed * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(hwts2.data(), dwts2, kUsed * sizeof(float),
                      hipMemcpyDeviceToHost));
      if (host_logits != host_logits2 || hids != hids2) {
        ++bad_exact;
        if (bad_exact <= 4) {
          int n_bits = 0;
          double max_rel = 0.0;
          int first = -1;
          for (std::uint32_t e = 0; e < s.n_exp; ++e) {
            if (host_logits[e] != host_logits2[e]) {
              ++n_bits;
              if (first < 0) first = static_cast<int>(e);
              const double ref =
                  std::fabs(static_cast<double>(host_logits[e])) + 1e-30;
              const double rel =
                  std::fabs(static_cast<double>(host_logits2[e]) -
                            host_logits[e]) /
                  ref;
              if (rel > max_rel) max_rel = rel;
            }
          }
          const bool stale = !prev_pair_logits.empty() &&
                             host_logits2 == prev_pair_logits;
          std::printf(
              "  round %d: logits DIFFER in %d/%u (first=%d max_rel=%.3e "
              "stale_prev=%s) ids_same=%d\n",
              round, n_bits, s.n_exp, first, max_rel, stale ? "yes" : "no",
              hids == hids2 ? 1 : 0);
        }
      }
      prev_pair_logits = host_logits;
      for (std::uint32_t u = 0; u < kUsed; ++u) {
        const double denom = std::fabs(hwts[u]) + 1e-30;
        if (std::fabs(hwts[u] - hwts2[u]) / denom > 1e-6) {
          ++bad_exact;
          if (bad_exact <= 8) {
            std::printf("  round %d: weight[%u] %.9g vs %.9g\n", round, u,
                        hwts[u], hwts2[u]);
          }
          break;
        }
      }
      const std::vector<int> oracle =
          OracleTopK(host_logits, hbias, &oracle_probs, s.n_exp, kGroups,
                     kGroupsUsed, kUsed);
      for (std::uint32_t u = 0; u < kUsed; ++u) {
        if (oracle[u] != hids[u]) {
          ++bad_oracle;
          std::printf("  round %d: oracle id[%u] %d vs gpu %d\n", round, u,
                      oracle[u], hids[u]);
          break;
        }
      }
    }
    std::int32_t ticket_after = -1;
    CHECK(hipMemcpy(&ticket_after, dticket, sizeof(std::int32_t),
                    hipMemcpyDeviceToHost));

    // ---- warm launch-loop timing ----
    const int kIters = 20000;
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    float ms_pair = 0.0F, ms_fused = 0.0F;
    for (int trial = 0; trial < 3; ++trial) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kIters; ++i) {
        GemvF16<<<(s.n_exp + 3U) / 4U, 128, 0, stream>>>(dw, dx, dlogits,
                                                         s.n_exp, s.cols);
        RouterTopKKernel<<<1, 512, smem, stream>>>(dlogits, dbias, s.n_exp,
                                                   kGroups, kGroupsUsed, kUsed,
                                                   kScale, dids, dwts);
      }
      CHECK(hipStreamSynchronize(stream));
      auto t1 = std::chrono::steady_clock::now();
      const float us_pair =
          std::chrono::duration<float, std::micro>(t1 - t0).count() / kIters;
      t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kIters; ++i) {
        RouterTopKFusedF16Kernel<<<(s.n_exp + 3U) / 4U, 128, smem, stream>>>(
            dw, dx, dlogits2, dbias, s.n_exp, s.cols, kGroups, kGroupsUsed,
            kUsed, kScale, dids2, dwts2, dticket);
      }
      CHECK(hipStreamSynchronize(stream));
      t1 = std::chrono::steady_clock::now();
      const float us_fused =
          std::chrono::duration<float, std::micro>(t1 - t0).count() / kIters;
      if (trial == 2) {
        ms_pair = us_pair;
        ms_fused = us_fused;
      }
    }
    CHECK(hipStreamDestroy(stream));
    std::printf("%s: exact %s (bad=%d) oracle_bad=%d ticket_after=%d | "
                "pair %.2f us  fused %.2f us  (%+.2f us, %+.1f%%)\n",
                s.name, bad_exact ? "FAIL" : "OK", bad_exact, bad_oracle,
                ticket_after, ms_pair, ms_fused, ms_fused - ms_pair,
                100.0 * (ms_fused - ms_pair) / ms_pair);

    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dbias));
    CHECK(hipFree(dlogits));
    CHECK(hipFree(dlogits2));
    CHECK(hipFree(dids));
    CHECK(hipFree(dids2));
    CHECK(hipFree(dwts));
    CHECK(hipFree(dwts2));
    CHECK(hipFree(dticket));
  }
  return 0;
}

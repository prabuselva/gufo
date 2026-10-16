#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ling3_flash::rocm {

/// Fused single-token operators for the Ling 3.0 Flash decode graph. Every
/// entry mirrors the CPU oracle in `src/models/ling3_flash/reference.cpp`
/// (RMSNorm/L2Norm/RoPE formulas from `cpu_ops.cpp`); the GEMV tier in
/// `gemv.hpp` supplies the projections these operators sit between.

/// out = x * rsqrt(mean(x^2) + eps) * w. `w` may be null (ones).
void RmsNorm(const float* x, const float* w, float eps, float* out,
             std::uint32_t n, hipStream_t stream);

/// out = a + b.
void Add(const float* a, const float* b, float* out, std::uint32_t n,
         hipStream_t stream);

/// out[0:n] = a, out[n:2n] = b (the MTP [embedding; hidden] concat).
void Concat2(const float* a, const float* b, float* out, std::uint32_t n,
             hipStream_t stream);

/// x = sigmoid(x), in place.
void SigmoidInPlace(float* x, std::uint32_t n, hipStream_t stream);

/// RoPE over the first `rotary_dim` entries of each head row on consecutive
/// pairs (2i, 2i+1) with frequency theta^(-2i/rotary_dim), in place.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta,
          hipStream_t stream);

/// Rope over rows of arbitrary stride: the rotary segment of row h starts at
/// `x + h * stride` (the MLA q tail, where the 64 rotary entries sit after the
/// 128 nope entries inside each 192-wide head row).
void RopeStrided(float* x, std::uint32_t heads, std::uint32_t stride,
                 std::uint32_t rotary_dim, std::uint32_t pos, float theta,
                 hipStream_t stream);

/// RoPE reading row h from `in + h * in_stride` and writing the rotated pairs
/// to `out + h * out_stride`; identical arithmetic to the in-place kernels.
void RopeTo(const float* in, float* out, std::uint32_t heads,
            std::uint32_t in_stride, std::uint32_t out_stride,
            std::uint32_t rotary_dim, std::uint32_t pos, float theta,
            hipStream_t stream);

/// One causal depthwise conv tap step for q/k/v: `conv_w` is [inner][kern]
/// (channel-major taps), `history` is [kern-1][inner] oldest-first and is
/// shifted in place, `out = silu(conv(proj))`.
void ConvSilu(const float* conv_w, const float* proj, float* history,
              float* out, std::uint32_t inner, std::uint32_t kern,
              hipStream_t stream);

/// Per-head L2 normalization of q and k (head rows of length `d`); q is then
/// scaled by `q_scale`. In place.
void KdaNormHeads(float* q, float* k, std::uint32_t heads, std::uint32_t d,
                  float eps, float q_scale, hipStream_t stream);

/// Fused KDA delta-rule decode step plus the per-head output norm:
/// decay_j = exp(gate_lb * sigmoid(a[h] * f[h*d+j])); S_j *= decay_j;
/// u_j = S_j . k_h; delta_j = (v[h*d+j] - u_j) * sigmoid(beta[h]);
/// S_j += delta_j * k_h; o_j = S_j . q_h;
/// out[h*d:] = rmsnorm(o) * norm_w * sigmoid(g[h*d:]).
/// `state` is [heads][d][d] and is updated in place.
void KdaDelta(const float* q, const float* k, const float* v, const float* f,
              const float* beta, const float* a, const float* norm_w,
              const float* g, float* state, float* out, std::uint32_t heads,
              std::uint32_t d, float eps, float gate_lb, hipStream_t stream);

/// --- Prefill tier: the same recurrence over a token chunk, sequential over
/// tokens but fully on-device. `q`, `k`, `v`, `f`, `beta` and `g` are chunked
/// as [tokens][heads*d] (the per-stream projected rows as consumed by the
/// decode chain); conv `history` is [kern-1][inner] oldest-first raw rows.

/// Batched causal depthwise conv + SiLU over a chunk for one stream; reads the
/// leading taps from `history`, leaves `history` untouched.
void KdaConvSiluPrefill(const float* conv_w, const float* proj,
                        const float* history, float* out, std::uint32_t tokens,
                        std::uint32_t inner, std::uint32_t kern,
                        hipStream_t stream);

/// Writes into `out` the [kern-1][inner] raw projected rows ending at the
/// chunk's last token (the next history), reading in-chunk `proj` rows and
/// falling back to `history` when the chunk is shorter than kern-1.
void KdaHistoryUpdate(const float* proj, const float* history, float* out,
                      std::uint32_t tokens, std::uint32_t inner,
                      std::uint32_t kern, hipStream_t stream);

/// Per-head L2 norm (q *= q_scale) of the convolved [tokens][heads*d] q and k,
/// in place; bit-identical to KdaNormHeads applied per token.
void KdaNormHeadsPrefill(float* q, float* k, std::uint32_t tokens,
                         std::uint32_t heads, std::uint32_t d, float eps,
                         float q_scale, hipStream_t stream);

/// Hoists the recurrence's transcendental inputs out of the serial token loop:
/// decay_pre[t*heads*d + h*d + i] = exp(gate_lb * sigmoid(a[h] * f[...])) (the
/// key-axis decay vector, same expression as KdaDelta), beta_pre[t*heads+h] =
/// sigmoid(beta[...]), and kq_pre[t*heads+h] = dot(kn, qn) over each head row
/// (qn is expected to already carry q_scale).
void KdaPrep(const float* f, const float* a, const float* beta, const float* qn,
             const float* kn, float gate_lb, float* decay_pre, float* beta_pre,
             float* kq_pre, std::uint32_t tokens, std::uint32_t heads,
             std::uint32_t d, hipStream_t stream);

/// KDA delta-rule recurrence over the whole chunk using the precomputed
/// decay/beta/kq; `v` is the convolved v stream, `state` [heads][d][d] is
/// updated in place, and raw (pre-out-norm) `attn` is [tokens][heads*d].
/// When `snap` is non-null, the state after each token (up to `snap_rows`) is
/// written to `snap[t][heads][d][d]`; the snapshot-free fast path requires
/// head_dim == 128.
void KdaDeltaLoop(const float* qn, const float* kn, const float* v,
                  const float* decay_pre, const float* beta_pre,
                  const float* kq_pre, float* state, float* attn,
                  std::uint32_t tokens, std::uint32_t heads,
                  std::uint32_t head_dim, float* snap, std::uint32_t snap_rows,
                  hipStream_t stream);

/// Gated output norm over the chunked raw attn rows, in place; same formula
/// and reduction as the decode KdaDelta tail.
void KdaOutNormPrefill(float* attn, const float* g, const float* norm_w,
                       std::uint32_t tokens, std::uint32_t heads,
                       std::uint32_t d, float eps, hipStream_t stream);

/// Sigmoid router with bias, group-limited top-k (ties to the smaller index,
/// matching ggml argsort_top_k): selection = sigmoid(logits) + bias, keep the
/// `groups_used` groups with the highest sum of their two best scores, take
/// the top `used` experts, weights = unbiased probs normalized by
/// max(sum, 6.103515625e-5) * weights_scale.
void RouterTopK(const float* logits, const float* bias, std::uint32_t n_exp,
                std::uint32_t groups, std::uint32_t groups_used,
                std::uint32_t used, float weights_scale, std::int32_t* ids,
                float* weights, hipStream_t stream);

/// out = clamp(silu(gate), -inf, limit) * clamp(up, -limit, limit) when
/// limit > 1e-6, else out = silu(gate) * up.
void SwigluClamped(const float* gate, const float* up, float* out,
                   std::uint32_t n, float limit, hipStream_t stream);

/// x[row*cols:] *= scales[row], in place (the MLA per-head output gate).
void RowScale(float* x, const float* scales, std::uint32_t rows,
              std::uint32_t cols, hipStream_t stream);

/// dst[i] = sum_r weights[r] * src[r*cols + i] (the MoE weighted expert-output
/// accumulation over the `used` selected experts).
void RowWeightedSum(const float* src, const float* weights, float* dst,
                    std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// --- Prefill batch tier
/// ------------------------------------------------------- Row-parallel
/// variants of the per-token operators that sit between the chunked GEMMs of
/// `Executor::Prefill`. Each is bit-identical to launching the decode-side
/// operator once per row.

/// RmsNormRows over `rows` independent rows of length `n`:
/// out[r*n:] = x[r*n:] * rsqrt(mean(x[r*n:]^2) + eps) * w.
void RmsNormRows(const float* x, const float* w, float eps, float* out,
                 std::uint32_t rows, std::uint32_t n, hipStream_t stream);

/// f[r*n:] += bias[0:n] for every row, in place (the per-token dt bias ahead
/// of KdaPrep, which consumes the biased f rows).
void AddBiasRows(float* f, const float* bias, std::uint32_t rows,
                 std::uint32_t n, hipStream_t stream);

/// RouterTopK over a chunk: one block per token row of logits
/// [tokens][n_exp]; ids/weights are [tokens][used].
void RouterTopKRows(const float* logits, const float* bias, std::uint32_t n_exp,
                    std::uint32_t groups, std::uint32_t groups_used,
                    std::uint32_t used, float weights_scale, std::int32_t* ids,
                    float* weights, std::uint32_t tokens, hipStream_t stream);

/// counts[e] += 1 for every non-negative slot id equal to e over `n` ids.
/// `counts` is zeroed by the launcher; block order does not affect counts.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n, std::uint32_t n_exp, hipStream_t stream);

/// ids[i] = i % pattern for i in [0, n) (the identity stack index vector for
/// the per-head attn_k_b/attn_v_b grouped GEMVs and their head-bucketed WMMA
/// prefill compact, pattern = num heads).
void FillModuloIds(std::int32_t* ids, std::uint32_t n, std::uint32_t pattern,
                   hipStream_t stream);

/// RowWeightedSum over a chunk: dst[t*cols:] =
/// sum_r weights[t*used + r] * src[(t*used + r)*cols:] per token.
void RowWeightedSumTokens(const float* src, const float* weights, float* dst,
                          std::uint32_t tokens, std::uint32_t used,
                          std::uint32_t cols, hipStream_t stream);

/// RoPE of the MLA q chunk over all heads (row stride `heads * head_stride`,
/// rotary segment of each head row at `+nope`), in place, with token t at
/// position `start_pos + t`; also writes the rotated segments packed as
/// qrp[(t * heads + h) * rotary_dim:]. Bit-identical to RopeStrided per token.
void MlaQRopeRows(float* q, float* qrp, std::uint32_t start_pos,
                  std::uint32_t tokens, std::uint32_t heads,
                  std::uint32_t head_stride, std::uint32_t nope,
                  std::uint32_t rotary_dim, float theta, hipStream_t stream);

/// MLA kv-a norm + rope over a chunk: token t writes the RMS-normalized
/// kv-rank segment of row a[t * (kv_rank + rotary_dim):] to
/// kv_cache[(start_pos + t) * kv_rank:] and the rope of its rotary segment to
/// kpe_cache[(start_pos + t) * rotary_dim:]. Bit-identical to the decode
/// RmsNorm + Rope pair.
void MlaKvANormRopeRows(const float* a, const float* gamma, float eps,
                        float* kv_cache, float* kpe_cache,
                        std::uint32_t start_pos, std::uint32_t tokens,
                        std::uint32_t kv_rank, std::uint32_t rotary_dim,
                        float theta, hipStream_t stream);

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_
#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private HIP launchers for the Qwen3.6-35B-A3B (qwen35moe) graph: the
/// fused operators outside the quantized GEMM tier. Activations are row-major
/// float32 [tokens][dim] unless noted; every launch is asynchronous on
/// `stream` (null uses the default stream). These encode the semantics the
/// scalar oracle in reference.cpp pins: plain `w * x` RMSNorm, partial NEOX
/// rotary, the Gated DeltaNet recurrence with its SiLU output gate and tiled
/// key heads, gated grouped-query attention, and the softmax top-k Mixture-of-
/// Experts combine. Nothing here is shared with another model.
namespace gufo::models::qwen36_a3b::rocm {

/// out[r][d] = (x[r][d] * rsqrt(mean_d x^2 + eps)) * gamma[d]. `rows`
/// independent norms of `dim` elements; `gamma` may be null (scale 1).
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t rows, std::uint32_t dim, float eps,
                 hipStream_t stream);

/// NEOX partial rotary on x [rows][heads][head_dim] at positions pos[row].
/// Only the leading `rotary_dim` channels of each head rotate; pair (i,
/// i + rotary_dim/2) turns by pos * theta^(-2i/rotary_dim).
void Rope(float* x, const std::uint32_t* pos, std::uint32_t rows,
          std::uint32_t heads, std::uint32_t head_dim, std::uint32_t rotary_dim,
          float theta, hipStream_t stream);

/// Splits the fused query/gate projection of a full-attention layer. `qg` is
/// [heads][2 * head_dim] with query and gate channels interleaved per head;
/// `q` and `gate` (each [heads * head_dim]) receive the two halves so the
/// query can be normalized and rotated on its own.
void SplitQGate(const float* qg, float* q, float* gate, std::uint32_t heads,
                std::uint32_t head_dim, std::uint32_t tokens,
                hipStream_t stream);

/// gate[i] = silu(gate[i]) * up[i], in place in `gate`, over `count` floats.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);

/// out[(t*k + s)*cols + i] = x[t*cols + i]: replicate each of the `tokens`
/// activation rows `k` times so the grouped expert GEMVs can index x by slot.
void DupRows(const float* x, std::uint32_t tokens, std::uint32_t k,
             std::uint32_t cols, float* out, hipStream_t stream);

/// x[i] *= sigmoid(g[i]) over `count` floats.
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);

/// a[i] += b[i] over `count` floats, the residual stream update.
void Add(float* a, const float* b, std::size_t count, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), keep the
/// top `k`, renormalize by their sum floored at 2^-14. `ids` [tokens][k] and
/// `weights` [tokens][k] receive the chosen expert indices (ascending index
/// order on ties) and their renormalized weights.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// counts[e] = number of (token, slot) pairs in `ids` (length tokens*k) routed
/// to expert e. Used to bound the routed MMQ column grid to the real largest
/// expert bucket instead of the full pair count.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[(t*k + s)][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);

/// Causal depthwise convolution with SiLU over one token, advancing the rolling
/// history. `qkv` is [channels], `conv_w` is [channels][kernel] (kernel index
/// contiguous), `history` is [kernel-1][channels] (oldest first) and is updated
/// in place to drop the oldest tap and append `qkv`. `convolved` [channels]
/// receives silu(conv). One launch per token.
void GdnConv(const float* qkv, const float* conv_w, float* history,
             float* convolved, std::uint32_t channels, std::uint32_t kernel,
             hipStream_t stream);

/// L2-normalizes the query and key halves of `convolved` per key head into
/// `qn` and `kn` (each [k_heads * head_dim]). The value half is left in place.
/// x / sqrt(sum x^2 + eps) over each head_dim slice.
void GdnNormQk(const float* convolved, float* qn, float* kn,
               std::uint32_t k_heads, std::uint32_t head_dim, float eps,
               hipStream_t stream);

/// One Gated DeltaNet recurrence step for every value head. `qn`/`kn` are the
/// normalized key-head vectors [k_heads * head_dim]; value head h reads key
/// head h % k_heads. `v` is [v_heads * head_dim]. `alpha`/`beta`/`a`/`dt` are
/// [v_heads]. `state` is [v_heads][head_dim][head_dim] (row j over the value
/// dim, column i over the key dim) and is updated in place. `attn`
/// [v_heads * head_dim] receives the raw recurrence output (before the norm
/// and gate). decay = exp(a[h] * softplus(alpha[h] + dt[h])),
/// b = sigmoid(beta[h]).
void GdnDelta(const float* qn, const float* kn, const float* v,
              const float* alpha, const float* beta, const float* a,
              const float* dt, float* state, float* attn, std::uint32_t k_heads,
              std::uint32_t v_heads, std::uint32_t head_dim,
              hipStream_t stream);

/// Per-head RMSNorm of `attn` (gamma `norm_w` [head_dim]) followed by the SiLU
/// output gate `attn[h*d + j] *= silu(z[h*d + j])`, in place. `z` is
/// [v_heads * head_dim].
void GdnOutNorm(float* attn, const float* z, const float* norm_w,
                std::uint32_t v_heads, std::uint32_t head_dim, float eps,
                hipStream_t stream);

/// Batched Gated DeltaNet prefill over a chunk of `tokens`. Buffers are
/// row-major [tokens][...]: `qkv`/`convolved` [tokens][channels], `qn`/`kn`
/// [tokens][k_heads * head_dim], `alpha`/`beta` [tokens][v_heads], `attn`/`z`
/// [tokens][v_heads * head_dim]. `a`/`dt`/`norm_w` are per-head/per-dim static.
/// `conv_state` ([kernel-1][channels]) is read by the convolution and advanced
/// past the chunk via `history_out` (a disjoint buffer the caller copies back).
/// `state` ([v_heads][head_dim][head_dim]) is updated in place. `channels` is
/// 2 * k_heads * head_dim + v_heads * head_dim. head_dim must be 256.
void GdnConvPrefill(const float* qkv, const float* conv_w, const float* history,
                    float* convolved, std::uint32_t tokens,
                    std::uint32_t channels, std::uint32_t kernel,
                    hipStream_t stream);
void GdnHistoryUpdate(const float* qkv, const float* history, float* out,
                      std::uint32_t tokens, std::uint32_t channels,
                      std::uint32_t kernel, hipStream_t stream);
void GdnNormQkPrefill(const float* convolved, float* qn, float* kn,
                      std::uint32_t tokens, std::uint32_t k_heads,
                      std::uint32_t channels, std::uint32_t head_dim, float eps,
                      hipStream_t stream);
/// When `snap` is non-null the state after each of the first `snap_rows`
/// tokens is also written to `snap` ([snap_rows][v_heads][head_dim][head_dim]),
/// slot t holding the state once tokens [0, t] are consumed. The live `state`
/// still ends the chunk fully advanced. Used by speculative verify to roll the
/// recurrent state back to the last accepted token.
void GdnDeltaLoop(const float* qn, const float* kn, const float* convolved,
                  const float* alpha, const float* beta, const float* a,
                  const float* dt, float* state, float* attn,
                  std::uint32_t tokens, std::uint32_t k_heads,
                  std::uint32_t v_heads, std::uint32_t head_dim,
                  std::uint32_t channels, float* snap, std::uint32_t snap_rows,
                  hipStream_t stream);
void GdnOutNormPrefill(float* attn, const float* z, const float* norm_w,
                       std::uint32_t tokens, std::uint32_t v_heads,
                       std::uint32_t head_dim, float eps, hipStream_t stream);

/// Builds the batched MTP input rows: `out[t]` = [`e[t]`][`h[t-1]`], with
/// `h[-1]` taken from `h_prev`. All buffers hold `tokens` rows of `hidden`
/// floats (`out` of 2 * `hidden`).
void MtpConcat(const float* e, const float* h, const float* h_prev, float* out,
               std::uint32_t tokens, std::uint32_t hidden, hipStream_t stream);

/// Single-query grouped-query causal attention with the sigmoid output gate.
/// `q` [heads][head_dim]; `k_cache`/`v_cache` [n_kv][kv_heads][head_dim]
/// (already rotated); `gate` [heads * head_dim]. `out` [heads * head_dim]
/// receives the gated context (before the output projection). `scale` is
/// 1/sqrt(head_dim). `scratch` holds heads * n_kv softmax weights (legacy
/// path). For head_dim == 256 a flash-decoding split runs instead: `part`
/// (heads * 32 * (head_dim + 2) floats) receives the per-split partials.
void AttentionDecode(const float* q, const float* k_cache, const float* v_cache,
                     const float* gate, float* out, float* scratch,
                     float* part, std::uint32_t n_kv, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim,
                     float scale, hipStream_t stream);

/// Multi-row variant of the head_dim == 256 flash-decoding path for the
/// speculative verify pass. `q`, `gate` and `out` are [rows][heads * head_dim]
/// with rows in [2, 5]; row o attends positions [0, n_kv0 + o). `part` holds
/// rows * heads * 32 * (head_dim + 2) floats.
void AttentionDecodeRows(const float* q, const float* k_cache,
                         const float* v_cache, const float* gate, float* out,
                         float* part, std::uint32_t n_kv0, std::uint32_t rows,
                         std::uint32_t heads, std::uint32_t kv_heads,
                         std::uint32_t head_dim, float scale,
                         hipStream_t stream);

/// Batched causal self-attention for a prefill chunk. `q` and `gate` are
/// [tokens][heads * head_dim] (already normed and rotated); `k_cache`/`v_cache`
/// hold absolute positions `[0][kv_heads * head_dim]` and must already contain
/// Mirror `count` contiguous FP32 key/value elements into the FP16 KV planes
/// the WMMA prefill kernel reads, starting at element `start` in both the FP32
/// cache and its FP16 mirror. The caller invokes this for every KV write
/// (prefill chunk, verify rows, decode token) so the FP16 mirror always matches
/// the FP32 cache. No-op when either FP16 pointer is null.
void KvCacheWriteF16(const float* k, const float* v, void* k_cache_f16,
                     void* v_cache_f16, std::size_t start, std::size_t count,
                     hipStream_t stream);

/// the chunk's own keys/values. Token `t` attends positions `[0, start + t]`.
/// `out` [tokens][heads * head_dim] receives the gated context. Uses a tiled
/// online softmax so no per-token scratch is needed.
///
/// When `k_cache_f16`/`v_cache_f16` are non-null and the shape is the model's
/// 16Q/2KV head_dim-256 GQA, the chunk's FP32 KV rows (already published at
/// `[start, start + tokens)` by the caller) are mirrored into the FP16 planes
/// and the causal pass runs on the WMMA matrix cores; otherwise the scalar
/// tiled/naive kernels read the FP32 cache directly.
void AttentionPrefill(const float* q, const float* k_cache,
                      const float* v_cache, void* k_cache_f16,
                      void* v_cache_f16, const float* gate, float* out,
                      std::uint32_t start, std::uint32_t tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, float scale, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_KERNELS_HPP_
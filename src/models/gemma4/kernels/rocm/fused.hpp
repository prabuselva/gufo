#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_FUSED_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_FUSED_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Fused elementwise and routing operators for the Gemma-4-26B-A4B (gemma4)
/// trunk: RMSNorm variants, the residual add, the softmax top-k router, the
/// expert histogram, the weighted MoE combine (with the per-expert down
/// scales) and the geglu (gelu-tanh times up) split of the fused gate/up
/// projection. Nothing here is shared with another model.
namespace gufo::models::gemma4::rocm {

/// out[r][d] = (x[r][d] * rsqrt(mean_d x^2 + eps)) * gamma[d]. `rows`
/// independent norms of `dim` elements; `gamma` may be null (scale 1).
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t rows, std::uint32_t dim, float eps,
                 hipStream_t stream);

/// Binary16-output RMSNorm: the same reduction and scale as `RmsNormRows` (the
/// sum-of-squares stays in double, the scale in float) but the row is stored as
/// F16. The wide-batch shared-FFN path uses it so the following dense F16 GEMMs
/// consume the normed activations directly, skipping the FP32->F16 narrowing
/// pass and halving the normed-row traffic. `gamma` may be null (scale 1).
void RmsNormRowsHalf(const float* x, const float* gamma, __half* out,
                     std::uint32_t rows, std::uint32_t dim, float eps,
                     hipStream_t stream);

/// Single-row residual add fused with the following RMSNorm: `x += addend`,
/// then `out = rmsnorm(x) * gamma`. Bit-identical to Add then RmsNormRows with
/// rows == 1. `out` may alias `x`; `gamma` may be null (scale 1).
void FusedAddRmsNorm(float* x, const float* addend, const float* gamma,
                     float* out, std::uint32_t dim, float eps,
                     hipStream_t stream);

/// a[i] += b[i] over `count` floats, the residual stream update.
void Add(float* a, const float* b, std::size_t count, hipStream_t stream);

/// x[i] *= k over `count` floats (embedding and layer-output scales).
void ScaleInPlace(float* x, float k, std::size_t count, hipStream_t stream);

/// x[i] *= scale[i % period] over `count` floats (the router's per-layer
/// scale of `period` elements, applied to every row of a multi-row buffer).
void MulInPlace(float* x, const float* scale, std::size_t count,
                std::size_t period, hipStream_t stream);

/// x[i] = cap * tanh(x[i] / cap) over `count` floats (the LM head softcap).
void SoftcapInPlace(float* x, float cap, std::size_t count, hipStream_t stream);

/// Per token: softmax over `n_experts` logits (row `logits + t * stride`),
/// then the `k` largest probabilities (lowest index wins ties) with the
/// weights renormalized over the selected sum, clamped to 6.103515625e-5.
/// `ids[token * k + s]` receives the expert and `weights[token * k + s]` the
/// renormalized probability. One block per token.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// counts[e] = number of slots with ids[slot] == e over tokens * k slots;
/// negative ids are skipped. `counts` is zeroed by the launcher.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t * k + s] * scale(ids[t * k + s]) *
///             expert_out[(t * k + s) * dim + i]. `expert_scale` may be null
/// (scale 1); it holds the artifact's per-expert down scales.
void MoeEpilogue(const float* expert_out, const float* weights,
                 const std::int32_t* ids, const float* expert_scale, float* out,
                 std::uint32_t tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream);

/// act[r][i] = gelu_tanh(gu[r][i]) * gu[r][ff + i] for the fused gate/up
/// projection: `gu` is [rows][2 * ff] (gate rows first), `act` is [rows][ff].
/// `count` is the number of act elements (rows * ff). F32 and F16 variants.
void GegluF32(const float* gu, float* act, std::size_t count, std::uint32_t ff,
              hipStream_t stream);
void GegluF16(const __half* gu, __half* act, std::size_t count,
              std::uint32_t ff, hipStream_t stream);

/// The shared-FFN variant with separate gate and up projections:
/// act[i] = gelu_tanh(gate[i]) * up[i] over `count` elements laid out as
/// [rows][ff] rows in all three buffers.
void GegluF32Separate(const float* gate, const float* up, float* act,
                      std::size_t count, hipStream_t stream);

/// Binary16-output shared-FFN geglu: reads the FP32 gate/up rows, stores the
/// act row as F16 so the following dense F16 down GEMM consumes it directly
/// (skipping the FP32->F16 narrowing pass and halving the act-row traffic).
void GegluF16Separate(const float* gate, const float* up, __half* act,
                      std::size_t count, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_FUSED_HPP_
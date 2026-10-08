#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMM_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/gemv.hpp"

/// Model-private batched (prefill) linear projections for the
/// Gemma-4-26B-A4B (gemma4) trunk. The artifact stores its projections as
/// Q8_0, F32 or BF16 rows (never dequantized at load), so the same three
/// encodings the GEMV tier decodes drive the GEMM tier: Q8_0 runs on the shared
/// mmq tensor-core path, F32 on hipBLAS SGEMM and BF16 on hipBLAS GemmEx with
/// the activations narrowed to BF16 in a scratch buffer. Weights always stay in
/// their native encoding. Nothing here is shared with another model.
namespace gufo::models::gemma4::rocm {

/// out[t][r] = sum_k W[r][k] * x[t][k] for t in [0, batch), r in [0, rows).
/// `base` points at the first row of a row-major [rows x cols] matrix stored in
/// `type`; `row_bytes` is the encoded size of one row and must match
/// TensorRef::RowBytes for that type and `cols`. `x` is [batch x cols] and
/// `out` is [batch x rows], both row-major. The launch is asynchronous on
/// `stream` (null uses the default stream).
void Gemm(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          std::uint32_t batch, hipStream_t stream);

/// True when a Q8_0 projection of shape [rows x cols] at `batch` rows falls in
/// the measured dense F16 WMMA winning window (the same bounds `Gemm` uses to
/// pick that route). Lets a caller gate a binary16 activation pipeline so every
/// GEMM in the pipeline takes the F16 route together, or none do.
bool DenseF16Window(std::uint32_t batch, std::uint32_t rows,
                    std::uint32_t cols);

/// The binary16-activation sibling of `Gemm`: runs the dense F16 WMMA GEMM
/// directly on pre-narrowed F16 activations `xh` ([batch x cols]), skipping the
/// FP32->F16 narrowing pass `Gemm` inserts. `out` is [batch x rows] F32.
/// Returns false (launching nothing) when `type` is not Q8_0 or the shape is
/// outside `DenseF16Window`, so the caller falls back to the FP32 `Gemm` path.
/// Asynchronous on `stream`.
bool GemmHalfIn(const void* base, GemvType type, std::uint32_t rows,
                std::uint32_t cols, const __half* xh, float* out,
                std::uint32_t batch, hipStream_t stream);

/// Grouped MoE projection. For each token t and slot s in [0, n_expert_used),
/// with e = ids[t*n_expert_used + s]:
///   out[(t*n_expert_used + s)][r] = sum_k W[e][r][k] * x[t][k].
/// `base` is a stacked [n_experts x rows x cols] matrix stored in `type`
/// (TensorRef::Expert layout); `x` is [n_tokens x cols]; `ids` is
/// [n_tokens x n_expert_used]; `out` is [n_tokens*n_expert_used x rows], all
/// row-major. Q8_0 runs on the mmq routed path. Asynchronous on `stream`.
void GemmMoe(const void* base, GemvType type, std::uint32_t rows,
             std::uint32_t cols, std::size_t row_bytes, const float* x,
             const std::int32_t* ids, float* out, std::uint32_t n_tokens,
             std::uint32_t n_experts, std::uint32_t n_expert_used,
             hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMM_HPP_
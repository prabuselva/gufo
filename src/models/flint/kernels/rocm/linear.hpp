#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_LINEAR_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_LINEAR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

/// Flint-owned linear algebra tiers. The artifact stores its projections
/// natively (Q8_0 for every matrix, BF16 or F32 for the router), so nothing
/// is dequantized at upload and every kernel decodes on the fly. The dense
/// batched path delegates Q8_0 to the shared mmq tensor-core tier and the
/// small F32/BF16 router shapes to a naive per-row batch kernel.
namespace gufo::models::flint::rocm {

/// out[r] = sum_k W[r][k] * x[k] for r in [0, rows). `base` is a row-major
/// [rows x cols] matrix (or one expert matrix) in `type`, `row_bytes` its
/// encoded row size. `type` must be kQ8_0, kF32 or kBF16.
void Gemv(const void* base, core::GgmlType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream);

/// Row-batched GEMV over consecutive activation rows:
/// out[t*rows + r] = sum_k W[r][k] * x[t*cols + k]. Supports the small
/// F32/BF16 router shapes and any Q8_0 projection for tokens [0, batch).
void GemvBatch(const void* base, core::GgmlType type, std::uint32_t rows,
               std::uint32_t cols, std::size_t row_bytes, const float* x,
               float* out, std::uint32_t batch, hipStream_t stream);

/// out[t][r] = sum_k W[r][k] * x[t][k]. Q8_0 runs on the shared mmq dense
/// path; F32/BF16 on GemvBatch. Asynchronous on `stream`.
void Gemm(const void* base, core::GgmlType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          std::uint32_t batch, hipStream_t stream);

/// Grouped routed projection: with e = ids[t*n_used + s],
/// out[(t*n_used + s)][r] = sum_k W[e][r][k] * x[t][k]. Q8_0 only; `base` is
/// the stacked [n_experts x rows x cols] matrix. When `h_counts` (host,
/// n_experts entries, as produced by ExpertCounts) is set it bounds the
/// routed mmq launch (max expert rows + count-selected column tile); the
/// values do not change results, only the grid.
void GemmMoe(const void* base, std::uint32_t rows, std::uint32_t cols,
             const float* x, const std::int32_t* ids, float* out,
             std::uint32_t n_tokens, std::uint32_t n_experts,
             std::uint32_t n_expert_used, hipStream_t stream,
             const unsigned int* h_counts = nullptr);

/// Zero-copy dequantization of one Q8_0/F32 row into a float buffer; the
/// executor embeds token rows with it.
void DequantRow(const void* base, core::GgmlType type, std::uint32_t row,
                std::uint32_t cols, std::size_t row_bytes, float* out,
                hipStream_t stream);

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_LINEAR_HPP_

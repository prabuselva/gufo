#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DENSE_F16_GEMM_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DENSE_F16_GEMM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

/// Model-private dense F16 WMMA GEMM for the Qwen3.6-35B-A3B (qwen35moe)
/// prefill projections. The Q8_0 weights stay in their native encoding and are
/// dequantized to F16 in LDS; the activations are pre-narrowed F16 rows, so the
/// matrix cores accumulate in F32 with no per-block requantize. This is the
/// wide-batch sibling of the int8 mmq dense path and wins wherever the
/// arithmetic intensity is low enough that halved activation traffic and no
/// FP32->Q8_1 requantize outweigh int8's 2x tensor throughput. The kernel is a
/// generic core (plain transposed store) ported into this model; nothing here
/// is shared with another model.
namespace gufo::models::qwen36_a3b::rocm {

/// out[t][r] = sum_k W[r][k] * x[t][k] for t in [0, batch), r in [0, m), with
/// `w` a row-major [m x k] Q8_0 matrix, `x` F16 [batch x k] and `out` F32
/// [batch x m]. Returns false (launching nothing) for an empty shape or a `k`
/// that is not a multiple of 32, so the caller can fall back to another path.
/// Asynchronous on `stream`.
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DENSE_F16_GEMM_HPP_
#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private quantized matrix-vector product for the Qwen3.6-35B-A3B
/// (qwen35moe) trunk. The artifact stores its linear projections as Q8_0, F32
/// or BF16 rows (never dequantized at load), so the GEMM tier dequantizes on
/// the fly. Nothing here is shared with another model.
namespace gufo::models::qwen36_a3b::rocm {

/// The quantized row formats the GEMV tier decodes. The executor maps the
/// artifact's GGUF type onto this; only these three appear in the model.
enum class GemvType { kQ8_0, kF32, kBF16 };

/// out[r] = sum_k W[r][k] * x[k] for r in [0, rows). `base` points at the first
/// row of a row-major [rows x cols] matrix stored in `type`; `row_bytes` is the
/// encoded size of one row and must match TensorRef::RowBytes for that type and
/// `cols`. For a stacked expert matrix the caller passes the expert's base
/// (TensorRef::Expert). One warp per output row; the launch is asynchronous on
/// `stream` (null uses the default stream).
void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_GEMV_HPP_
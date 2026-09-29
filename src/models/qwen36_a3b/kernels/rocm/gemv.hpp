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

/// Two-row variant of Gemv for the speculative verify pass:
/// out[o*out_stride + r] = sum_k W[r][k] * x[o*x_stride + k] for o in {0,1}.
/// One wave owns an output row and dots the weight row against both
/// activation rows in a single walk, so the matrix is read once for the two
/// tokens. Each per-row dot is bit-identical to the matching Gemv call.
void Gemv2(const void* base, GemvType type, std::uint32_t rows,
           std::uint32_t cols, std::size_t row_bytes, const float* x,
           std::uint32_t x_stride, float* out, std::uint32_t out_stride,
           hipStream_t stream);

/// Grouped decode GEMV over a stacked expert matrix. For slot s in [0, used)
/// with expert id ids[s]: out[s*rows + r] = sum_k W[e][r][k] *
/// x[s*x_stride + k]. `expert_stride` is the byte distance between stacked
/// experts (row_bytes * rows). One launch covers all selected experts, so the
/// caller never reads ids back to the host.
void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t rows, std::uint32_t cols, const float* x,
                 std::uint32_t x_stride, float* out, hipStream_t stream);

/// Fused gate+up pair of the grouped expert Gemv: both stacks must share the
/// shape and `expert_stride` (they do for the MoE gate/up pair). Returns
/// false for types without a fused kernel.
bool GemvGroupedPair(const void* wa, const void* wb, GemvType type,
                     std::size_t expert_stride, const std::int32_t* ids,
                     std::uint32_t used, std::uint32_t rows,
                     std::uint32_t cols, const float* x, std::uint32_t x_stride,
                     float* out_a, float* out_b, hipStream_t stream);

/// One projection of a fused multi launch (see GemvMulti).
struct GemvMultiProj {
  const void* base;
  GemvType type;
  std::uint32_t rows;
  std::uint32_t cols;
  float* out;
};

/// Fused launch of up to four projections sharing one activation row x (the
/// ssm qkv/gate/alpha/beta quartet, attention q/k/v, shared-expert
/// gate/up/gate_inp). The Q8_0 projections are folded into one Multi4 launch;
/// any non-Q8_0 projection (the F32 alpha/beta/gate_inp side vectors) runs as
/// a plain Gemv, and empty projections are skipped. Every per-row dot is
/// bit-identical to separate Gemv calls. Returns false only when n is outside
/// [1,4] (the caller falls back to separate Gemv calls).
bool GemvMulti(const GemvMultiProj* projs, std::uint32_t n, const float* x,
               hipStream_t stream);

/// out[i] = W[row][i] for i in [0, cols): dequantizes a single row of a
/// row-major [rows x cols] matrix stored in `type` (the token-embedding
/// lookup). `base` points at the matrix start; the row offset is derived from
/// `cols` and `type`. Asynchronous on `stream` (null uses the default stream).
void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_GEMV_HPP_
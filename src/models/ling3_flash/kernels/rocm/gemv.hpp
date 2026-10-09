#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private quantized matrix-vector product for the Ling 3.0 Flash
/// (bailingmoe3) trunk. The artifact stores its linear projections as Q4_K,
/// Q5_K, Q6_K, Q8_0 or F32 rows (never dequantized at load), and the MTP
/// draft block as Q4_0, so the GEMM tier dequantizes on the fly. Nothing here
/// is shared with another model.
namespace gufo::models::ling3_flash::rocm {

/// The quantized row formats the GEMV tier decodes. The executor maps the
/// artifact's GGUF type onto this. Q4_K backs most projections; Q5_K the
/// attention outputs; Q6_K the down projections and v_b stacks; Q8_0 the k_b
/// and shared-expert stacks; F32 the router, convolutions and side vectors;
/// Q4_0 the whole MTP draft block.
enum class GemvType { kQ4_0, kQ8_0, kQ4_K, kQ5_K, kQ6_K, kF32 };

/// out[r] = sum_k W[r][k] * x[k] for r in [0, rows). `base` points at the first
/// row of a row-major [rows x cols] matrix stored in `type`; `row_bytes` is the
/// encoded size of one row and must match TensorRef::RowBytes for that type and
/// `cols`. For a stacked expert matrix the caller passes the expert's base
/// (TensorRef::Expert). One warp per output row; the launch is asynchronous on
/// `stream` (null uses the default stream).
void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream);

/// Multi-row variant of Gemv for the speculative verify pass:
/// out[o*out_stride + r] = sum_k W[r][k] * x[o*x_stride + k] for o in
/// [0, tokens). `tokens` must be in [2, 5]. One wave owns an output row and
/// dots the weight row against all activation rows in a single walk, so the
/// matrix is read once for the whole token block. Each per-row dot is
/// bit-identical to the matching Gemv call.
void GemvRows(const void* base, GemvType type, std::uint32_t tokens,
              std::uint32_t rows, std::uint32_t cols, std::size_t row_bytes,
              const float* x, std::uint32_t x_stride, float* out,
              std::uint32_t out_stride, hipStream_t stream);

/// Grouped decode GEMV over a stacked expert matrix. For slot s in [0, used)
/// with expert id ids[s]: out[s*rows + r] = sum_k W[e][r][k] *
/// x[(s/slots_per_row)*x_stride + k]. `slots_per_row` is the number of
/// consecutive slots fed by one activation row (ids stay in slot order,
/// [rows][used] when rows > 1): pass the slot count of one row to keep every
/// slot reading the same x row. `expert_stride` is the byte distance between
/// stacked experts (row_bytes * rows). One launch covers all selected experts,
/// so the caller never reads ids back to the host.
void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t slots_per_row, std::uint32_t rows,
                 std::uint32_t cols, const float* x, std::uint32_t x_stride,
                 float* out, hipStream_t stream);

/// out[i] = W[row][i] for i in [0, cols): dequantizes a single row of a
/// row-major [rows x cols] matrix stored in `type` (the token-embedding
/// lookup). `base` points at the matrix start; the row offset is derived from
/// `cols` and `type`. Asynchronous on `stream` (null uses the default stream).
void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream);

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_GEMV_HPP_
#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_ROUTED_F16_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_ROUTED_F16_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Routed F16 WMMA expert GEMM for the Qwen3.6-35B-A3B MoE prefill, ported
/// from the Qwen3.8-Flash-Next route (src/models/qwen38_flash_next/kernels/
/// rocm/kernels.hip.cpp). The routed assignments are compacted by expert into
/// 16-row padded buckets; activations are narrowed to F16; one WMMA GEMM per
/// (expert, row tile) runs on the matrix cores with the weights dequantized to
/// F16 after the LDS read. Nothing here is shared with another model.
namespace gufo::models::qwen36_a3b::rocm {

/// GGUF type ids the routed F16 GEMM accepts for the expert weights.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
};

/// Narrows `count` F32 activations to F16 (or BF16 when `bf16`) in `out`.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// Rows a compacted bucket layout needs for `slots` routed assignments over
/// `n_experts` experts with 16-row padded buckets.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);

/// Compacts the routed assignments by expert into 16-row padded buckets.
/// `ids` [n_tokens * k] holds the chosen expert per (token, slot) or -1;
/// `counts` [n_experts] is the per-expert histogram. `pad_bounds`
/// [n_experts + 1] receives the exclusive scan of the padded bucket starts,
/// `cursors` [n_experts] is scratch, and `rows_token` / `rows_slot`
/// (RoutedCompactRows long) map each compacted row back to its token and slot.
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed F16 WMMA expert GEMM. `w` is the [n_experts][m][k] weight tensor in
/// `type`; `x` is the F16 activation rows indexed by `rows_in`; `tiles`
/// [n_tiles] packs the expert in the low 16 bits and the token macro-tile
/// index in the high 16. `pad_bounds` [n_experts + 1] gives the bucket starts.
/// Each compacted output row `rows_out[c]` receives the result as F32 `out`
/// (m-wide) or, when `out_half` is given, as F16 with the SwiGLU gate applied
/// from `swiglu_gate`. Exactly one of `out` / `out_half` is non-null.
/// `tile_rows` is 16, 48 or 64. Returns false for an unsupported shape.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Fused gate+up routed GEMM. `gate` and `up` are the [n_experts][m][k] weight
/// tensors in the same `type`; the kernel reads the F16 activation rows once
/// and writes the SwiGLU product `up * SiluF(gate)` as F16 to `out_half`
/// (indexed by `rows_out`). `tile_rows` is 64 or 128. Bit-identical to a gate
/// GEMM into F32 followed by an up GEMM with `swiglu_gate`, but one launch and
/// no gate round trip. Returns false for an unsupported shape.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out_half,
                        std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_ROUTED_F16_HPP_
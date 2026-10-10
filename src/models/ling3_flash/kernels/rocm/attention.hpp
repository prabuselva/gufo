#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_ATTENTION_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_ATTENTION_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ling3_flash::rocm {

/// Absorbed MLA decode attention over the latent cache, mirroring the CPU
/// oracle's MlaAttention softmax and latent-weighted sum. Split-KV: one
/// partial pass per position tile, then a combine pass per head.
///
/// `q_abs` is [heads][kv_rank + rotary] (wk_b-mapped nope half plus the
/// rotated rope half), `kv` is [n_pos][kv_rank] and `kpe` is [n_pos][rotary]
/// (both already containing the current row). The workspace holds
/// `tiles * heads * (kv_rank + 2)` floats: per (tile, head) the running max,
/// the sum of exponentials, and the unnormalized latent sum.

/// Number of position tiles for `n_pos` positions at `tile` positions each.
std::uint32_t MlaDecodeTiles(std::uint32_t n_pos, std::uint32_t tile);

/// Partial pass: writes per (tile, head) max/sumexp/latent into `workspace`.
void MlaDecodePartial(const float* q_abs, const float* kv, const float* kpe,
                      std::uint32_t n_pos, std::uint32_t tile, float scale,
                      float* workspace, std::uint32_t heads,
                      std::uint32_t kv_rank, std::uint32_t rotary,
                      hipStream_t stream);

/// Combine pass: reduces the tile partials into `latent` [heads][kv_rank].
void MlaDecodeCombine(const float* workspace, std::uint32_t tiles,
                      std::uint32_t heads, std::uint32_t kv_rank, float* latent,
                      hipStream_t stream);

/// Chunked causal absorbed-MLA prefill attention, one workgroup per head
/// per 32-token query tile, streaming KV tiles through shared memory with
/// an online softmax (adapted from the Qwen3.6 tiled prefill kernel; the
/// KV matrix doubles as the value matrix and there is no output gate).
///
/// `abs` is [tokens][heads][kv_rank] (the attn_k_b-absorbed Q rows) and
/// `qrp` is [tokens][heads][rotary] (the rotated rope halves), so the score
/// for query `t` at cache position `j` is `dot(abs[t][h], kv[j]) +
/// dot(qrp[t][h], kpe[j])`. `kv`/`kpe` are the global position caches and
/// must already contain rows up to `start_pos + tokens - 1`; position `j`
/// is visible to query `t` iff `j <= start_pos + t`. `latent` receives
/// [tokens][heads][kv_rank]. Requires kv_rank <= 512 and rotary <= 64.
void MlaPrefill(const float* q_abs, const float* q_rope, const float* kv,
                const float* kpe, float* latent, std::uint32_t start_pos,
                std::uint32_t tokens, std::uint32_t heads,
                std::uint32_t kv_rank, std::uint32_t rotary, float scale,
                hipStream_t stream);

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_ATTENTION_HPP_
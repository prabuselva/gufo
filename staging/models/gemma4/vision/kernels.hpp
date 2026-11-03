#ifndef GUFO_MODELS_GEMMA4_VISION_KERNELS_HPP_
#define GUFO_MODELS_GEMMA4_VISION_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Device operators for the `gemma4v` vision tower that are not covered by the
/// trunk's shared GEMM/RMSNorm tier. Every launcher mirrors the scalar oracle
/// in `vision/reference.cpp` exactly: the 16x16 patch im2col scaled to [-1,1],
/// the two stacked x/y positional tables, the 2D NEOX rope (theta 100, x over
/// dims [0,36), y over [36,72)), the geglu_quick FFN activation, the 3x3
/// average-pool merger with its sqrt(1152) scale, the per-channel
/// standardization and the non-causal attention with no 1/sqrt(d) scaling.
/// Buffers are device pointers; launches are asynchronous on `stream`.
namespace gufo::models::gemma4::vision {

/// Builds the im2col matrix [n_patches x 768] from CHW pixels in [0,1],
/// scaling each value to [-1,1]. Column order matches the patch-embed kernel's
/// [cin][ky][kx] layout with kx fastest: col = kx + 16*(ky + 16*cin).
void PatchEmbedIm2Col(const float* pixels, float* patchmat, std::uint32_t nx,
                      std::uint32_t ny, std::uint32_t n_px,
                      std::uint32_t n_patches, hipStream_t stream);

/// Adds the x and y positional rows to every patch: for patch p,
/// h[p] += pos_x[p % n_px] + pos_y[p / n_px] over `emb` features.
void PosAdd(float* h, const float* pos_x, const float* pos_y,
            std::uint32_t n_px, std::uint32_t n_patches, std::uint32_t emb,
            hipStream_t stream);

/// In-place 2D NEOX rope over a [n_patches x heads*hd] tensor. `inv_freq` holds
/// hd/4 entries (18 for hd=72); pass 0 rotates dims [0,hd/2) by the x grid
/// index, pass 1 rotates dims [hd/2,hd) by the y grid index.
void Rope2d(float* qk, const float* inv_freq, std::uint32_t n_px,
            std::uint32_t n_patches, std::uint32_t heads, std::uint32_t hd,
            hipStream_t stream);

/// act[i] = (g[i] / (1 + exp(-1.702 * g[i]))) * up[i] over `count` elements.
void GegluQuick(const float* gate, const float* up, float* act,
                std::size_t count, hipStream_t stream);

/// x[t][e] = (x[t][e] - bias[e]) * scale[e] over n_tokens rows of `emb`.
void StdNorm(float* x, const float* bias, const float* scale,
             std::uint32_t n_tokens, std::uint32_t emb, hipStream_t stream);

/// 3x3 stride-3 average pool over the [n_py x n_px] patch grid, each output
/// row scaled by `pool_scale`. Input [n_py*n_px x emb], output
/// [(n_py/3)*(n_px/3) x emb].
void AvgPool3(const float* h, float* pooled, std::uint32_t n_px,
              std::uint32_t n_py, std::uint32_t emb, float pool_scale,
              hipStream_t stream);

/// Non-causal single-head-group attention. One block per (query, head); online
/// softmax over all keys with no 1/sqrt(d) scaling. q/k/v/out are
/// [n_patches x heads*hd].
void VisionAttention(const float* q, const float* k, const float* v, float* out,
                     std::uint32_t n_patches, std::uint32_t heads,
                     std::uint32_t hd, hipStream_t stream);

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_KERNELS_HPP_
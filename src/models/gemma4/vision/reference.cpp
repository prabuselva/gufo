#include "src/models/gemma4/vision/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/gemma4/cpu_ops.hpp"

namespace gufo::models::gemma4::vision {
namespace {

[[nodiscard]] float GeluQuick(float x) noexcept {
  return x / (1.0F + std::exp(-1.702F * x));
}

// 2D NEOX rope over one head of width `head_dim`. Pass 1 rotates dims
// [0, half) by `pos_x`, pass 2 rotates dims [half, head_dim) by `pos_y`,
// where half = head_dim / 2 and inv_freq[i] = theta^(-2i/head_dim).
void Rope2d(float* head, std::uint32_t head_dim, std::uint32_t half,
            std::uint32_t pos_x, std::uint32_t pos_y, float theta) {
  const std::uint32_t quarter = half / 2;
  for (std::uint32_t pass = 0; pass < 2; ++pass) {
    const std::uint32_t base = pass * half;
    const float pos = static_cast<float>(pass == 0 ? pos_x : pos_y);
    for (std::uint32_t i = 0; i < quarter; ++i) {
      const float inv = std::pow(
          theta, -2.0F * static_cast<float>(i) / static_cast<float>(head_dim));
      const float angle = pos * inv;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float x0 = head[base + i];
      const float x1 = head[base + i + quarter];
      head[base + i] = x0 * c - x1 * s;
      head[base + i + quarter] = x0 * s + x1 * c;
    }
  }
}

}  // namespace

ReferenceEncoder::ReferenceEncoder(const VisionWeights& weights)
    : w_(weights) {}

bool ReferenceEncoder::Encode(const float* pixels, std::uint32_t nx,
                              std::uint32_t ny, std::vector<float>& out,
                              std::string* error_msg) {
  const Config& c = w_.config;
  const std::uint32_t patch = c.patch_size;
  const std::uint32_t merge = Config::kMergeSize;
  const std::uint32_t emb = c.embedding_length;
  const std::uint32_t heads = c.head_count;
  const std::uint32_t hd = c.HeadDim();
  const std::uint32_t half = hd / 2;
  const float eps = c.layer_norm_epsilon;
  if (nx % patch != 0 || ny % patch != 0 || (nx / patch) % merge != 0 ||
      (ny / patch) % merge != 0) {
    if (error_msg != nullptr) {
      *error_msg = "image grid is not a multiple of patch * merge";
    }
    return false;
  }
  const std::uint32_t n_px = nx / patch;
  const std::uint32_t n_py = ny / patch;
  const std::uint32_t n_patches = n_px * n_py;

  // Patch embedding: conv over pixels scaled to [-1,1]. Kernel is F32,
  // [cout][cin][ky][kx] with kx fastest (cols = 768). Output patch-major.
  std::vector<float> h(static_cast<std::size_t>(n_patches) * emb, 0.0F);
  const auto* kernel = static_cast<const float*>(w_.patch_embed.data);
  for (std::uint32_t p = 0; p < n_patches; ++p) {
    const std::uint32_t ox = p % n_px;
    const std::uint32_t oy = p / n_px;
    float* row = h.data() + static_cast<std::size_t>(p) * emb;
    for (std::uint32_t co = 0; co < emb; ++co) {
      const float* krow = kernel + static_cast<std::size_t>(co) * 768;
      float acc = 0.0F;
      for (std::uint32_t ci = 0; ci < 3; ++ci) {
        for (std::uint32_t ky = 0; ky < patch; ++ky) {
          for (std::uint32_t kx = 0; kx < patch; ++kx) {
            const float raw =
                pixels[(ci * ny + (oy * patch + ky)) * nx + (ox * patch + kx)];
            acc += (raw * 2.0F - 1.0F) * krow[kx + patch * (ky + patch * ci)];
          }
        }
      }
      row[co] = acc;
    }
  }

  // Learned 2D positional tables (F32, [1152][10240], feature fastest).
  const auto* pos_x = static_cast<const float*>(w_.pos_x.data);
  const auto* pos_y = static_cast<const float*>(w_.pos_y.data);
  for (std::uint32_t p = 0; p < n_patches; ++p) {
    const float* tx = pos_x + static_cast<std::size_t>(p % n_px) * emb;
    const float* ty = pos_y + static_cast<std::size_t>(p / n_px) * emb;
    float* row = h.data() + static_cast<std::size_t>(p) * emb;
    for (std::uint32_t e = 0; e < emb; ++e) {
      row[e] += tx[e] + ty[e];
    }
  }

  std::vector<float> xn(emb), attn(emb), tmp(emb);
  std::vector<float> g(c.feed_forward_length), cur(c.feed_forward_length);
  std::vector<float> q_all(static_cast<std::size_t>(n_patches) * emb);
  std::vector<float> k_all(static_cast<std::size_t>(n_patches) * emb);
  std::vector<float> v_all(static_cast<std::size_t>(n_patches) * emb);
  std::vector<float> scores(n_patches);

  for (std::uint32_t il = 0; il < c.block_count; ++il) {
    const BlockWeights& b = w_.blocks[il];
    const auto* qn = static_cast<const float*>(b.q_norm.data);
    const auto* kn = static_cast<const float*>(b.k_norm.data);

    // Pass A: pre-norm, QKV projections, per-head Q/K norm + 2D rope, V norm.
    for (std::uint32_t p = 0; p < n_patches; ++p) {
      const float* hp = h.data() + static_cast<std::size_t>(p) * emb;
      std::copy(hp, hp + emb, xn.begin());
      cpu::RmsNorm(std::span<float>(xn.data(), emb),
                   static_cast<const float*>(b.ln1.data), eps);
      float* qp = q_all.data() + static_cast<std::size_t>(p) * emb;
      float* kp = k_all.data() + static_cast<std::size_t>(p) * emb;
      float* vp = v_all.data() + static_cast<std::size_t>(p) * emb;
      cpu::MatVec(b.attn_q, 0, std::span<const float>(xn.data(), emb),
                  std::span<float>(qp, emb));
      cpu::MatVec(b.attn_k, 0, std::span<const float>(xn.data(), emb),
                  std::span<float>(kp, emb));
      cpu::MatVec(b.attn_v, 0, std::span<const float>(xn.data(), emb),
                  std::span<float>(vp, emb));
      const std::uint32_t gx = p % n_px;
      const std::uint32_t gy = p / n_px;
      for (std::uint32_t hh = 0; hh < heads; ++hh) {
        const std::size_t off = static_cast<std::size_t>(hh) * hd;
        cpu::RmsNorm(std::span<float>(qp + off, hd), qn, eps);
        cpu::RmsNorm(std::span<float>(kp + off, hd), kn, eps);
        Rope2d(qp + off, hd, half, gx, gy, Config::kRopeTheta);
        Rope2d(kp + off, hd, half, gx, gy, Config::kRopeTheta);
        cpu::RmsNorm(std::span<float>(vp + off, hd), nullptr, eps);
      }
    }

    // Pass B: non-causal attention per head (no 1/sqrt(d) scaling), then the
    // output projection, post-norm and residual add.
    for (std::uint32_t p = 0; p < n_patches; ++p) {
      float* ap = attn.data();
      const float* qp = q_all.data() + static_cast<std::size_t>(p) * emb;
      for (std::uint32_t hh = 0; hh < heads; ++hh) {
        const std::size_t off = static_cast<std::size_t>(hh) * hd;
        for (std::uint32_t i = 0; i < n_patches; ++i) {
          const float* ki =
              k_all.data() + static_cast<std::size_t>(i) * emb + off;
          const float* qi = qp + off;
          float dot = 0.0F;
          for (std::uint32_t d = 0; d < hd; ++d) {
            dot += ki[d] * qi[d];
          }
          scores[i] = dot;
        }
        cpu::Softmax(std::span<float>(scores.data(), n_patches));
        float* oh = ap + off;
        for (std::uint32_t d = 0; d < hd; ++d) {
          oh[d] = 0.0F;
        }
        for (std::uint32_t i = 0; i < n_patches; ++i) {
          const float* vi =
              v_all.data() + static_cast<std::size_t>(i) * emb + off;
          const float w = scores[i];
          for (std::uint32_t d = 0; d < hd; ++d) {
            oh[d] += w * vi[d];
          }
        }
      }
      float* hp = h.data() + static_cast<std::size_t>(p) * emb;
      cpu::MatVec(b.attn_out, 0, std::span<const float>(attn.data(), emb),
                  std::span<float>(tmp.data(), emb));
      cpu::RmsNorm(std::span<float>(tmp.data(), emb),
                   static_cast<const float*>(b.attn_post_norm.data), eps);
      for (std::uint32_t e = 0; e < emb; ++e) {
        hp[e] += tmp[e];
      }
    }

    // Pass C: geglu_quick FFN, post-norm and residual add.
    for (std::uint32_t p = 0; p < n_patches; ++p) {
      float* hp = h.data() + static_cast<std::size_t>(p) * emb;
      std::copy(hp, hp + emb, xn.begin());
      cpu::RmsNorm(std::span<float>(xn.data(), emb),
                   static_cast<const float*>(b.ln2.data), eps);
      cpu::MatVec(b.ffn_up, 0, std::span<const float>(xn.data(), emb),
                  std::span<float>(cur.data(), c.feed_forward_length));
      cpu::MatVec(b.ffn_gate, 0, std::span<const float>(xn.data(), emb),
                  std::span<float>(g.data(), c.feed_forward_length));
      for (std::uint32_t f = 0; f < c.feed_forward_length; ++f) {
        cur[f] = GeluQuick(g[f]) * cur[f];
      }
      cpu::MatVec(b.ffn_down, 0,
                  std::span<const float>(cur.data(), c.feed_forward_length),
                  std::span<float>(tmp.data(), emb));
      cpu::RmsNorm(std::span<float>(tmp.data(), emb),
                   static_cast<const float*>(b.ffn_post_norm.data), eps);
      for (std::uint32_t e = 0; e < emb; ++e) {
        hp[e] += tmp[e];
      }
    }
  }

  // Merger: 3x3 average pool over the [n_py][n_px] patch grid, then per-token
  // sqrt(emb) scale, per-channel standardization, weightless RMSNorm and the
  // clippable projection to the trunk width.
  const std::uint32_t out_x = n_px / merge;
  const std::uint32_t out_y = n_py / merge;
  const std::uint32_t n_tokens = out_x * out_y;
  const float pool_scale = std::sqrt(static_cast<float>(emb));
  const auto* std_bias = static_cast<const float*>(w_.std_bias.data);
  const auto* std_scale = static_cast<const float*>(w_.std_scale.data);
  out.assign(static_cast<std::size_t>(n_tokens) * c.projection_dim, 0.0F);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t px = t % out_x;
    const std::uint32_t py = t / out_x;
    for (std::uint32_t e = 0; e < emb; ++e) {
      float acc = 0.0F;
      for (std::uint32_t dy = 0; dy < merge; ++dy) {
        for (std::uint32_t dx = 0; dx < merge; ++dx) {
          const std::uint32_t gx = px * merge + dx;
          const std::uint32_t gy = py * merge + dy;
          acc += h[(static_cast<std::size_t>(gy) * n_px + gx) * emb + e];
        }
      }
      xn[e] = (acc / static_cast<float>(merge * merge)) * pool_scale;
    }
    for (std::uint32_t e = 0; e < emb; ++e) {
      xn[e] = (xn[e] - std_bias[e]) * std_scale[e];
    }
    cpu::RmsNorm(std::span<float>(xn.data(), emb), nullptr, eps);
    cpu::MatVec(w_.projection, 0, std::span<const float>(xn.data(), emb),
                std::span<float>(
                    out.data() + static_cast<std::size_t>(t) * c.projection_dim,
                    c.projection_dim));
  }
  return true;
}

}  // namespace gufo::models::gemma4::vision
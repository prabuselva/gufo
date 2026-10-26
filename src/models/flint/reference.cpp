#include "src/models/flint/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "src/models/flint/cpu_ops.hpp"

namespace gufo::models::flint {
namespace {

[[nodiscard]] const float* NormPtr(const TensorRef& t) noexcept {
  return static_cast<const float*>(t.data);
}

struct ExpertScore {
  float prob;
  std::uint32_t index;
};

/// llama.cpp norm_w clamp: the smallest F16 value (non-DEEPSEEK41 path).
constexpr float kWeightSumFloor = 6.103515625e-5F;

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights), c_(weights.config), max_context_(max_context) {
  layers_.resize(c_.num_layers);
}

void ReferenceModel::Reset() {
  position_ = 0;
  for (auto& s : layers_) {
    s.k.clear();
    s.v.clear();
  }
}

void ReferenceModel::Attention(const LayerWeights& l, AttentionState& s,
                               std::uint32_t layer, std::uint32_t pos,
                               std::span<const float> normed,
                               std::span<float> out) {
  const std::uint32_t head_dim = c_.head_dim;
  const std::uint32_t n_head = c_.num_heads;
  const std::uint32_t kv_dim = c_.AttentionKvDim();
  std::vector<float> q(c_.AttentionQDim());
  std::vector<float> k(kv_dim);
  std::vector<float> v(kv_dim);
  cpu::MatVec(l.attn_q, 0, normed, q);
  cpu::MatVec(l.attn_k, 0, normed, k);
  cpu::MatVec(l.attn_v, 0, normed, v);

  // LLM_NORM_RMS per head over one head_dim, before the gated rotary.
  for (std::uint32_t h = 0; h < n_head; ++h) {
    cpu::RmsNorm(
        std::span(q.data() + static_cast<std::size_t>(h) * head_dim, head_dim),
        NormPtr(l.attn_q_norm), c_.rms_eps);
  }
  for (std::uint32_t h = 0; h < c_.num_kv_heads; ++h) {
    cpu::RmsNorm(
        std::span(k.data() + static_cast<std::size_t>(h) * head_dim, head_dim),
        NormPtr(l.attn_k_norm), c_.rms_eps);
  }
  if (c_.HasRope(layer)) {
    cpu::Rope(q.data(), n_head, head_dim, pos, c_.rope_theta);
    cpu::Rope(k.data(), c_.num_kv_heads, head_dim, pos, c_.rope_theta);
  }

  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  const std::uint32_t start =
      (c_.IsWindowedLayer(layer) && pos + 1 > c_.sliding_window)
          ? pos + 1 - c_.sliding_window
          : 0;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::fill(out.begin(), out.end(), 0.0F);
  for (std::uint32_t h = 0; h < n_head; ++h) {
    const std::uint32_t kvh = h * c_.num_kv_heads / n_head;
    const float* qh = q.data() + static_cast<std::size_t>(h) * head_dim;
    const float sink = NormPtr(l.attn_sinks)[h];
    std::vector<float> scores(pos + 1);
    float max = sink;
    for (std::uint32_t p = start; p <= pos; ++p) {
      const float* kp =
          s.k.data() +
          (static_cast<std::size_t>(p) * c_.num_kv_heads + kvh) * head_dim;
      double acc = 0.0;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        acc += static_cast<double>(qh[d]) * kp[d];
      }
      scores[p] = static_cast<float>(acc) * scale;
      max = std::max(max, scores[p]);
    }
    double sum = std::exp(sink - max);  // sink as extra logit
    for (std::uint32_t p = start; p <= pos; ++p) {
      const float p_e = std::exp(scores[p] - max);
      sum += p_e;
      const float* vp =
          s.v.data() +
          (static_cast<std::size_t>(p) * c_.num_kv_heads + kvh) * head_dim;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        out[static_cast<std::size_t>(h) * head_dim + d] += p_e * vp[d];
      }
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      out[static_cast<std::size_t>(h) * head_dim + d] *= inv;
    }
  }
}

void ReferenceModel::Ffn(const LayerWeights& l, std::span<const float> normed,
                         std::span<float> out) {
  std::fill(out.begin(), out.end(), 0.0F);
  if (!l.moe) {
    std::vector<float> gate(c_.ffn_length);
    std::vector<float> up(c_.ffn_length);
    cpu::MatVec(l.ffn_gate, 0, normed, gate);
    cpu::MatVec(l.ffn_up, 0, normed, up);
    for (std::uint32_t i = 0; i < c_.ffn_length; ++i) {
      gate[i] = cpu::Silu(gate[i]) * up[i];
    }
    cpu::MatVec(l.ffn_down, 0, gate, out);
    return;
  }

  // Sigmoid router, top-k by probability (ties keep the lower index, like
  // ggml argsort), weights renormalised by the clamped top-k sum.
  std::vector<float> probs(c_.num_experts);
  cpu::MatVec(l.router, 0, normed, probs);
  for (float& p : probs) {
    p = cpu::Sigmoid(p);
  }
  std::vector<ExpertScore> scores(c_.num_experts);
  for (std::uint32_t i = 0; i < c_.num_experts; ++i) {
    scores[i] = {probs[i], i};
  }
  const std::size_t used = c_.num_experts_used;
  std::partial_sort(scores.begin(), scores.begin() + used, scores.end(),
                    [](const ExpertScore& a, const ExpertScore& b) {
                      return a.prob != b.prob ? a.prob > b.prob
                                              : a.index < b.index;
                    });
  std::vector<float> weights(used);
  float wsum = 0.0F;
  for (std::size_t i = 0; i < used; ++i) {
    weights[i] = probs[scores[i].index];
    wsum += weights[i];
  }
  if (c_.expert_weights_norm) {
    if (wsum < kWeightSumFloor) {
      wsum = kWeightSumFloor;
    }
    for (float& w : weights) {
      w /= wsum;
    }
  }
  if (c_.expert_weights_scale != 0.0F && c_.expert_weights_scale != 1.0F) {
    for (float& w : weights) {
      w *= c_.expert_weights_scale;
    }
  }

  std::vector<float> gate(c_.expert_ff);
  std::vector<float> up(c_.expert_ff);
  std::vector<float> down(c_.hidden_size);
  for (std::size_t i = 0; i < used; ++i) {
    const std::uint64_t e = scores[i].index;
    cpu::MatVec(l.ffn_gate_exps, e, normed, gate);
    cpu::MatVec(l.ffn_up_exps, e, normed, up);
    for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
      gate[j] = cpu::Silu(gate[j]) * up[j];
    }
    cpu::MatVec(l.ffn_down_exps, e, gate, down);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      out[j] += weights[i] * down[j];
    }
  }

  // One shared expert stream, added once outside the routed sum.
  std::vector<float> sgate(c_.shared_expert_ff);
  std::vector<float> sup(c_.shared_expert_ff);
  std::vector<float> sdown(c_.hidden_size);
  cpu::MatVec(l.shexp_gate, 0, normed, sgate);
  cpu::MatVec(l.shexp_up, 0, normed, sup);
  for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
    sgate[j] = cpu::Silu(sgate[j]) * sup[j];
  }
  cpu::MatVec(l.shexp_down, 0, sgate, sdown);
  for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
    out[j] += sdown[j];
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg) {
  if (token < 0 || static_cast<std::uint64_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id outside the vocabulary";
    }
    return false;
  }
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "reference cache exhausted";
    }
    return false;
  }
  if (logits.size() != c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "logits span has the wrong width";
    }
    return false;
  }

  std::vector<float> x(c_.hidden_size);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     x.data());

  std::vector<float> normed(c_.hidden_size);
  std::vector<float> attn(c_.AttentionQDim());
  std::vector<float> ffn(c_.hidden_size);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const auto& l = w_.layers[il];
    std::copy(x.begin(), x.end(), normed.begin());
    cpu::RmsNorm(normed, NormPtr(l.attn_norm), c_.rms_eps);
    Attention(l, layers_[il], il, position_, normed, attn);
    cpu::MatVec(l.attn_out, 0, attn, ffn);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      x[j] += ffn[j];
    }

    std::copy(x.begin(), x.end(), normed.begin());
    cpu::RmsNorm(normed, NormPtr(l.ffn_norm), c_.rms_eps);
    Ffn(l, normed, ffn);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      x[j] += ffn[j];
    }
  }

  cpu::RmsNorm(x, NormPtr(w_.output_norm), c_.rms_eps);
  cpu::MatVec(w_.output, 0, x, logits);
  ++position_;
  return true;
}

}  // namespace gufo::models::flint

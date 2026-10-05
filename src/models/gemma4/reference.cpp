#include "src/models/gemma4/reference.hpp"

#include <algorithm>
#include <cmath>

#include "src/models/gemma4/cpu_ops.hpp"

namespace gufo::models::gemma4 {
namespace {

// Division guard for the renormalized expert weights, matching the runtime.
constexpr float kMinWeightSum = 6.103515625e-5F;

[[nodiscard]] std::span<float> Head(std::span<float> x, std::uint32_t h,
                                    std::uint32_t head_dim) {
  return x.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
}

// The runtime stores the KV cache in F16, so the oracle narrows the cached K/V
// through the same round-to-nearest-even F16 representation. Without this the
// unscaled (no 1/sqrt(head_dim)) attention scores span hundreds, making the
// softmax hypersensitive to the F32-vs-F16 cache gap; the residual then
// reflects only the float-vs-double GEMV drift the parity bound already covers.
void RoundF16(std::span<float> x) {
  for (float& v : x) {
    v = static_cast<float>(static_cast<_Float16>(v));
  }
}

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights),
      c_(weights.config),
      max_context_(max_context),
      attention_(weights.config.num_layers) {}

void ReferenceModel::Reset() {
  position_ = 0;
  for (auto& state : attention_) {
    state.k.clear();
    state.v.clear();
  }
}

void ReferenceModel::Attention(const AttentionState& s,
                               std::span<const float> q,
                               std::uint32_t num_heads,
                               std::uint32_t num_kv_heads,
                               std::uint32_t head_dim, std::uint32_t window,
                               std::span<float> out) {
  const std::size_t kv_stride =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::uint32_t n_past =
      static_cast<std::uint32_t>(s.k.size() / kv_stride);
  const std::uint32_t lo =
      (window > 0 && n_past > window) ? n_past - window : 0;
  const std::uint32_t group = num_heads / num_kv_heads;
  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> scores(n_past - lo);
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q.data() + static_cast<std::size_t>(h) * head_dim;
    for (std::size_t p = lo; p < n_past; ++p) {
      const float* kp =
          s.k.data() + p * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
      double acc = 0.0;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        acc += static_cast<double>(qh[d]) * kp[d];
      }
      scores[p - lo] = static_cast<float>(acc);
    }
    cpu::Softmax(scores);
    for (std::size_t p = lo; p < n_past; ++p) {
      const float* vp =
          s.v.data() + p * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
      const float w = scores[p - lo];
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        out[static_cast<std::size_t>(h) * head_dim + d] += w * vp[d];
      }
    }
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::span<float> h_out, std::string* error_msg) {
  const std::size_t hidden = c_.hidden_size;
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }
  if (!logits.empty() && logits.size() < c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "logits span too small";
    }
    return false;
  }
  if (!h_out.empty() && h_out.size() < hidden) {
    if (error_msg != nullptr) {
      *error_msg = "h_out span too small";
    }
    return false;
  }

  std::vector<float> cur(hidden);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     cur.data());
  const float emb_scale = std::sqrt(static_cast<float>(hidden));
  for (float& v : cur) {
    v *= emb_scale;
  }

  std::vector<float> normed(hidden), q_proj, kv_proj, v_proj, attn, proj,
      attn_out(hidden), mlp, moe(hidden), gu, act, dn, router_in, router_logits;

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const auto& l = w_.layers[il];
    const std::uint32_t head_dim = c_.HeadDim(il);
    const std::uint32_t num_heads = c_.num_heads;
    const std::uint32_t num_kv_heads = c_.NumKvHeads(il);
    const bool swa = c_.IsSwa(il);

    normed.assign(cur.begin(), cur.end());
    cpu::RmsNorm(normed, static_cast<const float*>(l.attn_norm.data),
                 c_.rms_eps);

    // Q: project, per-head RMSNorm, NEOX rope (proportional factors on full
    // layers only).
    q_proj.assign(static_cast<std::size_t>(num_heads) * head_dim, 0.0F);
    cpu::MatVec(l.attn_q, 0, normed, q_proj);
    for (std::uint32_t h = 0; h < num_heads; ++h) {
      cpu::RmsNorm(Head(q_proj, h, head_dim),
                   static_cast<const float*>(l.attn_q_norm.data), c_.rms_eps);
    }
    cpu::RopeNeox(
        q_proj.data(), num_heads, head_dim, c_.RopeDim(il), position_,
        c_.RopeTheta(il),
        swa ? nullptr : static_cast<const float*>(w_.rope_freqs.data));

    // K: project, per-head RMSNorm, rope. V: weightless RMSNorm of the V (or
    // K on full layers) projection, never rotated.
    kv_proj.assign(static_cast<std::size_t>(num_kv_heads) * head_dim, 0.0F);
    cpu::MatVec(l.attn_k, 0, normed, kv_proj);
    if (l.attn_v.empty()) {
      v_proj = kv_proj;
    } else {
      v_proj.assign(static_cast<std::size_t>(num_kv_heads) * head_dim, 0.0F);
      cpu::MatVec(l.attn_v, 0, normed, v_proj);
    }
    for (std::uint32_t h = 0; h < num_kv_heads; ++h) {
      cpu::RmsNorm(Head(kv_proj, h, head_dim),
                   static_cast<const float*>(l.attn_k_norm.data), c_.rms_eps);
      cpu::RmsNorm(Head(v_proj, h, head_dim), nullptr, c_.rms_eps);
    }
    cpu::RopeNeox(
        kv_proj.data(), num_kv_heads, head_dim, c_.RopeDim(il), position_,
        c_.RopeTheta(il),
        swa ? nullptr : static_cast<const float*>(w_.rope_freqs.data));

    auto& state = attention_[il];
    RoundF16(kv_proj);
    RoundF16(v_proj);
    state.k.insert(state.k.end(), kv_proj.begin(), kv_proj.end());
    state.v.insert(state.v.end(), v_proj.begin(), v_proj.end());

    attn.assign(q_proj.size(), 0.0F);
    Attention(state, q_proj, num_heads, num_kv_heads, head_dim,
              swa ? c_.sliding_window : 0, attn);

    proj.assign(hidden, 0.0F);
    cpu::MatVec(l.attn_output, 0, attn, proj);
    cpu::RmsNorm(proj, static_cast<const float*>(l.post_attention_norm.data),
                 c_.rms_eps);
    for (std::size_t i = 0; i < hidden; ++i) {
      attn_out[i] = proj[i] + cur[i];
    }

    // Shared dense FFN: down( gelu(gate(x)) * up(x) ), then its post-norm.
    normed.assign(attn_out.begin(), attn_out.end());
    cpu::RmsNorm(normed, static_cast<const float*>(l.ffn_norm.data),
                 c_.rms_eps);
    const std::size_t ffn = c_.ffn_length;
    act.assign(ffn, 0.0F);
    std::vector<float> gate(ffn, 0.0F);
    cpu::MatVec(l.ffn_up, 0, normed, act);
    cpu::MatVec(l.ffn_gate, 0, normed, gate);
    for (std::size_t i = 0; i < ffn; ++i) {
      act[i] = cpu::Gelu(gate[i]) * act[i];
    }
    mlp.assign(hidden, 0.0F);
    cpu::MatVec(l.ffn_down, 0, act, mlp);
    cpu::RmsNorm(mlp, static_cast<const float*>(l.post_ffw_norm_1.data),
                 c_.rms_eps);

    // Router: softmax over the scaled RMS-normal attention output.
    router_in.assign(attn_out.begin(), attn_out.end());
    cpu::RmsNorm(router_in, nullptr, c_.rms_eps);
    const float inv_sqrt = 1.0F / std::sqrt(static_cast<float>(hidden));
    const auto* router_scale = static_cast<const float*>(l.router_scale.data);
    for (std::size_t i = 0; i < hidden; ++i) {
      router_in[i] *= inv_sqrt * router_scale[i];
    }
    router_logits.assign(c_.num_experts, 0.0F);
    cpu::MatVec(l.router, 0, router_in, router_logits);
    cpu::Softmax(router_logits);

    // Top-8 experts by probability, weights renormalized over the selected
    // probabilities.
    const std::size_t used = c_.num_experts_used;
    std::vector<std::uint32_t> selected(c_.num_experts);
    for (std::size_t i = 0; i < selected.size(); ++i) {
      selected[i] = static_cast<std::uint32_t>(i);
    }
    std::partial_sort(selected.begin(), selected.begin() + used, selected.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return router_logits[a] > router_logits[b];
                      });
    double weight_sum = 0.0;
    for (std::size_t k = 0; k < used; ++k) {
      weight_sum += router_logits[selected[k]];
    }
    const float wsum = static_cast<float>(
        std::max(weight_sum, static_cast<double>(kMinWeightSum)));

    normed.assign(attn_out.begin(), attn_out.end());
    cpu::RmsNorm(normed, static_cast<const float*>(l.pre_ffw_norm_2.data),
                 c_.rms_eps);
    const std::size_t expert_ff = c_.expert_ff;
    gu.assign(2 * expert_ff, 0.0F);
    act.assign(expert_ff, 0.0F);
    dn.assign(hidden, 0.0F);
    const auto* down_scale =
        static_cast<const float*>(l.ffn_down_exps_scale.data);
    std::fill(moe.begin(), moe.end(), 0.0F);
    for (std::size_t k = 0; k < used; ++k) {
      const std::uint32_t e = selected[k];
      cpu::MatVec(l.ffn_gate_up_exps, e, normed, gu);
      for (std::size_t i = 0; i < expert_ff; ++i) {
        act[i] = cpu::Gelu(gu[i]) * gu[expert_ff + i];
      }
      cpu::MatVec(l.ffn_down_exps, e, act, dn);
      const float w = router_logits[e] / wsum * down_scale[e];
      for (std::size_t i = 0; i < hidden; ++i) {
        moe[i] += w * dn[i];
      }
    }
    cpu::RmsNorm(moe, static_cast<const float*>(l.post_ffw_norm_2.data),
                 c_.rms_eps);

    for (std::size_t i = 0; i < hidden; ++i) {
      cur[i] = mlp[i] + moe[i];
    }
    cpu::RmsNorm(cur, static_cast<const float*>(l.post_ffw_norm.data),
                 c_.rms_eps);
    for (std::size_t i = 0; i < hidden; ++i) {
      cur[i] += attn_out[i];
    }
    const float layer_scale =
        static_cast<const float*>(l.layer_output_scale.data)[0];
    for (float& v : cur) {
      v *= layer_scale;
    }
  }

  cpu::RmsNorm(cur, static_cast<const float*>(w_.output_norm.data), c_.rms_eps);
  if (!h_out.empty()) {
    std::copy(cur.begin(), cur.end(), h_out.begin());
  }

  if (!logits.empty()) {
    cpu::MatVec(w_.output, 0, cur, logits.subspan(0, c_.vocab_size));
    if (c_.logit_softcap > 0.0F) {
      const float cap = c_.logit_softcap;
      for (std::size_t i = 0; i < c_.vocab_size; ++i) {
        logits[i] = cap * std::tanh(logits[i] / cap);
      }
    }
  }

  ++position_;
  return true;
}

bool ReferenceModel::DraftStep(const DraftWeights& draft, std::int32_t token,
                               std::span<const float> h,
                               std::span<float> logits, std::span<float> h_next,
                               std::string* error_msg) {
  const std::size_t hidden_out = c_.hidden_size;
  const std::size_t hidden = draft.config.hidden_size;
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (h.size() < hidden_out || logits.size() < c_.vocab_size ||
      h_next.size() < hidden_out) {
    if (error_msg != nullptr) {
      *error_msg = "draft span too small";
    }
    return false;
  }

  // xh = [target token embedding * sqrt(hidden_out), trunk hidden state].
  std::vector<float> xh(2 * hidden_out);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     xh.data());
  const float emb_scale = std::sqrt(static_cast<float>(hidden_out));
  for (std::size_t i = 0; i < hidden_out; ++i) {
    xh[i] *= emb_scale;
    xh[hidden_out + i] = h[i];
  }

  std::vector<float> cur(hidden, 0.0F);
  cpu::MatVec(draft.pre_projection, 0, xh, cur);

  std::vector<float> normed(hidden), q_proj, attn, proj(hidden), attn_out, gate,
      act;
  for (std::uint32_t il = 0; il < draft.config.num_layers; ++il) {
    const auto& l = draft.layers[il];
    const auto& dc = draft.config;
    const std::uint32_t head_dim = dc.HeadDim(il);
    const std::uint32_t num_heads = dc.num_heads;
    const bool swa = dc.IsSwa(il);

    normed.assign(cur.begin(), cur.end());
    cpu::RmsNorm(normed, static_cast<const float*>(l.attn_norm.data),
                 dc.rms_eps);

    q_proj.assign(static_cast<std::size_t>(num_heads) * head_dim, 0.0F);
    cpu::MatVec(l.attn_q, 0, normed, q_proj);
    for (std::uint32_t hh = 0; hh < num_heads; ++hh) {
      cpu::RmsNorm(Head(q_proj, hh, head_dim),
                   static_cast<const float*>(l.attn_q_norm.data), dc.rms_eps);
    }
    cpu::RopeNeox(
        q_proj.data(), num_heads, head_dim, dc.RopeDim(il), position_,
        dc.RopeTheta(il),
        swa ? nullptr : static_cast<const float*>(draft.rope_freqs.data));

    attn.assign(q_proj.size(), 0.0F);
    const std::uint32_t target = dc.DraftTargetLayer(il, c_);
    Attention(attention_[target], q_proj, num_heads, dc.NumKvHeads(il),
              head_dim, swa ? dc.sliding_window : 0, attn);

    cpu::MatVec(l.attn_output, 0, attn, proj);
    cpu::RmsNorm(proj, static_cast<const float*>(l.post_attention_norm.data),
                 dc.rms_eps);
    attn_out.resize(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
      attn_out[i] = proj[i] + cur[i];
    }

    normed.assign(attn_out.begin(), attn_out.end());
    cpu::RmsNorm(normed, static_cast<const float*>(l.ffn_norm.data),
                 dc.rms_eps);
    const std::size_t ffn = dc.ffn_length;
    act.assign(ffn, 0.0F);
    gate.assign(ffn, 0.0F);
    cpu::MatVec(l.ffn_up, 0, normed, act);
    cpu::MatVec(l.ffn_gate, 0, normed, gate);
    for (std::size_t i = 0; i < ffn; ++i) {
      act[i] = cpu::Gelu(gate[i]) * act[i];
    }
    proj.assign(hidden, 0.0F);
    cpu::MatVec(l.ffn_down, 0, act, proj);
    cpu::RmsNorm(proj, static_cast<const float*>(l.post_ffw_norm.data),
                 dc.rms_eps);
    for (std::size_t i = 0; i < hidden; ++i) {
      cur[i] = (proj[i] + attn_out[i]) *
               static_cast<const float*>(l.layer_output_scale.data)[0];
    }
  }

  cpu::RmsNorm(cur, static_cast<const float*>(draft.output_norm.data),
               draft.config.rms_eps);
  cpu::MatVec(draft.token_embd, 0, cur, logits.subspan(0, c_.vocab_size));
  cpu::MatVec(draft.post_projection, 0, cur, h_next.subspan(0, hidden_out));
  return true;
}

}  // namespace gufo::models::gemma4
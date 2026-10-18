#include "src/models/ling3_flash/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "src/models/ling3_flash/cpu_ops.hpp"

namespace gufo::models::ling3_flash {

using cpu::DequantizeRow;
using cpu::DotRow;
using cpu::L2Norm;
using cpu::MatVec;
using cpu::RmsNorm;
using cpu::Rope;
using cpu::Sigmoid;
using cpu::Silu;

namespace {

constexpr float kWeightsSumEps = 6.103515625e-5F;  // F16 smallest normal.
constexpr float kClampEps = 1e-6F;

/// Top-`k` indices of `v` by descending value, ties to the smaller index,
/// matching ggml `argsort_top_k`.
std::vector<std::uint32_t> TopK(std::span<const float> v, std::uint32_t k) {
  std::vector<std::uint32_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0U);
  std::stable_sort(
      idx.begin(), idx.end(),
      [&](std::uint32_t a, std::uint32_t b) { return v[a] > v[b]; });
  idx.resize(k);
  return idx;
}

float Clamp(float x, float lo, float hi) {
  return std::min(std::max(x, lo), hi);
}

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights), c_(weights.config), max_context_(max_context) {
  kda_.resize(c_.num_layers);
  mla_.resize(c_.num_layers);
  Reset();
}

void ReferenceModel::Reset() {
  position_ = 0;
  mtp_position_ = 0;
  mtp_state_ = {};
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsMlaLayer(il)) {
      auto& s = mla_[il];
      s.kv.clear();
      s.kpe.clear();
      s.kv.reserve(static_cast<std::size_t>(max_context_) * c_.kv_lora_rank);
      s.kpe.reserve(static_cast<std::size_t>(max_context_) * c_.rotary_dim);
    } else {
      auto& s = kda_[il];
      s.conv.assign(static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * 3 *
                        c_.SsmInnerSize(),
                    0.0F);
      s.state.assign(static_cast<std::size_t>(c_.num_heads) * c_.ssm_head_dim *
                         c_.ssm_head_dim,
                     0.0F);
    }
  }
}

void ReferenceModel::KdaAttention(const LayerWeights& l, KdaState& s,
                                  std::span<const float> x,
                                  std::span<float> out) {
  const std::uint32_t inner = c_.SsmInnerSize();
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  std::vector<float> q(inner);
  std::vector<float> k(inner);
  std::vector<float> v(inner);
  std::vector<float> f(inner);
  std::vector<float> g_proj(inner);
  std::vector<float> beta(heads);
  MatVec(l.ssm_q, 0, x, q);
  MatVec(l.ssm_k, 0, x, k);
  MatVec(l.ssm_v, 0, x, v);
  MatVec(l.ssm_f, 0, x, f);
  MatVec(l.ssm_g, 0, x, g_proj);
  MatVec(l.ssm_beta, 0, x, beta);

  const auto* dt = static_cast<const float*>(l.ssm_dt.data);
  for (std::uint32_t i = 0; i < inner; ++i) {
    f[i] += dt[i];
  }

  // Causal depthwise conv over the last `kern` projections, then SiLU. The
  // GGUF stores each conv as [kern, 1, inner] with the tap index contiguous
  // (ne[0]), so tap t of channel ch lives at conv_w[ch * kern + t]. History
  // rows are oldest first; the current projection is the newest tap.
  const float* convs[3] = {static_cast<const float*>(l.ssm_conv_q.data),
                           static_cast<const float*>(l.ssm_conv_k.data),
                           static_cast<const float*>(l.ssm_conv_v.data)};
  std::vector<float>* proj[3] = {&q, &k, &v};
  const std::size_t hist = static_cast<std::size_t>(kern - 1) * inner;
  for (int which = 0; which < 3; ++which) {
    const float* cw = convs[which];
    std::vector<float>& p = *proj[which];
    // The conv state holds the raw projected rows, not the activated output.
    const std::vector<float> raw = p;
    float* cs = s.conv.data() + static_cast<std::size_t>(which) * hist;
    for (std::uint32_t ch = 0; ch < inner; ++ch) {
      float acc = cw[static_cast<std::size_t>(ch) * kern + (kern - 1)] * p[ch];
      for (std::uint32_t t = 0; t + 1 < kern; ++t) {
        acc += cw[static_cast<std::size_t>(ch) * kern + t] *
               cs[static_cast<std::size_t>(t) * inner + ch];
      }
      p[ch] = Silu(acc);
    }
    if (kern > 1) {
      std::copy(cs + inner, cs + hist, cs);
      std::copy(raw.begin(), raw.end(), cs + hist - inner);
    }
  }

  for (std::uint32_t h = 0; h < heads; ++h) {
    L2Norm(std::span<float>(q.data() + static_cast<std::size_t>(h) * d, d),
           c_.rms_eps);
    L2Norm(std::span<float>(k.data() + static_cast<std::size_t>(h) * d, d),
           c_.rms_eps);
  }
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  for (float& value : q) {
    value *= q_scale;
  }

  const auto* a = static_cast<const float*>(l.ssm_a.data);
  const auto* norm_w = static_cast<const float*>(l.ssm_norm.data);
  std::vector<float> attn(inner);
  std::vector<float> u(d);
  std::vector<float> decay(d);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float* qh = q.data() + static_cast<std::size_t>(h) * d;
    const float* kh = k.data() + static_cast<std::size_t>(h) * d;
    const float* vh = v.data() + static_cast<std::size_t>(h) * d;
    float* S = s.state.data() + static_cast<std::size_t>(h) * d * d;
    const float b = Sigmoid(beta[h]);
    // The per-channel decay addresses the key axis i, broadcast across
    // each value row, as in the ggml gated_delta_net KDA path where
    // exp(g[i]) is applied element-wise to every stored row.
    for (std::uint32_t i = 0; i < d; ++i) {
      decay[i] =
          std::exp(c_.kda_gate_lower_bound *
                   Sigmoid(a[h] * f[static_cast<std::size_t>(h) * d + i]));
    }
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay[i];
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = attn.data() + static_cast<std::size_t>(h) * d;
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc);
    }
    // Per-head RMSNorm, the per-channel weight, then the sigmoid output gate.
    RmsNorm(std::span<float>(o, d), norm_w, c_.rms_eps);
    for (std::uint32_t j = 0; j < d; ++j) {
      o[j] *= Sigmoid(g_proj[static_cast<std::size_t>(h) * d + j]);
    }
  }
  MatVec(l.ssm_out, 0, attn, out);
}

void ReferenceModel::MlaAttention(const LayerWeights& l, MlaState& s,
                                  std::span<const float> x, std::uint32_t pos,
                                  std::span<float> out) {
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t nope = c_.MlaNopeDim();
  const std::uint32_t rot = c_.rotary_dim;
  const std::uint32_t kv_rank = c_.kv_lora_rank;
  const std::uint32_t v_dim = c_.mla_value_dim;
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(c_.mla_head_dim));

  std::vector<float> q_all(c_.MlaQDim());
  std::vector<float> kv_all(kv_rank + rot);
  if (c_.mla_q_split) {
    std::vector<float> q_latent(c_.mla_q_lora);
    MatVec(l.attn_q_a, 0, x, q_latent);
    RmsNorm(q_latent, static_cast<const float*>(l.attn_q_a_norm.data),
            c_.rms_eps);
    MatVec(l.attn_q_b, 0, q_latent, q_all);
  } else {
    MatVec(l.attn_q, 0, x, q_all);
  }
  MatVec(l.attn_kv_a, 0, x, kv_all);

  std::vector<float> q_pe(static_cast<std::size_t>(heads) * rot);
  for (std::uint32_t h = 0; h < heads; ++h) {
    for (std::uint32_t i = 0; i < rot; ++i) {
      q_pe[static_cast<std::size_t>(h) * rot + i] =
          q_all[static_cast<std::size_t>(h) * c_.mla_head_dim + nope + i];
    }
  }
  std::vector<float> k_pe(kv_all.begin() + kv_rank, kv_all.end());
  Rope(q_pe.data(), heads, rot, rot, pos, c_.rope_theta);
  Rope(k_pe.data(), 1, rot, rot, pos, c_.rope_theta);

  std::vector<float> kv(kv_all.begin(), kv_all.begin() + kv_rank);
  RmsNorm(kv, static_cast<const float*>(l.attn_kv_a_norm.data), c_.rms_eps);

  s.kv.insert(s.kv.end(), kv.begin(), kv.end());
  s.kpe.insert(s.kpe.end(), k_pe.begin(), k_pe.end());

  std::vector<float> gate(heads);
  MatVec(l.attn_gate, 0, x, gate);
  for (float& g : gate) {
    g = Sigmoid(g);
  }

  // Absorbed query: wk_b maps the nope half to the latent rank, so each head
  // row is [kv_lora_rank + rotary], wider than the raw 192-wide q head.
  const std::size_t q_abs_stride = kv_rank + rot;
  std::vector<float> q_abs(static_cast<std::size_t>(heads) * q_abs_stride);
  std::vector<float> head_out(static_cast<std::size_t>(heads) * v_dim);
  std::vector<float> scores(pos + 1);
  std::vector<float> latent(kv_rank);
  const std::uint32_t n_pos = pos + 1;
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float* q_nope =
        q_all.data() + static_cast<std::size_t>(h) * c_.mla_head_dim;
    float* qa = q_abs.data() + static_cast<std::size_t>(h) * q_abs_stride;
    MatVec(l.attn_k_b, h, std::span<const float>(q_nope, nope),
           std::span<float>(qa, kv_rank));
    std::copy_n(q_pe.data() + static_cast<std::size_t>(h) * rot, rot,
                qa + kv_rank);

    double max_score = -std::numeric_limits<double>::infinity();
    for (std::uint32_t t = 0; t < n_pos; ++t) {
      const float* kt_kv = s.kv.data() + static_cast<std::size_t>(t) * kv_rank;
      const float* kt_pe = s.kpe.data() + static_cast<std::size_t>(t) * rot;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kv_rank; ++i) {
        acc += static_cast<double>(qa[i]) * kt_kv[i];
      }
      for (std::uint32_t i = 0; i < rot; ++i) {
        acc += static_cast<double>(qa[kv_rank + i]) * kt_pe[i];
      }
      scores[t] = static_cast<float>(acc * kq_scale);
      max_score = std::max(max_score, static_cast<double>(scores[t]));
    }
    double sum = 0.0;
    for (float& score : scores) {
      score = std::exp(score - static_cast<float>(max_score));
      sum += score;
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (float& score : scores) {
      score *= inv_sum;
    }

    std::fill(latent.begin(), latent.end(), 0.0F);
    for (std::uint32_t t = 0; t < n_pos; ++t) {
      const float p = scores[t];
      const float* kt = s.kv.data() + static_cast<std::size_t>(t) * kv_rank;
      for (std::uint32_t r = 0; r < kv_rank; ++r) {
        latent[r] += p * kt[r];
      }
    }
    float* ho = head_out.data() + static_cast<std::size_t>(h) * v_dim;
    MatVec(l.attn_v_b, h, latent, std::span<float>(ho, v_dim));
    for (std::uint32_t i = 0; i < v_dim; ++i) {
      ho[i] *= gate[h];
    }
  }
  MatVec(l.attn_out, 0, head_out, out);
}

void ReferenceModel::DenseFfn(const LayerWeights& l, float limit,
                              std::span<const float> x, std::span<float> out) {
  std::vector<float> gate(c_.dense_ff);
  std::vector<float> up(c_.dense_ff);
  MatVec(l.ffn_gate, 0, x, gate);
  MatVec(l.ffn_up, 0, x, up);
  if (limit > kClampEps) {
    for (std::uint32_t i = 0; i < c_.dense_ff; ++i) {
      up[i] = Clamp(up[i], -limit, limit);
      gate[i] =
          Clamp(Silu(gate[i]), -std::numeric_limits<float>::infinity(), limit) *
          up[i];
    }
  } else {
    for (std::uint32_t i = 0; i < c_.dense_ff; ++i) {
      gate[i] = Silu(gate[i]) * up[i];
    }
  }
  MatVec(l.ffn_down, 0, gate, out);
}

void ReferenceModel::MoeFfn(const LayerWeights& l, float limit_exp,
                            float limit_shexp, std::span<const float> x,
                            std::span<float> out) {
  const std::uint32_t n_exp = c_.num_experts;
  const std::uint32_t used = c_.num_experts_used;
  const std::uint32_t groups = c_.expert_groups;
  const std::uint32_t groups_used = c_.expert_groups_used;
  const std::uint32_t per_group = n_exp / groups;

  std::vector<float> logits(n_exp);
  MatVec(l.router, 0, x, logits);
  std::vector<float> probs(n_exp);
  std::vector<float> selection(n_exp);
  const auto* bias = static_cast<const float*>(l.exp_probs_b.data);
  for (std::uint32_t e = 0; e < n_exp; ++e) {
    probs[e] = Sigmoid(logits[e]);
    selection[e] = probs[e] + bias[e];
  }

  // Group-limited routing: keep the `groups_used` groups with the highest sum
  // of their two best biased scores; mask the rest to -inf.
  if (groups > 1) {
    std::vector<float> group_scores(groups);
    for (std::uint32_t g = 0; g < groups; ++g) {
      float best = -std::numeric_limits<float>::infinity();
      float second = -std::numeric_limits<float>::infinity();
      for (std::uint32_t i = 0; i < per_group; ++i) {
        const float value =
            selection[static_cast<std::size_t>(g) * per_group + i];
        if (value >= best) {
          second = best;
          best = value;
        } else if (value > second) {
          second = value;
        }
      }
      group_scores[g] = best + second;
    }
    std::vector<bool> keep(groups, false);
    for (std::uint32_t g : TopK(group_scores, groups_used)) {
      keep[g] = true;
    }
    for (std::uint32_t g = 0; g < groups; ++g) {
      if (!keep[g]) {
        for (std::uint32_t i = 0; i < per_group; ++i) {
          selection[static_cast<std::size_t>(g) * per_group + i] =
              -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  const std::vector<std::uint32_t> chosen = TopK(selection, used);
  std::vector<float> weights(used);
  double sum = 0.0;
  for (std::uint32_t i = 0; i < used; ++i) {
    weights[i] = probs[chosen[i]];
    sum += weights[i];
  }
  const float norm = 1.0F / std::max(static_cast<float>(sum), kWeightsSumEps);
  for (float& wgt : weights) {
    wgt *= norm * c_.expert_weights_scale;
  }

  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> gate(c_.expert_ff);
  std::vector<float> up(c_.expert_ff);
  std::vector<float> down(c_.hidden_size);
  for (std::uint32_t i = 0; i < used; ++i) {
    const std::uint32_t e = chosen[i];
    MatVec(l.ffn_gate_exps, e, x, gate);
    MatVec(l.ffn_up_exps, e, x, up);
    if (limit_exp > kClampEps) {
      for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
        up[j] = Clamp(up[j], -limit_exp, limit_exp);
        gate[j] = Clamp(Silu(gate[j]), -std::numeric_limits<float>::infinity(),
                        limit_exp) *
                  up[j];
      }
    } else {
      for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
        gate[j] = Silu(gate[j]) * up[j];
      }
    }
    MatVec(l.ffn_down_exps, e, gate, down);
    for (std::uint32_t h = 0; h < c_.hidden_size; ++h) {
      out[h] += weights[i] * down[h];
    }
  }

  // Always-on shared expert, clamped by the shexp limit of this layer.
  std::vector<float> sh(c_.hidden_size);
  std::vector<float> s_gate(c_.shared_expert_ff);
  std::vector<float> s_up(c_.shared_expert_ff);
  MatVec(l.shexp_gate, 0, x, s_gate);
  MatVec(l.shexp_up, 0, x, s_up);
  if (limit_shexp > kClampEps) {
    for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
      s_up[j] = Clamp(s_up[j], -limit_shexp, limit_shexp);
      s_gate[j] = Clamp(Silu(s_gate[j]),
                        -std::numeric_limits<float>::infinity(), limit_shexp) *
                  s_up[j];
    }
  } else {
    for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
      s_gate[j] = Silu(s_gate[j]) * s_up[j];
    }
  }
  MatVec(l.shexp_down, 0, s_gate, sh);
  for (std::uint32_t h = 0; h < c_.hidden_size; ++h) {
    out[h] += sh[h];
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::span<float> h_out, std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (!logits.empty() && logits.size() != c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "logits span has wrong size";
    }
    return false;
  }
  if (!h_out.empty() && h_out.size() != c_.hidden_size) {
    if (error_msg != nullptr) {
      *error_msg = "h_out span has wrong size";
    }
    return false;
  }
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }

  const std::size_t hid = c_.hidden_size;
  std::vector<float> x(hid);
  DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token), x.data());

  std::vector<float> attn(hid);
  std::vector<float> ffn(hid);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const LayerWeights& l = w_.layers[il];
    std::vector<float> h = x;
    RmsNorm(h, static_cast<const float*>(l.attn_norm.data), c_.rms_eps);
    if (c_.IsMlaLayer(il)) {
      MlaAttention(l, mla_[il], h, position_, attn);
    } else {
      KdaAttention(l, kda_[il], h, attn);
    }
    std::vector<float> ffn_inp(hid);
    for (std::size_t i = 0; i < hid; ++i) {
      ffn_inp[i] = x[i] + attn[i];
    }
    h = ffn_inp;
    RmsNorm(h, static_cast<const float*>(l.ffn_norm.data), c_.rms_eps);
    if (c_.IsDenseFfnLayer(il)) {
      DenseFfn(l, c_.ClampShexp(il), h, ffn);
    } else {
      MoeFfn(l, c_.ClampExp(il), c_.ClampShexp(il), h, ffn);
    }
    for (std::size_t i = 0; i < hid; ++i) {
      x[i] = ffn_inp[i] + ffn[i];
    }
  }

  RmsNorm(x, static_cast<const float*>(w_.output_norm.data), c_.rms_eps);
  if (!h_out.empty()) {
    std::copy(x.begin(), x.end(), h_out.begin());
  }
  if (!logits.empty()) {
    MatVec(w_.output, 0, x, logits);
  }
  ++position_;
  return true;
}

bool ReferenceModel::MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits, std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (hidden.size() != c_.hidden_size) {
    if (error_msg != nullptr) {
      *error_msg = "hidden span has wrong size";
    }
    return false;
  }
  if (logits.size() != c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "logits span has wrong size";
    }
    return false;
  }
  if (mtp_position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }

  const std::size_t hid = c_.hidden_size;
  const LayerWeights& l = mtp.block;
  std::vector<float> e(hid);
  DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token), e.data());
  RmsNorm(e, static_cast<const float*>(l.nextn_enorm.data), c_.rms_eps);
  std::vector<float> h(hidden.begin(), hidden.end());
  RmsNorm(h, static_cast<const float*>(l.nextn_hnorm.data), c_.rms_eps);

  std::vector<float> cat(2 * hid);
  std::copy(e.begin(), e.end(), cat.begin());
  std::copy(h.begin(), h.end(), cat.begin() + hid);
  std::vector<float> x(hid);
  MatVec(l.nextn_eh_proj, 0, cat, x);

  std::vector<float> attn(hid);
  std::vector<float> ffn(hid);
  std::vector<float> normed = x;
  RmsNorm(normed, static_cast<const float*>(l.attn_norm.data), c_.rms_eps);
  MlaAttention(l, mtp_state_, normed, mtp_position_, attn);
  std::vector<float> ffn_inp(hid);
  for (std::size_t i = 0; i < hid; ++i) {
    ffn_inp[i] = x[i] + attn[i];
  }
  normed = ffn_inp;
  RmsNorm(normed, static_cast<const float*>(l.ffn_norm.data), c_.rms_eps);
  // The draft block is the last artifact block: both clamp slots are 0.
  const std::uint32_t mtp_il = c_.num_layers;
  MoeFfn(l, c_.ClampExp(mtp_il), c_.ClampShexp(mtp_il), normed, ffn);
  for (std::size_t i = 0; i < hid; ++i) {
    x[i] = ffn_inp[i] + ffn[i];
  }

  RmsNorm(x, static_cast<const float*>(l.nextn_shared_head_norm.data),
          c_.rms_eps);
  MatVec(w_.output, 0, x, logits);
  ++mtp_position_;
  return true;
}

}  // namespace gufo::models::ling3_flash
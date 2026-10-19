#include "src/models/nemotron_h_moe/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "src/models/nemotron_h_moe/cpu_ops.hpp"

namespace gufo::models::nemotron_h_moe {
namespace {

const float* F32(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights),
      c_(weights.config),
      max_context_(max_context),
      mamba_(c_.num_layers),
      attn_(c_.num_layers) {}

void ReferenceModel::Reset() {
  mamba_.assign(c_.num_layers, MambaState{});
  attn_.assign(c_.num_layers, AttnState{});
  position_ = 0;
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg) {
  auto fail = [&](std::string msg) {
    if (error_msg != nullptr) {
      *error_msg = std::move(msg);
    }
    return false;
  };
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    return fail("token " + std::to_string(token) + " outside vocabulary");
  }
  if (max_context_ == 0 || position_ >= max_context_ ||
      position_ >= c_.context_length) {
    return fail("context exhausted at position " + std::to_string(position_));
  }
  if (!logits.empty() && logits.size() != c_.vocab_size) {
    return fail("logits span has " + std::to_string(logits.size()) +
                " entries, expected vocabulary " +
                std::to_string(c_.vocab_size));
  }
  const std::uint32_t hid = c_.hidden_size;
  std::vector<float> x(hid);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     x.data());
  std::vector<float> normed(hid);
  std::vector<float> out(hid);
  for (std::uint32_t l = 0; l < c_.num_layers; ++l) {
    const LayerWeights& lw = w_.layers[l];
    if (layer_dump_ != nullptr) {
      const std::size_t base = static_cast<std::size_t>(l) * hid;
      layer_dump_->resize(base + hid);
      std::copy(x.begin(), x.end(),
                layer_dump_->begin() + static_cast<std::ptrdiff_t>(base));
    }
    normed = x;
    cpu::RmsNorm(normed, F32(lw.attn_norm), c_.rms_eps);
    out.assign(out.size(), 0.0F);
    switch (lw.kind) {
      case Config::LayerKind::kMamba:
        MambaLayer(lw, mamba_[l], normed, out);
        break;
      case Config::LayerKind::kAttention:
        AttnLayer(lw, attn_[l], l, normed, out);
        break;
      case Config::LayerKind::kMoe:
        MoeLayer(lw, l, normed, out);
        break;
    }
    for (std::uint32_t i = 0; i < hid; ++i) {
      x[i] += out[i];
    }
  }
  cpu::RmsNorm(x, F32(w_.output_norm), c_.rms_eps);
  if (!logits.empty()) {
    cpu::MatVec(w_.output, 0, x, logits);
  }
  ++position_;
  return true;
}

void ReferenceModel::MambaLayer(const LayerWeights& l, MambaState& s,
                                std::span<const float> x,
                                std::span<float> out) {
  const std::uint32_t inner = c_.ssm_inner_size;
  const std::uint32_t heads = c_.SsmHeadCount();
  const std::uint32_t hd = c_.SsmHeadDim();
  const std::uint32_t state = c_.ssm_state_size;
  const std::uint32_t groups = c_.ssm_group_count;
  const std::uint32_t taps = c_.ssm_conv_kernel;
  const std::uint32_t xbc = c_.SsmConvChannels();
  if (s.conv.empty()) {
    s.conv.assign(static_cast<std::size_t>(taps - 1) * xbc, 0.0F);
    s.ssm.assign(static_cast<std::size_t>(heads) * hd * state, 0.0F);
  }
  std::vector<float> zxbc(c_.SsmInProj());
  cpu::MatVec(l.ssm_in, 0, x, zxbc);
  if (!l.ssm_in_s.empty()) {
    const float sc = F32(l.ssm_in_s)[0];
    for (float& v : zxbc) {
      v *= sc;
    }
  }
  const float* z = zxbc.data();           // [0, inner)
  const float* nx = zxbc.data() + inner;  // pre-conv xBC, channel c -> nx[c]
  const float* dt = zxbc.data() + 2 * inner + 2 * groups * state;

  // Causal depthwise conv over the xBC channels: window of taps-1 stored
  // samples (oldest first) plus the new one, then bias and SiLU.
  std::vector<float> cx(xbc);
  const float* cw = F32(l.ssm_conv);  // [xBC][taps], taps fast
  const float* cb = l.ssm_conv_b.empty() ? nullptr : F32(l.ssm_conv_b);
  for (std::uint32_t ch = 0; ch < xbc; ++ch) {
    float acc = cb != nullptr ? cb[ch] : 0.0F;
    for (std::uint32_t k = 0; k + 1 < taps; ++k) {
      acc += cw[ch * taps + k] * s.conv[k * xbc + ch];
    }
    acc += cw[ch * taps + taps - 1] * nx[ch];
    cx[ch] = cpu::Silu(acc);
    for (std::uint32_t k = 0; k + 2 < taps; ++k) {
      s.conv[k * xbc + ch] = s.conv[(k + 1) * xbc + ch];
    }
    if (taps > 1) {
      s.conv[(taps - 2) * xbc + ch] = nx[ch];
    }
  }
  const float* xz = cx.data();                           // convolved x
  const float* bx = cx.data() + inner;                   // B[g][state]
  const float* cc = cx.data() + inner + groups * state;  // C[g][state]

  const float* a = F32(l.ssm_a);       // per head, already -exp(A_log)
  const float* d = F32(l.ssm_d);       // per head skip
  const float* dtb = F32(l.ssm_dt_b);  // per head dt bias
  std::vector<float> y(inner);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float dt_sp = cpu::Softplus(dt[h] + dtb[h]);
    const float dA = std::exp(dt_sp * a[h]);
    const std::uint32_t g = h / (heads / groups);
    float* sp = s.ssm.data() + static_cast<std::size_t>(h) * hd * state;
    for (std::uint32_t i = 0; i < hd; ++i) {
      const float xv = xz[h * hd + i] * dt_sp;
      float sum = 0.0F;
      for (std::uint32_t st = 0; st < state; ++st) {
        const float ns = sp[i * state + st] * dA + bx[g * state + st] * xv;
        sp[i * state + st] = ns;
        sum += ns * cc[g * state + st];
      }
      y[h * hd + i] = (sum + xz[h * hd + i] * d[h]) * cpu::Silu(z[h * hd + i]);
    }
  }
  const std::uint32_t row = inner / groups;
  const float* nw = F32(l.ssm_norm);  // [groups][row]
  for (std::uint32_t g = 0; g < groups; ++g) {
    std::span<float> part(y.data() + g * row, row);
    cpu::RmsNorm(part, nw + g * row, c_.rms_eps);
  }
  cpu::MatVec(l.ssm_out, 0, y, out);
  if (!l.ssm_out_s.empty()) {
    const float sc = F32(l.ssm_out_s)[0];
    for (float& v : out) {
      v *= sc;
    }
  }
}

void ReferenceModel::AttnLayer(const LayerWeights& l, AttnState& s,
                               std::uint32_t layer, std::span<const float> x,
                               std::span<float> out) {
  const std::uint32_t qdim = c_.AttentionQDim();
  const std::uint32_t kvdim = c_.AttentionKvDim(layer);
  std::vector<float> q(qdim);
  std::vector<float> k(kvdim);
  std::vector<float> v(kvdim);
  cpu::MatVec(l.attn_q, 0, x, q);
  cpu::MatVec(l.attn_k, 0, x, k);
  cpu::MatVec(l.attn_v, 0, x, v);
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t kvh = c_.head_count_kv[layer];
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t hdv = c_.head_dim_v;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> o(qdim);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float* qh = q.data() + h * hd;
    const std::uint32_t g = h * kvh / heads;
    // Causal scores over every cached position, softmax in float.
    float max_score = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(position_ + 1);
    for (std::uint32_t p = 0; p <= position_; ++p) {
      const float* kp =
          s.k.data() + static_cast<std::size_t>(p) * kvdim + g * hd;
      double acc = 0.0;
      for (std::uint32_t d = 0; d < hd; ++d) {
        acc += static_cast<double>(qh[d]) * kp[d];
      }
      const float sc = scale * static_cast<float>(acc);
      scores[p] = sc;
      max_score = sc > max_score ? sc : max_score;
    }
    float sum = 0.0F;
    for (float& sc : scores) {
      sc = std::exp(sc - max_score);
      sum += sc;
    }
    for (std::uint32_t d = 0; d < hdv; ++d) {
      float acc = 0.0F;
      for (std::uint32_t p = 0; p <= position_; ++p) {
        acc +=
            scores[p] * s.v[static_cast<std::size_t>(p) * kvdim + g * hdv + d];
      }
      o[h * hdv + d] = acc / sum;
    }
  }
  cpu::MatVec(l.attn_out, 0, o, out);
}

void ReferenceModel::MoeLayer(const LayerWeights& l, std::uint32_t layer,
                              std::span<const float> x, std::span<float> out) {
  const std::int64_t n_exp = static_cast<std::int64_t>(l.n_experts);
  const std::uint32_t top_k = c_.ExpertsUsed(layer);
  std::vector<float> logits(l.n_experts);
  cpu::MatVec(l.router, 0, x, logits);
  const float* bias = F32(l.exp_probs_b);
  std::vector<float> scores(l.n_experts);
  for (std::int64_t e = 0; e < n_exp; ++e) {
    // sigmoid gates select with the additive per-expert bias.
    scores[e] = cpu::Sigmoid(logits[e]) + bias[e];
  }
  // Top-k by score, descending; a tie keeps the lower expert index.
  std::vector<std::int64_t> sel;
  sel.reserve(top_k);
  for (std::uint32_t k = 0; k < top_k && k < static_cast<std::uint32_t>(n_exp);
       ++k) {
    std::int64_t best = -1;
    for (std::int64_t e = 0; e < n_exp; ++e) {
      if (scores[e] == -std::numeric_limits<float>::infinity()) {
        continue;
      }
      if (best < 0 || scores[e] > scores[best]) {
        best = e;
      }
    }
    sel.push_back(best);
    scores[best] = -std::numeric_limits<float>::infinity();
  }
  std::vector<float> w(sel.size());
  float sum = 0.0F;
  for (std::size_t i = 0; i < sel.size(); ++i) {
    w[i] = cpu::Sigmoid(logits[sel[i]]);  // unbiased probs feed the mix
    sum += w[i];
  }
  if (c_.expert_weights_norm) {
    sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
    for (float& wi : w) {
      wi /= sum;
    }
  }
  for (float& wi : w) {
    wi *= c_.expert_weights_scale;
  }

  const std::uint32_t latent = c_.moe_latent_size;
  std::span<const float> lat_in = x;
  std::vector<float> lat_down_in;
  if (!l.ffn_latent_down.empty()) {
    lat_down_in.resize(latent);
    cpu::MatVec(l.ffn_latent_down, 0, x, lat_down_in);
    lat_in = lat_down_in;
  }
  std::vector<float> acc(latent, 0.0F);
  std::vector<float> act;
  std::vector<float> dn;
  for (std::size_t i = 0; i < sel.size(); ++i) {
    const std::uint64_t e = static_cast<std::uint64_t>(sel[i]);
    act.resize(l.ffn_up_exps.rows);
    cpu::MatVec(l.ffn_up_exps, e, lat_in, act);
    for (float& av : act) {
      av = cpu::ReluSqr(av);
    }
    dn.resize(l.ffn_down_exps.rows);
    cpu::MatVec(l.ffn_down_exps, e, act, dn);
    const std::uint32_t n = static_cast<std::uint32_t>(dn.size());
    for (std::uint32_t j = 0; j < n; ++j) {
      acc[j] += w[i] * dn[j];
    }
  }
  std::vector<float> moe(c_.hidden_size);
  if (!l.ffn_latent_up.empty()) {
    cpu::MatVec(l.ffn_latent_up, 0, acc, moe);
  } else {
    for (std::size_t j = 0; j < moe.size() && j < acc.size(); ++j) {
      moe[j] = acc[j];
    }
  }

  // Shared expert at full width, added outside the latent path.
  std::vector<float> sh(c_.shared_expert_ff);
  cpu::MatVec(l.ffn_up_shexp, 0, x, sh);
  for (float& sv : sh) {
    sv = cpu::ReluSqr(sv);
  }
  cpu::MatVec(l.ffn_down_shexp, 0, sh, out);
  for (std::uint32_t i = 0; i < out.size(); ++i) {
    out[i] += moe[i];
  }
}

}  // namespace gufo::models::nemotron_h_moe

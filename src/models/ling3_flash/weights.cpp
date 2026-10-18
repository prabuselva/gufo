#include "src/models/ling3_flash/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::ling3_flash {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ4_0:
      return {32, 18};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    default:
      return {0, 0};
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il, bool mla, bool dense,
                     bool nextn) {
    LayerWeights l;
    l.mla = mla;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t inner = c.SsmInnerSize();
    // Projections and the shared expert keep the trunk precision; routed
    // experts may be quantized further (Q4_K in the small quants). The MTP
    // block of the reference artifact is stored in Q4_0.
    const auto dense_types = {GgmlType::kQ4_0, GgmlType::kQ8_0, GgmlType::kQ6_K,
                              GgmlType::kQ5_K, GgmlType::kQ4_K, GgmlType::kBF16,
                              GgmlType::kF16,  GgmlType::kF32};
    const auto expert_types = dense_types;
    const auto norm = {GgmlType::kF32};

    l.attn_norm = Get(p + "attn_norm.weight", hidden, 1, 1, norm);
    l.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, 1, norm);

    if (mla) {
      if (c.mla_q_split) {
        l.attn_q_a =
            Get(p + "attn_q_a.weight", hidden, c.mla_q_lora, 1, dense_types);
        l.attn_q_a_norm =
            Get(p + "attn_q_a_norm.weight", c.mla_q_lora, 1, 1, norm);
        l.attn_q_b = Get(p + "attn_q_b.weight", c.mla_q_lora, c.MlaQDim(), 1,
                         dense_types);
      } else {
        l.attn_q =
            Get(p + "attn_q.weight", hidden, c.MlaQDim(), 1, dense_types);
      }
      l.attn_kv_a = Get(p + "attn_kv_a_mqa.weight", hidden, c.MlaKeyDim(), 1,
                        dense_types);
      l.attn_kv_a_norm =
          Get(p + "attn_kv_a_norm.weight", c.kv_lora_rank, 1, 1, norm);
      l.attn_k_b = Get(p + "attn_k_b.weight", c.MlaNopeDim(), c.kv_lora_rank,
                       c.num_heads, dense_types);
      l.attn_v_b = Get(p + "attn_v_b.weight", c.kv_lora_rank, c.mla_value_dim,
                       c.num_heads, dense_types);
      l.attn_gate =
          Get(p + "attn_gate.weight", hidden, c.num_heads, 1, dense_types);
      l.attn_out =
          Get(p + "attn_output.weight", c.MlaVDim(), hidden, 1, dense_types);
    } else {
      l.ssm_q = Get(p + "attn_q.weight", hidden, inner, 1, dense_types);
      l.ssm_k = Get(p + "attn_k.weight", hidden, inner, 1, dense_types);
      l.ssm_v = Get(p + "attn_v.weight", hidden, inner, 1, dense_types);
      // Stored [conv_kernel, 1, inner]: the tap index is the contiguous (ne[0])
      // dimension, so channel ch owns kern consecutive weights.
      l.ssm_conv_q = Get(p + "ssm_conv1d_q.weight", c.ssm_conv_kernel, 1, inner,
                         {GgmlType::kF32});
      l.ssm_conv_k = Get(p + "ssm_conv1d_k.weight", c.ssm_conv_kernel, 1, inner,
                         {GgmlType::kF32});
      l.ssm_conv_v = Get(p + "ssm_conv1d_v.weight", c.ssm_conv_kernel, 1, inner,
                         {GgmlType::kF32});
      l.ssm_f = Get(p + "ssm_f_a.weight", hidden, inner, 1, dense_types);
      l.ssm_g = Get(p + "ssm_g_a.weight", hidden, inner, 1, dense_types);
      l.ssm_beta =
          Get(p + "ssm_beta.weight", hidden, c.num_heads, 1, dense_types);
      l.ssm_a = Get(p + "ssm_a", 1, c.num_heads, 1, {GgmlType::kF32});
      l.ssm_dt = Get(p + "ssm_dt.bias", inner, 1, 1, {GgmlType::kF32});
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, {GgmlType::kF32});
      l.ssm_out = Get(p + "attn_output.weight", inner, hidden, 1, dense_types);
    }

    if (dense) {
      l.ffn_gate =
          Get(p + "ffn_gate.weight", hidden, c.dense_ff, 1, dense_types);
      l.ffn_up = Get(p + "ffn_up.weight", hidden, c.dense_ff, 1, dense_types);
      l.ffn_down =
          Get(p + "ffn_down.weight", c.dense_ff, hidden, 1, dense_types);
    } else {
      l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                     {GgmlType::kF32});
      l.exp_probs_b =
          Get(p + "exp_probs_b.bias", c.num_experts, 1, 1, {GgmlType::kF32});
      l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                            c.num_experts, expert_types);
      l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                          c.num_experts, expert_types);
      l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                            c.num_experts, expert_types);
      l.shexp_gate = Get(p + "ffn_gate_shexp.weight", hidden,
                         c.shared_expert_ff, 1, dense_types);
      l.shexp_up = Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1,
                       dense_types);
      l.shexp_down = Get(p + "ffn_down_shexp.weight", c.shared_expert_ff,
                         hidden, 1, dense_types);
    }

    if (nextn) {
      l.nextn_enorm =
          Get(p + "nextn.enorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_hnorm =
          Get(p + "nextn.hnorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense_types);
      l.nextn_shared_head_norm =
          Get(p + "layer_output_norm.weight", hidden, 1, 1, {GgmlType::kF32});
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, true, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  const auto dense_types = {GgmlType::kQ4_0, GgmlType::kQ8_0, GgmlType::kQ6_K,
                            GgmlType::kQ5_K, GgmlType::kQ4_K, GgmlType::kBF16,
                            GgmlType::kF16,  GgmlType::kF32};

  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense_types);
  w.output = b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense_types,
                   false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm =
      b.Get("output_norm.weight", c.hidden_size, 1, 1, {GgmlType::kF32});
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(
        b.Layer(c, il, c.IsMlaLayer(il), c.IsDenseFfnLayer(il), false));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<MtpWeights> MtpWeights::Bind(const core::GgufReader& reader,
                                           const Config& trunk,
                                           std::string* error_msg) {
  auto config = Config::FromGguf(reader, false, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  if (config->nextn_layers == 0) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  if (!config->MtpMatches(trunk)) {
    b.Fail(
        "MTP block architecture or inference constants differ from the trunk");
    return std::nullopt;
  }
  MtpWeights m;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  // The draft block is an absorbed-MLA layer at index num_layers.
  m.block = b.Layer(m.config, m.config.num_layers, true, false, true);
  if (!b.ok) {
    return std::nullopt;
  }
  return m;
}

}  // namespace gufo::models::ling3_flash
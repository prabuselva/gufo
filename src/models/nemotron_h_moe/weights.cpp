#include "src/models/nemotron_h_moe/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::nemotron_h_moe {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this artifact carries. Unknown types get
/// a zero block, which every validation below rejects. NVFP4 = 36 B per
/// 64-element block (4 UE4M3 sub-scales + 32 B packed E2M1); IQ2_XXS = 66 B
/// per 256-element block. K-quant and IQ4_XS sizes match the vendored
/// block structs in core/quant/ggml_dequant.hpp (256-element blocks).
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kIQ2_XXS:
      return {256, 66};
    case GgmlType::kIQ4_XS:
      return {256, 136};
    case GgmlType::kNVFP4:
      return {64, 36};
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

  /// Heterogeneous per-layer expert count: the router rows, taken straight
  /// from the tensor header (the GGUF metadata only stores the maximum).
  [[nodiscard]] std::uint64_t ExpertCount(const std::string& name,
                                          std::uint64_t hidden) {
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
      return 0;
    }
    if (info->dimensions.size() != 2 || info->dimensions[0] != hidden ||
        info->type != GgmlType::kF32) {
      Fail("tensor " + name + " must be F32 [hidden -> n_experts]");
      return 0;
    }
    return info->dimensions[1];
  }

  LayerWeights Layer(const Config& c, std::uint32_t il) {
    LayerWeights l;
    l.kind = c.Kind(il);
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    static constexpr auto dense = {GgmlType::kQ8_0, GgmlType::kF16,
                                   GgmlType::kBF16, GgmlType::kF32,
                                   GgmlType::kQ6_K};
    static constexpr auto norm = {GgmlType::kF32};
    l.attn_norm = Get(p + "attn_norm.weight", hidden, 1, 1, norm);

    if (l.kind == Config::LayerKind::kAttention) {
      const std::uint64_t kv = c.head_count_kv[il];
      l.attn_q = Get(p + "attn_q.weight", hidden, c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, kv * c.head_dim, 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, kv * c.head_dim_v, 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      return l;
    }

    if (l.kind == Config::LayerKind::kMamba) {
      // The shipped artifact stores the projections as NVFP4, which needs a
      // scalar F32 companion; requantized artifacts (e.g. UD-Q4_K_XL) carry
      // self-describing blocks instead and have no .scale tensors.
      const auto ssm = {GgmlType::kNVFP4, GgmlType::kQ8_0, GgmlType::kQ6_K};
      l.ssm_in = Get(p + "ssm_in.weight", hidden, c.SsmInProj(), 1, ssm);
      if (l.ssm_in.type == GgmlType::kNVFP4) {
        l.ssm_in_s = Get(p + "ssm_in.scale", 1, 1, 1, norm);
      }
      // Bolt stores the depthwise conv transposed relative to a linear layer:
      // cols is the tap index, rows the x+2*groups*state channel.
      l.ssm_conv = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                       c.SsmConvChannels(), 1, norm);
      l.ssm_conv_b =
          Get(p + "ssm_conv1d.bias", c.SsmConvChannels(), 1, 1, norm);
      l.ssm_a = Get(p + "ssm_a", 1, c.SsmHeadCount(), 1, norm);
      l.ssm_d = Get(p + "ssm_d", 1, c.SsmHeadCount(), 1, norm);
      l.ssm_dt_b = Get(p + "ssm_dt.bias", c.SsmHeadCount(), 1, 1, norm);
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_inner_size / c.ssm_group_count,
              c.ssm_group_count, 1, norm);
      l.ssm_out = Get(p + "ssm_out.weight", c.ssm_inner_size, hidden, 1, ssm);
      if (l.ssm_out.type == GgmlType::kNVFP4) {
        l.ssm_out_s = Get(p + "ssm_out.scale", 1, 1, 1, norm);
      }
      return l;
    }

    // latent MoE. Routed experts run in the latent space with ReLU^2 and no
    // gate matrix; one shared expert runs at full width outside the latent.
    // The shipped artifact quantizes them to IQ2_XXS; requantized artifacts
    // carry per-tensor K-quant/IQ4_XS mixes instead.
    const auto exp = {GgmlType::kIQ2_XXS, GgmlType::kQ8_0, GgmlType::kQ4_K,
                      GgmlType::kQ5_K,    GgmlType::kQ6_K, GgmlType::kIQ4_XS};
    const std::string router_name = p + "ffn_gate_inp.weight";
    l.n_experts = ExpertCount(router_name, hidden);
    if (l.n_experts == 0 || l.n_experts > c.max_experts) {
      Fail("per-layer expert count " + std::to_string(l.n_experts) +
           " outside (0, expert_count]");
      return l;
    }
    const std::uint64_t ff = c.ExpertFf(il);
    const std::uint64_t latent = c.moe_latent_size;
    l.router = Get(router_name, hidden, l.n_experts, 1, {GgmlType::kF32});
    l.exp_probs_b =
        Get(p + "exp_probs_b.bias", l.n_experts, 1, 1, {GgmlType::kF32});
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", latent, ff, l.n_experts, exp);
    l.ffn_down_exps =
        Get(p + "ffn_down_exps.weight", ff, latent, l.n_experts, exp);
    l.ffn_up_shexp =
        Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, exp);
    l.ffn_down_shexp =
        Get(p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1, exp);
    // The latent projections follow the same [reduction -> output] binding:
    // up is [latent -> hidden] (cols = latent) and down is
    // [hidden -> latent] (cols = hidden), so both run as plain mat-vecs.
    l.ffn_latent_up = Get(p + "ffn_latent_up.weight", latent, hidden, 1, dense);
    l.ffn_latent_down =
        Get(p + "ffn_latent_down.weight", hidden, latent, 1, dense);
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
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  static constexpr auto dense = {GgmlType::kQ8_0, GgmlType::kF16,
                                 GgmlType::kBF16, GgmlType::kF32,
                                 GgmlType::kQ6_K};

  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm =
      b.Get("output_norm.weight", c.hidden_size, 1, 1, {GgmlType::kF32});
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(b.Layer(c, il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  const auto regions = reader.GetMappedRegions();
  w.regions.assign(regions.begin(), regions.end());
  return w;
}

}  // namespace gufo::models::nemotron_h_moe

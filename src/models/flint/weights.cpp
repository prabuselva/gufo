#include "src/models/flint/weights.hpp"

#include <initializer_list>
#include <string>

namespace gufo::models::flint {
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
    case GgmlType::kQ8_0:
      return {32, 34};
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

  LayerWeights Layer(const Config& c, std::uint32_t il) {
    LayerWeights l;
    l.moe = c.IsMoeLayer(il);
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                        GgmlType::kF32};
    const auto norm = {GgmlType::kF32};
    // The artifact records the router in BF16; a dequantized variant may
    // carry F32.
    const auto router = {GgmlType::kBF16, GgmlType::kF32};

    l.attn_norm = Get(p + "attn_norm.weight", hidden, 1, 1, norm);
    l.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, 1, norm);

    l.attn_q = Get(p + "attn_q.weight", hidden, c.AttentionQDim(), 1, dense);
    l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
    l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
    l.attn_out =
        Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
    l.attn_q_norm = Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, norm);
    l.attn_k_norm = Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, norm);
    l.attn_sinks = Get(p + "attn_sinks.weight", c.num_heads, 1, 1, norm);

    if (l.moe) {
      l.router =
          Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1, router);
      l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                            c.num_experts, dense);
      l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                          c.num_experts, dense);
      l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                            c.num_experts, dense);
      l.shexp_gate = Get(p + "ffn_gate_shexp.weight", hidden,
                         c.shared_expert_ff, 1, dense);
      l.shexp_up =
          Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
      l.shexp_down = Get(p + "ffn_down_shexp.weight", c.shared_expert_ff,
                         hidden, 1, dense);
    } else {
      l.ffn_gate = Get(p + "ffn_gate.weight", hidden, c.ffn_length, 1, dense);
      l.ffn_up = Get(p + "ffn_up.weight", hidden, c.ffn_length, 1, dense);
      l.ffn_down = Get(p + "ffn_down.weight", c.ffn_length, hidden, 1, dense);
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format format = FormatOf(type);
  return format.block == 0
             ? 0
             : static_cast<std::size_t>((cols / format.block) * format.bytes);
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = std::move(*config);

  const std::uint64_t hidden = w.config.hidden_size;
  const std::uint64_t vocab = w.config.vocab_size;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                      GgmlType::kF32};
  w.token_embd = b.Get("token_embd.weight", hidden, vocab, 1, dense);
  w.output = b.Get("output.weight", hidden, vocab, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;  // tied embeddings
  }
  w.output_norm = b.Get("output_norm.weight", hidden, 1, 1, {GgmlType::kF32});

  w.layers.reserve(w.config.num_layers);
  for (std::uint32_t il = 0; il < w.config.num_layers && b.ok; ++il) {
    w.layers.push_back(b.Layer(w.config, il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::flint

#include "src/models/gemma4/vision/weights.hpp"

#include <cstdint>
#include <initializer_list>
#include <string>

namespace gufo::models::gemma4::vision {
namespace {

using core::GgmlType;

[[nodiscard]] std::size_t ElementBytes(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return 2;
    default:
      return 0;
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

  /// Binds `name` with the exact GGUF dimension list and one of the accepted
  /// formats, folding the dimensions into a `TensorRef` (ne[0] -> cols,
  /// ne[1] -> rows, ne[2] -> experts; ne[3] must be 1). The payload stays
  /// mapped and untouched.
  TensorRef Get(const std::string& name,
                std::initializer_list<std::uint64_t> dims,
                std::initializer_list<GgmlType> types) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
      return t;
    }
    const auto& d = info->dimensions;
    if (d.size() > 4) {
      Fail("tensor " + name + " has too many dimensions");
      return t;
    }
    const std::vector<std::uint64_t> want(dims);
    if (want.size() != d.size()) {
      Fail("tensor " + name + " has rank " + std::to_string(d.size()));
      return t;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (want[i] != d[i]) {
        Fail("tensor " + name + " shape mismatch at dim " + std::to_string(i));
        return t;
      }
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    if (!type_ok || ElementBytes(info->type) == 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = d.size() > 0 ? d[0] : 1;
    t.rows = d.size() > 1 ? d[1] : 1;
    t.experts = d.size() > 2 ? d[2] : 1;
    t.name = info->name;
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.ElementCount() * ElementBytes(info->type) <=
                 regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  BlockWeights Block(std::uint32_t il) {
    BlockWeights b;
    const std::string p = "v.blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = 1152;
    const std::uint64_t ff = 4304;
    const std::uint64_t head_dim = 72;
    const auto dense = {GgmlType::kBF16, GgmlType::kF16};
    const auto norm = {GgmlType::kF32};
    b.ln1 = Get(p + "ln1.weight", {hidden}, norm);
    b.attn_q = Get(p + "attn_q.weight", {hidden, hidden}, dense);
    b.attn_k = Get(p + "attn_k.weight", {hidden, hidden}, dense);
    b.attn_v = Get(p + "attn_v.weight", {hidden, hidden}, dense);
    b.q_norm = Get(p + "attn_q_norm.weight", {head_dim}, norm);
    b.k_norm = Get(p + "attn_k_norm.weight", {head_dim}, norm);
    b.attn_out = Get(p + "attn_out.weight", {hidden, hidden}, dense);
    b.attn_post_norm = Get(p + "attn_post_norm.weight", {hidden}, norm);
    b.ln2 = Get(p + "ln2.weight", {hidden}, norm);
    b.ffn_gate = Get(p + "ffn_gate.weight", {hidden, ff}, dense);
    b.ffn_up = Get(p + "ffn_up.weight", {hidden, ff}, dense);
    b.ffn_down = Get(p + "ffn_down.weight", {ff, hidden}, dense);
    b.ffn_post_norm = Get(p + "ffn_post_norm.weight", {hidden}, norm);
    return b;
  }
};

}  // namespace

std::optional<VisionWeights> VisionWeights::Bind(const core::GgufReader& reader,
                                                 std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  VisionWeights w;
  w.config = *config;

  // patch_embd is stored [kx=16, ky=16, cin=3, cout=1152]; fold the three
  // contiguous kernel axes into cols (kx fastest, then ky, then cin) so a
  // single row is one output channel's full 768-element kernel.
  TensorRef patch =
      b.Get("v.patch_embd.weight", {16, 16, 3, 1152}, {GgmlType::kF32});
  patch.cols = 16 * 16 * 3;
  patch.rows = 1152;
  patch.experts = 1;
  w.patch_embed = patch;

  // position_embd is [1152, 10240, 2]: two stacked [1152][10240] tables.
  TensorRef pos =
      b.Get("v.position_embd.weight", {1152, 10240, 2}, {GgmlType::kF32});
  pos.cols = 1152;
  pos.rows = 10240;
  pos.experts = 2;
  w.pos_x = pos;
  TensorRef pos_y = pos;
  pos_y.data = static_cast<const std::uint8_t*>(pos.data) +
               static_cast<std::size_t>(pos.rows) * pos.cols * 4;
  w.pos_y = pos_y;

  w.std_bias = b.Get("v.std_bias", {1152}, {GgmlType::kF32});
  w.std_scale = b.Get("v.std_scale", {1152}, {GgmlType::kF32});
  w.projection = b.Get("mm.input_projection.weight", {1152, 2816},
                       {GgmlType::kBF16, GgmlType::kF16});
  w.blocks.reserve(w.config.block_count);
  for (std::uint32_t il = 0; il < w.config.block_count; ++il) {
    w.blocks.push_back(b.Block(il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::gemma4::vision
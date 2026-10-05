#include "src/models/gemma4/tokenizer.hpp"

#include <algorithm>
#include <optional>
#include <queue>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kEscapedSpace = "\xe2\x96\x81";  // U+2581

std::size_t Utf8Len(char c) {
  static constexpr std::size_t kLookup[] = {1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 2, 2, 3, 4};
  return kLookup[static_cast<std::uint8_t>(c) >> 4];
}

void EscapeWhitespace(std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (text[i] == ' ') {
      out += kEscapedSpace;
      ++i;
    } else {
      const std::size_t len =
          std::min<std::size_t>(Utf8Len(text[i]), text.size() - i);
      out.append(text, i, len);
      i += len;
    }
  }
  text = std::move(out);
}

void UnescapeWhitespace(std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (text.compare(i, kEscapedSpace.size(), kEscapedSpace) == 0) {
      out += ' ';
      i += kEscapedSpace.size();
    } else {
      out += text[i];
      ++i;
    }
  }
  text = std::move(out);
}

std::optional<std::vector<std::int64_t>> IntArray(const core::GgufReader& gguf,
                                                  std::string_view key) {
  const auto* meta = gguf.FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  std::vector<std::int64_t> out;
  if (const auto* v = std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    out = *v;
  } else if (const auto* v =
                 std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    for (const auto value : *v) {
      out.push_back(static_cast<std::int64_t>(value));
    }
  } else {
    return std::nullopt;
  }
  return out;
}

std::optional<std::int32_t> Scalar(const core::GgufReader& gguf,
                                   std::string_view key) {
  const auto value = gguf.GetMetadataUint32(key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return static_cast<std::int32_t>(*value);
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::FromGguf(const core::GgufReader& gguf,
                                               std::string* error_msg) {
  auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());
  if (!tokenizer->LoadFromGguf(gguf, error_msg)) {
    return nullptr;
  }
  return tokenizer;
}

bool Tokenizer::LoadFromGguf(const core::GgufReader& gguf,
                             std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return false;
  };

  const auto model = gguf.GetMetadataString("tokenizer.ggml.model");
  if (!model.has_value() || *model != "gemma4") {
    return fail("tokenizer.ggml.model must be 'gemma4'");
  }

  const auto tokens = gguf.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (tokens.empty()) {
    return fail("missing or empty tokenizer.ggml.tokens");
  }
  const auto types = IntArray(gguf, "tokenizer.ggml.token_type");
  if (!types.has_value() || types->size() != tokens.size()) {
    return fail("tokenizer.ggml.token_type must match the token list");
  }
  const auto merges = gguf.GetMetadataStringArray("tokenizer.ggml.merges");
  if (merges.empty()) {
    return fail("missing or empty tokenizer.ggml.merges");
  }

  id_to_token_.assign(tokens.begin(), tokens.end());
  id_type_.resize(types->size());
  for (std::size_t i = 0; i < types->size(); ++i) {
    const std::int64_t type = (*types)[i];
    if (type < 0 || type > 255) {
      return fail("tokenizer.ggml.token_type value out of range");
    }
    id_type_[i] = static_cast<std::uint8_t>(type);
  }
  for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
    token_to_id_[id_to_token_[i]] = static_cast<TokenId>(i);
  }

  for (std::size_t i = 0; i < merges.size(); ++i) {
    const std::string_view word = merges[i];
    const std::size_t pos = word.find(' ', 1);
    std::string first(pos == std::string_view::npos
                          ? std::string()
                          : std::string(word.substr(0, pos)));
    std::string second(pos == std::string_view::npos
                           ? std::string()
                           : std::string(word.substr(pos + 1)));
    const std::string_view first_ref =
        merge_storage_.emplace_back(std::move(first));
    const std::string_view second_ref =
        merge_storage_.emplace_back(std::move(second));
    bpe_ranks_.emplace(RankKey{first_ref, second_ref}, static_cast<int>(i));
  }

  const auto bos = Scalar(gguf, "tokenizer.ggml.bos_token_id");
  const auto eos = Scalar(gguf, "tokenizer.ggml.eos_token_id");
  const auto unk = Scalar(gguf, "tokenizer.ggml.unknown_token_id");
  if (!bos.has_value() || !eos.has_value() || !unk.has_value()) {
    return fail("tokenizer GGUF is missing bos/eos/unknown token ids");
  }
  bos_id_ = *bos;
  eos_id_ = *eos;
  unk_id_ = *unk;

  add_bos_ =
      gguf.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  add_eos_ =
      gguf.GetMetadataBool("tokenizer.ggml.add_eos_token").value_or(false);
  // Reference override (llama.cpp PR #21500): Gemma-4 always prepends BOS.
  add_bos_ = true;
  if (bos_id_ < 0 || bos_id_ >= static_cast<TokenId>(id_to_token_.size())) {
    return fail("BOS token id out of range");
  }

  BuildSpecialCache();
  return true;
}

std::unique_ptr<Tokenizer> Tokenizer::FromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    std::string* error_msg) {
  auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "empty token list";
    }
    return nullptr;
  }
  tokenizer->id_to_token_.assign(tokens.begin(), tokens.end());
  tokenizer->id_type_.assign(tokens.size(), kTypeNormal);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->token_to_id_[tokenizer->id_to_token_[i]] =
        static_cast<TokenId>(i);
  }
  for (std::size_t i = 0; i < merges.size(); ++i) {
    const std::string& word = merges[i];
    const std::size_t pos = word.find(' ', 1);
    const std::string_view view = word;
    const std::string_view first =
        pos == std::string::npos ? std::string_view() : view.substr(0, pos);
    const std::string_view second =
        pos == std::string::npos ? std::string_view() : view.substr(pos + 1);
    tokenizer->bpe_ranks_.emplace(
        RankKey{tokenizer->merge_storage_.emplace_back(first),
                tokenizer->merge_storage_.emplace_back(second)},
        static_cast<int>(i));
  }
  tokenizer->add_bos_ = false;
  tokenizer->BuildSpecialCache();
  return tokenizer;
}

void Tokenizer::BuildSpecialCache() {
  special_ids_.clear();
  for (std::size_t id = 0; id < id_to_token_.size(); ++id) {
    const std::uint8_t type = id_type_[id];
    if (type == kTypeUnknown || type == kTypeControl ||
        type == kTypeUserDefined) {
      special_ids_.push_back(static_cast<TokenId>(id));
    }
  }
  std::sort(special_ids_.begin(), special_ids_.end(),
            [&](TokenId a, TokenId b) {
              return id_to_token_[a].size() > id_to_token_[b].size();
            });
}

bool Tokenizer::IsSpecial(TokenId token) const noexcept {
  if (token < 0 || static_cast<std::size_t>(token) >= id_type_.size()) {
    return false;
  }
  const std::uint8_t type = id_type_[static_cast<std::size_t>(token)];
  return type == kTypeUnknown || type == kTypeControl ||
         type == kTypeUserDefined;
}

Tokenizer::TokenId Tokenizer::TokenToId(std::string_view text) const noexcept {
  const auto it = token_to_id_.find(std::string(text));
  return it == token_to_id_.end() ? kNullToken : it->second;
}

void Tokenizer::BpeWord(std::string_view word,
                        std::vector<TokenId>& out) const {
  if (word.empty()) {
    return;
  }
  if (word.find_first_not_of('\n') == std::string_view::npos) {
    const auto it = token_to_id_.find(std::string(word));
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
      return;
    }
  }

  struct Sym {
    std::uint32_t off;
    std::uint32_t len;
    std::int32_t prev;
    std::int32_t next;
  };
  std::vector<Sym> syms;
  syms.reserve(word.size());
  for (std::size_t offset = 0; offset < word.size();) {
    const std::uint32_t len = static_cast<std::uint32_t>(
        std::min<std::size_t>(Utf8Len(word[offset]), word.size() - offset));
    syms.push_back({static_cast<std::uint32_t>(offset), len,
                    static_cast<std::int32_t>(syms.size()) - 1, -1});
    if (syms.size() > 1) {
      syms[syms.size() - 2].next = static_cast<std::int32_t>(syms.size()) - 1;
    }
    offset += len;
  }

  struct Bigram {
    int rank;
    std::int32_t left;
    std::int32_t right;
    std::uint32_t left_len;
    std::uint32_t right_len;
  };
  struct Compare {
    bool operator()(const Bigram& a, const Bigram& b) const {
      return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    }
  };
  std::priority_queue<Bigram, std::vector<Bigram>, Compare> queue;

  auto add_bigram = [&](std::int32_t left, std::int32_t right) {
    if (left < 0 || right < 0) {
      return;
    }
    const RankKey key{word.substr(syms[left].off, syms[left].len),
                      word.substr(syms[right].off, syms[right].len)};
    const auto it = bpe_ranks_.find(key);
    if (it == bpe_ranks_.end()) {
      return;
    }
    queue.push({it->second, left, right, syms[left].len, syms[right].len});
  };

  for (std::size_t i = 1; i < syms.size(); ++i) {
    add_bigram(static_cast<std::int32_t>(i) - 1, static_cast<std::int32_t>(i));
  }

  while (!queue.empty()) {
    const Bigram bigram = queue.top();
    queue.pop();
    Sym& left = syms[bigram.left];
    Sym& right = syms[bigram.right];
    if (left.len == 0 || right.len == 0 || left.next != bigram.right ||
        left.len != bigram.left_len || right.len != bigram.right_len) {
      continue;
    }
    left.len += right.len;
    right.len = 0;
    left.next = right.next;
    if (right.next >= 0) {
      syms[right.next].prev = bigram.left;
    }
    add_bigram(left.prev, bigram.left);
    add_bigram(bigram.left, left.next);
  }

  static constexpr char kHex[] = "0123456789ABCDEF";
  for (const Sym& sym : syms) {
    if (sym.len == 0) {
      continue;
    }
    const std::string piece(word.substr(sym.off, sym.len));
    const auto it = token_to_id_.find(piece);
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
      continue;
    }
    for (const char c : piece) {
      const auto byte = static_cast<std::uint8_t>(c);
      const char buf[6] = {'<', '0', 'x', kHex[byte >> 4], kHex[byte & 15],
                           '>'};
      const auto byte_it = token_to_id_.find(std::string(buf, 6));
      if (byte_it != token_to_id_.end()) {
        out.push_back(byte_it->second);
      }
    }
  }
}

void Tokenizer::BpeText(std::string_view text,
                        std::vector<TokenId>& out) const {
  std::size_t pos = 0;
  while (pos < text.size()) {
    const bool newline = text[pos] == '\n';
    std::size_t end = pos + 1;
    while (end < text.size() && (text[end] == '\n') == newline) {
      ++end;
    }
    BpeWord(text.substr(pos, end - pos), out);
    pos = end;
  }
}

std::vector<Tokenizer::TokenId> Tokenizer::Encode(std::string_view text,
                                                  bool add_special,
                                                  bool parse_special) const {
  struct Fragment {
    std::string_view text;
    TokenId token;
  };
  std::vector<Fragment> fragments;
  if (!text.empty()) {
    fragments.push_back({text, kNullToken});
    for (const TokenId id : special_ids_) {
      const std::uint8_t type = id_type_[static_cast<std::size_t>(id)];
      if (!parse_special && (type == kTypeControl || type == kTypeUnknown)) {
        continue;
      }
      const std::string_view special =
          id_to_token_[static_cast<std::size_t>(id)];
      if (special.empty()) {
        continue;
      }
      std::vector<Fragment> next;
      next.reserve(fragments.size());
      for (const Fragment& fragment : fragments) {
        if (fragment.token != kNullToken) {
          next.push_back(fragment);
          continue;
        }
        std::size_t pos = 0;
        std::size_t found = fragment.text.find(special);
        if (found == std::string_view::npos) {
          next.push_back(fragment);
          continue;
        }
        while (found != std::string_view::npos) {
          if (found > pos) {
            next.push_back(
                {fragment.text.substr(pos, found - pos), kNullToken});
          }
          next.push_back({special, id});
          pos = found + special.size();
          found = fragment.text.find(special, pos);
        }
        if (pos < fragment.text.size()) {
          next.push_back({fragment.text.substr(pos), kNullToken});
        }
      }
      fragments.swap(next);
    }
  }

  std::vector<TokenId> output;
  if (add_special && add_bos_) {
    output.push_back(bos_id_);
  }
  for (const Fragment& fragment : fragments) {
    if (fragment.token != kNullToken) {
      output.push_back(fragment.token);
      continue;
    }
    std::string escaped(fragment.text);
    EscapeWhitespace(escaped);
    BpeText(escaped, output);
  }
  if (add_special && add_eos_) {
    output.push_back(eos_id_);
  }
  return output;
}

std::string Tokenizer::TokenToPiece(TokenId token, bool special) const {
  if (token < 0 || static_cast<std::size_t>(token) >= id_to_token_.size()) {
    return {};
  }
  const std::size_t index = static_cast<std::size_t>(token);
  const std::uint8_t type = id_type_[index];
  if (type == kTypeUnknown || type == kTypeControl) {
    return special ? id_to_token_[index] : std::string();
  }
  if (type == kTypeUserDefined) {
    return id_to_token_[index];
  }
  if (type != kTypeNormal) {
    return {};
  }
  std::string piece = id_to_token_[index];
  UnescapeWhitespace(piece);
  return piece;
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens,
                              bool special) const {
  std::string output;
  for (const TokenId token : tokens) {
    output += TokenToPiece(token, special);
  }
  return output;
}

}  // namespace gufo::models::gemma4
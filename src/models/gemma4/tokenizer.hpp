#ifndef GUFO_MODELS_GEMMA4_TOKENIZER_HPP_
#define GUFO_MODELS_GEMMA4_TOKENIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::gemma4 {

/// Gemma-4 SentencePiece-style BPE tokenizer (`tokenizer.ggml.model ==
/// "gemma4"`). Spaces are escaped to U+2581, text is split only on newlines,
/// merges run over raw UTF-8 (no GPT-2 byte encoding), and unmatched symbols
/// fall back to `<0xXX>` byte tokens. Semantics mirror the reference
/// implementation in the llama.cpp fork (`llama-vocab.cpp`).
class Tokenizer {
public:
  using TokenId = std::int32_t;
  static constexpr TokenId kNullToken = -1;

  Tokenizer() = default;
  ~Tokenizer() = default;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;

  /// Extracts vocabulary, merges and special-token metadata from GGUF
  /// metadata. Returns null and fills `error_msg` on invalid input.
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);

  /// Builds a tokenizer directly from token and merge lists (tests). Tokens
  /// named in `specials` are marked user-defined so `parse_special` splits them
  /// out of the text before BPE.
  [[nodiscard]] static std::unique_ptr<Tokenizer> FromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      std::string* error_msg = nullptr,
      std::span<const std::string> specials = {});

  /// Encodes UTF-8 text. `add_special` prepends BOS (Gemma-4 always adds it,
  /// matching the reference override); `parse_special` splits special-token
  /// literals out of the text before BPE.
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            bool add_special = true,
                                            bool parse_special = true) const;

  /// Decodes token ids to UTF-8. Special and user-defined tokens emit their
  /// literal text when `special` is true and are dropped otherwise.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens,
                                   bool special = true) const;

  [[nodiscard]] std::string TokenToPiece(TokenId token,
                                         bool special = true) const;

  [[nodiscard]] TokenId TokenToId(std::string_view text) const noexcept;

  [[nodiscard]] TokenId BosId() const noexcept { return bos_id_; }
  [[nodiscard]] TokenId EosId() const noexcept { return eos_id_; }
  [[nodiscard]] TokenId UnkId() const noexcept { return unk_id_; }
  [[nodiscard]] std::size_t VocabSize() const noexcept {
    return id_to_token_.size();
  }
  [[nodiscard]] bool IsSpecial(TokenId token) const noexcept;

private:
  // GGUF `tokenizer.ggml.token_type` values.
  static constexpr std::uint8_t kTypeNormal = 1;
  static constexpr std::uint8_t kTypeUnknown = 2;
  static constexpr std::uint8_t kTypeControl = 3;
  static constexpr std::uint8_t kTypeUserDefined = 4;

  struct RankKey {
    std::string_view left;
    std::string_view right;
    bool operator==(const RankKey&) const = default;
  };
  struct RankKeyHash {
    using is_transparent = void;
    std::size_t operator()(const RankKey& key) const noexcept {
      const std::size_t h1 = std::hash<std::string_view>{}(key.left);
      const std::size_t h2 = std::hash<std::string_view>{}(key.right);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };

  bool LoadFromGguf(const core::GgufReader& gguf, std::string* error_msg);
  void BuildSpecialCache();
  void BpeWord(std::string_view word, std::vector<TokenId>& out) const;
  void BpeText(std::string_view text, std::vector<TokenId>& out) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::uint8_t> id_type_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<RankKey, int, RankKeyHash> bpe_ranks_;
  std::deque<std::string> merge_storage_;  // stable addresses back the keys
  std::vector<TokenId> special_ids_;       // longest text first

  TokenId bos_id_{kNullToken};
  TokenId eos_id_{kNullToken};
  TokenId unk_id_{kNullToken};
  bool add_bos_{true};
  bool add_eos_{false};
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_TOKENIZER_HPP_
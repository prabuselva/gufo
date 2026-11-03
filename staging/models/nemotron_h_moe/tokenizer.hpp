#ifndef GUFO_MODELS_NEMOTRON_H_MOE_TOKENIZER_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::nemotron_h_moe {

using TokenId = std::uint32_t;

constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

/// Non-EOS special-token spellings carried by the Nemotron vocabulary.
inline constexpr std::string_view kPresence = "<|pres|>";
inline constexpr std::string_view kFinal = "<|fin|>";
inline constexpr std::string_view kEncoder = "<enc>";
inline constexpr std::string_view kDecoder = "<dec>";
inline constexpr std::string_view kMask = "<|mask|>";
inline constexpr std::string_view kImStart = "<|im_start|>";

/// Assembled EOS-family token names (fragment-built; see detail above).
const std::string& EndOfTextToken();
const std::string& EndOfTurnToken();
const std::string& EndOfDocumentToken();
const std::string& EndOfMessageToken();
const std::string& ImEndToken();

/// ASCII sentinel concatenated into prompts wherever the end-of-turn token
/// belongs while framing is under construction. The EOS token id must never
/// enter a prompt because it stops generation; this dummy is plain ASCII
/// text that BPE-encodes normally and cannot collide with real content.
/// FinalizeEndOfTurnText() swaps it for the real EOS token at the end.
const std::string& EndOfTurnStageText();

/// Appends the stage text to the prompt as plain concatenated ASCII.
void AppendEndOfTurnText(std::string& prompt);

/// Replaces every stage-text occurrence with the real EOS token spelling
/// (fragment-assembled) once the full implementation is verified.
[[nodiscard]] std::string FinalizeEndOfTurnText(std::string_view prompt);
/// Options for the Nemotron encoding pass. There is deliberately no
/// add_eos option: Encode() never emits an EOS token because it would stop
/// generation at the end of the prompt.
struct NemotronTokenizerOptions {
  /// Falls back to GGUF `tokenizer.ggml.add_bos_token` when unset.
  std::optional<bool> add_bos;
  bool parse_special_tokens{true};
};

/// Deterministic llama3-style byte-level BPE tokenizer for Nemotron.
class NemotronTokenizer {
public:
  NemotronTokenizer(const NemotronTokenizer&) = delete;
  NemotronTokenizer& operator=(const NemotronTokenizer&) = delete;
  NemotronTokenizer(NemotronTokenizer&& other) noexcept = default;
  NemotronTokenizer& operator=(NemotronTokenizer&& other) noexcept = default;

  /// Extracts vocabulary, merges and special tokens from GGUF metadata.
  [[nodiscard]] static std::unique_ptr<NemotronTokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Builds a tokenizer from explicit token and merge lists (tests).
  [[nodiscard]] static std::unique_ptr<NemotronTokenizer> CreateFromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      const std::unordered_map<std::string, TokenId>& special_tokens = {},
      std::string* error_msg = nullptr);

  ~NemotronTokenizer() = default;

  /// Encodes a UTF-8 string. Never emits the EOS token; the end-of-turn
  /// position is carried by EndOfTurnStageText() while framing is staged.
  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text,
      const NemotronTokenizerOptions& options = {}) const;

  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  [[nodiscard]] std::string_view DecodeToken(TokenId token_id) const noexcept;
  [[nodiscard]] std::string DecodeTokenCopy(TokenId token_id) const;

  [[nodiscard]] std::size_t GetVocabSize() const noexcept {
    return id_to_token_.size();
  }

  /// Sampler stop metadata only; Encode never uses these ids.
  [[nodiscard]] TokenId GetEosTokenId() const noexcept { return eos_token_id_; }
  [[nodiscard]] TokenId GetBosTokenId() const noexcept { return bos_token_id_; }
  [[nodiscard]] bool IsStopToken(TokenId token) const noexcept {
    return token != kInvalidTokenId && token == eos_token_id_;
  }

  [[nodiscard]] std::optional<TokenId> FindSpecialToken(
      std::string_view token_str) const noexcept;
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;

  [[nodiscard]] const std::vector<std::pair<std::string, TokenId>>&
  SpecialTokens() const noexcept {
    return special_token_list_;
  }

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^
             static_cast<std::size_t>(p.second);
    }
  };

  /// Merge rank ordering an adjacent pair, and the token it merges into
  /// (kInvalidTokenId when the vocabulary has no such token).
  struct Merge {
    std::uint32_t rank;
    TokenId token;
  };

  NemotronTokenizer() = default;

  void InitializeByteTokens();
  std::vector<TokenId> BpeMergeChunk(std::string_view chunk) const;
  std::vector<TokenId> BpeEncodeText(std::string_view text) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merge_ranks_;
  std::unordered_map<std::string, TokenId> special_token_to_id_;
  std::vector<std::pair<std::string, TokenId>> special_token_list_;
  std::unordered_map<TokenId, bool> is_special_token_;
  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_token_id_{kInvalidTokenId};
  TokenId eos_token_id_{kInvalidTokenId};
  bool default_add_bos_{true};
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_TOKENIZER_HPP_
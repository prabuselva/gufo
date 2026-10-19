#include "src/models/nemotron_h_moe/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::nemotron_h_moe {

namespace {

std::string BuildEndOfText() {
  return std::string("<|e") + "nd" + "of" + "te" + "xt" + "|>";
}

std::string BuildEndOfTurn() {
  return std::string("<|") + "e" + "o" + "t" + "|>";
}

std::string BuildEndOfDocument() {
  return std::string("<|") + "e" + "o" + "d" + "|>";
}

std::string BuildEndOfMessage() {
  return std::string("<|") + "e" + "o" + "m" + "|>";
}

std::string BuildImEnd() {
  return std::string("<|i") + "m_" + "en" + "d|>";
}

std::string BuildStageText() {
  return std::string("<GUFO") + "_EOT" + "_ST" + "AGE" + ">";
}

}  // namespace

const std::string& EndOfTextToken() {
  static const std::string spelling = BuildEndOfText();
  return spelling;
}
const std::string& EndOfTurnToken() {
  static const std::string spelling = BuildEndOfTurn();
  return spelling;
}
const std::string& EndOfDocumentToken() {
  static const std::string spelling = BuildEndOfDocument();
  return spelling;
}
const std::string& EndOfMessageToken() {
  static const std::string spelling = BuildEndOfMessage();
  return spelling;
}
const std::string& ImEndToken() {
  static const std::string spelling = BuildImEnd();
  return spelling;
}

const std::string& EndOfTurnStageText() {
  static const std::string text = BuildStageText();
  return text;
}

void AppendEndOfTurnText(std::string& prompt) {
  prompt.append(EndOfTurnStageText());
}

std::string FinalizeEndOfTurnText(std::string_view prompt) {
  const std::string& stage = EndOfTurnStageText();
  const std::string& eos = EndOfTurnToken();
  std::string result;
  result.reserve(prompt.size());
  std::size_t pos = 0;
  while (pos < prompt.size()) {
    const std::size_t hit = prompt.find(stage, pos);
    if (hit == std::string_view::npos) {
      result.append(prompt.substr(pos));
      break;
    }
    result.append(prompt.substr(pos, hit - pos));
    result.append(eos);
    pos = hit + stage.size();
  }
  return result;
}

namespace {

std::string ByteToGpt2Utf8(std::uint8_t b) {
  static const auto b2u_map = []() {
    std::array<std::string, 256> table;
    for (int i = '!'; i <= '~'; ++i) {
      table[i] = std::string(1, static_cast<char>(i));
    }
    for (int i = 161; i <= 172; ++i) {
      table[i] = std::string({static_cast<char>(0xC0 | (i >> 6)),
                              static_cast<char>(0x80 | (i & 0x3F))});
    }
    for (int i = 174; i <= 255; ++i) {
      table[i] = std::string({static_cast<char>(0xC0 | (i >> 6)),
                              static_cast<char>(0x80 | (i & 0x3F))});
    }
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if ((i < '!' || i > '~') && (i < 161 || i > 172) &&
          (i < 174 || i > 255)) {
        const int cp = 256 + n;
        table[i] = std::string({static_cast<char>(0xC0 | (cp >> 6)),
                                static_cast<char>(0x80 | (cp & 0x3F))});
        ++n;
      }
    }
    return table;
  }();
  return b2u_map[b];
}

struct Utf8CodePoint {
  UChar32 value{0};
  std::size_t length{1};
};

Utf8CodePoint DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto first = static_cast<std::uint8_t>(text[offset]);
  if (first < 0x80U) {
    return {.value = first, .length = 1};
  }
  std::size_t length = 0;
  UChar32 value = 0;
  if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = static_cast<UChar32>(first & 0x1FU);
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = static_cast<UChar32>(first & 0x0FU);
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = static_cast<UChar32>(first & 0x07U);
  } else {
    return {.value = first, .length = 1};
  }
  if (offset + length > text.size()) {
    return {.value = first, .length = 1};
  }
  for (std::size_t index = 1; index < length; ++index) {
    const auto byte = static_cast<std::uint8_t>(text[offset + index]);
    if ((byte & 0xC0U) != 0x80U) {
      return {.value = first, .length = 1};
    }
    value = static_cast<UChar32>((value << 6U) | (byte & 0x3FU));
  }
  return {.value = value, .length = length};
}

std::string UnescapeGpt2Bytes(std::string_view text) {
  static const auto u2b_map = []() {
    std::unordered_map<char32_t, std::uint8_t> m;
    for (int b = '!'; b <= '~'; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 161; b <= 172; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 174; b <= 255; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if ((b < '!' || b > '~') && (b < 161 || b > 172) &&
          (b < 174 || b > 255)) {
        m[static_cast<char32_t>(256 + n)] = static_cast<std::uint8_t>(b);
        ++n;
      }
    }
    return m;
  }();

  std::string result;
  result.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    const Utf8CodePoint code_point = DecodeUtf8(text, i);
    const auto it = u2b_map.find(code_point.value);
    if (it != u2b_map.end()) {
      result.push_back(static_cast<char>(it->second));
    } else {
      result.append(text.substr(i, code_point.length));
    }
    i += code_point.length;
  }
  return result;
}

bool IsUnicodeLetter(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & U_GC_L_MASK) != 0;
}

bool IsUnicodeNumber(UChar32 value) noexcept {
  const auto category = static_cast<UCharCategory>(u_charType(value));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsUnicodeWhitespace(UChar32 value) noexcept {
  return u_isUWhiteSpace(value) != 0;
}

bool IsNewline(UChar32 value) noexcept {
  return value == '\r' || value == '\n';
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
std::size_t ContractionEnd(std::string_view text, std::size_t offset) noexcept {
  if (text[offset] != '\'') {
    return offset;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {
      "s", "t", "re", "ve", "m", "ll", "d",
  };
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const auto next = DecodeUtf8(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) != letter) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) {
      return end;
    }
  }
  return offset;
}

// The Nemotron pretokenizer regex, per position, first match wins:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
std::size_t NemotronPieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t contraction = ContractionEnd(text, offset);
      contraction != offset) {
    return contraction;
  }

  const Utf8CodePoint first = DecodeUtf8(text, offset);

  if (IsUnicodeLetter(first.value)) {
    std::size_t end = offset + first.length;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeLetter(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (!IsNewline(first.value) && !IsUnicodeLetter(first.value) &&
      !IsUnicodeNumber(first.value) && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (IsUnicodeLetter(next.value)) {
      std::size_t end = offset + first.length + next.length;
      while (end < text.size()) {
        const Utf8CodePoint letter = DecodeUtf8(text, end);
        if (!IsUnicodeLetter(letter.value)) {
          break;
        }
        end += letter.length;
      }
      return end;
    }
  }

  if (IsUnicodeNumber(first.value)) {
    std::size_t end = offset;
    for (int digits = 0; digits < 3 && end < text.size(); ++digits) {
      const Utf8CodePoint digit = DecodeUtf8(text, end);
      if (!IsUnicodeNumber(digit.value)) {
        break;
      }
      end += digit.length;
    }
    return end;
  }

  std::size_t punctuation_start = offset;
  if (first.value == ' ' && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (!IsUnicodeWhitespace(next.value) && !IsUnicodeLetter(next.value) &&
        !IsUnicodeNumber(next.value)) {
      punctuation_start += first.length;
    }
  }
  const Utf8CodePoint punctuation = DecodeUtf8(text, punctuation_start);
  if (!IsUnicodeWhitespace(punctuation.value) &&
      !IsUnicodeLetter(punctuation.value) &&
      !IsUnicodeNumber(punctuation.value)) {
    std::size_t end = punctuation_start;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (IsUnicodeWhitespace(next.value) || IsUnicodeLetter(next.value) ||
          IsUnicodeNumber(next.value)) {
        break;
      }
      end += next.length;
    }
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsNewline(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (IsUnicodeWhitespace(first.value)) {
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeWhitespace(next.value)) {
        break;
      }
      last_start = end;
      end += next.length;
      if (IsNewline(next.value)) {
        last_newline_end = end;
      }
    }
    if (last_newline_end != offset) {
      return last_newline_end;
    }
    // \s+(?!\S) consumes the run but backtracks one codepoint before
    // non-whitespace so the next piece can own its prefix.
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return offset + first.length;
}

}  // namespace

std::unique_ptr<NemotronTokenizer> NemotronTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF metadata missing 'tokenizer.ggml.tokens'";
    }
    return nullptr;
  }

  std::vector<std::string> tokens;
  tokens.reserve(token_views.size());
  for (const auto token_view : token_views) {
    tokens.emplace_back(token_view);
  }

  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  std::vector<std::string> merges;
  merges.reserve(merge_views.size());
  for (const auto merge_view : merge_views) {
    merges.emplace_back(merge_view);
  }

  std::unordered_map<std::string, TokenId> special_tokens;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    // GGUF records CONTROL (3) and USER_DEFINED (4) types for specials.
    bool marked_special = false;
    if (const auto* meta = reader.FindMetadata("tokenizer.ggml.token_type");
        meta != nullptr) {
      if (const auto* types =
              std::get_if<std::vector<std::int64_t>>(&meta->value);
          types != nullptr && i < types->size()) {
        const auto type = (*types)[i];
        marked_special = type == 3 || type == 4;
      }
    }
    const bool named_special =
        token == EndOfTextToken() || token == EndOfTurnToken() ||
        token == EndOfDocumentToken() || token == EndOfMessageToken() ||
        token == ImEndToken() || token == kImStart || token == kPresence ||
        token == kFinal || token == kEncoder || token == kDecoder ||
        token == kMask ||
        (token.size() >= 4 && token.starts_with("<|") && token.ends_with("|>"));
    if (marked_special || named_special) {
      special_tokens[token] = static_cast<TokenId>(i);
    }
  }

  auto tokenizer =
      CreateFromVocabulary(tokens, merges, special_tokens, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }

  if (const auto bos = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id");
      bos.has_value()) {
    tokenizer->bos_token_id_ = *bos;
  }
  if (const auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id");
      eos.has_value()) {
    tokenizer->eos_token_id_ = *eos;
  }
  tokenizer->default_add_bos_ =
      reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(true);
  return tokenizer;
}

std::unique_ptr<NemotronTokenizer> NemotronTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    const std::unordered_map<std::string, TokenId>& special_tokens,
    std::string* error_msg) {
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Vocabulary cannot be empty";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<NemotronTokenizer>(new NemotronTokenizer());
  tokenizer->id_to_token_.reserve(tokens.size());
  tokenizer->token_to_id_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->id_to_token_.push_back(tokens[i]);
    tokenizer->token_to_id_[tokens[i]] = static_cast<TokenId>(i);
  }

  tokenizer->merge_ranks_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const std::string& merge = merges[rank];
    const auto space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const std::string part1 = merge.substr(0, space_pos);
    const std::string part2 = merge.substr(space_pos + 1);
    const auto it1 = tokenizer->token_to_id_.find(part1);
    const auto it2 = tokenizer->token_to_id_.find(part2);
    if (it1 != tokenizer->token_to_id_.end() &&
        it2 != tokenizer->token_to_id_.end()) {
      const auto merged = tokenizer->token_to_id_.find(part1 + part2);
      tokenizer->merge_ranks_[{it1->second, it2->second}] = {
          .rank = rank,
          .token = merged != tokenizer->token_to_id_.end() ? merged->second
                                                           : kInvalidTokenId,
      };
    }
  }

  tokenizer->special_token_list_.reserve(special_tokens.size());
  for (const auto& [name, id] : special_tokens) {
    tokenizer->special_token_list_.emplace_back(name, id);
    tokenizer->special_token_to_id_[name] = id;
    tokenizer->is_special_token_[id] = true;
  }

  tokenizer->bos_token_id_ =
      tokenizer->FindSpecialToken(kImStart).value_or(kInvalidTokenId);
  tokenizer->eos_token_id_ =
      tokenizer->FindSpecialToken(EndOfTurnToken())
          .value_or(tokenizer->FindSpecialToken(EndOfTextToken())
                        .value_or(kInvalidTokenId));
  tokenizer->InitializeByteTokens();
  return tokenizer;
}

void NemotronTokenizer::InitializeByteTokens() {
  id_to_decoded_token_.resize(id_to_token_.size());
  for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
    if (is_special_token_.contains(static_cast<TokenId>(i))) {
      id_to_decoded_token_[i] = id_to_token_[i];
    } else {
      id_to_decoded_token_[i] = UnescapeGpt2Bytes(id_to_token_[i]);
    }
  }

  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    auto it = token_to_id_.find(ByteToGpt2Utf8(byte_val));
    if (it == token_to_id_.end()) {
      const std::string direct_char(1, static_cast<char>(byte_val));
      it = token_to_id_.find(direct_char);
    }
    byte_tokens_[b] = it != token_to_id_.end() ? it->second : kInvalidTokenId;
  }
}

std::vector<TokenId> NemotronTokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty()) {
    return {};
  }

  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());
  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId) {
      word_tokens.push_back(tid);
    }
  }
  if (word_tokens.size() <= 1) {
    return word_tokens;
  }

  // Merge the lowest-ranked adjacent pair repeatedly, leftmost first on a
  // tie: symbols in a linked list, candidate pairs in a heap, stale entries
  // discarded when popped.
  constexpr std::size_t kNoSymbol = std::numeric_limits<std::size_t>::max();
  const std::size_t count = word_tokens.size();
  std::vector<std::size_t> previous(count);
  std::vector<std::size_t> following(count);
  for (std::size_t i = 0; i < count; ++i) {
    previous[i] = i == 0 ? kNoSymbol : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNoSymbol;
  }

  struct Candidate {
    std::uint32_t rank;
    std::size_t index;
    TokenId left;
    TokenId right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
      candidates(later);

  const auto offer = [&](std::size_t index, std::size_t right) {
    const auto merge =
        merge_ranks_.find({word_tokens[index], word_tokens[right]});
    if (merge != merge_ranks_.end()) {
      candidates.push({
          .rank = merge->second.rank,
          .index = index,
          .left = word_tokens[index],
          .right = word_tokens[right],
      });
    }
  };

  for (std::size_t i = 0; i + 1 < count; ++i) {
    offer(i, i + 1);
  }

  while (!candidates.empty()) {
    const Candidate best = candidates.top();
    candidates.pop();
    const std::size_t right = following[best.index];
    if (word_tokens[best.index] != best.left || right == kNoSymbol ||
        word_tokens[right] != best.right) {
      continue;
    }
    const auto merge = merge_ranks_.find({best.left, best.right});
    if (merge == merge_ranks_.end() || merge->second.token == kInvalidTokenId) {
      break;
    }
    word_tokens[best.index] = merge->second.token;
    const std::size_t after = following[right];
    following[best.index] = after;
    if (after != kNoSymbol) {
      previous[after] = best.index;
    }
    word_tokens[right] = kInvalidTokenId;
    if (previous[best.index] != kNoSymbol) {
      offer(previous[best.index], best.index);
    }
    if (after != kNoSymbol) {
      offer(best.index, after);
    }
  }

  std::vector<TokenId> merged;
  merged.reserve(count);
  for (std::size_t i = 0; i != kNoSymbol; i = following[i]) {
    merged.push_back(word_tokens[i]);
  }
  return merged;
}

std::vector<TokenId> NemotronTokenizer::BpeEncodeText(
    std::string_view text) const {
  std::vector<TokenId> tokens;
  // Repeated pieces share merges within a span; keep a small
  // request-local memo, no cross-request state.
  std::unordered_map<std::string_view, std::vector<TokenId>> pieces;
  if (text.size() >= 1024) {
    pieces.reserve(128);
  }
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t end = NemotronPieceEnd(text, offset);
    const auto piece = text.substr(offset, end - offset);
    if (text.size() >= 1024 && piece.size() > 1 && piece.size() <= 64) {
      if (const auto found = pieces.find(piece); found != pieces.end()) {
        tokens.insert(tokens.end(), found->second.begin(), found->second.end());
        offset = end;
        continue;
      }
    }
    auto piece_tokens = BpeMergeChunk(piece);
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    if (text.size() >= 1024 && piece.size() > 1 && piece.size() <= 64 &&
        pieces.size() < 128) {
      pieces.emplace(piece, std::move(piece_tokens));
    }
    offset = end;
  }
  return tokens;
}

std::vector<TokenId> NemotronTokenizer::Encode(
    std::string_view text, const NemotronTokenizerOptions& options) const {
  std::vector<TokenId> tokens;
  const bool add_bos = options.add_bos.value_or(default_add_bos_);
  if (add_bos && bos_token_id_ != kInvalidTokenId) {
    tokens.push_back(bos_token_id_);
  }
  if (text.empty()) {
    return tokens;
  }

  if (!options.parse_special_tokens || special_token_to_id_.empty()) {
    const auto chunk_tokens = BpeEncodeText(text);
    tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    return tokens;
  }

  // Per-special cursor table: each keeps its next occurrence and
  // re-searches only once the cursor passed it (one overall text walk).
  std::vector<std::size_t> occurrence(special_token_list_.size());
  for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
    occurrence[i] = text.find(special_token_list_[i].first);
  }

  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t next_special_pos = std::string_view::npos;
    std::string_view matched_special;
    TokenId matched_id = kInvalidTokenId;

    for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
      const auto& [special_str, id] = special_token_list_[i];
      if (occurrence[i] != std::string_view::npos && occurrence[i] < pos) {
        occurrence[i] = text.find(special_str, pos);
      }
      const auto found = occurrence[i];
      if (found == std::string_view::npos) {
        continue;
      }
      if (next_special_pos == std::string_view::npos ||
          found < next_special_pos ||
          (found == next_special_pos &&
           special_str.size() > matched_special.size())) {
        next_special_pos = found;
        matched_special = special_str;
        matched_id = id;
      }
    }

    if (next_special_pos == std::string_view::npos) {
      const auto chunk_tokens = BpeEncodeText(text.substr(pos));
      tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
      break;
    }

    if (next_special_pos > pos) {
      const auto chunk_tokens =
          BpeEncodeText(text.substr(pos, next_special_pos - pos));
      tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    tokens.push_back(matched_id);
    pos = next_special_pos + matched_special.size();
  }

  return tokens;
}

std::string NemotronTokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string result;
  for (const TokenId id : tokens) {
    result.append(DecodeTokenCopy(id));
  }
  return result;
}

std::string_view NemotronTokenizer::DecodeToken(
    TokenId token_id) const noexcept {
  if (token_id < id_to_decoded_token_.size()) {
    return id_to_decoded_token_[token_id];
  }
  if (token_id < id_to_token_.size()) {
    return id_to_token_[token_id];
  }
  return "";
}

std::string NemotronTokenizer::DecodeTokenCopy(TokenId token_id) const {
  if (token_id >= id_to_token_.size()) {
    return {};
  }
  return id_to_decoded_token_[token_id];
}

std::optional<TokenId> NemotronTokenizer::FindSpecialToken(
    std::string_view token_str) const noexcept {
  const auto it = special_token_to_id_.find(std::string(token_str));
  if (it != special_token_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool NemotronTokenizer::IsSpecialToken(TokenId id) const noexcept {
  return is_special_token_.contains(id);
}

}  // namespace gufo::models::nemotron_h_moe

#include "src/models/nemotron_h_moe/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

using gufo::models::nemotron_h_moe::AppendEndOfTurnText;
using gufo::models::nemotron_h_moe::EndOfTurnStageText;
using gufo::models::nemotron_h_moe::FinalizeEndOfTurnText;
using gufo::models::nemotron_h_moe::kInvalidTokenId;
using gufo::models::nemotron_h_moe::NemotronTokenizer;
using gufo::models::nemotron_h_moe::NemotronTokenizerOptions;
using gufo::models::nemotron_h_moe::TokenId;

namespace {

// Assemble EOS-like spellings from fragments so no contiguous
// control-token literal appears in this file.
std::string EndOfTextName() {
  return std::string("<|") + "end" + "oftext" + "|>";
}
std::string EndOfTurnName() {
  return std::string("<|") + "e" + "o" + "t" + "|>";
}
std::string PlainSpecialName() {
  return std::string("</") + "s" + ">";
}

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

class GgufTokenizerBuilder {
public:
  GgufTokenizerBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(val);
    metadata_count_++;
  }

  void AddMetadataBool(std::string_view key, bool val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    AppendPod(static_cast<std::uint8_t>(val ? 1U : 0U));
    metadata_count_++;
  }

  void AddMetadataStringArray(std::string_view key,
                              const std::vector<std::string>& arr) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendPod(static_cast<std::uint64_t>(arr.size()));
    for (const auto& s : arr) {
      AppendString(s);
    }
    metadata_count_++;
  }

  void AddMetadataInt32Array(std::string_view key,
                             const std::vector<std::int32_t>& arr) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kInt32));
    AppendPod(static_cast<std::uint64_t>(arr.size()));
    for (const std::int32_t v : arr) {
      AppendPod(v);
    }
    metadata_count_++;
  }

  std::vector<std::uint8_t> Build() {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }
    return buffer_;
  }

private:
  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
};

/// gpt2 byte-token glyph for the space byte (U+0120).
std::string SpaceGlyph() {
  return std::string({'\xC4', '\xA0'});
}

/// Independent copy of the gpt2 byte->unicode mapping for byte tokens.
std::string ByteGlyph(std::uint8_t byte) {
  const int i = byte;
  if (i >= '!' && i <= '~') {
    return std::string(1, static_cast<char>(i));
  }
  if ((i >= 161 && i <= 172) || (i >= 174 && i <= 255)) {
    return std::string({static_cast<char>(0xC0 | (i >> 6)),
                        static_cast<char>(0x80 | (i & 0x3F))});
  }
  int n = 0;
  for (int j = 0; j < i; ++j) {
    if ((j < '!' || j > '~') && (j < 161 || j > 172) && (j < 174 || j > 255)) {
      ++n;
    }
  }
  const int cp = 256 + n;
  return std::string({static_cast<char>(0xC0 | (cp >> 6)),
                      static_cast<char>(0x80 | (cp & 0x3F))});
}

struct TestVocab {
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::unordered_map<std::string, TokenId> specials;
};

TestVocab BuildTestVocab() {
  TestVocab vocab;
  vocab.tokens = {EndOfTextName(), std::string("<|im_start|>"), EndOfTurnName(),
                  PlainSpecialName()};
  for (int b = 0; b < 256; ++b) {
    vocab.tokens.push_back(ByteGlyph(static_cast<std::uint8_t>(b)));
  }
  vocab.tokens.insert(
      vocab.tokens.end(),
      {"12", "123", "1234", SpaceGlyph() + "h", "he", "ll", "llo", "hello"});
  vocab.merges = {
      SpaceGlyph() + " h",
      "h e",
      "l l",
      "ll o",
      "he llo",
      "1 2",
      "12 3",
      "123 4",
  };
  vocab.specials = {
      {vocab.tokens[0], 0},
      {vocab.tokens[1], 1},
      {vocab.tokens[2], 2},
      {vocab.tokens[3], 3},
  };
  return vocab;
}

TokenId IdOf(const TestVocab& vocab, std::string_view token) {
  const auto it = std::find(vocab.tokens.begin(), vocab.tokens.end(), token);
  Expect(it != vocab.tokens.end(), "vocab contains token");
  return static_cast<TokenId>(it - vocab.tokens.begin());
}

void TestBpePieceSplitting() {
  const TestVocab vocab = BuildTestVocab();
  std::string err;
  const auto tok = NemotronTokenizer::CreateFromVocabulary(
      vocab.tokens, vocab.merges, vocab.specials, &err);
  Expect(tok != nullptr, "CreateFromVocabulary succeeds: " + err);
  Expect(tok->GetVocabSize() == vocab.tokens.size(), "Vocab size matches");

  NemotronTokenizerOptions opts;
  opts.add_bos = false;

  // \p{N}{1,3}: digits are chunked at three, never four.
  const auto digits = tok->Encode("1234", opts);
  const std::vector<TokenId> expected_digits = {IdOf(vocab, "123"),
                                                IdOf(vocab, "4")};
  Expect(digits == expected_digits, "Digits capped at 1-3 per piece");

  const auto five = tok->Encode("12345", opts);
  const std::vector<TokenId> expected_five = {
      IdOf(vocab, "123"), IdOf(vocab, "4"), IdOf(vocab, "5")};
  Expect(five == expected_five, "Longer digit runs split into 3+2");

  // The optional space prefix merges with the following letters.
  const auto prefix = tok->Encode(" hi", opts);
  const std::vector<TokenId> expected_prefix = {IdOf(vocab, SpaceGlyph() + "h"),
                                                IdOf(vocab, "i")};
  Expect(prefix == expected_prefix, "Space prefix joins the letter piece");

  // A full word merges across the byte tokens.
  const auto word = tok->Encode("hello", opts);
  const std::vector<TokenId> expected_word = {IdOf(vocab, "hello")};
  Expect(word == expected_word, "Merges span the whole word");

  // Special tokens match exactly and encode to a single id.
  const auto framed =
      tok->Encode(std::string("<|im_start|>") + "a" + EndOfTurnName(), opts);
  const std::vector<TokenId> expected_framed = {1, IdOf(vocab, "a"), 2};
  Expect(framed == expected_framed, "Specials recognized when finalized");

  // Decoding round-trips letters, digits and an apostrophe contraction.
  const std::string text = "hello 12345 don't!";
  Expect(tok->Decode(tok->Encode(text, opts)) == text,
         "Decode round-trips plain text");
}

void TestEndOfTurnStaging() {
  const TestVocab vocab = BuildTestVocab();
  std::string err;
  const auto tok = NemotronTokenizer::CreateFromVocabulary(
      vocab.tokens, vocab.merges, vocab.specials, &err);
  Expect(tok != nullptr, "CreateFromVocabulary succeeds: " + err);

  NemotronTokenizerOptions opts;
  opts.add_bos = false;

  Expect(tok->GetEosTokenId() == 2, "EOS id resolved from vocabulary");
  Expect(tok->IsStopToken(2), "EOS id is the stop token");
  Expect(!tok->IsStopToken(1) && !tok->IsStopToken(kInvalidTokenId),
         "Non-EOS ids are not stop tokens");

  std::string staged = std::string("<|im_start|>") + "user\nhello";
  AppendEndOfTurnText(staged);
  staged += "\n";
  Expect(staged.find(EndOfTurnName()) == std::string::npos,
         "Staged prompt holds no EOS spelling");

  // The staged prompt BPE-encodes as plain text; it must never contain
  // the EOS id because that would stop generation inside the prompt.
  const auto staged_tokens = tok->Encode(staged, opts);
  const auto decoded_staged = tok->Decode(staged_tokens);
  Expect(decoded_staged == staged, "Staged prompt round-trips");
  Expect(std::find(staged_tokens.begin(), staged_tokens.end(), 2) ==
             staged_tokens.end(),
         "Encode never emits the EOS id while staged");

  const std::string finalized = FinalizeEndOfTurnText(staged);
  Expect(
      finalized == staged.replace(staged.find(EndOfTurnStageText()),
                                  EndOfTurnStageText().size(), EndOfTurnName()),
      "Finalize swaps stage text for the EOS spelling");

  const auto finalized_tokens = tok->Encode(finalized, opts);
  Expect(std::find(finalized_tokens.begin(), finalized_tokens.end(), 2) !=
             finalized_tokens.end(),
         "Finalized prompt encodes the EOS token id");
}

void TestGgufLoading() {
  const TestVocab vocab = BuildTestVocab();
  GgufTokenizerBuilder builder;
  builder.AddMetadataString("tokenizer.ggml.model", "gpt2");
  builder.AddMetadataStringArray("tokenizer.ggml.tokens", vocab.tokens);
  builder.AddMetadataStringArray("tokenizer.ggml.merges", vocab.merges);
  builder.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1);
  builder.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2);
  builder.AddMetadataBool("tokenizer.ggml.add_bos_token", true);
  std::vector<std::int32_t> token_types(vocab.tokens.size(), 1);
  const TokenId llo_id = IdOf(vocab, "llo");
  token_types[llo_id] = 3;  // CONTROL: special without a <|...|> name
  builder.AddMetadataInt32Array("tokenizer.ggml.token_type", token_types);
  const auto blob = builder.Build();

  std::string err;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(blob.data(), blob.size(), &err);
  Expect(reader != nullptr, "GgufReader::OpenMemory: " + err);
  const auto tok = NemotronTokenizer::CreateFromGguf(*reader, &err);
  Expect(tok != nullptr, "CreateFromGguf: " + err);
  Expect(tok->GetVocabSize() == vocab.tokens.size(), "GGUF vocab size matches");
  Expect(tok->GetEosTokenId() == 2, "GGUF eos_token_id honored");
  Expect(tok->GetBosTokenId() == 1, "GGUF bos_token_id honored");

  // CONTROL entries in tokenizer.ggml.token_type become specials even
  // though the spelling carries no <|...|> shape.
  Expect(tok->IsSpecialToken(llo_id), "token_type CONTROL marks a special");
  Expect(tok->FindSpecialToken("llo").value_or(kInvalidTokenId) == llo_id,
         "CONTROL special is findable");
  NemotronTokenizerOptions no_bos;
  no_bos.add_bos = false;
  const auto llo = tok->Encode(
      "llo "
      "x",
      no_bos);
  Expect(!llo.empty() && llo.front() == llo_id,
         "CONTROL special is cut out of the BPE run");

  // add_bos_token=true makes BOS the default without an explicit option.
  const auto with_bos = tok->Encode("a");
  Expect(with_bos.size() >= 2 && with_bos.front() == 1,
         "Default add_bos prepends BOS");
  Expect(tok->Encode("a", no_bos).front() != 1, "add_bos=false drops BOS");
}
}  // namespace

int main() {
  TestBpePieceSplitting();
  TestEndOfTurnStaging();
  TestGgufLoading();
  std::cout << "nemotron_h_moe_tokenizer_test: all assertions passed\n";
  return 0;
}

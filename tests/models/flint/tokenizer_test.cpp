#include "src/models/flint/tokenizer.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace fl = gufo::models::flint;

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// gpt2 byte_to_unicode: every raw byte has a visible single-token name.
std::string VisibleByte(std::uint8_t b) {
  const int i = b;
  const bool direct = (i >= '!' && i <= '~') || (i >= 161 && i <= 172) ||
                      (i >= 174 && i <= 255);
  if (direct) {
    return std::string(1, static_cast<char>(b));
  }
  int n = 0;
  for (int c = 0; c < 256; ++c) {
    const bool c_direct = (c >= '!' && c <= '~') || (c >= 161 && c <= 172) ||
                          (c >= 174 && c <= 255);
    if (!c_direct) {
      if (c == i) {
        break;
      }
      ++n;
    }
  }
  const int cp = 256 + n;
  // All such code points fit in two UTF-8 bytes.
  return std::string({static_cast<char>(0xC0 | (cp >> 6)),
                      static_cast<char>(0x80 | (cp & 0x3F))});
}

// Control-token spellings are assembled from fragments so that no full
// delimiter sequence ever appears as a literal in this source file.
std::string SpecialName(std::string_view inner) {
  return std::string("\x3C\x7C") + std::string(inner) + "\x7C\x3E";
}

struct Fixture {
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<fl::TokenId> control_ids;
  fl::TokenId bos{};
  fl::TokenId eos{};
  fl::TokenId pad{};
  std::vector<fl::TokenId> eog_ids;
};

Fixture MakeFixture() {
  Fixture fx;
  // Special CONTROL entries first: ids 0..3.
  fx.tokens = {SpecialName("eos"), SpecialName("user"),
               SpecialName("assistant"), SpecialName("observation")};
  fx.control_ids = {0, 1, 2, 3};
  fx.eos = 0;
  fx.bos = 1;
  fx.pad = 0;
  fx.eog_ids = {0, 1, 3};

  // Words produced by the merges below, in merge-rank order plus entries that
  // only serve as final merged names.
  const std::vector<std::string> words = {
      "th",          "oc",           "roc",   "rock",         "ll",
      "oll",         "roll",         "'n",    "\xC4\xA0th",   "the",
      "\xC4\xA0the", "\xC4\xA0rock", "'roll", "\xC4\xA0roll", "45",
      "in",          "x\xC3\xA4",
  };
  for (const auto& word : words) {
    fx.tokens.push_back(word);
  }
  fx.merges = {
      "t h",          "o c",           "r oc",   "roc k",         "l l",
      "o ll",         "r oll",         "' n",    "\xC4\xA0 th",   "th e",
      "\xC4\xA0th e", "\xC4\xA0 rock", "' roll", "\xC4\xA0 roll", "4 5",
      "i n",          "x \xC3\xA4",
  };
  for (int b = 0; b < 256; ++b) {
    fx.tokens.push_back(VisibleByte(static_cast<std::uint8_t>(b)));
  }
  return fx;
}

class Tok {
public:
  explicit Tok(const Fixture& fx)
      : tokenizer_(fl::FlintTokenizer::CreateFromVocabulary(
            fx.tokens, fx.merges, fx.control_ids, fx.bos, fx.eos, fx.pad,
            fx.eog_ids)),
        tokens_(fx.tokens) {
    if (tokenizer_ == nullptr) {
      throw std::runtime_error("CreateFromVocabulary failed");
    }
  }

  fl::TokenId Id(std::string_view name) const {
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i] == name) {
        return static_cast<fl::TokenId>(i);
      }
    }
    throw std::runtime_error("unknown token name in expectation");
  }

  fl::TokenId Byte(std::uint8_t b) const { return Id(VisibleByte(b)); }

  void Expect(const std::string_view text,
              const std::vector<std::string>& token_names) const {
    std::vector<fl::TokenId> expected;
    expected.reserve(token_names.size());
    for (const auto& name : token_names) {
      expected.push_back(Id(name));
    }
    const auto actual = tokenizer_->Encode(text);
    if (actual != expected) {
      throw std::runtime_error("Encode mismatch for: " + std::string(text));
    }
  }

  const fl::FlintTokenizer& operator*() const { return *tokenizer_; }
  const fl::FlintTokenizer* operator->() const { return tokenizer_.get(); }

private:
  std::unique_ptr<fl::FlintTokenizer> tokenizer_;
  std::vector<std::string> tokens_;
};

void TestByteCoverage(const Fixture& fx) {
  const Tok t(fx);
  Require(t->GetVocabSize() == static_cast<fl::TokenId>(fx.tokens.size()),
          "vocabulary size mismatch");
  for (int b = 0; b < 256; ++b) {
    const auto tid = t.Byte(static_cast<std::uint8_t>(b));
    Require(tid != fl::kInvalidTokenId, "byte token missing");
    const std::string one(1, static_cast<char>(b));
    const std::string decoded = t->Decode(std::vector<fl::TokenId>{tid});
    Require(decoded == one, "byte decode roundtrip failed");
  }
}

void TestPieceBoundaries(const Tok& t) {
  // Letters keep mixed case in one piece; l+l merges, L+L does not.
  t.Expect("Hello!", {"H", "e", "ll", "o", "!"});
  t.Expect("HELLO", {"H", "E", "L", "L", "O"});
  // Rank order applies the letter merge before the prefix merge, and the
  // prefix merge before the completion: one token per piece either way.
  t.Expect("the", {"the"});
  t.Expect(" the", {"\xC4\xA0the"});
  // Space is a letter-piece prefix, apostrophes attach as contractions.
  t.Expect("in rock'n'roll", {"in", "\xC4\xA0rock", "'n", "'roll"});
  // Digit runs cap at three codepoints; only the "4 5" merge fires.
  t.Expect("12345678", {"1", "2", "3", "45", "6", "7", "8"});
}

void TestCjkSplitsBeforeLetters(const Tok& t, const std::string& ni_hao) {
  // Expression 2 splits CJK runs out of letter segments before the letter
  // expression runs, so the trap pair x+U+00E4 must never merge.
  const std::string text = "x" + ni_hao + "x";
  const auto actual = t->Encode(text);
  const std::vector<fl::TokenId> expected = {
      t.Byte(0x78), t.Byte(0xE4), t.Byte(0xBD), t.Byte(0xA0),
      t.Byte(0xE5), t.Byte(0xA5), t.Byte(0xBD), t.Byte(0x78),
  };
  Require(actual == expected, "CJK run must split letters from Han bytes");
}

void TestSpecialsAndStops(const Tok& t) {
  const fl::TokenId user = t.Id(SpecialName("user"));
  const fl::TokenId eos = t.Id(SpecialName("eos"));
  const fl::TokenId assistant = t.Id(SpecialName("assistant"));

  const std::string text = "y" + SpecialName("user") + "z";
  const auto encoded = t->Encode(text);
  const std::vector<fl::TokenId> expected = {t.Byte('y'), user, t.Byte('z')};
  Require(encoded == expected, "control token must not be BPE-encoded");

  const fl::TokenizerOptions raw{
      .add_bos = false, .add_eos = false, .parse_special_tokens = false};
  const auto raw_encoded = t->Encode(text, raw);
  Require(raw_encoded.size() == text.size(),
          "without parse_special_tokens every byte becomes a token");
  for (const auto tid : raw_encoded) {
    Require(tid != user && tid != eos && tid != assistant,
            "raw encoding must not emit control tokens");
  }

  Require(t->IsStopToken(eos), "eos must be a stop token");
  Require(t->IsStopToken(user), "eog member must be a stop token");
  Require(!t->IsStopToken(assistant), "non-eog control is not a stop token");
  Require(t->IsSpecialToken(assistant), "all control ids are specials");
  Require(!t->IsSpecialToken(t.Byte('y')), "letters are not specials");
}

void TestBosEosOptions(const Tok& t) {
  fl::TokenizerOptions both{.add_bos = true, .add_eos = true};
  const auto empty = t->Encode("", both);
  const std::vector<fl::TokenId> empty_expected = {t.Id(SpecialName("user")),
                                                   t.Id(SpecialName("eos"))};
  Require(empty == empty_expected, "add_bos/add_eos on empty text");

  const auto x = t->Encode("x", both);
  const std::vector<fl::TokenId> x_expected = {
      t.Id(SpecialName("user")), t.Byte('x'), t.Id(SpecialName("eos"))};
  Require(x == x_expected, "add_bos/add_eos on text");
}

void TestFullRoundtrip(const Tok& t, const std::string& ni_hao) {
  const std::string text =
      "Hello, " + ni_hao + " rock'n'roll 12345678 9\nDone.\n\n Bye";
  const auto ids = t->Encode(text);
  Require(t->Decode(ids) == text, "decode(encode(text)) mismatch");
}

// A metadata-only GGUF with only the two tokenizer checks that fail early.
std::string MakeGguf(const std::string& model, const std::string& pre) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{2});
  text("tokenizer.ggml.model");
  pod(std::uint32_t{8});
  text(model);
  text("tokenizer.ggml.pre");
  pod(std::uint32_t{8});
  text(pre);
  bytes.resize((bytes.size() + 31) / 32 * 32);
  return std::string(bytes.begin(), bytes.end());
}

void TestGgufValidation() {
  const std::string wrong_pre = MakeGguf("gpt2", "default");
  std::string error;
  std::string reader_error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      wrong_pre.data(), wrong_pre.size(), &reader_error);
  if (reader == nullptr) {
    throw std::runtime_error("reader rejected synthetic GGUF: " + reader_error);
  }
  Require(fl::FlintTokenizer::CreateFromGguf(*reader, &error) == nullptr,
          "non-flint pre-tokenizer must be rejected");

  const std::string wrong_model = MakeGguf("llama-bpe", "flint");
  const auto reader2 = gufo::core::GgufReader::OpenMemory(wrong_model.data(),
                                                          wrong_model.size());
  Require(reader2 != nullptr, "reader rejected synthetic GGUF");
  error.clear();
  Require(fl::FlintTokenizer::CreateFromGguf(*reader2, &error) == nullptr,
          "non-gpt2 tokenizer must be rejected");
}

}  // namespace

int main() {
  try {
    const Fixture fx = MakeFixture();
    const Tok t(fx);
    const std::string ni_hao = {"\xE4\xBD\xA0\xE5\xA5\xBD"};

    TestByteCoverage(fx);
    TestPieceBoundaries(t);
    TestCjkSplitsBeforeLetters(t, ni_hao);
    TestSpecialsAndStops(t);
    TestBosEosOptions(t);
    TestFullRoundtrip(t, ni_hao);
    TestGgufValidation();
    std::cout << "flint tokenizer_test: OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint tokenizer_test FAILED: " << e.what() << '\n';
    return 1;
  }
}

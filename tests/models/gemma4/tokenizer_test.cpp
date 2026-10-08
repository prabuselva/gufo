// Tokenizer parity against golden vectors captured from the llama.cpp fork
// (`llama_tokenize` with the real Gemma-4 GGUF, vocab-only load). Covers BOS
// prepending, space escaping, newline word splitting, special-token
// partitioning (parse_special on and off), multilingual text and decode
// round trips. Skips (77) unless GUFO_GEMMA4_GGUF points at a Gemma-4 trunk
// GGUF.
#include "src/models/gemma4/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace g4 = gufo::models::gemma4;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

using Ids = std::vector<std::int32_t>;

void ExpectIds(const Ids& got, const Ids& want, const std::string& message) {
  if (got != want) {
    std::cerr << "FAIL: " << message << "\n  want:";
    for (const auto id : want) {
      std::cerr << ' ' << id;
    }
    std::cerr << "\n  got: ";
    for (const auto id : got) {
      std::cerr << ' ' << id;
    }
    std::cerr << "\n";
    ++failures;
  }
}

struct Golden {
  std::string text;
  Ids parse_special;
  Ids literal;  // parse_special = false
};

// Captured with the fork build at llama.cpp/build-cpu against
// gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf (TOKS / TOKN lines).
std::vector<Golden> GoldenVectors() {
  return {
      {"Hello, world!",
       {2, 9259, 236764, 1902, 236888},
       {2, 9259, 236764, 1902, 236888}},
      {"The quick brown fox jumps over the lazy dog.",
       {2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761},
       {2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761}},
      {"  leading and trailing  ",
       {2, 138, 26016, 532, 45330, 138},
       {2, 138, 26016, 532, 45330, 138}},
      {"line1\nline2\n\n\nline3",
       {2, 1257, 236770, 107, 1257, 236778, 109, 1257, 236800},
       {2, 1257, 236770, 107, 1257, 236778, 109, 1257, 236800}},
      {"<|turn>user\nhello<turn|>\n<|turn>model\nhi<turn|>",
       {2, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107, 2202, 106},
       {2,      236820, 236909, 887,    236813, 2364,   107, 23391,
        236820, 887,    111038, 107,    236820, 236909, 887, 236813,
        4368,   107,    2202,   236820, 887,    111038}},
      {"日本語のテキストと emoji 🚀",
       {2, 94951, 236945, 95830, 237032, 64334, 236743, 242015},
       {2, 94951, 236945, 95830, 237032, 64334, 236743, 242015}},
      {"a b  c   d",
       {2, 236746, 518, 138, 236755, 139, 236753},
       {2, 236746, 518, 138, 236755, 139, 236753}},
      {"tab\tand\tcarriage",
       {2, 4823, 255968, 624, 255968, 81712},
       {2, 4823, 255968, 624, 255968, 81712}},
      {"Numbers 1234567890 and mixed123abc",
       {2, 37091, 236743, 236770, 236778, 236800, 236812, 236810, 236825,
        236832, 236828, 236819, 236771, 532, 9726, 236770, 236778, 236800,
        28180},
       {2, 37091, 236743, 236770, 236778, 236800, 236812, 236810, 236825,
        236832, 236828, 236819, 236771, 532, 9726, 236770, 236778, 236800,
        28180}},
  };
}

// Merge engine sanity without an external artifact.
void CheckSynthetic() {
  const std::vector<std::string> tokens = {"a", "b", "ab", "c", "abc", "d"};
  const std::vector<std::string> merges = {"a b", "ab c"};
  std::string error;
  const auto tokenizer = g4::Tokenizer::FromVocabulary(tokens, merges, &error);
  Expect(tokenizer != nullptr, "synthetic tokenizer: " + error);
  if (tokenizer == nullptr) {
    return;
  }
  ExpectIds(tokenizer->Encode("abc", false), {4}, "synthetic abc merges");
  ExpectIds(tokenizer->Encode("ab", false), {2}, "synthetic ab merge");
  ExpectIds(tokenizer->Encode("abcd", false), {4, 5}, "synthetic abcd");
  ExpectIds(tokenizer->Encode("ad", false), {0, 5}, "synthetic no merge");
  const Ids joined = {4, 5};
  Expect(tokenizer->Decode(joined, true) == "abcd", "synthetic decode");
}

}  // namespace

int main() {
  CheckSynthetic();
  if (failures > 0) {
    return 1;
  }

  const char* path = std::getenv("GUFO_GEMMA4_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_GGUF is not set\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "FAIL: open " << path << ": " << error << "\n";
    return 1;
  }
  const auto tokenizer = g4::Tokenizer::FromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cerr << "FAIL: tokenizer load: " << error << "\n";
    return 1;
  }

  Expect(tokenizer->BosId() == 2, "bos id");
  Expect(tokenizer->EosId() == 106, "eos id");
  Expect(tokenizer->UnkId() == 3, "unk id");
  Expect(tokenizer->VocabSize() == 262144, "vocab size");

  for (const Golden& golden : GoldenVectors()) {
    const std::string tag = "'" + golden.text.substr(0, 24) + "'";
    ExpectIds(tokenizer->Encode(golden.text, true, true), golden.parse_special,
              "parse_special " + tag);
    ExpectIds(tokenizer->Encode(golden.text, true, false), golden.literal,
              "literal " + tag);
    const std::string decoded = tokenizer->Decode(golden.parse_special, true);
    Expect(decoded == "<bos>" + golden.text, "decode round trip " + tag);
  }

  if (failures > 0) {
    std::cerr << failures << " tokenizer check(s) failed\n";
    return 1;
  }
  std::cout << "gemma4 tokenizer parity OK\n";
  return 0;
}
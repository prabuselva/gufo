// Runs the scalar Gemma-4 reference over a prompt and writes its tokens and
// per-position logits for the KL harness (tools/gemma4/compare_logits.py),
// against a GPU-runtime logit file or another Storage mode of the oracle:
//   gemma4_reference_probe --model GGUF (--text TEXT | --chat MESSAGE |
//       --tokens-in T.i32) --tokens-out T.i32 --logits-out L.g4lg
//       [--half-kv] [--q8-activations]
// With neither storage flag the oracle keeps K/V exact (Storage::kFloat32);
// --half-kv mirrors the GPU binary16 cache and --q8-activations additionally
// rounds every non-F32 projection input through Q8_1. The oracle is a
// single-row engine, so chunked and bulk prefill share its rows; the GPU side
// of a bulk-vs-exact comparison is captured by the runtime probe.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/reference.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, text, chat, tokens_in, tokens_out, logits_out;
  bool half_kv = false;
  bool q8 = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model") {
      model = value();
    } else if (arg == "--text") {
      text = value();
    } else if (arg == "--chat") {
      chat = value();
    } else if (arg == "--tokens-in") {
      tokens_in = value();
    } else if (arg == "--tokens-out") {
      tokens_out = value();
    } else if (arg == "--logits-out") {
      logits_out = value();
    } else if (arg == "--half-kv") {
      half_kv = true;
    } else if (arg == "--q8-activations") {
      q8 = true;
    } else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(!model.empty() && !tokens_out.empty() && !logits_out.empty() &&
                (!text.empty()) + (!chat.empty()) + (!tokens_in.empty()) == 1,
            "usage: --model GGUF (--text T | --chat M | --tokens-in F) "
            "--tokens-out F --logits-out F [--half-kv] [--q8-activations]");
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    Require(reader != nullptr, error);
    const auto weights = g4::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), error);
    const auto tokenizer = g4::Tokenizer::FromGguf(*reader, &error);
    Require(tokenizer != nullptr, error);

    std::vector<g4::Tokenizer::TokenId> tokens;
    if (!tokens_in.empty()) {
      std::ifstream in(tokens_in, std::ios::binary);
      for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), 4);) {
        tokens.push_back(t);
      }
    } else if (!chat.empty()) {
      g4::ChatMessage message;
      message.role = "user";
      message.content = chat;
      tokens = tokenizer->Encode(g4::RenderChat(std::span(&message, 1)), false,
                                 true);
    } else {
      tokens = tokenizer->Encode(text, true, true);
    }
    gemma4_test::WriteTokens(tokens_out, tokens);

    g4::ReferenceModel reference(
        *weights, static_cast<std::uint32_t>(tokens.size()),
        q8 ? g4::ReferenceModel::Storage::kQ8Activations
           : (half_kv ? g4::ReferenceModel::Storage::kHalfKv
                      : g4::ReferenceModel::Storage::kFloat32));
    gemma4_test::LogitFile file;
    file.vocab = weights->config.vocab_size;
    std::vector<float> logits(weights->config.vocab_size);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t p = 0; p < tokens.size(); ++p) {
      Require(reference.Step(tokens[p], logits),
              "reference step failed at position " + std::to_string(p));
      file.positions.push_back(static_cast<std::uint32_t>(p));
      file.logits.insert(file.logits.end(), logits.begin(), logits.end());
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    gemma4_test::WriteLogits(logits_out, file);
    std::cout << tokens.size() << " tokens in " << seconds << " s\n";
  });
}

// Captures GPU-runtime Gemma-4 logits for the KL harness against the oracle
// file from gemma4_reference_probe (tools/gemma4/compare_logits.py aligns
// common positions):
//   gemma4_runtime_probe --model GGUF (--chat MESSAGE | --tokens-in T.i32)
//       --logits-out L.g4lg [--stride N] [--bulk-final]
// Default rows follow the oracle framing: after feeding token p the session
// logits are written for position p. --stride N keeps every Nth row so long
// conversations stay on disk. --bulk-final instead bulk-syncs the whole
// prefix and writes only the last position, the one logits row the bulk
// prefill exposes today (per-row bulk dumps arrive with the prefill hook).
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/engine.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/logit_file.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

int main(int argc, char** argv) {
  std::string model, chat, tokens_in, logits_out;
  std::size_t stride = 1;
  bool bulk_final = false;
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
    } else if (arg == "--chat") {
      chat = value();
    } else if (arg == "--tokens-in") {
      tokens_in = value();
    } else if (arg == "--logits-out") {
      logits_out = value();
    } else if (arg == "--stride") {
      stride = std::stoul(value());
    } else if (arg == "--bulk-final") {
      bulk_final = true;
    } else {
      std::cerr << "unknown argument " << arg << '\n';
      return 2;
    }
  }
  return gemma4_test::Run([&] {
    Require(!model.empty() && !logits_out.empty() && stride >= 1 &&
                (!chat.empty()) + (!tokens_in.empty()) == 1,
            "usage: --model GGUF (--chat M | --tokens-in F) --logits-out F "
            "[--stride N] [--bulk-final]");
    std::string error;
    std::vector<std::int32_t> tokens;
    std::shared_ptr<const gufo::core::GgufReader> reader(
        gufo::core::GgufReader::OpenFile(model, &error));
    Require(reader != nullptr, error);
    if (!tokens_in.empty()) {
      std::ifstream in(tokens_in, std::ios::binary);
      for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), 4);) {
        tokens.push_back(t);
      }
      Require(!tokens.empty(), "no tokens in " + tokens_in);
    }
    const auto loaded = g4::Model::Load(
        model,
        g4::ModelOptions{
            .max_context = static_cast<std::uint32_t>(tokens.size() + 64),
            .draft_path = "",
            .vision_model_path = ""},
        &error);
    Require(loaded != nullptr, error);
    g4::Model& engine = *loaded;
    if (tokens.empty()) {
      g4::ChatMessage message;
      message.role = "user";
      message.content = chat;
      tokens = engine.EncodeChat(std::span(&message, 1), {}, {});
      Require(!tokens.empty(), "chat prompt encoded to no tokens");
    }
    auto session = engine.CreateSession(
        static_cast<std::uint32_t>(tokens.size() + 64), &error);
    Require(session != nullptr, error);

    gemma4_test::LogitFile file;
    file.vocab = engine.VocabSize();
    if (bulk_final) {
      Require(session->Sync(std::span(tokens), &error), "bulk sync failed");
      const auto logits = session->Logits();
      Require(logits.size() == file.vocab, "logit row width mismatch");
      file.positions.push_back(static_cast<std::uint32_t>(tokens.size() - 1));
      file.logits.assign(logits.begin(), logits.end());
    } else {
      Require(session->Sync(std::span(tokens).first(1), &error),
              "first-token sync failed");
      for (std::size_t p = 0; p < tokens.size(); ++p) {
        if (p > 0) {
          Require(session->Evaluate(tokens[p], &error),
                  "evaluate failed at position " + std::to_string(p));
        }
        if ((p + 1) % stride != 0 && p + 1 != tokens.size()) {
          continue;
        }
        const auto logits = session->Logits();
        Require(logits.size() == file.vocab, "logit row width mismatch");
        file.positions.push_back(static_cast<std::uint32_t>(p));
        file.logits.insert(file.logits.end(), logits.begin(), logits.end());
      }
    }
    gemma4_test::WriteLogits(logits_out, file);
    std::cout << file.positions.size() << " rows of " << tokens.size()
              << " tokens written\n";
  });
}

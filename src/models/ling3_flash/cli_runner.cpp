#include "src/models/ling3_flash/cli_runner.hpp"

#include <chrono>
#include <iostream>
#include <vector>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/cli/prompt/prompt.hpp"
#include "src/core/sampling.hpp"
#include "src/models/ling3_flash/chat_template.hpp"
#include "src/models/ling3_flash/engine.hpp"
#endif

namespace gufo::models::ling3_flash {

bool IsLing3Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "bailingmoe3";
}

#if defined(ENGINE_ENABLE_HIP)

namespace {

constexpr std::uint32_t kDefaultContext = 4096;

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

std::shared_ptr<Model> LoadModel(const cli::PromptOptions& opt,
                                 std::chrono::steady_clock::time_point start) {
  std::string error;
  auto model = Model::Load(
      opt.model_path, ModelOptions{.max_context = kDefaultContext}, &error);
  PrintModelLoadTime(start, model != nullptr);
  if (model == nullptr) {
    std::cerr << "Ling 3.0 Flash load failed: " << error << '\n';
  }
  return model;
}

// Streams one generation turn from a synced session, echoing the decoded
// text and optionally accumulating it into `reply`.
int GenerateTurn(const Model& model, Session& session,
                 const cli::PromptOptions& opt, std::string* reply) {
  std::vector<sampling::TokenId> history;
  for (const std::int32_t token : session.Tokens()) {
    history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, history);
  std::size_t generated = 0;
  std::size_t drafted = 0;
  std::size_t accepted = 0;
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  while (generated < opt.max_tokens) {
    Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated, sampler, &decoded,
                            &error)) {
      std::cerr << "Ling 3.0 Flash decode failed: " << error << '\n';
      return 1;
    }
    drafted += decoded.drafted;
    accepted += decoded.accepted;
    for (const std::int32_t token : decoded.tokens) {
      const std::string piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply != nullptr) {
        reply->append(piece);
      }
      ++generated;
    }
    if (decoded.stop) {
      break;
    }
  }
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cerr << "Generated " << generated << " tokens (" << generated / seconds
              << " tok/s)";
    if (drafted > 0U) {
      std::cerr << " MTP accepted " << accepted << "/" << drafted;
    }
    std::cerr << '\n';
  }
  return 0;
}

}  // namespace

int RunLing3FlashPrompt(const cli::PromptOptions& opt,
                        const core::GgufReader& reader,
                        std::chrono::steady_clock::time_point load_start,
                        const ReasoningOptions& reasoning) {
  (void)reader;
  if (opt.force_cpu ||
      (!opt.speculative_backend.empty() && opt.speculative_backend != "mtp")) {
    std::cerr << "Ling 3.0 Flash requires ROCm and supports only "
                 "--speculative mtp\n";
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
    std::cerr << "Ling 3.0 Flash does not support image input\n";
    return 2;
  }
  auto model = LoadModel(opt, load_start);
  if (model == nullptr) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (session == nullptr) {
    std::cerr << "Ling 3.0 Flash session failed: " << error << '\n';
    return 1;
  }
  if (opt.speculative_backend == "mtp") {
    if (!model->HasMtp()) {
      std::cerr << "Ling 3.0 Flash model has no MTP block; omit "
                   "--speculative mtp\n";
      return 1;
    }
    session->SetMtpEnabled(true);
    session->SetDraftLimits(opt.min_draft_tokens, opt.draft_tokens);
  }

  std::vector<std::int32_t> ids;
  std::vector<tokenization::ChatMessage> messages;
  if (opt.use_chat_template) {
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto rendered = RenderChat(
        messages, ResolveLing3ChatOptions(reasoning), /*add_generation_prompt=*/
        true);
    if (!rendered.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
    ids = model->Tokenize(*rendered);
    if (ids.empty() ||
        ids.size() >= static_cast<std::size_t>(kDefaultContext) ||
        opt.max_tokens >
            static_cast<std::size_t>(kDefaultContext) - ids.size()) {
      std::cerr << "Ling 3.0 Flash prompt and output exceed the "
                << kDefaultContext << "-token CLI context\n";
      return 1;
    }
  } else {
    ids = model->Tokenize(opt.prompt_text);
  }
  if (!session->Sync(ids, &error)) {
    std::cerr << "Ling 3.0 Flash prefill failed: " << error << '\n';
    return 1;
  }
  if (GenerateTurn(*model, *session, opt, nullptr) != 0) {
    return 1;
  }
  std::cout << '\n';
  return 0;
}

int RunLing3FlashChat(const cli::PromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point load_start,
                      const ReasoningOptions& reasoning) {
  (void)reader;
  if (opt.force_cpu ||
      (!opt.speculative_backend.empty() && opt.speculative_backend != "mtp")) {
    std::cerr << "Ling 3.0 Flash requires ROCm and supports only "
                 "--speculative mtp\n";
    PrintModelLoadTime(load_start, false);
    return 1;
  }
  if (!opt.use_chat_template) {
    std::cerr << "Ling 3.0 Flash interactive chat requires chat framing; use "
                 "prompt --raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (model == nullptr) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (session == nullptr) {
    std::cerr << "Ling 3.0 Flash session failed: " << error << '\n';
    return 1;
  }
  if (model->HasMtp()) {
    session->SetMtpEnabled(true);
    session->SetDraftLimits(opt.min_draft_tokens, opt.draft_tokens);
  }

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }
  const auto options = ResolveLing3ChatOptions(reasoning);
  std::cout << "=== Gufo Interactive Chat (Ling 3.0 Flash) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.push_back({tokenization::ChatRole::kUser, input, "", ""});
    const auto rendered =
        RenderChat(history, options, /*add_generation_prompt=*/
                   true);
    if (!rendered.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
    const auto ids = model->Tokenize(*rendered);
    if (ids.empty() ||
        ids.size() >= static_cast<std::size_t>(kDefaultContext)) {
      std::cerr << "Ling 3.0 Flash chat history exceeds the " << kDefaultContext
                << "-token CLI context\n";
      return 1;
    }
    if (!session->Sync(ids, &error)) {
      std::cerr << "Ling 3.0 Flash prefill failed: " << error << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateTurn(*model, *session, opt, &reply) != 0) {
      return 1;
    }
    tokenization::ChatMessage response;
    response.role = tokenization::ChatRole::kAssistant;
    if (options.enable_thinking) {
      const auto boundary = reply.find(kThinkEnd);
      if (boundary == std::string::npos) {
        response.content = std::move(reply);
      } else {
        response.thought = reply.substr(0, boundary);
        response.content = reply.substr(boundary + kThinkEnd.size());
      }
    } else {
      response.content = std::move(reply);
    }
    history.push_back(std::move(response));
    std::cout << '\n';
  }
  return 0;
}

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::ling3_flash

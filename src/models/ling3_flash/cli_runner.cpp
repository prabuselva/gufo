#include "src/models/ling3_flash/cli_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/cli/bench/bench.hpp"
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

struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {.mean = 0.0, .stddev = 0.0};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

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
    // The end-of-turn framing token is the stop signal, not content; drop
    // it so the reply never carries the raw control text.
    if (decoded.stop && !decoded.tokens.empty() &&
        model.IsStopToken(decoded.tokens.back())) {
      decoded.tokens.pop_back();
    }
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

int RunLing3FlashBenchmark(
    const cli::BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Ling 3.0 Flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  if (std::max({max_depth, max_prompt, max_generation}) >
      std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{kDefaultContext}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1});
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Ling 3.0 Flash supports only --speculative mtp or "
                 "off\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Ling 3.0 Flash bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Ling 3.0 Flash context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = Model::Load(
      options.model_path,
      ModelOptions{.max_context = static_cast<std::uint32_t>(required_context)},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Ling 3.0 Flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  if (mtp && !model->HasMtp()) {
    std::cerr << "Error: --speculative mtp requested but the model has no MTP "
                 "head\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  // Repeat a fixed token pattern for reproducible timing.
  std::vector<std::int32_t> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = pattern[i % pattern.size()];
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        // The model owns one shared executor; a fresh session must reset it so
        // each timed prefill starts at position 0 (the serve path does the same
        // in Invalidate()).
        session->Reset();
        if (mtp) {
          // Under --speculative mtp this row must measure the same spec Sync
          // the serve path runs (batched draft-cache prefill), not a plain
          // prefill with the draft block skipped.
          session->SetMtpEnabled(true);
        }
        if (depth > 0 &&
            !session->Sync(std::span(tokens).first(depth), &error)) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        session->Reset();
        if (mtp) {
          // Speculative decode consumes the draft cache, so it must be filled
          // during the prefill below; enable it before Sync. The pp loop
          // enables it too, so every --speculative row measures the serve
          // path's spec Sync.
          session->SetMtpEnabled(true);
          session->SetDraftLimits(options.min_draft_tokens,
                                  options.draft_tokens);
        }
        if (!session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        std::size_t drafted = 0;
        std::size_t accepted = 0;
        const std::vector<sampling::TokenId> history(
            tokens.begin(), tokens.begin() + prefix_length);
        sampling::SamplerState sampler(options.sampling, history);
        const auto start = std::chrono::steady_clock::now();
        while (generated.size() < generation_length) {
          Session::DecodeResult decoded;
          // A one-token budget disables the draft path (greedy); a larger
          // budget lets the session chain up to the adaptive draft count.
          const std::size_t budget =
              mtp ? generation_length - generated.size() : 1;
          if (!session->DecodeStep(budget, sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running Ling 3.0 Flash decode: " << error
                      << '\n';
            return 1;
          }
          generated.insert(generated.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
          drafted += decoded.drafted;
          accepted += decoded.accepted;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          std::cerr << "Ling 3.0 Flash tg depth=" << depth
                    << " drafted=" << drafted << " accepted=" << accepted
                    << " output_sha256="
                    << crypto::Sha256Hex(
                           std::span(reinterpret_cast<const std::uint8_t*>(
                                         generated.data()),
                                     generated.size() * sizeof(std::int32_t)))
                    << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::ling3_flash

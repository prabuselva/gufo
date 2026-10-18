#include "src/models/gemma4/cli_runner.hpp"

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

bool IsGemma4(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "gemma4";
}

}  // namespace gufo::models::gemma4

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "src/core/reasoning.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::gemma4 {
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

// Hash emitted IDs in a portable byte order, after the timed generation.
// Decoded text alone can hide different token sequences.
void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

ReasoningOptions PromptReasoningOptions(const cli::PromptOptions& options) {
  ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

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

struct KlReport {
  bool finite{true};
  bool top1_match{true};
  double kl{0.0};
  double tv{0.0};
};

// Same metric family as tools/gemma4/compare_logits.py: KL of the batched
// distribution against the sequential reference sum_v p·(log p − log q), in
// nats per position, plus total variation and the top-1 agreement.
KlReport KlDivergence(std::span<const float> reference,
                      std::span<const float> candidate) {
  KlReport report;
  double ref_max = -std::numeric_limits<double>::infinity();
  double cand_max = -std::numeric_limits<double>::infinity();
  for (const float value : reference) {
    if (!std::isfinite(value)) {
      report.finite = false;
    }
    ref_max = std::max(ref_max, static_cast<double>(value));
  }
  for (const float value : candidate) {
    if (!std::isfinite(value)) {
      report.finite = false;
    }
    cand_max = std::max(cand_max, static_cast<double>(value));
  }
  if (!report.finite || reference.size() != candidate.size() ||
      reference.empty()) {
    return report;
  }
  const std::size_t argmax = static_cast<std::size_t>(
      std::max_element(reference.begin(), reference.end()) - reference.begin());
  const std::size_t cand_argmax = static_cast<std::size_t>(
      std::max_element(candidate.begin(), candidate.end()) - candidate.begin());
  report.top1_match = argmax == cand_argmax;
  double log_z_ref = 0.0;
  double log_z_cand = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    log_z_ref += std::exp(static_cast<double>(reference[i]) - ref_max);
    log_z_cand += std::exp(static_cast<double>(candidate[i]) - cand_max);
  }
  log_z_ref += ref_max;
  log_z_cand += cand_max;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double log_p = static_cast<double>(reference[i]) - log_z_ref;
    const double log_q = static_cast<double>(candidate[i]) - log_z_cand;
    const double p = std::exp(log_p);
    report.kl += p * (log_p - log_q);
    report.tv += 0.5 * std::abs(p - std::exp(log_q));
  }
  return report;
}

// Bulk batched prefill against one-token-at-a-time evaluation of the same
// prefix; both run on the GPU, so this checks the batched kernels against
// the exact path with KL, not only rmse/argmax.
bool ValidateGemma4Prefill(Model& model,
                           const std::vector<std::int32_t>& tokens,
                           std::size_t prompt_length, std::uint32_t context,
                           std::string* error) {
  if (prompt_length == 0 || prompt_length >= context) {
    *error = "--validate-prefill must be between 1 and the context size";
    return false;
  }
  const auto sequential = model.CreateSession(context, error);
  const auto batched = model.CreateSession(context, error);
  const auto prefix = std::span(tokens).first(prompt_length);
  if (!sequential || !batched || !sequential->Sync(prefix.first(1), error) ||
      !batched->Sync(prefix, error)) {
    if (error->empty()) {
      *error = "session prefill failed";
    }
    return false;
  }
  for (const std::int32_t token : prefix.subspan(1)) {
    if (!sequential->Evaluate(token, error)) {
      if (error->empty()) {
        *error = "sequential reference failed";
      }
      return false;
    }
  }
  const auto report = KlDivergence(sequential->Logits(), batched->Logits());
  // Known bulk-prefill drift class (fork report: bulk 0.027, top-1
  // 1489/1542). PASS is the fork-optimized exactness level (3.7e-5 nats);
  // drift inside the recorded baseline is reported but not a run failure;
  // larger divergence means the bulk path actually broke.
  constexpr double kExactKl = 3.7e-5;
  constexpr double kRecordedBulkKl = 5e-2;
  const char* status = "RECORDED";
  if (!report.finite || report.kl > kRecordedBulkKl) {
    status = "FAIL";
  } else if (report.top1_match && report.kl <= kExactKl) {
    status = "PASS";
  }
  std::cout << std::fixed << std::setprecision(10)
            << "[Prefill Validation] tokens=" << prompt_length
            << " finite=" << (report.finite ? "yes" : "no")
            << " top1_match=" << (report.top1_match ? "yes" : "no")
            << " kl=" << report.kl << " tv=" << report.tv
            << " status=" << status << '\n';
  if (status == std::string_view("FAIL")) {
    std::ostringstream message;
    message << (!report.finite ? "non-finite logits"
                               : "bulk-vs-exact KL exceeds the recorded "
                                 "baseline and the bulk path must be fixed");
    *error = message.str();
    return false;
  }
  return true;
}

}  // namespace

int RunGemma4Benchmark(const cli::BenchOptions& options,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Gemma-4-26B-A4B\n";
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
  if (std::max({max_depth, max_prompt, max_generation,
                options.validate_prefill_tokens}) >
      std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Gemma-4-26B-A4B supports only --speculative mtp or "
                 "off\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Gemma-4-26B-A4B bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Gemma-4-26B-A4B context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = Model::Load(
      options.model_path,
      ModelOptions{.max_context = static_cast<std::uint32_t>(required_context),
                   .draft_path = mtp ? options.mtp_model_path : "",
                   .vision_model_path = ""},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Gemma-4-26B-A4B model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  if (mtp && !model->HasMtp()) {
    std::cerr << "Error: --speculative mtp requested but the draft sidecar "
                 "failed to load\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

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

  if (options.validate_prefill_tokens != 0) {
    if (!ValidateGemma4Prefill(*model, tokens, options.validate_prefill_tokens,
                               static_cast<std::uint32_t>(required_context),
                               &error)) {
      std::cerr << "Gemma-4-26B-A4B prefill validation failed: " << error
                << '\n';
      return 1;
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
        session->Reset();
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
          // Speculative decode reads the trunk KV the prefill below writes, so
          // enable it before Sync; the pp loop leaves it off.
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
          // A one-token budget disables the draft path (greedy); a larger one
          // lets the session chain up to the pinned draft count.
          const std::size_t budget =
              mtp ? generation_length - generated.size() : 1;
          if (!session->DecodeStep(budget, sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running Gemma-4 decode: " << error << '\n';
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
          std::cerr << "Gemma-4-26B-A4B tg depth=" << depth
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

namespace {

std::shared_ptr<Model> LoadGemma4Model(
    const cli::PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (opt.force_cpu) {
    std::cerr << "Gemma-4-26B-A4B is supported only by the ROCm backend\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  const bool mtp = opt.speculative_backend == "mtp";
  if (!opt.speculative_backend.empty() && !mtp) {
    std::cerr << "Gemma-4-26B-A4B supports only --speculative mtp or off\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (mtp && opt.mtp_model_path.empty()) {
    std::cerr << "--speculative mtp requires --mtp-model\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  std::string template_error;
  if (!ValidateGgufTemplate(reader, &template_error)) {
    std::cerr << "Unsupported Gemma-4 chat template: " << template_error
              << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  std::string error;
  auto model =
      Model::Load(opt.model_path,
                  ModelOptions{.max_context = kDefaultContext,
                               .draft_path = mtp ? opt.mtp_model_path : "",
                               .vision_model_path = opt.vision_model_path},
                  &error);
  if (model == nullptr) {
    std::cerr << "Error creating Gemma-4-26B-A4B model: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (mtp && !model->HasMtp()) {
    std::cerr << "--speculative mtp requested but the draft sidecar failed to "
                 "load\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  PrintModelLoadTime(load_start);
  return model;
}

int GenerateGemma4Response(
    const cli::PromptOptions& opt, const std::shared_ptr<Model>& model,
    Session& session, std::span<const std::int32_t> prompt_tokens,
    std::string* reply = nullptr,
    const std::vector<vision::VisionSlot>* vision_slots = nullptr) {
  std::string error;
  const auto emit = [&](std::int32_t token) {
    const auto piece = model->TokenText(token);
    if (reply != nullptr)
      reply->append(piece);
    std::cout << piece << std::flush;
  };
  if (prompt_tokens.empty()) {
    std::cerr << "Gemma-4-26B-A4B prompt produced no tokens\n";
    return 1;
  }
  if (prompt_tokens.size() >= kDefaultContext ||
      opt.max_tokens >= kDefaultContext - prompt_tokens.size()) {
    std::cerr << "Gemma-4-26B-A4B prompt and output exceed the 4096-token "
                 "CLI context\n";
    return 1;
  }
  session.Reset();
  const bool synced = vision_slots != nullptr && !vision_slots->empty()
                          ? session.Sync(prompt_tokens, *vision_slots, &error)
                          : session.Sync(prompt_tokens, &error);
  if (!synced) {
    std::cerr << "Gemma-4-26B-A4B prefill failed: " << error << '\n';
    return 1;
  }
  if (opt.verbose) {
    std::cout << "[Engine]: Gemma-4-26B-A4B ROCm (gfx1151)\n"
              << "Model: " << model->ModelName() << '\n'
              << "Prompt tokens: " << prompt_tokens.size() << '\n'
              << "Max tokens: " << opt.max_tokens << '\n'
              << "--- Generation Output ---\n";
  }
  std::vector<sampling::TokenId> sampling_history;
  sampling_history.reserve(prompt_tokens.size());
  for (const std::int32_t token : prompt_tokens) {
    sampling_history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, sampling_history);

  const auto generation_start = std::chrono::steady_clock::now();
  const bool mtp = opt.speculative_backend == "mtp";
  std::size_t generated = 0;
  std::vector<tokenization::TokenId> generated_ids;
  while (generated < opt.max_tokens) {
    // A one-token budget keeps the autoregressive path; a larger one lets the
    // session chain drafts through the MTP block and verify them in one pass.
    const std::size_t budget = mtp ? opt.max_tokens - generated : 1;
    Session::DecodeResult decoded;
    if (!session.DecodeStep(budget, sampler, &decoded, &error, true)) {
      std::cerr << "\nGemma-4-26B-A4B decode failed: " << error << '\n';
      return 1;
    }
    if (decoded.tokens.empty()) {
      break;
    }
    for (const std::int32_t token : decoded.tokens) {
      if (model->IsStopToken(token)) {
        decoded.stop = true;
        break;
      }
      emit(token);
      if (opt.verbose)
        generated_ids.push_back(token);
      if (++generated >= opt.max_tokens) {
        break;
      }
    }
    if (decoded.stop) {
      break;
    }
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      generation_start)
            .count();
    std::cout << "Generated " << generated << " tokens on ROCm in " << seconds
              << "s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
    PrintTokenTrace(generated_ids);
  }
  return 0;
}

ChatTemplateOptions Gemma4ChatOptions(const cli::PromptOptions& opt) {
  const auto reasoning = PromptReasoningOptions(opt);
  return ChatTemplateOptions{
      .add_generation_prompt = true,
      .enable_thinking = reasoning.enabled.value_or(false),
      .preserve_thinking = reasoning.preserve_thinking.value_or(false),
  };
}

}  // namespace

int RunGemma4Prompt(const cli::PromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start) {
  auto model = LoadGemma4Model(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma-4 session creation failed: " << error << '\n';
    return 1;
  }
  if (opt.speculative_backend == "mtp") {
    session->SetMtpEnabled(true);
    session->SetDraftLimits(opt.min_draft_tokens, opt.draft_tokens);
  }
  std::vector<std::int32_t> prompt_tokens;
  std::vector<vision::VisionSlot> vision_slots;
  if (opt.use_chat_template) {
    std::vector<ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back({.role = "system", .content = opt.system_prompt});
    }
    std::vector<core::Image> images;
    for (const auto& path : opt.image_paths) {
      images.push_back(core::DecodeImage(core::ReadImageFile(path)));
    }
    if (!images.empty()) {
      if (!model->HasVision()) {
        std::cerr << "Gemma-4 image input requires --mmproj\n";
        return 1;
      }
      messages.push_back(
          {.role = "user",
           .content = opt.prompt_text,
           .image_count = static_cast<std::uint32_t>(images.size())});
      auto prepared = model->EncodeChatVision(
          messages, {}, Gemma4ChatOptions(opt), images, &error);
      if (prepared.tokens.empty()) {
        std::cerr << "Gemma-4 vision prompt failed: " << error << '\n';
        return 1;
      }
      prompt_tokens = std::move(prepared.tokens);
      vision_slots = std::move(prepared.images);
    } else {
      messages.push_back({.role = "user", .content = opt.prompt_text});
      prompt_tokens = model->EncodeChat(messages, {}, Gemma4ChatOptions(opt));
    }
  } else {
    if (!opt.image_paths.empty()) {
      std::cerr << "Gemma-4 image input requires the chat template; drop "
                   "--raw\n";
      return 1;
    }
    prompt_tokens = model->Tokenize(opt.prompt_text);
  }
  return GenerateGemma4Response(opt, model, *session, prompt_tokens, nullptr,
                                vision_slots.empty() ? nullptr : &vision_slots);
}

int RunGemma4Chat(const cli::PromptOptions& opt, const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Gemma-4 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadGemma4Model(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma-4 session creation failed: " << error << '\n';
    return 1;
  }
  if (opt.speculative_backend == "mtp") {
    session->SetMtpEnabled(true);
    session->SetDraftLimits(opt.min_draft_tokens, opt.draft_tokens);
  }
  std::vector<ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back({.role = "system", .content = opt.system_prompt});
  }
  const auto chat_options = Gemma4ChatOptions(opt);
  std::cout << "=== Gufo Interactive Chat (Gemma-4-26B-A4B) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;
    history.push_back({.role = "user", .content = input});
    const auto tokens = model->EncodeChat(history, {}, chat_options);
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateGemma4Response(opt, model, *session, tokens, &reply) != 0)
      return 1;
    ChatMessage response;
    response.role = "assistant";
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "<channel|>";
      const auto boundary = reply.find(end);
      response.reasoning_content = reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        response.content = reply.substr(boundary + end.size());
      }
    } else {
      response.content = std::move(reply);
    }
    history.push_back(std::move(response));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace gufo::models::gemma4

#endif  // defined(ENGINE_ENABLE_HIP)
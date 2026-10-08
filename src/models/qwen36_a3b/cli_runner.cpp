#include "src/models/qwen36_a3b/cli_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::qwen36_a3b {
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

}  // namespace

bool IsQwen36A3B(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "qwen35moe";
}

int RunQwen36A3BBenchmark(
    const cli::BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  namespace q36 = models::qwen36_a3b;
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Qwen3.6-35B-A3B\n";
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
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1});
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Qwen3.6-35B-A3B supports only --speculative mtp or "
                 "off\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Qwen3.6-35B-A3B bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Qwen3.6-35B-A3B context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = q36::Model::Load(
      options.model_path,
      q36::ModelOptions{.max_context =
                            static_cast<std::uint32_t>(required_context)},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Qwen3.6-35B-A3B model: " << error << '\n';
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
          // during the prefill below; enable it before Sync. The pp loop leaves
          // it off so plain prefill skips the draft block.
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
          q36::Session::DecodeResult decoded;
          // A one-token budget disables the draft path (greedy); a larger
          // budget lets the session chain up to the pinned draft count.
          const std::size_t budget =
              mtp ? generation_length - generated.size() : 1;
          if (!session->DecodeStep(budget, sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running Qwen3.6-35B-A3B decode: " << error
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
          std::cerr << "Qwen3.6-35B-A3B tg depth=" << depth
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

std::shared_ptr<Model> LoadQwen36A3BModel(
    const cli::PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  if (opt.force_cpu || !opt.speculative_backend.empty()) {
    std::cerr << "Qwen3.6-35B-A3B requires ROCm and does not support "
                 "speculative decoding\n";
    return nullptr;
  }
  if (opt.use_chat_template &&
      !tokenization::QwenChatTemplate::ValidateGgufTemplate(reader, &error)) {
    std::cerr << "Unsupported Qwen3.6-35B-A3B chat template: " << error << '\n';
    return nullptr;
  }
  auto model = models::qwen36_a3b::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .attn_window = opt.attn_window,
       .attn_sink = opt.attn_sink,
       .vision_model_path = opt.vision_model_path},
      &error);
  PrintModelLoadTime(load_start, model != nullptr);
  if (!model)
    std::cerr << "Qwen3.6-35B-A3B load failed: " << error << '\n';
  return model;
}

int GenerateQwen36A3BResponse(const cli::PromptOptions& opt, const Model& model,
                              Session& session,
                              std::span<const tokenization::TokenId> prompt,
                              std::string* reply) {
  if (prompt.empty() || prompt.size() >= session.ContextSize() ||
      opt.max_tokens > session.ContextSize() - prompt.size()) {
    std::cerr << "Qwen3.6-35B-A3B prompt and output exceed the 4096-token CLI "
                 "context\n";
    return 1;
  }
  const std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  std::string error;
  if (!session.Sync(input, &error)) {
    std::cerr << "Qwen3.6-35B-A3B prefill failed: " << error << '\n';
    return 1;
  }
  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<tokenization::TokenId> generated;
  const auto start = std::chrono::steady_clock::now();
  while (generated.size() < opt.max_tokens) {
    models::qwen36_a3b::Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated.size(), sampler,
                            &decoded, &error)) {
      std::cerr << "Qwen3.6-35B-A3B decode failed: " << error << '\n';
      return 1;
    }
    for (const auto token : decoded.tokens) {
      const auto piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply)
        reply->append(piece);
      generated.push_back(static_cast<tokenization::TokenId>(token));
    }
    if (decoded.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cerr << "Generated " << generated.size() << " tokens ("
              << generated.size() / seconds << " tok/s)\n";
    PrintTokenTrace(generated);
  }
  return 0;
}

}  // namespace gufo::models::qwen36_a3b

#endif  // defined(ENGINE_ENABLE_HIP)
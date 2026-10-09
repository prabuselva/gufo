// End-to-end semantic gate for the MTP draft block. Parity tests pin the
// draft logits against the oracle for one step, but only real text shows the
// draft chain is aligned with the trunk: fed the trunk's hidden state and the
// freshly sampled token, the draft must predict the trunk's next greedy
// token most of the time. A position, cache or gate misalignment leaves the
// formulas intact and collapses agreement to near zero. Skips (77) unless
// GUFO_LING3_FLASH_GGUF points at a Ling 3.0 Flash GGUF.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling3_flash/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace l3 = gufo::models::ling3_flash;
namespace rocm = l3::rocm;
namespace ht = gufo::tests::ling3_flash;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> values(count);
  ht::CheckHip(hipMemcpy(values.data(), device, count * sizeof(float),
                         hipMemcpyDeviceToHost),
               "download");
  return values;
}

std::int32_t ArgMax(const std::vector<float>& v) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(best);
}

constexpr int kDecodeTokens = 32;
// Observed agreement on this artifact is well above this floor; a broken
// draft chain sits near the single-token chance level for a 157k vocab.
constexpr double kMinAcceptRate = 0.6;

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_LING3_FLASH_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_LING3_FLASH_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cerr << "tokenizer load failed: " << error << "\n";
    return 1;
  }
  const auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto mtp = l3::MtpWeights::Bind(*reader, weights->config, &error);
  if (!mtp.has_value()) {
    std::cerr << "MTP bind failed: " << error << "\n";
    return 1;
  }
  const auto device =
      rocm::DeviceModel::Upload(*weights, *reader, &*mtp, &error);
  if (device == nullptr) {
    std::cerr << "device upload failed: " << error << "\n";
    return 1;
  }
  constexpr std::uint32_t kContext = 128;
  const auto executor = rocm::Executor::Create(*device, kContext, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }
  const auto session = executor->CreateSession(kContext, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }

  std::vector<std::int32_t> prompt;
  for (const auto id : tokenizer->Encode("The capital of France is")) {
    prompt.push_back(static_cast<std::int32_t>(id));
  }
  if (prompt.size() < 4) {
    std::cerr << "prompt tokenized to " << prompt.size() << " tokens\n";
    return 1;
  }

  // The draft cache must track the trunk one entry per token, so every prompt
  // token feeds both blocks; the last prompt token's draft pair is produced
  // by the first sampled token below.
  const std::size_t vocab = weights->config.vocab_size;
  if (!executor->Step(*session, prompt[0], &error)) {
    std::cerr << "prefill step failed: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 1; i < prompt.size(); ++i) {
    if (!executor->MtpStep(*session, prompt[i], &error)) {
      std::cerr << "prefill draft step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*session, prompt[i], &error)) {
      std::cerr << "prefill step " << i << " failed: " << error << "\n";
      return 1;
    }
  }

  const std::int32_t eos =
      static_cast<std::int32_t>(tokenizer->GetEosTokenId());
  const std::int32_t pad =
      static_cast<std::int32_t>(tokenizer->GetPadTokenId());
  int agreed = 0;
  int total = 0;
  for (int i = 0; i < kDecodeTokens; ++i) {
    const auto token = ArgMax(Download(executor->logits(), vocab));
    if (token == eos || token == pad) {
      break;
    }
    if (!executor->MtpStep(*session, token, &error)) {
      std::cerr << "draft step failed: " << error << "\n";
      return 1;
    }
    const auto draft = ArgMax(Download(executor->mtp_logits(), vocab));
    if (!executor->Step(*session, token, &error)) {
      std::cerr << "decode step failed: " << error << "\n";
      return 1;
    }
    const auto actual = ArgMax(Download(executor->logits(), vocab));
    ++total;
    if (draft == actual) {
      ++agreed;
    }
  }
  if (total < kDecodeTokens / 2) {
    std::cerr << "only " << total << " decode steps measured\n";
    ++failures;
  }
  const double rate = total == 0 ? 0.0 : static_cast<double>(agreed) / total;
  std::cout << "MTP greedy draft agreement: " << agreed << "/" << total << " ("
            << rate * 100.0 << "%)\n";
  Expect(rate >= kMinAcceptRate,
         "draft agrees with the trunk most of the time");

  // Speculative rounds must reproduce the plain decode exactly: every
  // commit is either the target's own token or a target-confirmed draft,
  // and RollbackVerify has to restore the recurrent trunk state to the
  // committed prefix. Any position, snapshot or cache misalignment shows
  // up as the first mismatching greedy token.
  constexpr int kParityTokens = 48;
  constexpr std::size_t kK = 3;
  std::vector<std::int32_t> oracle;
  {
    const auto plain = executor->CreateSession(kContext, &error);
    if (plain == nullptr) {
      std::cerr << "parity oracle session failed: " << error << "\n";
      return 1;
    }
    if (!executor->Prefill(*plain, prompt.data(),
                           static_cast<std::uint32_t>(prompt.size()), &error)) {
      std::cerr << "parity oracle prefill failed: " << error << "\n";
      return 1;
    }
    for (int i = 0; i < kParityTokens; ++i) {
      const auto logits = Download(executor->logits(), vocab);
      const auto token = ArgMax(logits);
      if (token == eos || token == pad) {
        break;
      }
      oracle.push_back(token);
      if (!executor->Step(*plain, token, &error)) {
        std::cerr << "parity oracle step failed: " << error << "\n";
        return 1;
      }
    }
  }

  std::vector<std::int32_t> spec;
  {
    const auto sp = executor->CreateSession(kContext, &error);
    if (sp == nullptr) {
      std::cerr << "parity spec session failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*sp, prompt[0], &error)) {
      std::cerr << "parity spec step failed: " << error << "\n";
      return 1;
    }
    for (std::size_t i = 1; i < prompt.size(); ++i) {
      if (!executor->MtpStep(*sp, prompt[i], &error) ||
          !executor->Step(*sp, prompt[i], &error)) {
        std::cerr << "parity spec prefill failed: " << error << "\n";
        return 1;
      }
    }
    std::vector<float> vrows((kK + 1U) * vocab);
    const auto vrow = [&](std::size_t r) { return vrows.data() + r * vocab; };
    const auto argmax = [](const float* values, std::size_t count) {
      std::size_t best = 0;
      for (std::size_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) {
          best = i;
        }
      }
      return static_cast<std::int32_t>(best);
    };
    std::vector<float> logits = Download(executor->logits(), vocab);
    int mismatch = -1;
    std::size_t rounds = 0;
    std::size_t accepted_total = 0;
    while (spec.size() < oracle.size()) {
      const auto token = argmax(logits.data(), logits.size());
      if (token == eos || token == pad) {
        break;
      }
      ++rounds;
      if (!executor->MtpStep(*sp, token, &error)) {
        std::cerr << "parity spec draft failed: " << error << "\n";
        return 1;
      }
      std::int32_t drafts[kK];
      for (std::size_t i = 0; i < kK; ++i) {
        if (i > 0U && !executor->MtpDraft(*sp, drafts[i - 1U], &error)) {
          std::cerr << "parity spec chain failed: " << error << "\n";
          return 1;
        }
        drafts[i] = ArgMax(Download(executor->mtp_logits(), vocab));
      }
      if (!executor->Verify(*sp, token, drafts, static_cast<std::uint32_t>(kK),
                            &error)) {
        std::cerr << "parity spec verify failed: " << error << "\n";
        return 1;
      }
      ht::CheckHip(
          hipMemcpy(vrows.data(), executor->verify_logits(),
                    vrows.size() * sizeof(float), hipMemcpyDeviceToHost),
          "upload verify logits");
      std::size_t accepted = 0;
      std::int32_t correction = -1;
      while (accepted < kK &&
             argmax(vrow(accepted), vocab) == drafts[accepted]) {
        ++accepted;
      }
      if (accepted < kK) {
        correction = argmax(vrow(accepted), vocab);
      }
      accepted_total += accepted;
      spec.push_back(token);
      for (std::size_t i = 0; i < accepted; ++i) {
        spec.push_back(drafts[i]);
      }
      if (accepted == kK) {
        if (!executor->MtpAdvance(*sp, drafts[kK - 1U],
                                  executor->verify_hidden() +
                                      (kK - 1U) * weights->config.hidden_size,
                                  &error)) {
          std::cerr << "parity spec advance failed: " << error << "\n";
          return 1;
        }
        logits.assign(vrow(kK), vrow(kK) + vocab);
      } else {
        executor->RollbackVerify(*sp,
                                 static_cast<std::uint32_t>(accepted + 1U));
        if (!executor->Step(*sp, correction, &error)) {
          std::cerr << "parity spec correction step failed: " << error << "\n";
          return 1;
        }
        spec.push_back(correction);
        logits = Download(executor->logits(), vocab);
      }
    }
    for (std::size_t i = 0; i < spec.size() && i < oracle.size(); ++i) {
      if (spec[i] != oracle[i]) {
        mismatch = static_cast<int>(i);
        break;
      }
    }
    if (spec.size() < oracle.size()) {
      std::cerr << "spec run produced " << spec.size() << " tokens, oracle "
                << oracle.size() << "\n";
    }
    std::cout << "Speculative greedy parity: " << spec.size() << "/"
              << oracle.size() << " tokens, accepted " << accepted_total << "/"
              << rounds * kK << "\n";
    if (mismatch >= 0) {
      std::cerr << "first mismatch at token " << mismatch << ": oracle "
                << oracle[mismatch] << ", spec " << spec[mismatch] << "\n";
    }
    Expect(mismatch < 0 && spec.size() >= oracle.size(),
           "speculative decoding reproduces the plain greedy text");
  }

  // A verify batch over the greedy token sequence must produce per-row
  // logits bit-identical to stepping the trunk sequentially, and a rollback
  // to row `keep` followed by the next token must land on the same hidden
  // state as the sequential run. Any differing float localizes the parity
  // break to either the batch kernels or the snapshot restore.
  {
    constexpr int kRefTokens = 24;
    const auto ref = executor->CreateSession(kContext, &error);
    if (ref == nullptr) {
      std::cerr << "bisect ref session failed: " << error << "\n";
      return 1;
    }
    if (!executor->Prefill(*ref, prompt.data(),
                           static_cast<std::uint32_t>(prompt.size()), &error)) {
      std::cerr << "bisect ref prefill failed: " << error << "\n";
      return 1;
    }
    std::vector<std::vector<float>> ref_logits;
    std::vector<std::int32_t> ref_tokens;
    for (int i = 0; i < kRefTokens; ++i) {
      ref_logits.push_back(Download(executor->logits(), vocab));
      const auto token = ArgMax(ref_logits.back());
      if (token == eos || token == pad) {
        break;
      }
      ref_tokens.push_back(token);
      if (!executor->Step(*ref, token, &error)) {
        std::cerr << "bisect ref step failed: " << error << "\n";
        return 1;
      }
    }
    const auto bt = executor->CreateSession(kContext, &error);
    if (bt == nullptr) {
      std::cerr << "bisect spec session failed: " << error << "\n";
      return 1;
    }
    if (!executor->Prefill(*bt, prompt.data(),
                           static_cast<std::uint32_t>(prompt.size()), &error)) {
      std::cerr << "bisect spec prefill failed: " << error << "\n";
      return 1;
    }
    const auto diff = [](const std::vector<float>& a,
                         const std::vector<float>& b) {
      std::size_t differs = 0;
      float max_abs = 0.0F;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
          ++differs;
          max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
        }
      }
      return std::pair{differs, max_abs};
    };
    std::size_t row_checks = 0;
    std::size_t row_bad = 0;
    std::size_t step_checks = 0;
    std::size_t step_bad = 0;
    const std::uint32_t keeps[4] = {1U, 2U, 1U, 2U};
    std::size_t p = 0;
    std::size_t ci = 0;
    std::vector<float> vrows(kRefTokens * vocab);
    while (p + 1U < ref_tokens.size()) {
      const auto rows = static_cast<std::uint32_t>(
          std::min<std::size_t>(4U, ref_tokens.size() - 1U - p));
      if (rows < 2U) {
        break;
      }
      const std::int32_t* drafts = ref_tokens.data() + p + 1U;
      if (!executor->Verify(*bt, ref_tokens[p], drafts, rows - 1U, &error)) {
        std::cerr << "bisect verify failed at " << p << ": " << error << "\n";
        return 1;
      }
      ht::CheckHip(
          hipMemcpy(vrows.data(), executor->verify_logits(),
                    static_cast<std::size_t>(rows) * vocab * sizeof(float),
                    hipMemcpyDeviceToHost),
          "upload bisect rows");
      for (std::uint32_t r = 0; r < rows; ++r) {
        const std::vector<float> got(
            vrows.data() + static_cast<std::size_t>(r) * vocab,
            vrows.data() + static_cast<std::size_t>(r + 1U) * vocab);
        const auto d = diff(got, ref_logits[p + r + 1U]);
        ++row_checks;
        if (d.first != 0) {
          ++row_bad;
          std::cerr << "verify row " << r << " at " << p << " differs in "
                    << d.first << " floats (max " << d.second << ")\n";
        }
      }
      const auto keep = std::min<std::uint32_t>(keeps[ci++ % 4U], rows - 1U);
      executor->RollbackVerify(*bt, keep);
      p += keep;
      while (p + 1U < ref_tokens.size()) {
        if (!executor->Step(*bt, ref_tokens[p], &error)) {
          std::cerr << "bisect step failed at " << p << ": " << error << "\n";
          return 1;
        }
        const auto d =
            diff(Download(executor->logits(), vocab), ref_logits[p + 1U]);
        ++step_checks;
        if (d.first != 0) {
          ++step_bad;
          std::cerr << "step after rollback differs at " << p << " in "
                    << d.first << " floats (max " << d.second << ")\n";
          break;
        }
        ++p;
        if (p + 1U >= ref_tokens.size()) {
          break;
        }
        if ((ci - 1U) % 4U == 3U) {
          break;
        }
      }
      if (p + 1U >= ref_tokens.size() || step_bad != 0) {
        break;
      }
    }
    std::cout << "Rollback bit-parity: rows " << row_checks - row_bad << "/"
              << row_checks << ", post-rollback steps "
              << step_checks - step_bad << "/" << step_checks
              << " bit-identical\n";
    Expect(row_bad == 0, "verify rows are bit-identical to sequential steps");
    Expect(step_bad == 0, "rollback restores the sequential trunk state");
  }

  if (failures != 0) {
    std::cerr << failures << " MTP acceptance checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 Flash MTP drafts track the trunk.\n";
  return 0;
}

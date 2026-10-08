// MTP draft forward parity: the ROCm draft step against the scalar oracle on
// the real Gemma-4-26B-A4B trunk plus its MTP sidecar. The draft is dense (no
// router, no experts), so unlike the trunk it has no near-tied expert flips:
// the only legitimate divergence between the float32 GPU path and the double
// oracle is the trunk KV cache it reads read-only, which the two trunk
// implementations populate to their own rounding. To isolate the draft kernel
// from that, both paths consume the *same* hidden input: the GPU's post-norm
// trunk stream, downloaded once and fed to the oracle too. The contract is
// then finiteness, an exactly matching argmax (the draft's actual prediction),
// a scale-relative logit bound far below the ~1.0 a wrong projection, wrong
// rope, wrong target layer or an applied softcap produces, and a matching
// next-step hidden. A second identical step must be bit-for-bit reproducible.
// Skips (77) unless GUFO_GEMMA4_GGUF and GUFO_GEMMA4_MTP_GGUF point at the
// trunk and draft GGUFs; needs the full GPU (stop the resident server first).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/reference.hpp"
#include "src/models/gemma4/weights.hpp"
#include "tests/models/gemma4/hip_test.hpp"

namespace g4 = gufo::models::gemma4;
namespace q = g4::rocm;
namespace t = gufo::tests::gemma4;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::uint32_t ArgMax(std::span<const float> v) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

// The draft reads the trunk KV cache, which the float32 GPU trunk and the
// double oracle populate to their own rounding: the unscaled attention and the
// top-8 MoE let a near-tied router decision flip an expert and shift the trunk
// K/V rows (rocm_forward_test holds the trunk to <= 2.0 abs on *softcapped*
// logits). The draft is dense and uncapped, so that same drift is not squashed
// by the +-30 softcap and shows up as a larger absolute error on logits that
// reach ~37. The robust gates are therefore finiteness and an exactly matching
// argmax (the draft's actual prediction, which catches a wrong projection,
// rope, target layer or residual order), plus a scale-relative bound that
// tolerates the trunk-KV drift floor (~0.16 observed) yet stays far below the
// ~1.0 a layout or indexing fault produces. A softcap is monotonic, so argmax
// cannot detect one being wrongly applied; instead the GPU draft is compared to
// both the uncapped oracle and a softcapped copy of it, and must be closer to
// the uncapped one. Top-K relative figures are printed for visibility.
void CheckDraftLogits(const std::vector<float>& oracle,
                      const std::vector<float>& gpu, float softcap,
                      const std::string& tag) {
  Expect(oracle.size() == gpu.size(), tag + " logit width");
  bool finite = true;
  for (const float v : gpu) {
    if (!std::isfinite(v)) {
      finite = false;
      break;
    }
  }
  Expect(finite, tag + " draft logits are finite");
  const double rel_scale = t::WorstRelativeToScale(oracle, gpu);
  // Distance to a softcapped oracle: if the GPU wrongly applied the trunk's
  // softcap, this would be smaller than the distance to the uncapped oracle.
  std::vector<float> capped(oracle.size());
  for (std::size_t i = 0; i < oracle.size(); ++i) {
    capped[i] = softcap * std::tanh(oracle[i] / softcap);
  }
  const double rel_capped = t::WorstRelativeToScale(capped, gpu);
  double worst_abs = 0.0;
  double max_oracle = 0.0;
  for (std::size_t i = 0; i < oracle.size(); ++i) {
    const double d = std::fabs(oracle[i] - gpu[i]);
    worst_abs = std::max(worst_abs, d);
    max_oracle =
        std::max(max_oracle, std::fabs(static_cast<double>(oracle[i])));
  }
  std::vector<std::uint32_t> order(oracle.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<std::uint32_t>(i);
  }
  const std::size_t k = std::min<std::size_t>(32, order.size());
  const auto greater = [&](std::uint32_t a, std::uint32_t b) {
    return oracle[a] > oracle[b];
  };
  std::partial_sort(order.begin(), order.begin() + k, order.end(), greater);
  const auto rel = [&](std::uint32_t i) {
    const double d = std::fabs(oracle[i] - gpu[i]);
    const double m = std::fabs(static_cast<double>(oracle[i]));
    return d / std::max(m, 1e-3);
  };
  const auto topk_rel = [&](std::size_t kk) {
    double m = 0.0;
    for (std::size_t j = 0; j < kk; ++j) {
      m = std::max(m, rel(order[j]));
    }
    return m;
  };
  std::cout << tag << ": worst abs " << worst_abs << ", rel-to-scale "
            << rel_scale << " (vs capped " << rel_capped << "), top1 "
            << topk_rel(1) << ", top5 " << topk_rel(5) << ", top8 "
            << topk_rel(8) << ", top32 " << topk_rel(32) << ", max|oracle| "
            << max_oracle << ", argmax " << ArgMax(gpu) << " vs "
            << ArgMax(oracle) << "\n";
  Expect(rel_scale <= 0.30, tag + " draft logits match the oracle");
  Expect(rel_scale < rel_capped, tag + " draft logits are uncapped");
  Expect(ArgMax(gpu) == ArgMax(oracle), tag + " draft argmax agreement");
}

}  // namespace

int main() {
  const char* trunk_path = std::getenv("GUFO_GEMMA4_GGUF");
  const char* draft_path = std::getenv("GUFO_GEMMA4_MTP_GGUF");
  if (trunk_path == nullptr || trunk_path[0] == '\0' || draft_path == nullptr ||
      draft_path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_GGUF and GUFO_GEMMA4_MTP_GGUF\n";
    return 77;
  }
  std::string error;
  const auto trunk_reader =
      gufo::core::GgufReader::OpenFile(trunk_path, &error);
  if (trunk_reader == nullptr) {
    std::cerr << "cannot open " << trunk_path << ": " << error << "\n";
    return 1;
  }
  const auto weights = g4::ModelWeights::Bind(*trunk_reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;

  const auto model = q::DeviceModel::Upload(*weights, *trunk_reader, &error);
  if (model == nullptr) {
    std::cerr << "trunk upload failed: " << error << "\n";
    return 1;
  }
  std::cout << "uploaded trunk " << model->resident_bytes() / (1024U * 1024U)
            << " MiB\n";

  const auto draft_reader =
      gufo::core::GgufReader::OpenFile(draft_path, &error);
  if (draft_reader == nullptr) {
    std::cerr << "cannot open " << draft_path << ": " << error << "\n";
    return 1;
  }
  const auto draft_weights = g4::DraftWeights::Bind(*draft_reader, c, &error);
  if (!draft_weights.has_value()) {
    std::cerr << "draft bind failed: " << error << "\n";
    return 1;
  }
  const auto draft =
      q::DeviceDraft::Upload(*draft_weights, *draft_reader, &error);
  if (draft == nullptr) {
    std::cerr << "draft upload failed: " << error << "\n";
    return 1;
  }
  std::cout << "uploaded draft " << draft->resident_bytes() / (1024U * 1024U)
            << " MiB\n";

  const std::uint32_t max_context = 64;
  const auto executor = q::Executor::Create(*model, max_context, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }
  if (!executor->AttachDraft(*draft, &error)) {
    std::cerr << "draft attach failed: " << error << "\n";
    return 1;
  }
  Expect(executor->has_draft(), "executor reports a draft");

  // Prefill the trunk so the draft has a KV cache to read, then take the GPU
  // post-norm hidden as the shared draft input.
  const std::vector<std::int32_t> prompt = {100, 200, 300, 400,
                                            500, 600, 700, 800};
  const auto session = executor->CreateSession(max_context, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }
  if (!executor->Prefill(*session, prompt.data(),
                         static_cast<std::uint32_t>(prompt.size()), &error)) {
    std::cerr << "prefill failed: " << error << "\n";
    return 1;
  }
  Expect(session->position() == prompt.size(), "prefill position");

  std::vector<float> gpu_hidden(c.hidden_size);
  t::CheckHip(
      hipMemcpy(gpu_hidden.data(), executor->h_out(),
                gpu_hidden.size() * sizeof(float), hipMemcpyDeviceToHost),
      "hidden download");

  // Drive the oracle trunk over the same prompt so its KV cache (which the
  // oracle draft reads) matches the GPU's to trunk rounding.
  std::vector<float> ref_logits(c.vocab_size);
  std::vector<float> ref_hidden(c.hidden_size);
  g4::ReferenceModel oracle(*weights, max_context);
  for (const std::int32_t token : prompt) {
    if (!oracle.Step(token, ref_logits, ref_hidden, &error)) {
      std::cerr << "oracle prefill step failed: " << error << "\n";
      return 1;
    }
  }
  Expect(oracle.Position() == session->position(), "oracle position matches");

  // Two chained draft steps. Each consumes the same hidden input on both
  // paths: the GPU reads its own h_out device buffer, the oracle reads the
  // downloaded gpu_hidden. The second step chains the draft's own h_next so a
  // wrong residual or projection in the loop shows up.
  std::vector<float> gpu_draft_logits(c.vocab_size);
  std::vector<float> gpu_h_next(c.hidden_size);
  std::vector<float> ref_draft_logits(c.vocab_size);
  std::vector<float> ref_h_next(c.hidden_size);

  const std::int32_t draft_token = 400;
  if (!executor->DraftStep(*session, draft_token, executor->h_out(), &error)) {
    std::cerr << "gpu draft step failed: " << error << "\n";
    return 1;
  }
  t::CheckHip(
      hipMemcpy(gpu_draft_logits.data(), executor->draft_logits(),
                gpu_draft_logits.size() * sizeof(float), hipMemcpyDeviceToHost),
      "draft logit download");
  t::CheckHip(
      hipMemcpy(gpu_h_next.data(), executor->draft_h_next(),
                gpu_h_next.size() * sizeof(float), hipMemcpyDeviceToHost),
      "draft hidden download");
  if (!oracle.DraftStep(*draft_weights, draft_token, gpu_hidden,
                        ref_draft_logits, ref_h_next, &error)) {
    std::cerr << "oracle draft step failed: " << error << "\n";
    return 1;
  }
  CheckDraftLogits(ref_draft_logits, gpu_draft_logits, c.logit_softcap,
                   "draft step 1");
  const double h_next_rel = t::WorstRelativeToScale(ref_h_next, gpu_h_next);
  std::cout << "draft h_next 1: rel-to-scale " << h_next_rel << "\n";
  Expect(h_next_rel <= 0.30, "draft h_next matches the oracle");

  // Determinism: a second identical step must reproduce the first bit-for-bit.
  std::vector<float> gpu_draft_logits2(c.vocab_size);
  if (!executor->DraftStep(*session, draft_token, executor->h_out(), &error)) {
    std::cerr << "gpu draft step 2 failed: " << error << "\n";
    return 1;
  }
  t::CheckHip(hipMemcpy(gpu_draft_logits2.data(), executor->draft_logits(),
                        gpu_draft_logits2.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "draft logit download 2");
  Expect(gpu_draft_logits == gpu_draft_logits2, "draft step is deterministic");

  // Chained step: feed the draft's own h_next back in on both paths. The
  // session position is unchanged (the draft never writes KV), so the oracle
  // and GPU still read the same trunk cache.
  std::vector<float> gpu_hidden_chain = gpu_h_next;
  if (!executor->DraftStep(*session, draft_token + 1, executor->draft_h_next(),
                           &error)) {
    std::cerr << "gpu chained draft step failed: " << error << "\n";
    return 1;
  }
  t::CheckHip(
      hipMemcpy(gpu_draft_logits.data(), executor->draft_logits(),
                gpu_draft_logits.size() * sizeof(float), hipMemcpyDeviceToHost),
      "chained draft logit download");
  t::CheckHip(
      hipMemcpy(gpu_h_next.data(), executor->draft_h_next(),
                gpu_h_next.size() * sizeof(float), hipMemcpyDeviceToHost),
      "chained draft hidden download");
  if (!oracle.DraftStep(*draft_weights, draft_token + 1, gpu_hidden_chain,
                        ref_draft_logits, ref_h_next, &error)) {
    std::cerr << "oracle chained draft step failed: " << error << "\n";
    return 1;
  }
  CheckDraftLogits(ref_draft_logits, gpu_draft_logits, c.logit_softcap,
                   "draft step 2");
  const double h_next_rel2 = t::WorstRelativeToScale(ref_h_next, gpu_h_next);
  std::cout << "draft h_next 2: rel-to-scale " << h_next_rel2 << "\n";
  Expect(h_next_rel2 <= 0.30, "chained draft h_next matches the oracle");

  if (failures != 0) {
    std::cerr << failures << " MTP draft parity checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4-26B-A4B MTP draft parity passed.\n";
  return 0;
}
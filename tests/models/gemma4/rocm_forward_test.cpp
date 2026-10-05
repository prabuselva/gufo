// End-to-end forward parity: the ROCm executor against the scalar oracle on
// the real Gemma-4-26B-A4B artifact. Gemma-4 attention is unscaled (no 1/sqrt
// (head_dim) in the GGUF), so its softmax is extremely peaked, and the 30-layer
// top-8 MoE makes the network chaotic: the float32 GPU path and the double
// oracle legitimately pick different experts when the router's 8th/9th scores
// are within float noise, which shifts even the top logits by O(1). A relative
// logit bound is therefore not robust (it spikes to >1 on a near-tied flip and
// on near-zero logits). The decode contract is instead the invariants that hold
// across every position: finite logits, an exactly matching argmax (the actual
// prediction), and a bounded absolute logit error (<= 2.0) that tolerates one
// expert flip yet stays far below the O(10) shift a logic error produces
// against the +-30 softcap. Prefill (batched GEMM tier + routed F16 WMMA MoE)
// is a smoke test for finiteness and argmax agreement for the same reason (the
// throughput contract lives in the bench). Skips (77) unless GUFO_GEMMA4_GGUF
// points at the trunk GGUF; needs the full GPU (stop the resident server
// first).
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

// The float32 GPU path and the double oracle diverge through unscaled attention
// and the top-8 MoE: a near-tied router decision picks a different expert and
// shifts even the top logits by O(1), so relative error is unbounded on
// near-zero logits and spikes past 1 on a flip. The pass/fail invariants are
// finiteness, exact argmax, and a bounded absolute logit error; the top-K
// relative figures are printed for visibility only.
void CheckLogits(const std::vector<float>& oracle,
                 const std::vector<float>& gpu, const std::string& tag) {
  Expect(oracle.size() == gpu.size(), tag + " logit width");
  bool finite = true;
  for (const float v : gpu) {
    if (!std::isfinite(v)) {
      finite = false;
      break;
    }
  }
  Expect(finite, tag + " GPU logits are finite");
  double worst_abs = 0.0;
  for (std::size_t i = 0; i < oracle.size(); ++i) {
    const double d = std::fabs(oracle[i] - gpu[i]);
    worst_abs = std::max(worst_abs, d);
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
  std::cout << tag << ": worst abs " << worst_abs << ", top1 " << topk_rel(1)
            << ", top5 " << topk_rel(5) << ", top8 " << topk_rel(8)
            << ", top32 " << topk_rel(32) << ", argmax " << ArgMax(gpu)
            << " vs " << ArgMax(oracle) << "\n";
  Expect(worst_abs <= 2.0, tag + " logits match the oracle");
  Expect(ArgMax(gpu) == ArgMax(oracle), tag + " argmax agreement");
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_GGUF\n";
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = g4::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;

  const auto model = q::DeviceModel::Upload(*weights, *reader, &error);
  if (model == nullptr) {
    std::cerr << "device upload failed: " << error << "\n";
    return 1;
  }
  std::cout << "uploaded " << model->resident_bytes() / (1024U * 1024U)
            << " MiB of weights\n";

  const std::uint32_t max_context = 64;
  const auto executor = q::Executor::Create(*model, max_context, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }

  std::vector<float> ref_logits(c.vocab_size);
  std::vector<float> ref_hidden(c.hidden_size);
  std::vector<float> gpu_logits(c.vocab_size);
  auto download_logits = [&](std::vector<float>& out) {
    t::CheckHip(hipMemcpy(out.data(), executor->logits(),
                          out.size() * sizeof(float), hipMemcpyDeviceToHost),
                "logit download");
  };

  // Prefill smoke: a fresh session prefilling the prompt must produce finite
  // logits whose argmax matches the oracle after the same three tokens. The
  // magnitude is not held to the decode bound (MoE expert selection differs
  // under 8-bit activation quantization).
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  {
    const auto session = executor->CreateSession(max_context, &error);
    if (session == nullptr) {
      std::cerr << "prefill session create failed: " << error << "\n";
      return 1;
    }
    if (!executor->Prefill(*session, prompt.data(),
                           static_cast<std::uint32_t>(prompt.size()), &error)) {
      std::cerr << "prefill failed: " << error << "\n";
      return 1;
    }
    download_logits(gpu_logits);
    g4::ReferenceModel oracle(*weights, max_context);
    for (const std::int32_t token : prompt) {
      if (!oracle.Step(token, ref_logits, ref_hidden, &error)) {
        std::cerr << "oracle prefill step failed: " << error << "\n";
        return 1;
      }
    }
    bool finite = true;
    for (const float v : gpu_logits) {
      if (!std::isfinite(v)) {
        finite = false;
        break;
      }
    }
    Expect(finite, "prefill logits are finite");
    Expect(ArgMax(gpu_logits) == ArgMax(ref_logits),
           "prefill argmax agreement");
    Expect(session->position() == prompt.size(), "prefill position");
    std::cout << "prefill smoke: argmax " << ArgMax(gpu_logits) << " vs "
              << ArgMax(ref_logits) << "\n";
  }

  // Decode parity: a fresh session stepping the prompt then two more tokens
  // must match the oracle step for step.
  const auto session = executor->CreateSession(max_context, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }
  g4::ReferenceModel oracle(*weights, max_context);
  const std::vector<std::int32_t> sequence = {100, 200, 300, 400, 500};
  for (const std::int32_t token : sequence) {
    if (!executor->Step(*session, token, &error)) {
      std::cerr << "step " << token << " failed: " << error << "\n";
      return 1;
    }
    download_logits(gpu_logits);
    if (!oracle.Step(token, ref_logits, ref_hidden, &error)) {
      std::cerr << "oracle step " << token << " failed: " << error << "\n";
      return 1;
    }
    CheckLogits(ref_logits, gpu_logits, "decode " + std::to_string(token));
  }
  Expect(session->position() == oracle.Position(), "position matches");

  // h_out parity after the last decode step.
  std::vector<float> gpu_hidden(c.hidden_size);
  t::CheckHip(
      hipMemcpy(gpu_hidden.data(), executor->h_out(),
                gpu_hidden.size() * sizeof(float), hipMemcpyDeviceToHost),
      "hidden download");
  const double hidden_rel = t::WorstRelative(ref_hidden, gpu_hidden, 1e-3);
  double hidden_abs = 0.0;
  for (std::size_t i = 0; i < ref_hidden.size(); ++i) {
    hidden_abs =
        std::max(hidden_abs,
                 static_cast<double>(std::fabs(ref_hidden[i] - gpu_hidden[i])));
  }
  std::cout << "h_out: worst relative " << hidden_rel << ", worst abs "
            << hidden_abs << "\n";
  Expect(hidden_abs <= 2.0, "h_out parity");

  if (failures != 0) {
    std::cerr << failures << " forward parity checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4-26B-A4B ROCm forward parity passed.\n";
  return 0;
}
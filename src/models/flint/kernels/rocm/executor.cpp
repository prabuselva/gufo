#include "src/models/flint/kernels/rocm/executor.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/flint/kernels/rocm/kernels.hpp"
#include "src/models/flint/kernels/rocm/linear.hpp"

namespace gufo::models::flint {
namespace {

[[nodiscard]] bool HipOk(hipError_t err, std::string* error, const char* what) {
  if (err == hipSuccess) {
    return true;
  }
  if (error != nullptr) {
    *error = std::string(what) + ": " + hipGetErrorName(err);
  }
  return false;
}

constexpr std::uint32_t kUsedMax = 6;

}  // namespace

struct FlintExecutor::State {
  // Device weight buffers, one per bound tensor.
  std::vector<void*> weight_buffers;
  std::vector<TensorRef> weights;  // device-side rewrites of the table
  ModelWeights dev;
  ModelWeights host;

  // Activations ([batch_capacity] rows unless annotated).
  float* x{nullptr};
  float* normed{nullptr};
  float* q{nullptr};
  float* k{nullptr};
  float* v{nullptr};
  float* attn{nullptr};
  float* o{nullptr};
  float* ffn_a{nullptr};
  float* ffn_b{nullptr};
  float* probs{nullptr};
  std::int32_t* ids{nullptr};
  float* wts{nullptr};
  float* eg{nullptr};
  float* eu{nullptr};
  float* ed{nullptr};
  float* sg{nullptr};
  float* su{nullptr};
  float* sd{nullptr};
  float* logits{nullptr};
  std::size_t* dev_tokens{nullptr};
  bool layer_capture{false};
  std::vector<std::vector<float>> hidden;
  int moe_trace_layer{-1};
  FlintExecutor::MoETrace moe_trace;

  __half* k_plane{nullptr};
  __half* v_plane{nullptr};
  std::size_t layer_stride{0};
  std::uint32_t kv_capacity{0};
};

namespace {

[[nodiscard]] bool Alloc(void** p, std::size_t bytes, std::string* error) {
  if (bytes == 0) {
    *p = nullptr;
    return true;
  }
  const hipError_t err = hipMalloc(p, bytes);
  return HipOk(err, error, "hipMalloc");
}

}  // namespace

FlintExecutor::~FlintExecutor() {
  if (state_ == nullptr) {
    return;
  }
  for (void* p : state_->weight_buffers) {
    hipFree(p);
  }
  hipFree(state_->x);
  hipFree(state_->normed);
  hipFree(state_->q);
  hipFree(state_->k);
  hipFree(state_->v);
  hipFree(state_->attn);
  hipFree(state_->o);
  hipFree(state_->ffn_a);
  hipFree(state_->ffn_b);
  hipFree(state_->probs);
  hipFree(state_->ids);
  hipFree(state_->wts);
  hipFree(state_->eg);
  hipFree(state_->eu);
  hipFree(state_->ed);
  hipFree(state_->sg);
  hipFree(state_->su);
  hipFree(state_->sd);
  hipFree(state_->logits);
  hipFree(state_->k_plane);
  hipFree(state_->v_plane);
  delete state_;
}

namespace {

/// Allocates one device buffer per bound tensor and streams the file bytes
/// into it through the shared WeightUpload pipeline. The returned ModelWeights
/// mirrors the table with `data` replaced by device pointers.
[[nodiscard]] bool UploadWeights(const core::GgufReader& reader,
                                 const ModelWeights& host, ModelWeights* dev,
                                 std::vector<void*>* buffers,
                                 std::string* error) {
  auto pipeline = hip::WeightUpload::Create(reader.GetMappedRegions(), error);
  if (pipeline == nullptr) {
    return false;
  }
  auto copy = [&](const TensorRef& src, TensorRef* dst) -> bool {
    void* p = nullptr;
    if (!Alloc(&p, src.SizeBytes(), error)) {
      return false;
    }
    buffers->push_back(p);
    if (!pipeline->Copy(src.shard, src.file_offset, src.SizeBytes(), p,
                        error)) {
      return false;
    }
    *dst = src;
    dst->data = p;
    return true;
  };
  if (!copy(host.token_embd, &dev->token_embd) ||
      !copy(host.output, &dev->output) ||
      !copy(host.output_norm, &dev->output_norm)) {
    return false;
  }
  dev->layers.resize(host.layers.size());
  for (std::size_t i = 0; i < host.layers.size(); ++i) {
    const LayerWeights& l = host.layers[i];
    LayerWeights d;
    d.moe = l.moe;
    const TensorRef* pairs[] = {
        &l.attn_norm, &l.ffn_norm,    &l.attn_q,      &l.attn_k,     &l.attn_v,
        &l.attn_out,  &l.attn_q_norm, &l.attn_k_norm, &l.attn_sinks,
    };
    TensorRef* out_pairs[] = {
        &d.attn_norm, &d.ffn_norm,    &d.attn_q,      &d.attn_k,     &d.attn_v,
        &d.attn_out,  &d.attn_q_norm, &d.attn_k_norm, &d.attn_sinks,
    };
    for (std::size_t j = 0; j < sizeof(pairs) / sizeof(pairs[0]); ++j) {
      if (!copy(*pairs[j], out_pairs[j])) {
        return false;
      }
    }
    if (l.moe) {
      if (!copy(l.router, &d.router) ||
          !copy(l.ffn_gate_exps, &d.ffn_gate_exps) ||
          !copy(l.ffn_up_exps, &d.ffn_up_exps) ||
          !copy(l.ffn_down_exps, &d.ffn_down_exps) ||
          !copy(l.shexp_gate, &d.shexp_gate) ||
          !copy(l.shexp_up, &d.shexp_up) ||
          !copy(l.shexp_down, &d.shexp_down)) {
        return false;
      }
    } else if (!copy(l.ffn_gate, &d.ffn_gate) || !copy(l.ffn_up, &d.ffn_up) ||
               !copy(l.ffn_down, &d.ffn_down)) {
      return false;
    }
    dev->layers[i] = d;
  }
  return pipeline->Finish(error);
}

/// Copies row 0 of a device activation buffer into the capture log. The
/// first entry holds the post-embedding row, then one row per layer.
void CaptureRow(const float* p, std::size_t hidden, std::uint32_t layer,
                bool is_layer, std::vector<std::vector<float>>* out) {
  std::vector<float> h(hidden);
  if (hipMemcpy(h.data(), p, h.size() * 4, hipMemcpyDeviceToHost) !=
      hipSuccess) {
    std::printf("DBG capture failed at %s %u\n",
                is_layer ? "layer" : "embedding", layer);
    return;
  }
  out->push_back(std::move(h));
}

/// Env-gated finite-range probe on a device float buffer; the C4 debug path
/// for locating the first poisoned activation.
void DebugStats(const char* tag, const float* p, std::size_t n) {
  static const bool on = std::getenv("GUFO_FLINT_DEBUG") != nullptr;
  if (!on) {
    return;
  }
  if (n == 0) {
    return;
  }
  std::vector<float> h(n);
  if (hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost) != hipSuccess) {
    std::printf("DBG %s: readback failed\n", tag);
    return;
  }
  std::size_t nan = 0;
  float lo = h[0];
  float hi = h[0];
  for (const float v : h) {
    if (!std::isfinite(v)) {
      ++nan;
      continue;
    }
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  std::printf("DBG %s n=%zu nan=%zu lo=%g hi=%g v0=%g\n", tag, n, nan, lo, hi,
              h[0]);
}

}  // namespace

std::unique_ptr<FlintExecutor> FlintExecutor::Create(
    const core::GgufReader& reader, const ModelWeights& weights,
    std::uint32_t batch_capacity, std::uint32_t kv_capacity,
    std::string* error) {
  static std::once_flag mmq_once;
  std::call_once(mmq_once, [] { (void)qfn_mmq_init(0); });

  std::unique_ptr<FlintExecutor> ex(new FlintExecutor());
  State* st = new State();
  ex->state_ = st;
  st->host = weights;

  if (!UploadWeights(reader, weights, &st->dev, &st->weight_buffers, error)) {
    return nullptr;  // ~FlintExecutor frees state_
  }

  const Config& c = weights.config;
  const std::size_t B = batch_capacity;
  const std::size_t H = c.hidden_size;
  const std::size_t QD = c.AttentionQDim();
  const std::size_t KVD = c.AttentionKvDim();
  const std::size_t E = c.num_experts_used;
  bool ok = true;
  ok = ok && Alloc(reinterpret_cast<void**>(&st->x), B * H * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->normed), B * H * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->q), B * QD * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->k), B * KVD * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->v), B * KVD * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->attn), B * QD * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->o), B * H * 4, error);
  ok = ok &&
       Alloc(reinterpret_cast<void**>(&st->ffn_a), B * c.ffn_length * 4, error);
  ok = ok &&
       Alloc(reinterpret_cast<void**>(&st->ffn_b), B * c.ffn_length * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->probs), B * c.num_experts * 4,
                   error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->ids), B * E * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->wts), B * E * 4, error);
  ok = ok &&
       Alloc(reinterpret_cast<void**>(&st->eg), B * E * c.expert_ff * 4, error);
  ok = ok &&
       Alloc(reinterpret_cast<void**>(&st->eu), B * E * c.expert_ff * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->ed), B * E * H * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->sg),
                   B * c.shared_expert_ff * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->su),
                   B * c.shared_expert_ff * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->sd), B * H * 4, error);
  ok = ok &&
       Alloc(reinterpret_cast<void**>(&st->logits), c.vocab_size * 4, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->k_plane),
                   c.num_layers * kv_capacity * KVD * 2, error);
  ok = ok && Alloc(reinterpret_cast<void**>(&st->v_plane),
                   c.num_layers * kv_capacity * KVD * 2, error);
  st->layer_stride = static_cast<std::size_t>(kv_capacity) * KVD;
  st->kv_capacity = kv_capacity;
  if (!ok) {
    if (error != nullptr && error->empty()) {
      *error = "flint: activation allocation failed";
    }
    return nullptr;  // ~FlintExecutor frees state_
  }
  ex->position_ = 0;
  return ex;
}

namespace {

/// out = base @ x for rows (batch <= capacity), dispatching decode through
/// the plain GEMV and wider batches through the GEMM tier.
void Project(const TensorRef& w, const float* x, float* out,
             std::uint32_t batch, hipStream_t stream) {
  if (batch == 1) {
    rocm::Gemv(w.data, w.type, static_cast<std::uint32_t>(w.rows),
               static_cast<std::uint32_t>(w.cols), w.RowBytes(), x, out,
               stream);
    return;
  }
  // The fp32-accumulating SIMT tier keeps batched prefill inside the
  // oracle's numeric envelope; the tensor-core mmq dense path rounds its
  // activations to f16 and is reserved for the speed phase.
  rocm::GemvBatch(w.data, w.type, static_cast<std::uint32_t>(w.rows),
                  static_cast<std::uint32_t>(w.cols), w.RowBytes(), x, out,
                  batch, stream);
}

}  // namespace

bool FlintExecutor::Forward(const std::int32_t* tokens, std::uint32_t count,
                            std::vector<float>* logits, std::string* error) {
  State* st = state_;
  if (st == nullptr || count == 0) {
    if (error != nullptr) {
      *error = "flint: forward without state or tokens";
    }
    return false;
  }
  const Config& c = st->host.config;
  if (static_cast<std::uint64_t>(position_) + count > st->kv_capacity) {
    if (error != nullptr) {
      *error = "flint: forward beyond the cache capacity";
    }
    return false;
  }
  if (st->layer_capture && count != 1) {
    if (error != nullptr) {
      *error = "flint: layer capture requires a single token";
    }
    return false;
  }
  const std::uint32_t pos0 = position_;
  const auto stream = static_cast<hipStream_t>(nullptr);
  const std::size_t H = c.hidden_size;
  const std::size_t QD = c.AttentionQDim();
  const std::size_t KVD = c.AttentionKvDim();
  const std::uint32_t B = count;

  if (std::getenv("GUFO_FLINT_DEBUG") != nullptr) {
    std::printf("DBG tokens[0]=%d pos0=%u B=%u\n", tokens[0], pos0, B);
  }
  for (std::uint32_t t = 0; t < B; ++t) {
    const std::int32_t tok = tokens[t];
    if (tok < 0 || static_cast<std::uint64_t>(tok) >= c.vocab_size) {
      if (error != nullptr) {
        *error = "flint: token id outside the vocabulary";
      }
      return false;
    }
    rocm::DequantRow(st->dev.token_embd.data, st->dev.token_embd.type,
                     static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(H),
                     st->dev.token_embd.RowBytes(), st->x + t * H, stream);
  }

  DebugStats("embd", st->x, B * H);
  if (st->layer_capture) {
    st->hidden.clear();
    CaptureRow(st->x, H, 0, false, &st->hidden);
  }
  for (std::uint32_t L = 0; L < c.num_layers; ++L) {
    const LayerWeights& l = st->dev.layers[L];
    const bool dbg = L == 0 && std::getenv("GUFO_FLINT_DEBUG") != nullptr;
    rocm::RmsNorm(st->x, static_cast<const float*>(l.attn_norm.data), c.rms_eps,
                  st->normed, B, static_cast<std::uint32_t>(H), stream);
    Project(l.attn_q, st->normed, st->q, B, stream);
    Project(l.attn_k, st->normed, st->k, B, stream);
    Project(l.attn_v, st->normed, st->v, B, stream);
    rocm::QkEpilogue(
        st->q, st->k, st->v, static_cast<const float*>(l.attn_q_norm.data),
        static_cast<const float*>(l.attn_k_norm.data), c.rms_eps, c.rope_theta,
        c.HasRope(L), pos0, B, c.num_heads, c.num_kv_heads, c.head_dim,
        st->k_plane + static_cast<std::size_t>(L) * st->layer_stride,
        st->v_plane + static_cast<std::size_t>(L) * st->layer_stride, KVD,
        stream);
    if (dbg) {
      DebugStats("L0 normed", st->normed, B * H);
      DebugStats("L0 q", st->q, B * QD);
    }
    rocm::Attention(
        st->q, st->k_plane + static_cast<std::size_t>(L) * st->layer_stride,
        st->v_plane + static_cast<std::size_t>(L) * st->layer_stride,
        static_cast<const float*>(l.attn_sinks.data),
        1.0F / std::sqrt(static_cast<float>(c.head_dim)), pos0, B,
        c.IsWindowedLayer(L) ? c.sliding_window : 0U, c.num_heads,
        c.num_kv_heads, c.head_dim, KVD, st->attn, stream);
    if (dbg) {
      DebugStats("L0 attn", st->attn, B * QD);
    }
    Project(l.attn_out, st->attn, st->o, B, stream);
    rocm::AddResidual(st->x, st->o, B, static_cast<std::uint32_t>(H), stream);
    if (dbg) {
      DebugStats("L0 o", st->o, B * H);
      DebugStats("L0 x", st->x, B * H);
    }

    rocm::RmsNorm(st->x, static_cast<const float*>(l.ffn_norm.data), c.rms_eps,
                  st->normed, B, static_cast<std::uint32_t>(H), stream);
    if (!l.moe) {
      Project(l.ffn_gate, st->normed, st->ffn_a, B, stream);
      Project(l.ffn_up, st->normed, st->ffn_b, B, stream);
      rocm::SiluMul(st->ffn_a, st->ffn_b, st->ffn_a,
                    static_cast<std::size_t>(B) * c.ffn_length, stream);
      Project(l.ffn_down, st->ffn_a, st->o, B, stream);
      rocm::AddResidual(st->x, st->o, B, static_cast<std::uint32_t>(H), stream);
      if (st->layer_capture) {
        CaptureRow(st->x, H, L, true, &st->hidden);
      }
      continue;
    }
    Project(l.router, st->normed, st->probs, B, stream);
    if (dbg) {
      DebugStats("L0 router", st->probs, B * c.num_experts);
    }
    rocm::RouterTopK(st->probs, st->ids, st->wts, B,
                     static_cast<std::uint32_t>(c.num_experts_used),
                     c.expert_weights_norm, c.expert_weights_scale, stream);
    const bool trace =
        L == static_cast<std::size_t>(st->moe_trace_layer) && B == 1;
    if (trace) {
      st->moe_trace = {};
      auto& tr = st->moe_trace;
      tr.normed.resize(H);
      tr.probs.resize(c.num_experts);
      tr.ids.resize(c.num_experts_used);
      tr.wts.resize(c.num_experts_used);
      const auto pull = [&](void* dst, const void* src, std::size_t bytes) {
        hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
      };
      pull(tr.normed.data(), st->normed, H * 4);
      pull(tr.probs.data(), st->probs, c.num_experts * 4);
      pull(tr.ids.data(), st->ids, c.num_experts_used * 4);
      pull(tr.wts.data(), st->wts, c.num_experts_used * 4);
    }
    rocm::GemmMoe(l.ffn_gate_exps.data, static_cast<std::uint32_t>(c.expert_ff),
                  static_cast<std::uint32_t>(H), st->normed, st->ids, st->eg, B,
                  c.num_experts, c.num_experts_used, stream);
    rocm::GemmMoe(l.ffn_up_exps.data, static_cast<std::uint32_t>(c.expert_ff),
                  static_cast<std::uint32_t>(H), st->normed, st->ids, st->eu, B,
                  c.num_experts, c.num_experts_used, stream);
    rocm::SiluMul(
        st->eg, st->eu, st->eg,
        static_cast<std::size_t>(B) * c.num_experts_used * c.expert_ff, stream);
    // down consumes the slot-indexed SiLU output, so each routed pair acts
    // as its own token with a single expert slot (ids stay pair-indexed).
    rocm::GemmMoe(l.ffn_down_exps.data, static_cast<std::uint32_t>(H),
                  static_cast<std::uint32_t>(c.expert_ff), st->eg, st->ids,
                  st->ed, B * c.num_experts_used, c.num_experts, 1U, stream);
    Project(l.shexp_gate, st->normed, st->sg, B, stream);
    Project(l.shexp_up, st->normed, st->su, B, stream);
    rocm::SiluMul(st->sg, st->su, st->sg,
                  static_cast<std::size_t>(B) * c.shared_expert_ff, stream);
    Project(l.shexp_down, st->sg, st->sd, B, stream);
    if (dbg) {
      DebugStats("L0 ed", st->ed, B * c.num_experts_used * H);
      DebugStats("L0 sd", st->sd, B * H);
    }
    if (trace) {
      auto& tr = st->moe_trace;
      tr.routed.resize(static_cast<std::size_t>(c.num_experts_used) * H);
      tr.shared.resize(H);
      hipMemcpy(tr.routed.data(), st->ed, tr.routed.size() * 4,
                hipMemcpyDeviceToHost);
      hipMemcpy(tr.shared.data(), st->sd, H * 4, hipMemcpyDeviceToHost);
    }
    rocm::SumExperts(st->x, st->ed, st->wts, st->sd, B,
                     static_cast<std::uint32_t>(H),
                     static_cast<std::uint32_t>(c.num_experts_used), stream);
    if (dbg) {
      DebugStats("L0 x2", st->x, B * H);
    }
    if (st->layer_capture) {
      CaptureRow(st->x, H, L, true, &st->hidden);
    }
  }

  DebugStats("x final", st->x, B * H);
  rocm::RmsNorm(st->x + static_cast<std::size_t>(B - 1) * H,
                static_cast<const float*>(st->dev.output_norm.data), c.rms_eps,
                st->normed, 1, static_cast<std::uint32_t>(H), stream);
  rocm::Gemv(st->dev.output.data, st->dev.output.type,
             static_cast<std::uint32_t>(c.vocab_size),
             static_cast<std::uint32_t>(H), st->dev.output.RowBytes(),
             st->normed, st->logits, stream);

  logits->resize(c.vocab_size);
  if (!HipOk(hipMemcpy(logits->data(), st->logits, c.vocab_size * 4,
                       hipMemcpyDeviceToHost),
             error, "flint: logits readback") ||
      !HipOk(hipGetLastError(), error, "flint: kernel launch")) {
    return false;
  }
  position_ += B;
  return true;
}

void FlintExecutor::EnableMoeTrace(int layer) noexcept {
  if (state_ != nullptr) {
    state_->moe_trace_layer = layer;
  }
}

const FlintExecutor::MoETrace& FlintExecutor::LastMoeTrace() const noexcept {
  return state_->moe_trace;
}

void FlintExecutor::EnableLayerCapture(bool enable) noexcept {
  if (state_ != nullptr) {
    state_->layer_capture = enable;
    state_->hidden.clear();
  }
}

const std::vector<std::vector<float>>& FlintExecutor::CapturedHidden() const {
  static const std::vector<std::vector<float>> kEmpty;
  return state_ != nullptr ? state_->hidden : kEmpty;
}

}  // namespace gufo::models::flint

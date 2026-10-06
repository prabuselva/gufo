#include "src/models/gemma4/vision/encoder.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/kernels/rocm/fused.hpp"
#include "src/models/gemma4/kernels/rocm/gemm.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/vision/kernels.hpp"
#include "src/models/gemma4/vision/weights.hpp"

namespace g4r = gufo::models::gemma4::rocm;

namespace gufo::models::gemma4::vision {
namespace {

struct DTensor {
  void* data{nullptr};
  g4r::GemvType type{};
  std::uint32_t rows{0};
  std::uint32_t cols{0};
  std::size_t row_bytes{0};
};

struct DBlock {
  float* ln1{nullptr};
  float* q_norm{nullptr};
  float* k_norm{nullptr};
  float* attn_post_norm{nullptr};
  float* ln2{nullptr};
  float* ffn_post_norm{nullptr};
  DTensor attn_q;
  DTensor attn_k;
  DTensor attn_v;
  DTensor attn_out;
  DTensor ffn_gate;
  DTensor ffn_up;
  DTensor ffn_down;
};

}  // namespace

struct Encoder::Impl {
  Config config;
  DTensor patch_embed;
  DTensor projection;
  float* pos_x{nullptr};
  float* pos_y{nullptr};
  float* std_bias{nullptr};
  float* std_scale{nullptr};
  float* inv_freq{nullptr};
  std::vector<DBlock> blocks;
  std::vector<void*> allocs;
  std::size_t bytes{0};

  ~Impl() {
    for (void* p : allocs) {
      (void)hipFree(p);
    }
  }
};

namespace {

struct Uploader {
  std::vector<void*>& allocs;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& msg) {
    if (ok && error != nullptr) {
      *error = msg;
    }
    ok = false;
  }

  void* CopyRaw(const void* src, std::size_t size, const char* what) {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) != hipSuccess) {
      Fail(std::string("hipMalloc failed for ") + what);
      return nullptr;
    }
    allocs.push_back(ptr);
    bytes += size;
    if (hipMemcpy(ptr, src, size, hipMemcpyHostToDevice) != hipSuccess) {
      Fail(std::string("upload failed for ") + what);
      return nullptr;
    }
    return ptr;
  }

  float* Raw(const TensorRef& t) {
    if (t.empty() || !ok) {
      return nullptr;
    }
    // Vision F32 tensors are single blocks; the stacked position table carries
    // experts=2 but each lookup table is one rows*cols block, so copy that
    // rather than SizeBytes (which would double-count and over-read pos_y).
    const std::size_t size =
        static_cast<std::size_t>(t.rows) * t.cols * sizeof(float);
    return static_cast<float*>(CopyRaw(t.data, size, t.name.data()));
  }

  // Upload a projection as F32. BF16 weights are upcast on the host so the
  // vision tower computes in full precision, matching the CPU oracle; the
  // tower is tiny next to the text model, so the extra bytes cost nothing.
  DTensor Proj(const TensorRef& t) {
    DTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t elems = static_cast<std::size_t>(t.rows) * t.cols;
    if (t.type == core::GgmlType::kBF16) {
      std::vector<float> host(elems);
      const auto* bits = static_cast<const std::uint16_t*>(t.data);
      for (std::size_t i = 0; i < elems; ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(bits[i]) << 16;
        std::memcpy(&host[i], &u, sizeof(float));
      }
      void* ptr = CopyRaw(host.data(), elems * sizeof(float), t.name.data());
      if (ptr == nullptr) {
        return d;
      }
      d.data = ptr;
    } else if (t.type == core::GgmlType::kF32) {
      d.data = CopyRaw(t.data, elems * sizeof(float), t.name.data());
      if (d.data == nullptr) {
        return d;
      }
    } else {
      Fail("unsupported projection type for " + std::string(t.name));
      return d;
    }
    d.type = g4r::GemvType::kF32;
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.row_bytes = static_cast<std::size_t>(t.cols) * sizeof(float);
    return d;
  }

  float* Norm(const TensorRef& t) {
    return Raw(t);
  }
};

}  // namespace

Encoder::Encoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::Open(const std::string& mmproj_path,
                                       std::string* error_msg) {
  const auto reader = core::GgufReader::OpenFile(mmproj_path, error_msg);
  if (reader == nullptr) {
    return nullptr;
  }
  const auto weights = VisionWeights::Bind(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->config = weights->config;
  Uploader up{impl->allocs, impl->bytes, error_msg};
  impl->patch_embed = up.Proj(weights->patch_embed);
  impl->pos_x = up.Norm(weights->pos_x);
  impl->pos_y = up.Norm(weights->pos_y);
  impl->std_bias = up.Norm(weights->std_bias);
  impl->std_scale = up.Norm(weights->std_scale);
  impl->projection = up.Proj(weights->projection);
  impl->blocks.reserve(weights->blocks.size());
  for (const BlockWeights& b : weights->blocks) {
    DBlock d;
    d.ln1 = up.Norm(b.ln1);
    d.q_norm = up.Norm(b.q_norm);
    d.k_norm = up.Norm(b.k_norm);
    d.attn_post_norm = up.Norm(b.attn_post_norm);
    d.ln2 = up.Norm(b.ln2);
    d.ffn_post_norm = up.Norm(b.ffn_post_norm);
    d.attn_q = up.Proj(b.attn_q);
    d.attn_k = up.Proj(b.attn_k);
    d.attn_v = up.Proj(b.attn_v);
    d.attn_out = up.Proj(b.attn_out);
    d.ffn_gate = up.Proj(b.ffn_gate);
    d.ffn_up = up.Proj(b.ffn_up);
    d.ffn_down = up.Proj(b.ffn_down);
    impl->blocks.push_back(d);
    if (!up.ok) {
      return nullptr;
    }
  }
  // 2D rope inverse frequencies: inv[i] = theta^(-2i/head_dim), i in [0, hd/4).
  const std::uint32_t hd = impl->config.HeadDim();
  const std::uint32_t quarter = hd / 4;
  std::vector<float> inv(quarter);
  for (std::uint32_t i = 0; i < quarter; ++i) {
    inv[i] = static_cast<float>(std::pow(
        static_cast<double>(Config::kRopeTheta),
        -2.0 * static_cast<double>(i) / static_cast<double>(hd)));
  }
  if (hipMalloc(&impl->inv_freq, quarter * sizeof(float)) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "hipMalloc failed for inv_freq";
    }
    return nullptr;
  }
  impl->allocs.push_back(impl->inv_freq);
  if (hipMemcpy(impl->inv_freq, inv.data(), quarter * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "upload failed for inv_freq";
    }
    return nullptr;
  }
  if (!up.ok) {
    return nullptr;
  }
  return std::unique_ptr<Encoder>(new Encoder(std::move(impl)));
}

bool Encoder::Encode(const float* pixels, std::uint32_t nx, std::uint32_t ny,
                     std::vector<float>& out, std::string* error_msg) {
  Impl& m = *impl_;
  const Config& c = m.config;
  const std::uint32_t patch = c.patch_size;
  const std::uint32_t merge = Config::kMergeSize;
  const std::uint32_t emb = c.embedding_length;
  const std::uint32_t heads = c.head_count;
  const std::uint32_t hd = c.HeadDim();
  const std::uint32_t ff = c.feed_forward_length;
  const float eps = c.layer_norm_epsilon;
  if (nx % patch != 0 || ny % patch != 0 || (nx / patch) % merge != 0 ||
      (ny / patch) % merge != 0) {
    if (error_msg != nullptr) {
      *error_msg = "image grid is not a multiple of patch * merge";
    }
    return false;
  }
  const std::uint32_t n_px = nx / patch;
  const std::uint32_t n_py = ny / patch;
  const std::uint32_t n_patches = n_px * n_py;
  const std::uint32_t out_x = n_px / merge;
  const std::uint32_t out_y = n_py / merge;
  const std::uint32_t n_tokens = out_x * out_y;

  // Scratch, allocated per image and released on scope exit.
  std::vector<void*> scratch;
  auto alloc = [&](std::size_t floats) -> float* {
    float* p = nullptr;
    if (hipMalloc(&p, floats * sizeof(float)) != hipSuccess) {
      return nullptr;
    }
    scratch.push_back(p);
    return p;
  };
  const std::size_t row = static_cast<std::size_t>(n_patches) * emb;
  const std::size_t frow = static_cast<std::size_t>(n_patches) * ff;
  float* pix_dev = alloc(static_cast<std::size_t>(3) * ny * nx);
  float* patchmat = alloc(static_cast<std::size_t>(n_patches) * 768);
  float* h = alloc(row);
  float* xn = alloc(row);
  float* q = alloc(row);
  float* qn = alloc(row);
  float* k = alloc(row);
  float* kn = alloc(row);
  float* v = alloc(row);
  float* vn = alloc(row);
  float* attn = alloc(row);
  float* tmp = alloc(row);
  float* tmp2 = alloc(row);
  float* up = alloc(frow);
  float* gate = alloc(frow);
  float* act = alloc(frow);
  float* pooled = alloc(static_cast<std::size_t>(n_tokens) * emb);
  float* pooled2 = alloc(static_cast<std::size_t>(n_tokens) * emb);
  float* out_dev = alloc(static_cast<std::size_t>(n_tokens) * c.projection_dim);
  auto release = [&]() {
    for (void* p : scratch) {
      (void)hipFree(p);
    }
  };
  if (pix_dev == nullptr || patchmat == nullptr || h == nullptr ||
      xn == nullptr || q == nullptr || qn == nullptr || k == nullptr ||
      kn == nullptr || v == nullptr || vn == nullptr || attn == nullptr ||
      tmp == nullptr || tmp2 == nullptr || up == nullptr || gate == nullptr ||
      act == nullptr || pooled == nullptr || pooled2 == nullptr ||
      out_dev == nullptr) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "vision scratch allocation failed";
    }
    return false;
  }

  hipStream_t s = nullptr;
  if (hipMemcpyAsync(pix_dev, pixels, static_cast<std::size_t>(3) * ny * nx *
                                          sizeof(float),
                     hipMemcpyHostToDevice, s) != hipSuccess) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "pixel upload failed";
    }
    return false;
  }

  PatchEmbedIm2Col(pix_dev, patchmat, nx, ny, n_px, n_patches, s);
  g4r::Gemm(m.patch_embed.data, m.patch_embed.type, emb, 768,
            m.patch_embed.row_bytes, patchmat, h, n_patches, s);
  PosAdd(h, m.pos_x, m.pos_y, n_px, n_patches, emb, s);

  for (const DBlock& b : m.blocks) {
    g4r::RmsNormRows(h, b.ln1, xn, n_patches, emb, eps, s);
    g4r::Gemm(b.attn_q.data, b.attn_q.type, emb, emb, b.attn_q.row_bytes, xn, q,
              n_patches, s);
    g4r::Gemm(b.attn_k.data, b.attn_k.type, emb, emb, b.attn_k.row_bytes, xn, k,
              n_patches, s);
    g4r::Gemm(b.attn_v.data, b.attn_v.type, emb, emb, b.attn_v.row_bytes, xn, v,
              n_patches, s);
    g4r::RmsNormRows(q, b.q_norm, qn, static_cast<std::size_t>(n_patches) * heads,
                     hd, eps, s);
    g4r::RmsNormRows(k, b.k_norm, kn, static_cast<std::size_t>(n_patches) * heads,
                     hd, eps, s);
    g4r::RmsNormRows(v, nullptr, vn, static_cast<std::size_t>(n_patches) * heads,
                     hd, eps, s);
    Rope2d(qn, m.inv_freq, n_px, n_patches, heads, hd, s);
    Rope2d(kn, m.inv_freq, n_px, n_patches, heads, hd, s);
    VisionAttention(qn, kn, vn, attn, n_patches, heads, hd, s);
    g4r::Gemm(b.attn_out.data, b.attn_out.type, emb, emb, b.attn_out.row_bytes,
              attn, tmp, n_patches, s);
    g4r::RmsNormRows(tmp, b.attn_post_norm, tmp2, n_patches, emb, eps, s);
    g4r::Add(h, tmp2, row, s);

    g4r::RmsNormRows(h, b.ln2, xn, n_patches, emb, eps, s);
    g4r::Gemm(b.ffn_up.data, b.ffn_up.type, ff, emb, b.ffn_up.row_bytes, xn, up,
              n_patches, s);
    g4r::Gemm(b.ffn_gate.data, b.ffn_gate.type, ff, emb, b.ffn_gate.row_bytes,
              xn, gate, n_patches, s);
    GegluQuick(gate, up, act, frow, s);
    g4r::Gemm(b.ffn_down.data, b.ffn_down.type, emb, ff, b.ffn_down.row_bytes,
              act, tmp, n_patches, s);
    g4r::RmsNormRows(tmp, b.ffn_post_norm, tmp2, n_patches, emb, eps, s);
    g4r::Add(h, tmp2, row, s);
  }

  AvgPool3(h, pooled, n_px, n_py, emb, std::sqrt(static_cast<float>(emb)), s);
  StdNorm(pooled, m.std_bias, m.std_scale, n_tokens, emb, s);
  g4r::RmsNormRows(pooled, nullptr, pooled2, n_tokens, emb, eps, s);
  g4r::Gemm(m.projection.data, m.projection.type, c.projection_dim, emb,
            m.projection.row_bytes, pooled2, out_dev, n_tokens, s);

  const hipError_t sync = hipStreamSynchronize(s);
  if (sync != hipSuccess) {
    release();
    if (error_msg != nullptr) {
      *error_msg = "vision stream synchronize failed";
    }
    return false;
  }
  const std::size_t out_floats =
      static_cast<std::size_t>(n_tokens) * c.projection_dim;
  out.assign(out_floats, 0.0F);
  const hipError_t copy = hipMemcpy(out.data(), out_dev,
                                    out_floats * sizeof(float),
                                    hipMemcpyDeviceToHost);
  release();
  if (copy != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "vision output download failed";
    }
    return false;
  }
  return true;
}

const Config& Encoder::config() const noexcept { return impl_->config; }
std::size_t Encoder::resident_bytes() const noexcept { return impl_->bytes; }

}  // namespace gufo::models::gemma4::vision
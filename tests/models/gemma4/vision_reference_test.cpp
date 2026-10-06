// Binds the real Gemma-4 mmproj (`clip` + `gemma4v` projector) and runs the
// scalar vision oracle over a small deterministic image, checking that every
// tensor resolves with the expected shape and that the tower produces a finite
// [n_tokens x 2816] embedding. This is the CPU oracle the vision kernels are
// later validated against; it skips (77) without GUFO_GEMMA4_MMPROJ_GGUF.
#include "src/models/gemma4/vision/reference.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace g4v = gufo::models::gemma4::vision;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void ExpectBound(const gufo::models::gemma4::TensorRef& t,
                 const std::string& what) {
  Expect(!t.empty(), what + " bound");
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MMPROJ_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_MMPROJ_GGUF\n";
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = g4v::VisionWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "vision bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;
  Expect(c.projection_dim == 2816, "projection_dim");
  Expect(c.embedding_length == 1152, "embedding_length");
  Expect(c.block_count == 27, "block_count");
  Expect(c.head_count == 16, "head_count");
  Expect(c.HeadDim() == 72, "head_dim");
  Expect(weights->blocks.size() == c.block_count, "block vector size");
  ExpectBound(weights->patch_embed, "patch_embed");
  ExpectBound(weights->pos_x, "pos_x");
  ExpectBound(weights->pos_y, "pos_y");
  Expect(weights->pos_x.data != weights->pos_y.data, "pos tables distinct");
  ExpectBound(weights->std_bias, "std_bias");
  ExpectBound(weights->std_scale, "std_scale");
  ExpectBound(weights->projection, "projection");
  for (std::uint32_t il = 0; il < c.block_count; ++il) {
    const auto& b = weights->blocks[il];
    const std::string tag = "block " + std::to_string(il);
    ExpectBound(b.ln1, tag + " ln1");
    ExpectBound(b.attn_q, tag + " attn_q");
    ExpectBound(b.attn_k, tag + " attn_k");
    ExpectBound(b.attn_v, tag + " attn_v");
    ExpectBound(b.q_norm, tag + " q_norm");
    ExpectBound(b.k_norm, tag + " k_norm");
    ExpectBound(b.attn_out, tag + " attn_out");
    ExpectBound(b.attn_post_norm, tag + " attn_post_norm");
    ExpectBound(b.ln2, tag + " ln2");
    ExpectBound(b.ffn_gate, tag + " ffn_gate");
    ExpectBound(b.ffn_up, tag + " ffn_up");
    ExpectBound(b.ffn_down, tag + " ffn_down");
    ExpectBound(b.ffn_post_norm, tag + " ffn_post_norm");
  }

  // A deterministic 96x96 gradient (grid 6x6 patches -> 2x2 = 4 tokens).
  const std::uint32_t nx = 96;
  const std::uint32_t ny = 96;
  std::vector<float> pixels(static_cast<std::size_t>(3) * nx * ny);
  for (std::uint32_t ch = 0; ch < 3; ++ch) {
    for (std::uint32_t y = 0; y < ny; ++y) {
      for (std::uint32_t x = 0; x < nx; ++x) {
        pixels[(ch * ny + y) * nx + x] =
            static_cast<float>((x + y + ch * 7) % 256) / 255.0F;
      }
    }
  }
  g4v::ReferenceEncoder encoder(*weights);
  std::vector<float> out;
  if (!encoder.Encode(pixels.data(), nx, ny, out, &error)) {
    std::cerr << "vision encode failed: " << error << "\n";
    return 1;
  }
  const std::uint32_t n_tokens = (nx / (c.patch_size * g4v::Config::kMergeSize)) *
                                 (ny / (c.patch_size * g4v::Config::kMergeSize));
  Expect(n_tokens == 4, "token count for a 96x96 image");
  Expect(out.size() == static_cast<std::size_t>(n_tokens) * c.projection_dim,
         "output size is n_tokens x projection_dim");
  bool all_finite = true;
  double checksum = 0.0;
  for (const float value : out) {
    if (!std::isfinite(value)) {
      all_finite = false;
    }
    checksum += value;
  }
  Expect(all_finite, "all output values are finite");
  Expect(std::abs(checksum) > 1e-3, "output is non-trivial");
  std::cout << "vision oracle checksum: " << checksum << "\n";

  if (failures != 0) {
    std::cerr << failures << " vision checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4 vision oracle passed.\n";
  return 0;
}
// CPU contract for the gemma4v preprocessing: the smart-resize policy and the
// CHW normalization. The resize targets are checked against hand-computed
// values from the reference `calc_size_preserved_ratio` (align 48, min 40
// tokens = 92160 px, max 280 tokens = 645120 px), and a solid-color image must
// survive resize unchanged in every channel, which pins the CHW layout and the
// /255 scaling. No GPU, no model file.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/vision/prompt.hpp"

namespace v = gufo::models::gemma4::vision;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void CheckSize(std::uint32_t w, std::uint32_t h, std::uint32_t exp_w,
               std::uint32_t exp_h, const std::string& tag) {
  const v::Size got = v::CalcSizePreservedRatio(w, h, {});
  const std::string label = tag + " (" + std::to_string(w) + "x" +
                            std::to_string(h) + " -> " +
                            std::to_string(got.width) + "x" +
                            std::to_string(got.height) + ", want " +
                            std::to_string(exp_w) + "x" + std::to_string(exp_h) +
                            ")";
  Expect(got.width == exp_w && got.height == exp_h, label);
}

gufo::core::Image Solid(std::uint32_t w, std::uint32_t h, std::uint8_t r,
                        std::uint8_t g, std::uint8_t b) {
  gufo::core::Image img;
  img.width = w;
  img.height = h;
  img.pixels.assign(static_cast<std::size_t>(w) * h * 3, 0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(w) * h; ++i) {
    img.pixels[i * 3] = r;
    img.pixels[i * 3 + 1] = g;
    img.pixels[i * 3 + 2] = b;
  }
  return img;
}

// A tiny vocabulary with the three image markers marked special so
// `parse_special` splits them out; `a`/`b` are single-char normal tokens.
std::unique_ptr<gufo::models::gemma4::Tokenizer> MarkerTokenizer() {
  const std::vector<std::string> tokens = {"<unk>", "<|image>", "<|image|>",
                                           "<image|>", "a", "b"};
  const std::vector<std::string> specials = {"<|image>", "<|image|>",
                                             "<image|>"};
  return gufo::models::gemma4::Tokenizer::FromVocabulary(tokens, {}, nullptr,
                                                         specials);
}

void CheckBuildPrompt() {
  using gufo::models::gemma4::Tokenizer;
  const std::unique_ptr<Tokenizer> tok = MarkerTokenizer();
  Expect(tok != nullptr, "marker tokenizer built");
  if (tok == nullptr) {
    return;
  }
  const Tokenizer::TokenId begin = tok->TokenToId("<|image>");
  const Tokenizer::TokenId marker = tok->TokenToId("<|image|>");
  const Tokenizer::TokenId end = tok->TokenToId("<image|>");

  // Every slot must be bracketed by begin/end and filled with marker tokens,
  // independent of how the surrounding text tokenizes.
  auto check_slot = [&](const v::PreparedPrompt& p, const v::VisionSlot& s,
                        const std::string& tag) {
    Expect(s.offset >= 1 && s.offset + s.count < p.tokens.size(),
           tag + " slot within bounds");
    if (s.offset < 1 || s.offset + s.count >= p.tokens.size()) {
      return;
    }
    Expect(p.tokens[s.offset - 1] == begin, tag + " begin before fillers");
    Expect(p.tokens[s.offset + s.count] == end, tag + " end after fillers");
    bool fillers_ok = true;
    for (std::uint32_t i = 0; i < s.count; ++i) {
      fillers_ok = fillers_ok && (p.tokens[s.offset + i] == marker);
    }
    Expect(fillers_ok, tag + " fillers are marker tokens");
    Expect(s.pixels.size() == 3 * static_cast<std::size_t>(s.nx) * s.ny,
           tag + " pixel buffer size");
  };

  // One marker between two text tokens expands to begin + N fillers + end.
  const std::vector<gufo::core::Image> one = {Solid(224, 224, 10, 20, 30)};
  const std::size_t base1 =
      tok->Encode("a<|image|>b", false, true).size();
  const v::PreparedPrompt p1 =
      v::BuildPrompt(*tok, "a<|image|>b", one, {}, nullptr);
  Expect(p1.images.size() == 1, "one image slot");
  if (p1.images.size() == 1) {
    const v::VisionSlot& s = p1.images[0];
    Expect(s.count == 49, "one image filler count 49");
    // One marker (1 token) becomes begin + count + end (count + 2): net +count+1.
    Expect(p1.tokens.size() == base1 + s.count + 1, "one image total length");
    check_slot(p1, s, "one");
  }

  // Two markers, two images: slots are ordered and non-overlapping.
  const std::vector<gufo::core::Image> two = {Solid(224, 224, 1, 2, 3),
                                              Solid(224, 224, 4, 5, 6)};
  const std::size_t base2 =
      tok->Encode("<|image|>a<|image|>", false, true).size();
  const v::PreparedPrompt p2 =
      v::BuildPrompt(*tok, "<|image|>a<|image|>", two, {}, nullptr);
  Expect(p2.images.size() == 2, "two image slots");
  if (p2.images.size() == 2) {
    const v::VisionSlot& s0 = p2.images[0];
    const v::VisionSlot& s1 = p2.images[1];
    Expect(s0.offset < s1.offset, "slots ordered");
    Expect(s0.offset + s0.count + 1 <= s1.offset, "slots non-overlapping");
    std::uint32_t total = 0;
    for (const v::VisionSlot& s : p2.images) {
      total += s.count + 1;
    }
    Expect(p2.tokens.size() == base2 + total, "two image total length");
    check_slot(p2, s0, "two[0]");
    check_slot(p2, s1, "two[1]");
  }

  // Marker/image count mismatch is rejected with an empty prompt.
  std::string err;
  const v::PreparedPrompt bad =
      v::BuildPrompt(*tok, "a<|image|>b", {}, {}, &err);
  Expect(bad.tokens.empty() && bad.images.empty(), "mismatch rejected");
  Expect(!err.empty(), "mismatch reports an error");
}

}  // namespace

int main() {
  // Smart-resize policy (align 48, min 92160 px, max 645120 px).
  CheckSize(224, 224, 336, 336, "square below min upscales");
  CheckSize(48, 48, 336, 336, "one block upscales");
  CheckSize(640, 480, 624, 480, "4:3 within budget keeps aligned size");
  CheckSize(1024, 1024, 768, 768, "large square downscales under max");
  CheckSize(0, 100, 0, 0, "zero width");
  CheckSize(100, 0, 0, 0, "zero height");

  // Token count and CHW normalization on a solid red image.
  const auto red = Solid(100, 100, 255, 0, 0);
  const v::PreparedImage prepared = v::PrepareImage(red, {});
  Expect(prepared.nx == 336 && prepared.ny == 336, "solid resize dims");
  Expect(prepared.tokens == 49, "solid token count");
  const std::size_t plane = static_cast<std::size_t>(prepared.nx) * prepared.ny;
  Expect(prepared.pixels.size() == 3 * plane, "pixel buffer size");
  bool red_ok = true;
  for (std::size_t i = 0; i < plane; ++i) {
    if (std::fabs(prepared.pixels[i] - 1.0F) > 1e-4F ||
        std::fabs(prepared.pixels[plane + i]) > 1e-4F ||
        std::fabs(prepared.pixels[2 * plane + i]) > 1e-4F) {
      red_ok = false;
      break;
    }
  }
  Expect(red_ok, "solid red stays red in CHW [0,1]");

  // A solid mid-gray image maps to ~0.5 in every channel.
  const auto gray = Solid(200, 120, 128, 128, 128);
  const v::PreparedImage gp = v::PrepareImage(gray, {});
  const std::size_t gplane = static_cast<std::size_t>(gp.nx) * gp.ny;
  bool gray_ok = gp.nx % 48 == 0 && gp.ny % 48 == 0;
  for (std::size_t i = 0; i < gplane && gray_ok; ++i) {
    for (int c = 0; c < 3; ++c) {
      if (std::fabs(gp.pixels[static_cast<std::size_t>(c) * gplane + i] -
                    128.0F / 255.0F) > 1e-3F) {
        gray_ok = false;
        break;
      }
    }
  }
  Expect(gray_ok, "solid gray maps to 128/255 in every channel");

  CheckBuildPrompt();

  if (failures != 0) {
    std::cerr << failures << " vision prompt checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4 vision prompt preprocessing checks passed.\n";
  return 0;
}
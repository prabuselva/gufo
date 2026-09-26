// Quantized matrix-vector product checks: synthetic Q8_0, F32 and BF16
// matrices through the GEMV tier against a CPU reference that dequantizes and
// dots in double. No model artifact is required; the real-artifact layout is
// exercised by the forward check.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/gemv.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// Model geometry: the trunk's linear projections are [hidden x out] with
// hidden 2048, so cols is the reduction width and rows the output width.
constexpr std::uint32_t kRows = 32;
constexpr std::uint32_t kCols = 2048;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

bool TestQ8_0() {
  constexpr std::uint32_t nblocks = kCols / 32;
  const auto x = t::MakeValues(kCols, 0x11112222U, 1.0F);

  // Encode a random Q8_0 matrix: per block a half scale and 32 int8 codes.
  std::vector<__half> scales(static_cast<std::size_t>(kRows) * nblocks);
  std::vector<std::int8_t> codes(static_cast<std::size_t>(kRows) * nblocks *
                                 32);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(kRows) * nblocks * 34);
  std::uint32_t seed = 0x0BADF00DU;
  for (std::size_t idx = 0; idx < scales.size(); ++idx) {
    const float scale =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    scales[idx] = __float2half(scale);
    const auto* scale_bytes =
        reinterpret_cast<const std::uint8_t*>(&scales[idx]);
    bytes.push_back(scale_bytes[0]);
    bytes.push_back(scale_bytes[1]);
    for (int i = 0; i < 32; ++i) {
      const std::int8_t code =
          static_cast<std::int8_t>(t::NextRandom(&seed) & 0xFFU);
      codes[idx * 32 + i] = code;
      bytes.push_back(static_cast<std::uint8_t>(code));
    }
  }

  t::HipBuffer<std::uint8_t> d_w(bytes.size());
  t::CheckHip(
      hipMemcpy(d_w.get(), bytes.data(), bytes.size(), hipMemcpyHostToDevice),
      "upload q8_0 weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  q::Gemv(d_w.get(), q::GemvType::kQ8_0, kRows, kCols, 34, d_x.get(),
          d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemvQ8_0 synchronization");
  const auto got = t::Download(&d_out, kRows);

  std::vector<float> ref(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double acc = 0.0;
    for (std::uint32_t b = 0; b < nblocks; ++b) {
      const float d =
          __half2float(scales[static_cast<std::size_t>(r) * nblocks + b]);
      for (int i = 0; i < 32; ++i) {
        const std::size_t c =
            (static_cast<std::size_t>(r) * nblocks + b) * 32 + i;
        acc += static_cast<double>(d) * codes[c] * x[b * 32 + i];
      }
    }
    ref[r] = static_cast<float>(acc);
  }
  return Check("GemvQ8_0", t::WorstRelative(ref, got), 1e-3);
}

bool TestF32() {
  const auto x = t::MakeValues(kCols, 0x33334444U, 1.0F);
  const auto w =
      t::MakeValues(static_cast<std::size_t>(kRows) * kCols, 0x55556666U, 1.0F);
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  q::Gemv(d_w.get(), q::GemvType::kF32, kRows, kCols, 4, d_x.get(), d_out.get(),
          nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemvF32 synchronization");
  const auto got = t::Download(&d_out, kRows);

  std::vector<float> ref(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < kCols; ++i) {
      acc += static_cast<double>(w[static_cast<std::size_t>(r) * kCols + i]) *
             x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  return Check("GemvF32", t::WorstRelative(ref, got), 1e-3);
}

// BF16 is the top 16 bits of a float32. Encode with round-to-nearest-even and
// decode exactly (a bf16 value is a float32 with the low 16 bits zeroed), so
// the host reference matches the kernel's hip_bfloat16 conversion bit-for-bit
// without needing the device conversion intrinsics on the host.
std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(std::uint16_t bits) {
  const std::uint32_t fbits = static_cast<std::uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &fbits, sizeof(f));
  return f;
}

bool TestBf16() {
  const auto x = t::MakeValues(kCols, 0x77778888U, 1.0F);
  std::vector<std::uint16_t> w(static_cast<std::size_t>(kRows) * kCols);
  std::uint32_t seed = 0x9A8B7C6DU;
  for (auto& value : w) {
    const float f =
        1.0F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    value = FloatToBf16Bits(f);
  }
  t::HipBuffer<std::uint16_t> d_w(w.size());
  t::CheckHip(
      hipMemcpy(d_w.get(), w.data(), d_w.bytes(), hipMemcpyHostToDevice),
      "upload bf16 weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  q::Gemv(d_w.get(), q::GemvType::kBF16, kRows, kCols, 2, d_x.get(),
          d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemvBf16 synchronization");
  const auto got = t::Download(&d_out, kRows);

  std::vector<float> ref(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < kCols; ++i) {
      acc += static_cast<double>(
                 Bf16BitsToFloat(w[static_cast<std::size_t>(r) * kCols + i])) *
             x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  return Check("GemvBf16", t::WorstRelative(ref, got), 1e-3);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = TestQ8_0() && ok;
    ok = TestF32() && ok;
    ok = TestBf16() && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
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

// Encode a random Q8_0 matrix (same scheme as TestQ8_0) into host bytes.
std::vector<std::uint8_t> EncodeQ8_0(std::uint32_t rows, std::uint32_t cols,
                                     std::uint32_t seed) {
  const std::uint32_t nblocks = cols / 32;
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(rows) * nblocks * 34);
  for (std::size_t idx = 0; idx < static_cast<std::size_t>(rows) * nblocks;
       ++idx) {
    const float scale =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const __half h = __float2half(scale);
    const auto* scale_bytes = reinterpret_cast<const std::uint8_t*>(&h);
    bytes.push_back(scale_bytes[0]);
    bytes.push_back(scale_bytes[1]);
    for (int i = 0; i < 32; ++i) {
      bytes.push_back(static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU));
    }
  }
  return bytes;
}

// The fused launches must be bit-identical to the separate ones: the
// executor relies on this to keep the forward-test token parity.
bool TestMulti() {
  // Unequal rows and a narrower last matrix exercise the per-projection
  // offsets and block counts; the n=2 call covers the pair case.
  constexpr std::uint32_t kRowsA = 5, kRowsB = 3, kRowsC = 1, kRowsD = 7;
  constexpr std::uint32_t kColsD = 512;
  const auto x = t::MakeValues(kCols, 0xABCD0123U, 1.0F);
  const auto wa = EncodeQ8_0(kRowsA, kCols, 0x11110001U);
  const auto wb = EncodeQ8_0(kRowsB, kCols, 0x22220002U);
  const auto wc = EncodeQ8_0(kRowsC, kCols, 0x33330003U);
  const auto wd = EncodeQ8_0(kRowsD, kColsD, 0x44440004U);
  t::HipBuffer<std::uint8_t> d_wa(wa.size());
  t::HipBuffer<std::uint8_t> d_wb(wb.size());
  t::HipBuffer<std::uint8_t> d_wc(wc.size());
  t::HipBuffer<std::uint8_t> d_wd(wd.size());
  t::CheckHip(
      hipMemcpy(d_wa.get(), wa.data(), wa.size(), hipMemcpyHostToDevice),
      "upload multi matrix A");
  t::CheckHip(
      hipMemcpy(d_wb.get(), wb.data(), wb.size(), hipMemcpyHostToDevice),
      "upload multi matrix B");
  t::CheckHip(
      hipMemcpy(d_wc.get(), wc.data(), wc.size(), hipMemcpyHostToDevice),
      "upload multi matrix C");
  t::CheckHip(
      hipMemcpy(d_wd.get(), wd.data(), wd.size(), hipMemcpyHostToDevice),
      "upload multi matrix D");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_ref_a(kRowsA);
  t::HipBuffer<float> d_ref_b(kRowsB);
  t::HipBuffer<float> d_ref_c(kRowsC);
  t::HipBuffer<float> d_ref_d(kRowsD);
  q::Gemv(d_wa.get(), q::GemvType::kQ8_0, kRowsA, kCols, 34, d_x.get(),
          d_ref_a.get(), nullptr);
  q::Gemv(d_wb.get(), q::GemvType::kQ8_0, kRowsB, kCols, 34, d_x.get(),
          d_ref_b.get(), nullptr);
  q::Gemv(d_wc.get(), q::GemvType::kQ8_0, kRowsC, kCols, 34, d_x.get(),
          d_ref_c.get(), nullptr);
  q::Gemv(d_wd.get(), q::GemvType::kQ8_0, kRowsD, kColsD, 34, d_x.get(),
          d_ref_d.get(), nullptr);
  t::HipBuffer<float> d_out_a(kRowsA);
  t::HipBuffer<float> d_out_b(kRowsB);
  t::HipBuffer<float> d_out_c(kRowsC);
  t::HipBuffer<float> d_out_d(kRowsD);
  const q::GemvMultiProj projs[4] = {
      {d_wa.get(), q::GemvType::kQ8_0, kRowsA, kCols, d_out_a.get()},
      {d_wb.get(), q::GemvType::kQ8_0, kRowsB, kCols, d_out_b.get()},
      {d_wc.get(), q::GemvType::kQ8_0, kRowsC, kCols, d_out_c.get()},
      {d_wd.get(), q::GemvType::kQ8_0, kRowsD, kColsD, d_out_d.get()},
  };
  if (!q::GemvMulti(projs, 4U, d_x.get(), nullptr)) {
    std::cerr << "GemvMulti unexpectedly returned false for Q8_0\n";
    return false;
  }
  t::CheckHip(hipDeviceSynchronize(), "GemvQ8_0Multi4 synchronization");
  const bool ok = t::Download(&d_ref_a, kRowsA) ==
                      t::Download(&d_out_a, kRowsA) &&
                  t::Download(&d_ref_b, kRowsB) ==
                      t::Download(&d_out_b, kRowsB) &&
                  t::Download(&d_ref_c, kRowsC) ==
                      t::Download(&d_out_c, kRowsC);
  std::cout << "GemvQ8_0Multi4 bit-exact: " << (ok ? "yes" : "no") << '\n';
  // n=2 (the gate/up pair shape) and a non-Q8_0 rejection.
  t::HipBuffer<float> d_pair_a(kRowsA);
  t::HipBuffer<float> d_pair_b(kRowsB);
  const q::GemvMultiProj pair[2] = {
      {d_wa.get(), q::GemvType::kQ8_0, kRowsA, kCols, d_pair_a.get()},
      {d_wb.get(), q::GemvType::kQ8_0, kRowsB, kCols, d_pair_b.get()},
  };
  const bool pair_ok = q::GemvMulti(pair, 2U, d_x.get(), nullptr) &&
                       t::Download(&d_ref_a, kRowsA) ==
                           t::Download(&d_pair_a, kRowsA) &&
                       t::Download(&d_ref_b, kRowsB) ==
                           t::Download(&d_pair_b, kRowsB);
  std::cout << "GemvMulti n=2: " << (pair_ok ? "yes" : "no") << '\n';
  // Mixed Q8_0 + F32 group (the ssm quartet shape): the F32 side vector
  // runs as a plain Gemv and must still match bit-for-bit.
  constexpr std::uint32_t kRowsF = 3;
  const auto wf = t::MakeValues(kRowsF * kCols, 0x99AA0006U, 1.0F);
  t::HipBuffer<float> d_wf(wf.size());
  t::Upload(&d_wf, wf);
  t::HipBuffer<float> d_ref_f(kRowsF);
  q::Gemv(d_wf.get(), q::GemvType::kF32, kRowsF, kCols, 0U, d_x.get(),
          d_ref_f.get(), nullptr);
  t::HipBuffer<float> d_out_f(kRowsF);
  const q::GemvMultiProj mixed[3] = {
      {d_wa.get(), q::GemvType::kQ8_0, kRowsA, kCols, d_pair_a.get()},
      {d_wf.get(), q::GemvType::kF32, kRowsF, kCols, d_out_f.get()},
      {d_wb.get(), q::GemvType::kQ8_0, kRowsB, kCols, d_pair_b.get()},
  };
  const bool mixed_ok = q::GemvMulti(mixed, 3U, d_x.get(), nullptr) &&
                        t::Download(&d_ref_a, kRowsA) ==
                            t::Download(&d_pair_a, kRowsA) &&
                        t::Download(&d_ref_f, kRowsF) ==
                            t::Download(&d_out_f, kRowsF) &&
                        t::Download(&d_ref_b, kRowsB) ==
                            t::Download(&d_pair_b, kRowsB);
  const bool rejects = !q::GemvMulti(pair, 0U, d_x.get(), nullptr) &&
                       !q::GemvMulti(pair, 5U, d_x.get(), nullptr);
  std::cout << "GemvMulti mixed Q8_0+F32 and rejection: "
            << (mixed_ok && rejects ? "yes" : "no") << '\n';
  return ok && pair_ok && mixed_ok && rejects;
}

bool TestGroupedPair() {
  constexpr std::uint32_t kExperts = 4;
  constexpr std::uint32_t kUsed = 2;
  const std::int32_t ids[kUsed] = {2, 0};
  const std::size_t expert_bytes =
      static_cast<std::size_t>(kRows) * (kCols / 32) * 34;
  const auto x = t::MakeValues(kCols, 0x55AA0204U, 1.0F);
  const auto wa = EncodeQ8_0(kExperts * kRows, kCols, 0x33330003U);
  const auto wb = EncodeQ8_0(kExperts * kRows, kCols, 0x44440004U);
  t::HipBuffer<std::uint8_t> d_wa(wa.size());
  t::CheckHip(
      hipMemcpy(d_wa.get(), wa.data(), wa.size(), hipMemcpyHostToDevice),
      "upload grouped matrix A");
  t::HipBuffer<std::uint8_t> d_wb(wb.size());
  t::CheckHip(
      hipMemcpy(d_wb.get(), wb.data(), wb.size(), hipMemcpyHostToDevice),
      "upload grouped matrix B");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<std::int32_t> d_ids(kUsed);
  t::CheckHip(hipMemcpy(d_ids.get(), ids, sizeof(ids), hipMemcpyHostToDevice),
              "upload expert ids");
  t::HipBuffer<float> d_ref_a(kUsed * kRows);
  t::HipBuffer<float> d_ref_b(kUsed * kRows);
  q::GemvGrouped(d_wa.get(), q::GemvType::kQ8_0, expert_bytes, d_ids.get(),
                 kUsed, kRows, kCols, d_x.get(), 0U, d_ref_a.get(), nullptr);
  q::GemvGrouped(d_wb.get(), q::GemvType::kQ8_0, expert_bytes, d_ids.get(),
                 kUsed, kRows, kCols, d_x.get(), 0U, d_ref_b.get(), nullptr);
  t::HipBuffer<float> d_out_a(kUsed * kRows);
  t::HipBuffer<float> d_out_b(kUsed * kRows);
  if (!q::GemvGroupedPair(d_wa.get(), d_wb.get(), q::GemvType::kQ8_0,
                         expert_bytes, d_ids.get(), kUsed, kRows, kCols,
                         d_x.get(), 0U, d_out_a.get(), d_out_b.get(),
                         nullptr)) {
    std::cerr << "GemvGroupedPair unexpectedly returned false for Q8_0\n";
    return false;
  }
  t::CheckHip(hipDeviceSynchronize(), "GemvGroupedQ8_0Pair synchronization");
  const auto ref_a = t::Download(&d_ref_a, kUsed * kRows);
  const auto ref_b = t::Download(&d_ref_b, kUsed * kRows);
  const auto out_a = t::Download(&d_out_a, kUsed * kRows);
  const auto out_b = t::Download(&d_out_b, kUsed * kRows);
  const bool ok_a = ref_a == out_a;
  const bool ok_b = ref_b == out_b;
  std::cout << "GemvGroupedQ8_0Pair bit-exact: "
            << (ok_a && ok_b ? "yes" : "no") << '\n';
  return ok_a && ok_b;
}

// The multi-row verify GEMV must be bit-identical to single-row Gemv calls
// for every row count in [2, 5] and every stored type: same per-row walk,
// weight row read once for all activation rows.
bool TestMultiRow() {
  bool ok = true;

  const auto run = [&](const char* name, q::GemvType type,
                       const std::vector<std::uint8_t>& w,
                       std::size_t row_bytes) {
    for (std::uint32_t rows = 2U; rows <= 5U; ++rows) {
      std::vector<float> x(static_cast<std::size_t>(rows) * kCols);
      for (std::uint32_t o = 0; o < rows; ++o) {
        const auto xo = t::MakeValues(kCols, 0x11223344U + o * 0x01010101U,
                                      1.0F);
        std::copy(xo.begin(), xo.end(), x.begin() + o * kCols);
      }
      t::HipBuffer<std::uint8_t> d_w(w.size());
      t::CheckHip(
          hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
          "upload multi-row weights");
      t::HipBuffer<float> d_x(x.size());
      t::Upload(&d_x, x);
      t::HipBuffer<float> d_ref(static_cast<std::size_t>(rows) * kRows);
      for (std::uint32_t o = 0; o < rows; ++o) {
        q::Gemv(d_w.get(), type, kRows, kCols, row_bytes, d_x.get() + o * kCols,
                d_ref.get() + static_cast<std::size_t>(o) * kRows, nullptr);
      }
      t::HipBuffer<float> d_out(static_cast<std::size_t>(rows) * kRows);
      q::GemvRows(d_w.get(), type, rows, kRows, kCols, row_bytes, d_x.get(),
                  kCols, d_out.get(), kRows, nullptr);
      t::CheckHip(hipDeviceSynchronize(), "GemvRows synchronization");
      const auto ref =
          t::Download(&d_ref, static_cast<std::size_t>(rows) * kRows);
      const auto got =
          t::Download(&d_out, static_cast<std::size_t>(rows) * kRows);
      const bool exact = ref == got;
      std::cout << name << " rows=" << rows
                << " bit-exact: " << (exact ? "yes" : "no") << '\n';
      ok = ok && exact;
    }
  };

  run("GemvRowsQ8_0", q::GemvType::kQ8_0,
      EncodeQ8_0(kRows, kCols, 0x66778899U), 34);

  const auto w_f32 =
      t::MakeValues(static_cast<std::size_t>(kRows) * kCols, 0x55556666U, 1.0F);
  std::vector<std::uint8_t> w_f32_bytes(w_f32.size() * sizeof(float));
  std::memcpy(w_f32_bytes.data(), w_f32.data(), w_f32_bytes.size());
  run("GemvRowsF32", q::GemvType::kF32, w_f32_bytes, 4);

  std::vector<std::uint16_t> w_bf16(static_cast<std::size_t>(kRows) * kCols);
  std::uint32_t seed = 0x9A8B7C6DU;
  for (auto& value : w_bf16) {
    const float f = 1.0F *
                    static_cast<float>(
                        static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                        32768) /
                    32768.0F;
    value = FloatToBf16Bits(f);
  }
  std::vector<std::uint8_t> w_bf16_bytes(w_bf16.size() * sizeof(std::uint16_t));
  std::memcpy(w_bf16_bytes.data(), w_bf16.data(), w_bf16_bytes.size());
  run("GemvRowsBf16", q::GemvType::kBF16, w_bf16_bytes, 2);
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = TestQ8_0() && ok;
    ok = TestF32() && ok;
    ok = TestBf16() && ok;
    ok = TestMulti() && ok;
    ok = TestGroupedPair() && ok;
    ok = TestMultiRow() && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
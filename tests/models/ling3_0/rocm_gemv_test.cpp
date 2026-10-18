// Quantized matrix-vector product checks: synthetic Q4_0, Q8_0, Q4_K, Q5_K,
// Q6_K and F32 matrices through the GEMV tier against a CPU reference that
// dequantizes with the canonical ggml dequantizers and dots in double. No
// model artifact is required; the real-artifact layout is exercised by the
// forward check.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/ling3_0/kernels/rocm/gemv.hpp"
#include "tests/models/ling3_0/hip_test.hpp"

namespace q = gufo::models::ling3_0::rocm;
namespace t = gufo::tests::ling3_0;

namespace {

// Model geometry: every trunk projection reduces over the hidden width 2560,
// so cols is the reduction width and rows the output width. 2560 is a multiple
// of both the 32-element Q_0 super-block and the 256-element K super-block.
constexpr std::uint32_t kRows = 32;
constexpr std::uint32_t kCols = 2560;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

std::uint16_t HalfBits(float f) {
  const __half h = __float2half(f);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

// Encode a random Q4_0 matrix: per 32-element block a half scale and 16 packed
// nibble bytes. The oracle's DequantizeQ4_0 defines the semantics.
std::vector<std::uint8_t> EncodeQ4_0(std::uint32_t rows, std::uint32_t cols,
                                     std::uint32_t seed) {
  using gufo::quant::block_q4_0;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 32);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q4_0));
  auto* blk = reinterpret_cast<block_q4_0*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    blk[i].d =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    for (int s = 0; s < 16; ++s) {
      blk[i].qs[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
  }
  return bytes;
}

// Encode a random Q8_0 matrix: per 32-element block a half scale and 32 int8
// codes.
std::vector<std::uint8_t> EncodeQ8_0(std::uint32_t rows, std::uint32_t cols,
                                     std::uint32_t seed) {
  using gufo::quant::block_q8_0;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 32);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q8_0));
  auto* blk = reinterpret_cast<block_q8_0*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    blk[i].d =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    for (int s = 0; s < 32; ++s) {
      blk[i].qs[s] = static_cast<std::int8_t>(t::NextRandom(&seed) & 0xFFU);
    }
  }
  return bytes;
}

// Encode a random Q4_K matrix: per 256-element super-block a half d, half
// dmin, twelve packed (scale, minimum) bytes and 128 packed nibble bytes.
std::vector<std::uint8_t> EncodeQ4K(std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t seed) {
  using gufo::quant::block_q4_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 256);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q4_K));
  auto* blk = reinterpret_cast<block_q4_K*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    blk[i].d =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    blk[i].dmin =
        HalfBits(0.25F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    for (int s = 0; s < 12; ++s) {
      blk[i].scales[s] =
          static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    for (int s = 0; s < 128; ++s) {
      blk[i].qs[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
  }
  return bytes;
}

// Encode a random Q5_K matrix: the Q4_K super-block plus 32 high-bit bytes.
std::vector<std::uint8_t> EncodeQ5K(std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t seed) {
  using gufo::quant::block_q5_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 256);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q5_K));
  auto* blk = reinterpret_cast<block_q5_K*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    blk[i].d =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    blk[i].dmin =
        HalfBits(0.25F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    for (int s = 0; s < 12; ++s) {
      blk[i].scales[s] =
          static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    for (int s = 0; s < 32; ++s) {
      blk[i].qh[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    for (int s = 0; s < 128; ++s) {
      blk[i].qs[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
  }
  return bytes;
}

// Encode a random Q6_K matrix: per 256-element super-block 128 low-nibble
// bytes, 64 packed high-bit bytes, 16 int8 sub-scales and a half d.
std::vector<std::uint8_t> EncodeQ6K(std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t seed) {
  using gufo::quant::block_q6_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 256);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q6_K));
  auto* blk = reinterpret_cast<block_q6_K*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    for (int s = 0; s < 128; ++s) {
      blk[i].ql[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    for (int s = 0; s < 64; ++s) {
      blk[i].qh[s] = static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    for (int s = 0; s < 16; ++s) {
      blk[i].scales[s] = static_cast<std::int8_t>(t::NextRandom(&seed) & 0xFFU);
    }
    blk[i].d =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
  }
  return bytes;
}

std::vector<std::uint8_t> EncodeF32(std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t seed) {
  const auto values =
      t::MakeValues(static_cast<std::size_t>(rows) * cols, seed, 1.0F);
  std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void DequantizeRow(q::GemvType type, const void* row, float* dst) {
  switch (type) {
    case q::GemvType::kQ4_0:
      gufo::quant::DequantizeQ4_0(row, dst, kCols);
      break;
    case q::GemvType::kQ8_0:
      gufo::quant::DequantizeQ8_0(row, dst, kCols);
      break;
    case q::GemvType::kQ4_K:
      gufo::quant::DequantizeQ4_K(row, dst, kCols);
      break;
    case q::GemvType::kQ5_K:
      gufo::quant::DequantizeQ5_K(row, dst, kCols);
      break;
    case q::GemvType::kQ6_K:
      gufo::quant::DequantizeQ6_K(row, dst, kCols);
      break;
    case q::GemvType::kF32:
      std::memcpy(dst, row, kCols * sizeof(float));
      break;
  }
}

std::size_t RowBytes(q::GemvType type) {
  switch (type) {
    case q::GemvType::kQ4_0:
      return (kCols / 32) * sizeof(gufo::quant::block_q4_0);
    case q::GemvType::kQ8_0:
      return (kCols / 32) * sizeof(gufo::quant::block_q8_0);
    case q::GemvType::kQ4_K:
      return (kCols / 256) * sizeof(gufo::quant::block_q4_K);
    case q::GemvType::kQ5_K:
      return (kCols / 256) * sizeof(gufo::quant::block_q5_K);
    case q::GemvType::kQ6_K:
      return (kCols / 256) * sizeof(gufo::quant::block_q6_K);
    case q::GemvType::kF32:
      return kCols * sizeof(float);
  }
  return 0;
}

// Dense single-vector GEMV: the kernel's per-row dot must match the oracle
// dequantize + double dot within float accumulation noise.
bool TestDense(const char* name, q::GemvType type,
               const std::vector<std::uint8_t>& w) {
  const auto x = t::MakeValues(kCols, 0x11112222U, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  q::Gemv(d_w.get(), type, kRows, kCols, RowBytes(type), d_x.get(), d_out.get(),
          nullptr);
  t::CheckHip(hipDeviceSynchronize(), name);
  const auto got = t::Download(&d_out, kRows);

  std::vector<float> ref(kRows);
  std::vector<float> deq(kCols);
  const std::size_t row_bytes = RowBytes(type);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    DequantizeRow(type, w.data() + static_cast<std::size_t>(r) * row_bytes,
                  deq.data());
    double acc = 0.0;
    for (std::uint32_t i = 0; i < kCols; ++i) {
      acc += static_cast<double>(deq[i]) * x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  return Check(name, t::WorstRelative(ref, got), 2e-3);
}

// The grouped expert GEMV (the decode route for the routed experts) must
// match a CPU reference over the selected expert rows.
bool TestGrouped(const char* name, q::GemvType type,
                 const std::vector<std::uint8_t>& w) {
  // The uploaded weight tensor covers four experts; ids pick experts 2 and 0.
  constexpr std::uint32_t kUsed = 2;
  const std::int32_t ids[kUsed] = {2, 0};
  const std::size_t row_bytes = RowBytes(type);
  const std::size_t expert_bytes = static_cast<std::size_t>(kRows) * row_bytes;
  const auto x = t::MakeValues(kCols, 0x55AA0204U, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload grouped weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<std::int32_t> d_ids(kUsed);
  t::CheckHip(hipMemcpy(d_ids.get(), ids, sizeof(ids), hipMemcpyHostToDevice),
              "upload expert ids");
  t::HipBuffer<float> d_out(kUsed * kRows);
  q::GemvGrouped(d_w.get(), type, expert_bytes, d_ids.get(), kUsed, kUsed,
                 kRows, kCols, d_x.get(), 0U, d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), name);
  const auto got = t::Download(&d_out, kUsed * kRows);

  std::vector<float> ref(static_cast<std::size_t>(kUsed) * kRows);
  std::vector<float> deq(kCols);
  for (std::uint32_t s = 0; s < kUsed; ++s) {
    for (std::uint32_t r = 0; r < kRows; ++r) {
      const std::size_t row_index =
          static_cast<std::size_t>(ids[s]) * kRows + r;
      DequantizeRow(type, w.data() + row_index * row_bytes, deq.data());
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kCols; ++i) {
        acc += static_cast<double>(deq[i]) * x[i];
      }
      ref[static_cast<std::size_t>(s) * kRows + r] = static_cast<float>(acc);
    }
  }
  return Check(name, t::WorstRelative(ref, got), 2e-3);
}

// The multi-row verify GEMV must be bit-identical to single-row Gemv calls
// for every row count in [2, 5] and every stored type: same per-row walk,
// weight row read once for all activation rows.
bool TestMultiRow(const char* name, q::GemvType type,
                  const std::vector<std::uint8_t>& w) {
  bool ok = true;
  const std::size_t row_bytes = RowBytes(type);
  for (std::uint32_t rows = 2U; rows <= 5U; ++rows) {
    std::vector<float> x(static_cast<std::size_t>(rows) * kCols);
    for (std::uint32_t o = 0; o < rows; ++o) {
      const auto xo = t::MakeValues(kCols, 0x11223344U + o * 0x01010101U, 1.0F);
      std::copy(xo.begin(), xo.end(), x.begin() + o * kCols);
    }
    t::HipBuffer<std::uint8_t> d_w(w.size());
    t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
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
  return ok;
}

// The token-embedding lookup must reproduce the oracle dequantization of the
// requested row exactly for the first, a middle and the last row.
bool TestEmbedRow(const char* name, q::GemvType type, const void* d_w,
                  const std::vector<std::uint8_t>& w, std::uint32_t rows) {
  bool ok = true;
  const std::size_t row_bytes = RowBytes(type);
  const std::uint32_t probes[3] = {0U, rows / 2, rows - 1U};
  for (const std::uint32_t row : probes) {
    t::HipBuffer<float> d_out(kCols);
    q::EmbedRow(d_w, type, row, kCols, d_out.get(), nullptr);
    t::CheckHip(hipDeviceSynchronize(), "EmbedRow synchronization");
    const auto got = t::Download(&d_out, kCols);
    std::vector<float> ref(kCols);
    DequantizeRow(type, w.data() + static_cast<std::size_t>(row) * row_bytes,
                  ref.data());
    const double worst = t::WorstRelative(ref, got, 1e-6);
    std::cout << name << " row=" << row << " worst error " << worst << '\n';
    ok = ok && worst <= 1e-5;
  }
  return ok;
}

// Upload the matrix once and run the embedding-lookup probes against it.
bool RunEmbedRow(const char* name, q::GemvType type,
                 const std::vector<std::uint8_t>& w) {
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload embed weights");
  return TestEmbedRow(name, type, d_w.get(), w, kRows);
}

}  // namespace

int main() {
  try {
    const auto w_q4_0 = EncodeQ4_0(kRows, kCols, 0x0BADF00DU);
    const auto w_q8_0 = EncodeQ8_0(kRows, kCols, 0x1BADF00DU);
    const auto w_q4_k = EncodeQ4K(kRows, kCols, 0x66AA0004U);
    const auto w_q5_k = EncodeQ5K(kRows, kCols, 0x77AA0005U);
    const auto w_q6_k = EncodeQ6K(kRows, kCols, 0x88AA0006U);
    const auto w_f32 = EncodeF32(kRows, kCols, 0x55556666U);

    bool ok = true;
    ok = TestDense("GemvQ4_0", q::GemvType::kQ4_0, w_q4_0) && ok;
    ok = TestDense("GemvQ8_0", q::GemvType::kQ8_0, w_q8_0) && ok;
    ok = TestDense("GemvQ4_K", q::GemvType::kQ4_K, w_q4_k) && ok;
    ok = TestDense("GemvQ5_K", q::GemvType::kQ5_K, w_q5_k) && ok;
    ok = TestDense("GemvQ6_K", q::GemvType::kQ6_K, w_q6_k) && ok;
    ok = TestDense("GemvF32", q::GemvType::kF32, w_f32) && ok;

    ok = TestGrouped("GemvGroupedQ4_K", q::GemvType::kQ4_K,
                     EncodeQ4K(4 * kRows, kCols, 0x66AA0104U)) &&
         ok;
    ok = TestGrouped("GemvGroupedQ6_K", q::GemvType::kQ6_K,
                     EncodeQ6K(4 * kRows, kCols, 0x88AA0106U)) &&
         ok;
    ok = TestGrouped("GemvGroupedQ4_0", q::GemvType::kQ4_0,
                     EncodeQ4_0(4 * kRows, kCols, 0x0BAD010DU)) &&
         ok;
    ok = TestGrouped("GemvGroupedQ8_0", q::GemvType::kQ8_0,
                     EncodeQ8_0(4 * kRows, kCols, 0x1BAD010DU)) &&
         ok;
    ok = TestGrouped("GemvGroupedF32", q::GemvType::kF32,
                     EncodeF32(4 * kRows, kCols, 0x55550106U)) &&
         ok;

    ok = TestMultiRow("GemvRowsQ4_0", q::GemvType::kQ4_0, w_q4_0) && ok;
    ok = TestMultiRow("GemvRowsQ8_0", q::GemvType::kQ8_0, w_q8_0) && ok;
    ok = TestMultiRow("GemvRowsQ4_K", q::GemvType::kQ4_K, w_q4_k) && ok;
    ok = TestMultiRow("GemvRowsQ5_K", q::GemvType::kQ5_K, w_q5_k) && ok;
    ok = TestMultiRow("GemvRowsQ6_K", q::GemvType::kQ6_K, w_q6_k) && ok;
    ok = TestMultiRow("GemvRowsF32", q::GemvType::kF32, w_f32) && ok;

    ok = RunEmbedRow("EmbedRowQ4_0", q::GemvType::kQ4_0, w_q4_0) && ok;
    ok = RunEmbedRow("EmbedRowQ8_0", q::GemvType::kQ8_0, w_q8_0) && ok;
    ok = RunEmbedRow("EmbedRowQ4_K", q::GemvType::kQ4_K, w_q4_k) && ok;
    ok = RunEmbedRow("EmbedRowQ5_K", q::GemvType::kQ5_K, w_q5_k) && ok;
    ok = RunEmbedRow("EmbedRowQ6_K", q::GemvType::kQ6_K, w_q6_k) && ok;
    ok = RunEmbedRow("EmbedRowF32", q::GemvType::kF32, w_f32) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
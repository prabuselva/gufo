// Batched (prefill) projection checks: synthetic Q8_0, F32 and BF16 matrices
// through the GEMM tier against a CPU reference that dequantizes and dots in
// double. The Q8_0 paths run on the shared mmq tensor-core route (which
// quantizes the activations to Q8_1), so their tolerance is looser than the
// exact-activation F32 path. The routed Q4_K/Q5_K checks drive the WMMA expert
// GEMM (the prefill route for a Q4_K_XL artifact) against the canonical
// dequantizer and prove the 64-row tile width the executor uses is bit-
// identical to the 48-row width. No model artifact is required.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/gemm.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/routed_f16.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

// A row-major [rows x cols] Q8_0 matrix with its decoded scales and codes kept
// alongside so the host reference can reproduce the exact stored values.
struct Q8Matrix {
  std::vector<std::uint8_t> bytes;
  std::vector<__half> scales;
  std::vector<std::int8_t> codes;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;

  float At(std::uint32_t r, std::uint32_t k) const {
    const std::uint32_t nb = cols / 32;
    const float d = __half2float(scales[static_cast<std::size_t>(r) * nb + k / 32]);
    return d * static_cast<float>(codes[static_cast<std::size_t>(r) * cols + k]);
  }
};

Q8Matrix MakeQ8(std::uint32_t rows, std::uint32_t cols, std::uint32_t seed) {
  Q8Matrix m;
  m.rows = rows;
  m.cols = cols;
  const std::uint32_t nb = cols / 32;
  m.scales.assign(static_cast<std::size_t>(rows) * nb, __float2half(0.0F));
  m.codes.assign(static_cast<std::size_t>(rows) * cols, 0);
  m.bytes.reserve(static_cast<std::size_t>(rows) * nb * 34);
  for (std::size_t idx = 0; idx < m.scales.size(); ++idx) {
    const float scale =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    m.scales[idx] = __float2half(scale);
    const auto* sb = reinterpret_cast<const std::uint8_t*>(&m.scales[idx]);
    m.bytes.push_back(sb[0]);
    m.bytes.push_back(sb[1]);
    for (int i = 0; i < 32; ++i) {
      const std::int8_t code =
          static_cast<std::int8_t>(t::NextRandom(&seed) & 0xFFU);
      m.codes[idx * 32 + i] = code;
      m.bytes.push_back(static_cast<std::uint8_t>(code));
    }
  }
  return m;
}

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

// ---- Dense --------------------------------------------------------------

bool DenseQ8(std::uint32_t rows, std::uint32_t cols, std::uint32_t batch) {
  const Q8Matrix w = MakeQ8(rows, cols, 0x1234ABCDU);
  const auto x = t::MakeValues(static_cast<std::size_t>(batch) * cols,
                               0x0F0F0F0FU, 1.0F);
  t::HipBuffer<std::uint8_t> d_w(w.bytes.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.bytes.data(), w.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload q8_0");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(batch) * rows);
  q::Gemm(d_w.get(), q::GemvType::kQ8_0, rows, cols, 34, d_x.get(),
          d_out.get(), batch, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemmQ8_0 sync");
  const auto got = t::Download(&d_out, static_cast<std::size_t>(batch) * rows);

  std::vector<float> ref(static_cast<std::size_t>(batch) * rows);
  for (std::uint32_t tb = 0; tb < batch; ++tb) {
    for (std::uint32_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k) {
        acc += static_cast<double>(w.At(r, k)) *
               x[static_cast<std::size_t>(tb) * cols + k];
      }
      ref[static_cast<std::size_t>(tb) * rows + r] = static_cast<float>(acc);
    }
  }
  return Check("GemmQ8_0 dense", t::WorstRelativeToScale(ref, got), 1.5e-2);
}

bool DenseF32(std::uint32_t rows, std::uint32_t cols, std::uint32_t batch) {
  const auto w = t::MakeValues(static_cast<std::size_t>(rows) * cols,
                               0x55556666U, 1.0F);
  const auto x = t::MakeValues(static_cast<std::size_t>(batch) * cols,
                               0x77778888U, 1.0F);
  t::HipBuffer<float> d_w(w.size());
  t::Upload(&d_w, w);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(batch) * rows);
  q::Gemm(d_w.get(), q::GemvType::kF32, rows, cols, 4, d_x.get(), d_out.get(),
          batch, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemmF32 sync");
  const auto got = t::Download(&d_out, static_cast<std::size_t>(batch) * rows);

  std::vector<float> ref(static_cast<std::size_t>(batch) * rows);
  for (std::uint32_t tb = 0; tb < batch; ++tb) {
    for (std::uint32_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k) {
        acc += static_cast<double>(w[static_cast<std::size_t>(r) * cols + k]) *
               x[static_cast<std::size_t>(tb) * cols + k];
      }
      ref[static_cast<std::size_t>(tb) * rows + r] = static_cast<float>(acc);
    }
  }
  return Check("GemmF32 dense", t::WorstRelative(ref, got), 1e-3);
}

bool DenseBf16(std::uint32_t rows, std::uint32_t cols, std::uint32_t batch) {
  std::vector<std::uint16_t> w(static_cast<std::size_t>(rows) * cols);
  std::uint32_t seed = 0x9A8B7C6DU;
  for (auto& value : w) {
    const float f = 1.0F *
                    static_cast<float>(
                        static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                        32768) /
                    32768.0F;
    value = FloatToBf16Bits(f);
  }
  const auto x = t::MakeValues(static_cast<std::size_t>(batch) * cols,
                               0x2468ACE1U, 1.0F);
  t::HipBuffer<std::uint16_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), d_w.bytes(), hipMemcpyHostToDevice),
              "upload bf16");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(batch) * rows);
  q::Gemm(d_w.get(), q::GemvType::kBF16, rows, cols, 2, d_x.get(), d_out.get(),
          batch, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemmBf16 sync");
  const auto got = t::Download(&d_out, static_cast<std::size_t>(batch) * rows);

  std::vector<float> ref(static_cast<std::size_t>(batch) * rows);
  for (std::uint32_t tb = 0; tb < batch; ++tb) {
    for (std::uint32_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k) {
        acc += static_cast<double>(
                   Bf16BitsToFloat(w[static_cast<std::size_t>(r) * cols + k])) *
               x[static_cast<std::size_t>(tb) * cols + k];
      }
      ref[static_cast<std::size_t>(tb) * rows + r] = static_cast<float>(acc);
    }
  }
  return Check("GemmBf16 dense", t::WorstRelativeToScale(ref, got), 1.5e-2);
}

// ---- MoE ----------------------------------------------------------------

bool MoeQ8(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts,
           std::uint32_t used, std::uint32_t tokens) {
  const Q8Matrix w = MakeQ8(experts * rows, cols, 0x31415926U);
  const auto x = t::MakeValues(static_cast<std::size_t>(tokens) * cols,
                               0x27182818U, 1.0F);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * used);
  std::uint32_t seed = 0xABCDEF01U;
  for (auto& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&seed) % experts);
  }
  t::HipBuffer<std::uint8_t> d_w(w.bytes.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.bytes.data(), w.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload q8_0 moe");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<std::int32_t> d_ids(ids.size());
  t::CheckHip(hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(),
                        hipMemcpyHostToDevice),
              "upload ids");
  const std::size_t pairs = static_cast<std::size_t>(tokens) * used;
  t::HipBuffer<float> d_out(pairs * rows);
  q::GemmMoe(d_w.get(), q::GemvType::kQ8_0, rows, cols, 34, d_x.get(),
             d_ids.get(), d_out.get(), tokens, experts, used, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemmMoeQ8_0 sync");
  const auto got = t::Download(&d_out, pairs * rows);

  std::vector<float> ref(pairs * rows);
  for (std::uint32_t tb = 0; tb < tokens; ++tb) {
    for (std::uint32_t s = 0; s < used; ++s) {
      const std::uint32_t pair = tb * used + s;
      const std::uint32_t e = ids[pair];
      for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::uint32_t k = 0; k < cols; ++k) {
          acc += static_cast<double>(w.At(e * rows + r, k)) *
                 x[static_cast<std::size_t>(tb) * cols + k];
        }
        ref[static_cast<std::size_t>(pair) * rows + r] = static_cast<float>(acc);
      }
    }
  }
  return Check("GemmMoeQ8_0", t::WorstRelativeToScale(ref, got), 1.5e-2);
}

bool MoeBf16(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts,
             std::uint32_t used, std::uint32_t tokens) {
  std::vector<std::uint16_t> w(static_cast<std::size_t>(experts) * rows * cols);
  std::uint32_t seed = 0x13579BDFU;
  for (auto& value : w) {
    const float f = 1.0F *
                    static_cast<float>(
                        static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                        32768) /
                    32768.0F;
    value = FloatToBf16Bits(f);
  }
  const auto x = t::MakeValues(static_cast<std::size_t>(tokens) * cols,
                               0x9E3779B9U, 1.0F);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * used);
  for (auto& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&seed) % experts);
  }
  t::HipBuffer<std::uint16_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), d_w.bytes(), hipMemcpyHostToDevice),
              "upload bf16 moe");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<std::int32_t> d_ids(ids.size());
  t::CheckHip(hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(),
                        hipMemcpyHostToDevice),
              "upload ids");
  const std::size_t pairs = static_cast<std::size_t>(tokens) * used;
  t::HipBuffer<float> d_out(pairs * rows);
  q::GemmMoe(d_w.get(), q::GemvType::kBF16, rows, cols, 2, d_x.get(),
             d_ids.get(), d_out.get(), tokens, experts, used, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemmMoeBf16 sync");
  const auto got = t::Download(&d_out, pairs * rows);

  std::vector<float> ref(pairs * rows);
  for (std::uint32_t tb = 0; tb < tokens; ++tb) {
    for (std::uint32_t s = 0; s < used; ++s) {
      const std::uint32_t pair = tb * used + s;
      const std::uint32_t e = ids[pair];
      for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::uint32_t k = 0; k < cols; ++k) {
          acc += static_cast<double>(Bf16BitsToFloat(
                     w[(static_cast<std::size_t>(e) * rows + r) * cols + k])) *
                 x[static_cast<std::size_t>(tb) * cols + k];
        }
        ref[static_cast<std::size_t>(pair) * rows + r] = static_cast<float>(acc);
      }
    }
  }
  return Check("GemmMoeBf16", t::WorstRelative(ref, got), 2e-2);
}

// ---- Routed WMMA (Q4_K / Q5_K) ------------------------------------------

// Encode a random Q4_K matrix: per 256-element super-block a half d, half dmin,
// twelve packed (scale, minimum) bytes and 128 packed nibble bytes. The
// oracle's DequantizeQ4_K defines the semantics the WMMA kernel must reproduce.
std::vector<std::uint8_t> EncodeQ4K(std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t seed) {
  using gufo::quant::block_q4_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (cols / 256);
  std::vector<std::uint8_t> bytes(nblocks * sizeof(block_q4_K));
  auto* blk = reinterpret_cast<block_q4_K*>(bytes.data());
  for (std::size_t i = 0; i < nblocks; ++i) {
    const float d =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const float dmin =
        0.25F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const __half dh = __float2half(d);
    const __half dmh = __float2half(dmin);
    std::memcpy(&blk[i].d, &dh, sizeof(blk[i].d));
    std::memcpy(&blk[i].dmin, &dmh, sizeof(blk[i].dmin));
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
    const float d =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const float dmin =
        0.25F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const __half dh = __float2half(d);
    const __half dmh = __float2half(dmin);
    std::memcpy(&blk[i].d, &dh, sizeof(blk[i].d));
    std::memcpy(&blk[i].dmin, &dmh, sizeof(blk[i].dmin));
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

// The routed F16 WMMA expert GEMM (the prefill route for a Q4_K_XL artifact's
// routed experts) must match a CPU reference that dequantizes each selected
// expert row with the canonical oracle and dots in double, and must produce
// bit-identical output at the 64-row tile width the executor uses and the
// 48-row width the kernel was validated at.
bool RoutedQuant(bool q5k) {
  constexpr std::uint32_t kExperts = 8;
  constexpr std::uint32_t kUsed = 4;
  constexpr std::uint32_t kTokens = 40;
  constexpr std::uint32_t kM = 201;  // ragged m, exercises the partial tile
  constexpr std::uint32_t kK = 512;  // multiple of the 256-wide super-block
  const std::size_t slots = static_cast<std::size_t>(kTokens) * kUsed;
  const std::size_t block =
      q5k ? sizeof(gufo::quant::block_q5_K) : sizeof(gufo::quant::block_q4_K);
  const std::size_t row_blocks = kK / 256;

  // Skewed top-k without replacement so some experts see more than 48 rows and
  // the 64- and 48-row tile maps genuinely differ.
  std::vector<std::int32_t> ids(slots);
  std::uint32_t state = 0x5A5A5A5AU;
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < kUsed) {
      const std::uint32_t r = t::NextRandom(&state);
      const auto e = static_cast<std::int32_t>(
          (r % 3 == 0 ? r / 3 % 4 : r / 3) % kExperts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * kUsed);
  }
  std::vector<std::uint32_t> counts(kExperts, 0);
  for (std::int32_t e : ids) {
    ++counts[e];
  }

  const auto w = q5k ? EncodeQ5K(kExperts * kM, kK, 0x77AA0005U)
                     : EncodeQ4K(kExperts * kM, kK, 0x66AA0004U);
  const auto x =
      t::MakeValues(static_cast<std::size_t>(kTokens) * kK, 0x27182818U, 2.0F);
  std::vector<__half> x_half(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x_half[i] = __float2half(x[i]);
  }

  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload routed quant weights");
  t::HipBuffer<__half> d_xh(x_half.size());
  t::CheckHip(
      hipMemcpy(d_xh.get(), x_half.data(), d_xh.bytes(), hipMemcpyHostToDevice),
      "upload routed activations");
  t::HipBuffer<std::int32_t> d_ids(ids.size());
  t::CheckHip(
      hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(), hipMemcpyHostToDevice),
      "upload routed ids");
  t::HipBuffer<std::uint32_t> d_counts(counts.size());
  t::CheckHip(hipMemcpy(d_counts.get(), counts.data(), d_counts.bytes(),
                        hipMemcpyHostToDevice),
              "upload routed counts");

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  t::HipBuffer<std::int32_t> d_bounds(kExperts + 1);
  t::HipBuffer<std::int32_t> d_cursors(kExperts);
  t::HipBuffer<std::int32_t> d_rows_token(compact);
  t::HipBuffer<std::int32_t> d_rows_slot(compact);
  t::CheckHip(hipMemset(d_bounds.get(), 0, d_bounds.bytes()), "zero bounds");
  t::CheckHip(hipMemset(d_cursors.get(), 0, d_cursors.bytes()), "zero cursors");
  t::CheckHip(hipMemset(d_rows_token.get(), 0, d_rows_token.bytes()),
              "zero rows_token");
  t::CheckHip(hipMemset(d_rows_slot.get(), 0, d_rows_slot.bytes()),
              "zero rows_slot");
  q::RoutedCompact(d_ids.get(), d_counts.get(), d_bounds.get(), d_cursors.get(),
                   d_rows_token.get(), d_rows_slot.get(), kTokens, kUsed,
                   kExperts, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "RoutedCompact sync");

  const auto build_tiles = [&](std::uint32_t tr) {
    std::vector<std::int32_t> tiles;
    for (std::size_t e = 0; e < kExperts; ++e) {
      const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
      for (std::uint32_t j = 0; j < (padded + tr - 1) / tr; ++j) {
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
      }
    }
    return tiles;
  };
  const auto run = [&](std::uint32_t tr, std::vector<float>* out) {
    const auto tiles = build_tiles(tr);
    t::HipBuffer<std::int32_t> d_tiles(tiles.size());
    t::CheckHip(hipMemcpy(d_tiles.get(), tiles.data(), d_tiles.bytes(),
                          hipMemcpyHostToDevice),
                "upload tiles");
    t::HipBuffer<float> d_out(slots * kM);
    t::CheckHip(hipMemset(d_out.get(), 0, d_out.bytes()), "zero out");
    if (!q::RoutedF16Gemm(
            d_w.get(), q5k ? q::WeightType::kQ5_K : q::WeightType::kQ4_K,
            d_xh.get(), d_tiles.get(), static_cast<std::uint32_t>(tiles.size()),
            tr, d_bounds.get(), d_rows_token.get(), d_rows_slot.get(), nullptr,
            d_out.get(), nullptr, kM, kK, nullptr)) {
      throw std::runtime_error("routed F16 GEMM rejected the shape");
    }
    t::CheckHip(hipDeviceSynchronize(), "routed F16 GEMM sync");
    *out = t::Download(&d_out, slots * kM);
  };

  std::vector<float> out64;
  run(64, &out64);
  std::vector<float> out48;
  run(48, &out48);
  if (std::memcmp(out48.data(), out64.data(), out64.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("routed tile widths change output bits");
  }

  std::vector<float> ref(slots * kM);
  std::vector<float> deq(kK);
  for (std::size_t s = 0; s < slots; ++s) {
    const std::size_t tok = s / kUsed;
    const std::int32_t e = ids[s];
    for (std::uint32_t row = 0; row < kM; ++row) {
      const std::size_t row_index = static_cast<std::size_t>(e) * kM + row;
      const void* row_bytes = w.data() + row_index * row_blocks * block;
      if (q5k) {
        gufo::quant::DequantizeQ5_K(row_bytes, deq.data(), kK);
      } else {
        gufo::quant::DequantizeQ4_K(row_bytes, deq.data(), kK);
      }
      double acc = 0.0;
      const float* xr = x.data() + tok * kK;
      for (std::uint32_t i = 0; i < kK; ++i) {
        acc += static_cast<double>(deq[i]) * xr[i];
      }
      ref[s * kM + row] = static_cast<float>(acc);
    }
  }
  for (std::size_t i = 0; i < out64.size(); ++i) {
    if (!std::isfinite(out64[i])) {
      throw std::runtime_error("routed F16 output is not finite");
    }
  }
  return Check(q5k ? "RoutedF16Q5K" : "RoutedF16Q4K",
               t::WorstRelativeToScale(ref, out64), 2e-3);
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0) {
      std::cerr << "qfn_mmq_init failed\n";
      return 1;
    }
    bool ok = true;
    ok = DenseQ8(64, 2048, 5) && ok;
    ok = DenseF32(64, 2048, 5) && ok;
    ok = DenseBf16(64, 2048, 5) && ok;
    ok = MoeQ8(64, 2048, 8, 2, 4) && ok;
    ok = MoeBf16(64, 2048, 8, 2, 4) && ok;
    ok = RoutedQuant(false) && ok;
    ok = RoutedQuant(true) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
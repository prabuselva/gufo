// Batched (prefill) routed projection checks: synthetic Q4_K, Q5_K, Q6_K and
// Q8_0 expert matrices through the F16 WMMA tier against a CPU reference that
// dequantizes with the canonical ggml dequantizers (gufo_core) and dots in
// double. The Q6_K case proves the merged 6-bit code unpack, and the SwiGLU
// cases pin the epilogue: the tile widths must agree bit-for-bit, the limit
// must reproduce the reference Clamp(Silu(gate)) * Clamp(up), and the fused
// gate+up launch must agree with the separate launches. No model artifact is
// required.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/ling3_flash/kernels/rocm/routed_f16.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace q = gufo::models::ling3_flash::rocm;
namespace t = gufo::tests::ling3_flash;

namespace {

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

// Worst |ref - got| scaled by the largest |ref|. The F16 operands the WMMA
// fragments consume inject an absolute rounding floor into every dot; it is
// independent of how small one particular output is, so a per-element floor
// on tiny cancelled sums would measure the metric, not the kernel. A wrong
// decode shifts a full weight term (~1e-2 of the scale) and still trips.
double WorstScaled(const std::vector<float>& ref,
                   const std::vector<float>& got) {
  double scale = 0.0;
  for (const float r : ref) {
    if (std::isfinite(r)) {
      scale = std::max(scale, std::abs(static_cast<double>(r)));
    }
  }
  if (scale == 0.0) {
    scale = 1.0;
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
      // F16-out overflow is legitimate: matching infinities are exact.
      if (ref[i] != got[i]) {
        worst = 2.0;
      }
      continue;
    }
    worst =
        std::max(worst, std::abs(static_cast<double>(ref[i]) - got[i]) / scale);
  }
  return worst;
}

std::uint16_t HalfBits(float f) {
  const __half h = __float2half(f);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

// hip_test.hpp only downloads float buffers; the F16-out epilogue needs the
// half variant converted back to float.
std::vector<float> DownloadHalf(t::HipBuffer<__half>* source,
                                std::size_t count) {
  std::vector<__half> raw(count);
  t::CheckHip(hipMemcpy(raw.data(), source->get(), count * sizeof(__half),
                        hipMemcpyDeviceToHost),
              "download half");
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = __half2float(raw[i]);
  }
  return values;
}

constexpr std::uint32_t kExperts = 8;
constexpr std::uint32_t kUsed = 4;
constexpr std::uint32_t kTokens = 40;
constexpr std::uint32_t kM = 201;  // ragged m, exercises the partial tile
constexpr std::uint32_t kK = 512;  // multiple of the 256-wide super-block
constexpr std::size_t kSlots = static_cast<std::size_t>(kTokens) * kUsed;

// Skewed top-k without replacement so some experts see more than 48 rows and
// the different tile maps genuinely differ.
std::vector<std::int32_t> MakeIds() {
  std::vector<std::int32_t> ids(kSlots);
  std::uint32_t state = 0x5A5A5A5AU;
  for (std::uint32_t tok = 0; tok < kTokens; ++tok) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < kUsed) {
      const std::uint32_t r = t::NextRandom(&state);
      const auto e = static_cast<std::int32_t>(
          (r % 3 == 0 ? r / 3 % 4 : r / 3) % kExperts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + tok * kUsed);
  }
  return ids;
}

// ---- Encoders ------------------------------------------------------------

std::vector<std::uint8_t> EncodeQ4K(std::uint32_t rows, std::uint32_t seed) {
  using gufo::quant::block_q4_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (kK / 256);
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
    const std::uint16_t dh = HalfBits(d);
    const std::uint16_t dmh = HalfBits(dmin);
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

std::vector<std::uint8_t> EncodeQ5K(std::uint32_t rows, std::uint32_t seed) {
  using gufo::quant::block_q5_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (kK / 256);
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
    const std::uint16_t dh = HalfBits(d);
    const std::uint16_t dmh = HalfBits(dmin);
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

std::vector<std::uint8_t> EncodeQ6K(std::uint32_t rows, std::uint32_t seed) {
  using gufo::quant::block_q6_K;
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (kK / 256);
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
    const std::uint16_t dh =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    std::memcpy(&blk[i].d, &dh, sizeof(blk[i].d));
  }
  return bytes;
}

std::vector<std::uint8_t> EncodeQ8(std::uint32_t rows, std::uint32_t seed) {
  const std::size_t nblocks = static_cast<std::size_t>(rows) * (kK / 32);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(nblocks * 34);
  for (std::size_t i = 0; i < nblocks; ++i) {
    const std::uint16_t dh =
        HalfBits(0.5F *
                 static_cast<float>(
                     static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) - 32768) /
                 32768.0F);
    bytes.push_back(static_cast<std::uint8_t>(dh & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>(dh >> 8));
    for (int s = 0; s < 32; ++s) {
      bytes.push_back(static_cast<std::uint8_t>(t::NextRandom(&seed) & 0xFFU));
    }
  }
  return bytes;
}

std::size_t RowBytesCount(q::WeightType type) {
  switch (type) {
    case q::WeightType::kQ4_K:
      return (kK / 256) * sizeof(gufo::quant::block_q4_K);
    case q::WeightType::kQ5_K:
      return (kK / 256) * sizeof(gufo::quant::block_q5_K);
    case q::WeightType::kQ6_K:
      return (kK / 256) * sizeof(gufo::quant::block_q6_K);
    default:
      return (kK / 32) * 34;
  }
}

void DequantizeRow(q::WeightType type, const void* row, float* dst) {
  switch (type) {
    case q::WeightType::kQ4_K:
      gufo::quant::DequantizeQ4_K(row, dst, kK);
      break;
    case q::WeightType::kQ5_K:
      gufo::quant::DequantizeQ5_K(row, dst, kK);
      break;
    case q::WeightType::kQ6_K:
      gufo::quant::DequantizeQ6_K(row, dst, kK);
      break;
    default:
      gufo::quant::DequantizeQ8_0(row, dst, kK);
      break;
  }
}

const char* TypeName(q::WeightType type) {
  switch (type) {
    case q::WeightType::kQ4_K:
      return "RoutedF16Q4K";
    case q::WeightType::kQ5_K:
      return "RoutedF16Q5K";
    case q::WeightType::kQ6_K:
      return "RoutedF16Q6K";
    default:
      return "RoutedF16Q8_0";
  }
}

// ---- Compaction + launches -----------------------------------------------

struct RoutedEnv {
  std::vector<std::int32_t> ids;
  std::vector<std::uint32_t> counts;
  std::size_t compact = 0;
  t::HipBuffer<std::int32_t> d_ids;
  t::HipBuffer<std::uint32_t> d_counts;
  t::HipBuffer<std::int32_t> d_bounds;
  t::HipBuffer<std::int32_t> d_cursors;
  t::HipBuffer<std::int32_t> d_rows_token;
  t::HipBuffer<std::int32_t> d_rows_slot;
  t::HipBuffer<__half> d_xh;
  std::vector<float> x;

  explicit RoutedEnv(std::uint32_t x_scale)
      : ids(MakeIds()),
        counts(kExperts, 0),
        compact(q::RoutedCompactRows(kSlots, kExperts)),
        d_ids(ids.size()),
        d_counts(counts.size()),
        d_bounds(kExperts + 1),
        d_cursors(kExperts),
        d_rows_token(compact),
        d_rows_slot(compact),
        d_xh(static_cast<std::size_t>(kTokens) * kK),
        x(t::MakeValues(static_cast<std::size_t>(kTokens) * kK, 0x27182818U,
                        x_scale)) {
    for (std::int32_t e : ids) {
      ++counts[e];
    }
    std::vector<__half> x_half(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x_half[i] = __float2half(x[i]);
      // Feed the oracle the half-rounded activations the kernel actually
      // multiplies with, so the metric measures kernel error only.
      x[i] = __half2float(x_half[i]);
    }
    t::CheckHip(hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(),
                          hipMemcpyHostToDevice),
                "upload routed ids");
    t::CheckHip(hipMemcpy(d_counts.get(), counts.data(), d_counts.bytes(),
                          hipMemcpyHostToDevice),
                "upload routed counts");
    t::CheckHip(hipMemcpy(d_xh.get(), x_half.data(), d_xh.bytes(),
                          hipMemcpyHostToDevice),
                "upload routed activations");
    t::CheckHip(hipMemset(d_bounds.get(), 0, d_bounds.bytes()), "zero bounds");
    t::CheckHip(hipMemset(d_cursors.get(), 0, d_cursors.bytes()),
                "zero cursors");
    q::RoutedCompact(d_ids.get(), d_counts.get(), d_bounds.get(),
                     d_cursors.get(), d_rows_token.get(), d_rows_slot.get(),
                     kTokens, kUsed, kExperts, nullptr);
    t::CheckHip(hipDeviceSynchronize(), "RoutedCompact sync");
  }

  std::vector<std::int32_t> BuildTiles(std::uint32_t tr) const {
    std::vector<std::int32_t> tiles;
    for (std::size_t e = 0; e < kExperts; ++e) {
      const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
      for (std::uint32_t j = 0; j < (padded + tr - 1U) / tr; ++j) {
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
      }
    }
    return tiles;
  }
};

struct DeviceTiles {
  t::HipBuffer<std::int32_t> buf;
  std::uint32_t count = 0;
  explicit DeviceTiles(const std::vector<std::int32_t>& tiles)
      : buf(tiles.size()), count(static_cast<std::uint32_t>(tiles.size())) {
    t::CheckHip(
        hipMemcpy(buf.get(), tiles.data(), buf.bytes(), hipMemcpyHostToDevice),
        "upload tiles");
  }
};

// A CPU double dot for expert e's row against token tok's (dequantized) row.
void ExpertRef(const std::vector<std::uint8_t>& w, q::WeightType type,
               const std::vector<float>& x, std::int32_t e, std::size_t tok,
               std::uint32_t row, float* dst) {
  std::vector<float> deq(kK);
  const std::size_t row_index = static_cast<std::size_t>(e) * kM + row;
  const void* row_bytes = w.data() + row_index * RowBytesCount(type);
  DequantizeRow(type, row_bytes, deq.data());
  // The kernel hands the tensor cores F16-rounded weights: reproduce that
  // rounding so the dot matches the operands the WMMA fragments actually see.
  for (std::uint32_t i = 0; i < kK; ++i) {
    deq[i] = __half2float(__float2half(deq[i]));
  }
  double acc = 0.0;
  const float* xr = x.data() + tok * kK;
  for (std::uint32_t i = 0; i < kK; ++i) {
    acc += static_cast<double>(deq[i]) * xr[i];
  }
  *dst = static_cast<float>(acc);
}

// Width equivalence + oracle reference for every quantized type.
bool RoutedQuant(q::WeightType type, std::uint32_t seed) {
  const std::uint32_t rows = kExperts * kM;
  std::vector<std::uint8_t> w;
  switch (type) {
    case q::WeightType::kQ4_K:
      w = EncodeQ4K(rows, seed);
      break;
    case q::WeightType::kQ5_K:
      w = EncodeQ5K(rows, seed);
      break;
    case q::WeightType::kQ6_K:
      w = EncodeQ6K(rows, seed);
      break;
    default:
      w = EncodeQ8(rows, seed);
      break;
  }
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload routed weights");
  RoutedEnv env(2.0F);

  const auto run = [&](std::uint32_t tr) {
    DeviceTiles tiles(env.BuildTiles(tr));
    t::HipBuffer<float> d_out(kSlots * kM);
    // 0x7F7F7F7F is a NaN: outputs left at the sentinel prove the kernel
    // never wrote them, while zeros would mean it ran with zeroed weights.
    t::CheckHip(hipMemset(d_out.get(), 0x7F, d_out.bytes()), "sentinel out");
    if (!q::RoutedF16Gemm(
            d_w.get(), type, env.d_xh.get(), tiles.buf.get(), tiles.count, tr,
            env.d_bounds.get(), env.d_rows_token.get(), env.d_rows_slot.get(),
            nullptr, 0.0F, d_out.get(), nullptr, kM, kK, 0, nullptr)) {
      throw std::runtime_error("routed F16 GEMM rejected the shape");
    }
    t::CheckHip(hipGetLastError(), "routed F16 GEMM launch");
    t::CheckHip(hipDeviceSynchronize(), "routed F16 GEMM sync");
    return t::Download(&d_out, kSlots * kM);
  };

  const auto out64 = run(64);
  const auto out48 = run(48);
  const auto out16 = run(16);
  if (std::memcmp(out48.data(), out64.data(), out64.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("routed 48-row width changes output bits");
  }
  if (std::memcmp(out16.data(), out64.data(), out64.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("routed 16-row width changes output bits");
  }

  std::vector<float> ref(kSlots * kM);
  for (std::size_t s = 0; s < kSlots; ++s) {
    const std::size_t tok = s / kUsed;
    for (std::uint32_t row = 0; row < kM; ++row) {
      ExpertRef(w, type, env.x, env.ids[s], tok, row, &ref[s * kM + row]);
    }
  }
  std::size_t unwritten = 0;
  for (std::size_t i = 0; i < out64.size(); ++i) {
    if (!std::isfinite(out64[i])) {
      ++unwritten;
    }
  }
  if (unwritten != 0) {
    throw std::runtime_error(std::string(TypeName(type)) + ": " +
                             std::to_string(unwritten) + "/" +
                             std::to_string(out64.size()) +
                             " outputs kept the no-write NaN sentinel");
  }
  return Check(TypeName(type), WorstScaled(ref, out64), 2e-3);
}

// Reference SwiGLU in double: Clamp(Silu(gate), -inf, limit) * Clamp(up).
double SwigluRef(double up, double gate, double limit) {
  double silu = gate / (1.0 + std::exp(-gate));
  double up_ref = up;
  if (limit > 0.0) {
    silu = std::min(silu, limit);
    up_ref = std::min(std::max(up, -limit), limit);
  }
  return up_ref * silu;
}

// The SwiGLU epilogue: an F16-out up projection gated by the gate GEMM's F32
// output. The epilogue is judged against the kernel's own GEMM results (the
// gate F32 buffer and an identical ungated up F32 launch feeding the same
// accumulators) so the product/clamp arithmetic is isolated from the dot
// rounding, which RoutedQuant already checks. The unclamped variant runs with
// small activations so F16 output stays far from overflow and no product sits
// on the overflow boundary.
bool RoutedSwiglu(float limit) {
  constexpr std::uint32_t kSeedG = 0x4AF00001U;
  constexpr std::uint32_t kSeedU = 0xBEEF0002U;
  const auto gate_w = EncodeQ4K(kExperts * kM, kSeedG);
  const auto up_w = EncodeQ4K(kExperts * kM, kSeedU);
  t::HipBuffer<std::uint8_t> d_gate(gate_w.size());
  t::HipBuffer<std::uint8_t> d_up(up_w.size());
  t::CheckHip(hipMemcpy(d_gate.get(), gate_w.data(), gate_w.size(),
                        hipMemcpyHostToDevice),
              "upload gate weights");
  t::CheckHip(
      hipMemcpy(d_up.get(), up_w.data(), up_w.size(), hipMemcpyHostToDevice),
      "upload up weights");
  RoutedEnv env(limit > 0.0F ? 1.0F : 0.01F);
  DeviceTiles tiles(env.BuildTiles(64));
  DeviceTiles tiles16(env.BuildTiles(16));

  // Ungated F32 launch: the reference factors, with the gate buffer kept on
  // device for the epilogue launches.
  const auto launch_f32 = [&](t::HipBuffer<std::uint8_t>& weights,
                              t::HipBuffer<float>& d_out, const char* name) {
    t::CheckHip(hipMemset(d_out.get(), 0x7F, d_out.bytes()), name);
    if (!q::RoutedF16Gemm(weights.get(), q::WeightType::kQ4_K, env.d_xh.get(),
                          tiles.buf.get(), tiles.count, 64, env.d_bounds.get(),
                          env.d_rows_token.get(), env.d_rows_slot.get(),
                          nullptr, 0.0F, d_out.get(), nullptr, kM, kK, 0,
                          nullptr)) {
      throw std::runtime_error(std::string(name) + " GEMM rejected the shape");
    }
    t::CheckHip(hipGetLastError(), name);
    t::CheckHip(hipDeviceSynchronize(), name);
    auto values = t::Download(&d_out, kSlots * kM);
    for (const float v : values) {
      if (!std::isfinite(v)) {
        throw std::runtime_error(std::string(name) +
                                 " GEMM left slot bits unwritten");
      }
    }
    return values;
  };

  // Gate and ungated up run first into F32 exactly as the executor does.
  t::HipBuffer<float> d_gate_out(kSlots * kM);
  t::HipBuffer<float> d_up_out(kSlots * kM);
  const auto gate_ref = launch_f32(d_gate, d_gate_out, "gate");
  const auto up_ref = launch_f32(d_up, d_up_out, "up reference");

  const auto run_up = [&](DeviceTiles& tk, std::uint32_t tr,
                          float clamp_limit) {
    t::HipBuffer<__half> d_out(kSlots * kM);
    t::CheckHip(hipMemset(d_out.get(), 0, d_out.bytes()), "zero out_half");
    if (!q::RoutedF16Gemm(d_up.get(), q::WeightType::kQ4_K, env.d_xh.get(),
                          tk.buf.get(), tk.count, tr, env.d_bounds.get(),
                          env.d_rows_token.get(), env.d_rows_slot.get(),
                          d_gate_out.get(), clamp_limit, nullptr, d_out.get(),
                          kM, kK, 0, nullptr)) {
      throw std::runtime_error("up GEMM rejected the shape");
    }
    t::CheckHip(hipGetLastError(), "up GEMM launch");
    t::CheckHip(hipDeviceSynchronize(), "up GEMM sync");
    return DownloadHalf(&d_out, kSlots * kM);
  };

  const auto got64 = run_up(tiles, 64, limit);
  const auto got16 = run_up(tiles16, 16, limit);
  if (std::memcmp(got64.data(), got16.data(), got64.size() * sizeof(float)) !=
      0) {
    throw std::runtime_error("SwiGLU epilogues differ between tile widths");
  }

  // Epilogue formula reference: the kernel's own F32 factors through double
  // SwiGLU, rounded to the F16 the epilogue writes. Residual disagreement is
  // at most one F16 ulp where the kernel's fast SiluF and the double silu
  // straddle an F16 rounding boundary, so the tolerance is 1e-2 of scale.
  std::vector<float> ref(kSlots * kM);
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ref[i] = __half2float(__float2half(static_cast<float>(SwigluRef(
        static_cast<double>(up_ref[i]), static_cast<double>(gate_ref[i]),
        static_cast<double>(limit)))));
  }
  return Check(limit > 0.0F ? "RoutedSwigluClamped" : "RoutedSwigluPlain",
               WorstScaled(ref, got64), 1e-2);
}

// The fused gate+up launch must match the separate gate (F32) + up (SwiGLU)
// launches bit-for-bit, which is what lets the prefill pipeline swap them.
bool RoutedGatedPair(float limit) {
  const auto gate_w = EncodeQ4K(kExperts * kM, 0x11223344U);
  const auto up_w = EncodeQ4K(kExperts * kM, 0x55667788U);
  t::HipBuffer<std::uint8_t> d_gate(gate_w.size());
  t::HipBuffer<std::uint8_t> d_up(up_w.size());
  t::CheckHip(hipMemcpy(d_gate.get(), gate_w.data(), gate_w.size(),
                        hipMemcpyHostToDevice),
              "upload pair gate weights");
  t::CheckHip(
      hipMemcpy(d_up.get(), up_w.data(), up_w.size(), hipMemcpyHostToDevice),
      "upload pair up weights");
  RoutedEnv env(1.0F);
  DeviceTiles tiles(env.BuildTiles(64));

  t::HipBuffer<float> d_gate_out(kSlots * kM);
  if (!q::RoutedF16Gemm(d_gate.get(), q::WeightType::kQ4_K, env.d_xh.get(),
                        tiles.buf.get(), tiles.count, 64, env.d_bounds.get(),
                        env.d_rows_token.get(), env.d_rows_slot.get(), nullptr,
                        0.0F, d_gate_out.get(), nullptr, kM, kK, 0, nullptr)) {
    throw std::runtime_error("pair gate GEMM rejected the shape");
  }
  t::HipBuffer<__half> d_sep(kSlots * kM);
  t::CheckHip(hipMemset(d_sep.get(), 0, d_sep.bytes()), "zero separate out");
  if (!q::RoutedF16Gemm(d_up.get(), q::WeightType::kQ4_K, env.d_xh.get(),
                        tiles.buf.get(), tiles.count, 64, env.d_bounds.get(),
                        env.d_rows_token.get(), env.d_rows_slot.get(),
                        d_gate_out.get(), limit, nullptr, d_sep.get(), kM, kK,
                        0, nullptr)) {
    throw std::runtime_error("pair up GEMM rejected the shape");
  }
  t::CheckHip(hipGetLastError(), "pair GEMM launch");
  t::CheckHip(hipDeviceSynchronize(), "separate launches sync");

  t::HipBuffer<__half> d_pair(kSlots * kM);
  t::CheckHip(hipMemset(d_pair.get(), 0, d_pair.bytes()), "zero pair out");
  if (!q::RoutedGatedF16Gemm(d_gate.get(), d_up.get(), q::WeightType::kQ4_K,
                             env.d_xh.get(), tiles.buf.get(), tiles.count, 64,
                             env.d_bounds.get(), env.d_rows_token.get(),
                             env.d_rows_slot.get(), limit, d_pair.get(), kM, kK,
                             nullptr)) {
    throw std::runtime_error("fused gate+up GEMM rejected the shape");
  }
  t::CheckHip(hipDeviceSynchronize(), "fused pair sync");

  const auto sep = DownloadHalf(&d_sep, kSlots * kM);
  const auto pair = DownloadHalf(&d_pair, kSlots * kM);
  const double diffs =
      std::memcmp(sep.data(), pair.data(), pair.size() * sizeof(float)) == 0
          ? 0.0
          : 1.0;
  return Check(limit > 0.0F ? "RoutedGatedPairClamped" : "RoutedGatedPairPlain",
               diffs, 0.0);
}

// ---- Dense token tier -----------------------------------------------------

// The identity-bucket dense projection against the same double-dot oracle:
// F32-out rows at a padded out_stride must land exactly, the half epilogue
// must agree, and the wrapper must reject an unsupported k.
bool DensePlain(std::uint32_t tokens, q::WeightType type, std::uint32_t seed) {
  std::vector<std::uint8_t> w;
  switch (type) {
    case q::WeightType::kQ4_K:
      w = EncodeQ4K(kM, seed);
      break;
    case q::WeightType::kQ5_K:
      w = EncodeQ5K(kM, seed);
      break;
    case q::WeightType::kQ6_K:
      w = EncodeQ6K(kM, seed);
      break;
    default:
      w = EncodeQ8(kM, seed);
      break;
  }
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload dense weights");
  auto x =
      t::MakeValues(static_cast<std::size_t>(tokens) * kK, seed + 1U, 0.5F);
  std::vector<__half> x_half(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x_half[i] = __float2half(x[i]);
    // The oracle dots the half-rounded activations the kernel multiplies.
    x[i] = __half2float(x_half[i]);
  }
  t::HipBuffer<float> d_xf(static_cast<std::size_t>(tokens) * kK);
  t::HipBuffer<__half> d_xh(static_cast<std::size_t>(tokens) * kK);
  t::Upload(&d_xf, x);
  q::NarrowActivations(d_xf.get(), d_xh.get(), false, x.size(), nullptr);

  const std::uint32_t padded = (tokens + 15U) / 16U * 16U;
  t::HipBuffer<std::int32_t> d_rows(padded), d_bounds(2), d_tiles(padded / 16U);
  q::DenseRowPrepare(d_rows.get(), d_bounds.get(), d_tiles.get(), tokens,
                     nullptr);
  t::CheckHip(hipDeviceSynchronize(), "DenseRowPrepare sync");

  std::vector<float> ref(static_cast<std::size_t>(tokens) * kM);
  for (std::uint32_t tok = 0; tok < tokens; ++tok) {
    for (std::uint32_t row = 0; row < kM; ++row) {
      ExpertRef(w, type, x, 0, tok, row,
                &ref[static_cast<std::size_t>(tok) * kM + row]);
    }
  }

  const auto run = [&](std::size_t out_stride) {
    const std::size_t plane =
        out_stride == 0 ? static_cast<std::size_t>(kM) : out_stride;
    t::HipBuffer<float> d_out(static_cast<std::size_t>(tokens) * plane);
    t::CheckHip(hipMemset(d_out.get(), 0x7F, d_out.bytes()),
                "sentinel dense out");
    if (!q::DenseF16Gemm(d_w.get(), type, d_xh.get(), d_rows.get(),
                         d_bounds.get(), d_tiles.get(), tokens, d_out.get(),
                         nullptr, kM, kK, out_stride, nullptr)) {
      throw std::runtime_error("dense F16 GEMM rejected the shape");
    }
    t::CheckHip(hipGetLastError(), "dense F16 GEMM launch");
    t::CheckHip(hipDeviceSynchronize(), "dense F16 GEMM sync");
    return std::make_pair(
        t::Download(&d_out, static_cast<std::size_t>(tokens) * plane), plane);
  };

  const std::string name =
      std::string(TypeName(type)) + "Dense" + std::to_string(tokens);
  const auto plain = run(0);
  bool ok = Check(name.c_str(), WorstScaled(ref, plain.first), 2e-3);

  // The same launch scattered into a wider plane: rows must land at the
  // requested stride and the padding must keep the no-write sentinel.
  const std::size_t wide = static_cast<std::size_t>(kM) + 8;
  const auto strided = run(wide);
  std::vector<float> packed(static_cast<std::size_t>(tokens) * kM);
  std::vector<float> pad_gaps;
  for (std::uint32_t tok = 0; tok < tokens; ++tok) {
    for (std::uint32_t row = 0; row < kM; ++row) {
      packed[static_cast<std::size_t>(tok) * kM + row] =
          strided.first[static_cast<std::size_t>(tok) * wide + row];
    }
    for (std::uint32_t row = kM; row < wide; ++row) {
      pad_gaps.push_back(
          strided.first[static_cast<std::size_t>(tok) * wide + row]);
    }
  }
  ok = Check(std::string(name + "/stride").c_str(), WorstScaled(ref, packed),
             2e-3) &&
       ok;
  bool untouched = true;
  for (const float v : pad_gaps) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    untouched = untouched && bits == 0x7F7F7F7FU;
  }
  std::cout << name
            << "/stride padding kept sentinel: " << (untouched ? "yes" : "NO")
            << '\n';
  ok = untouched && ok;

  // F16 output plane for one mid-size chunk (the narrow epilogue path).
  {
    const std::size_t plane = static_cast<std::size_t>(kM) + 8;
    t::HipBuffer<__half> d_h(static_cast<std::size_t>(tokens) * plane);
    t::CheckHip(hipMemset(d_h.get(), 0, d_h.bytes()), "zero dense half");
    if (!q::DenseF16Gemm(d_w.get(), type, d_xh.get(), d_rows.get(),
                         d_bounds.get(), d_tiles.get(), tokens, nullptr,
                         d_h.get(), kM, kK, plane, nullptr)) {
      throw std::runtime_error("dense F16-out GEMM rejected the shape");
    }
    t::CheckHip(hipDeviceSynchronize(), "dense half sync");
    const auto half_out =
        DownloadHalf(&d_h, static_cast<std::size_t>(tokens) * plane);
    std::vector<float> packed_h(static_cast<std::size_t>(tokens) * kM);
    for (std::uint32_t tok = 0; tok < tokens; ++tok) {
      for (std::uint32_t row = 0; row < kM; ++row) {
        packed_h[static_cast<std::size_t>(tok) * kM + row] =
            half_out[static_cast<std::size_t>(tok) * plane + row];
      }
    }
    ok = Check(std::string(name + "/half").c_str(), WorstScaled(ref, packed_h),
               4e-3) &&
         ok;
  }
  return ok;
}

// The wrapper contract: zero tokens and a k that is not a whole quant block
// are rejected instead of launching a partial dot.
bool DenseRejects() {
  const auto w = EncodeQ4K(kM, 0xD1000001U);
  t::HipBuffer<std::uint8_t> d_w(w.size());
  t::CheckHip(hipMemcpy(d_w.get(), w.data(), w.size(), hipMemcpyHostToDevice),
              "upload reject weights");
  t::HipBuffer<__half> d_xh(static_cast<std::size_t>(16) * kK);
  t::HipBuffer<std::int32_t> d_rows(16), d_bounds(2), d_tiles(1);
  t::HipBuffer<float> d_out(16 * kM);
  q::DenseRowPrepare(d_rows.get(), d_bounds.get(), d_tiles.get(), 16, nullptr);
  bool ok = true;
  if (q::DenseF16Gemm(d_w.get(), q::WeightType::kQ4_K, d_xh.get(), d_rows.get(),
                      d_bounds.get(), d_tiles.get(), 0, d_out.get(), nullptr,
                      kM, kK, 0, nullptr)) {
    std::cout << "DenseRejects: tokens=0 was accepted\n";
    ok = false;
  }
  if (q::DenseF16Gemm(d_w.get(), q::WeightType::kQ4_K, d_xh.get(), d_rows.get(),
                      d_bounds.get(), d_tiles.get(), 16, d_out.get(), nullptr,
                      kM, kK + 8, 0, nullptr)) {
    std::cout << "DenseRejects: k not a multiple of 256 was accepted\n";
    ok = false;
  }
  std::cout << "DenseRejects " << (ok ? "passed" : "FAILED") << '\n';
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  try {
    ok &= RoutedQuant(q::WeightType::kQ4_K, 0x04040404U);
    ok &= RoutedQuant(q::WeightType::kQ5_K, 0x05050505U);
    ok &= RoutedQuant(q::WeightType::kQ6_K, 0x06060606U);
    ok &= RoutedQuant(q::WeightType::kQ8_0, 0x08080808U);
    ok &= RoutedSwiglu(0.0F);
    ok &= RoutedSwiglu(4.0F);
    ok &= RoutedGatedPair(0.0F);
    ok &= RoutedGatedPair(7.0F);
    ok &= DensePlain(3, q::WeightType::kQ4_K, 0xD0000001U);
    ok &= DensePlain(20, q::WeightType::kQ4_K, 0xD0000002U);
    ok &= DensePlain(48, q::WeightType::kQ4_K, 0xD0000003U);
    ok &= DensePlain(130, q::WeightType::kQ4_K, 0xD0000004U);
    ok &= DensePlain(3, q::WeightType::kQ8_0, 0xD0000005U);
    ok &= DensePlain(49, q::WeightType::kQ8_0, 0xD0000006U);
    ok &= DensePlain(20, q::WeightType::kQ5_K, 0xD0000007U);
    ok &= DensePlain(130, q::WeightType::kQ6_K, 0xD0000008U);
    ok &= DenseRejects();
  } catch (const std::exception& error) {
    std::cerr << "ling3_flash rocm_gemm: " << error.what() << '\n';
    return 1;
  }
  return ok ? 0 : 1;
}

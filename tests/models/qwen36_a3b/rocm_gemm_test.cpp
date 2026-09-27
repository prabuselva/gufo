// Batched (prefill) projection checks: synthetic Q8_0, F32 and BF16 matrices
// through the GEMM tier against a CPU reference that dequantizes and dots in
// double. The Q8_0 paths run on the shared mmq tensor-core route (which
// quantizes the activations to Q8_1), so their tolerance is looser than the
// exact-activation F32 path. No model artifact is required.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "qfn_mmq.h"
#include "src/models/qwen36_a3b/kernels/rocm/gemm.hpp"
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
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
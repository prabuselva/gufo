// MMQ/MMVQ operator checks for the vendored nmb_ library: synthetic IQ2_XXS
// and NVFP4 matrices through the vector tier (dense-vec and routed moe-vec)
// and the MMQ tier against a CPU reference that dequantizes with the scalar
// oracle's own model-private dequantizers and dots in double. The GPU result
// quantizes activations to Q8_1, so the tolerance is the Q8_1 floor (3%),
// matching tools/nvfp4_probe.hip. No model artifact required.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "nmb_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron_h_moe/cpu_ops.hpp"
#include "src/models/nemotron_h_moe/weights.hpp"
#include "tests/models/nemotron_h_moe/hip_test.hpp"

namespace {

using gufo::models::nemotron_h_moe::TensorRef;
using gufo::models::nemotron_h_moe::cpu::DequantizeRow;
using namespace gufo::tests::nemotron_h_moe;

constexpr double kTolerance = 0.03;  // Q8_1 activation-quant floor.

// Fills `count` bytes with deterministic patterns that are valid for the
// given model-private format (finite scales).
std::vector<std::uint8_t> MakeWeightBytes(int kind, int cols, int matrices,
                                          std::uint32_t seed) {
  const std::size_t block_elems = kind == 0 ? 256 : (kind == 1 ? 64 : 32);
  const std::size_t block_bytes = kind == 0 ? 66 : (kind == 1 ? 36 : 34);
  const std::size_t per_row = cols / block_elems * block_bytes;
  std::vector<std::uint8_t> bytes(per_row * matrices);
  for (std::uint8_t& byte : bytes) {
    byte = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFU);
  }
  for (std::size_t i = 0; i < per_row * matrices; ++i) {
    if (kind != 1 && i % block_bytes == 0) {
      // fp16 scale head (IQ2_XXS / Q8_0): keep a small positive finite value.
      bytes[i] = 0x30 | static_cast<std::uint8_t>(bytes[i] & 0x01U);
    } else if (kind == 1 && i % 36 < 4) {
      // block_nvfp4: four UE4M3 sub-scales head every 36-byte block; clamp
      // off the NaN/Inf code points.
      bytes[i] &= 0x7EU;
    }
  }
  return bytes;
}

TensorRef MakeRef(std::vector<std::uint8_t>* bytes, int kind, int cols,
                  int rows, int experts) {
  TensorRef t;
  t.data = bytes->data();
  t.type = kind == 0   ? gufo::core::GgmlType::kIQ2_XXS
           : kind == 1 ? gufo::core::GgmlType::kNVFP4
                       : gufo::core::GgmlType::kQ8_0;
  t.cols = static_cast<std::uint64_t>(cols);
  t.rows = static_cast<std::uint64_t>(rows);
  t.experts = static_cast<std::uint64_t>(experts);
  return t;
}

// reference[t * rows + r] = dot(dequant(expert_of(t), row r), x[t])
std::vector<float> Reference(const TensorRef& w, const std::vector<float>& x,
                             int tokens,
                             const std::vector<std::int32_t>& expert_of) {
  const int cols = static_cast<int>(w.cols);
  const int rows = static_cast<int>(w.rows);
  std::vector<float> reference(static_cast<std::size_t>(tokens) * rows, 0.0F);
  std::vector<float> row(cols);
  for (int t = 0; t < tokens; ++t) {
    for (int r = 0; r < rows; ++r) {
      DequantizeRow(w, static_cast<std::uint64_t>(expert_of[t]),
                    static_cast<std::uint64_t>(r), row.data());
      double sum = 0.0;
      for (int k = 0; k < cols; ++k) {
        sum += static_cast<double>(row[k]) *
               static_cast<double>(x[static_cast<std::size_t>(t) * cols + k]);
      }
      reference[static_cast<std::size_t>(t) * rows + r] =
          static_cast<float>(sum);
    }
  }
  return reference;
}

bool Compare(const std::vector<float>& reference, const std::vector<float>& gpu,
             int rows, int tokens, const char* name) {
  // Error measured against the largest reference entry, as nvfp4_probe does:
  // individual dots can vanish, where a per-element relative error would
  // punish the Q8_1 quantization noise far beyond its actual size.
  double scale = 0.0;
  for (const float v : reference) {
    scale = std::max(scale, std::abs(static_cast<double>(v)));
  }
  double worst = 0.0;
  for (const bool token_major : {true, false}) {
    double abs_err = 0.0;
    for (int t = 0; t < tokens; ++t) {
      for (int r = 0; r < rows; ++r) {
        const std::size_t ref_i = static_cast<std::size_t>(t) * rows + r;
        const std::size_t gpu_i =
            token_major ? ref_i : static_cast<std::size_t>(r) * tokens + t;
        abs_err =
            std::max(abs_err, std::abs(static_cast<double>(reference[ref_i]) -
                                       static_cast<double>(gpu[gpu_i])));
      }
    }
    worst = worst == 0.0 ? abs_err : std::min(worst, abs_err);
  }
  const double rel = worst / std::max(scale, 1e-30);
  const bool ok = rel < kTolerance;
  std::printf("%-44s %s  rel/scale=%.3e\n", name, ok ? "PASS" : "FAIL", rel);
  return ok;
}

bool RunFormat(int kind, const char* fmt) {
  bool ok = true;
  const int cols = 512;
  std::uint32_t seed = kind == 0 ? 0xB017u : kind == 1 ? 0xC0FFEu : 0xD0D0Fu;

  // Dense vector tier: N activations, one matrix.
  for (const int rows : {128, 512}) {
    for (const int tokens : {1, 2, 3, 4, 8}) {
      auto bytes = MakeWeightBytes(kind, cols, rows, seed);
      auto x =
          MakeValues(static_cast<std::size_t>(tokens) * cols, seed ^ 5, 1.0F);
      TensorRef w = MakeRef(&bytes, kind, cols, rows, 1);
      HipBuffer<std::uint8_t> w_dev(bytes.size());
      HipBuffer<float> x_dev(x.size());
      HipBuffer<float> out_dev(static_cast<std::size_t>(rows) * tokens);
      Upload(w_dev.get(), bytes.data(), bytes.size());
      Upload(&x_dev, x);
      std::vector<std::int32_t> expert_of(static_cast<std::size_t>(tokens), 0);
      const int rc =
          kind == 0 ? nmb_mmq_iq2_xxs_dense_vec(w_dev.get(), x_dev.get(),
                                                out_dev.get(), rows, tokens,
                                                cols, nullptr)
          : kind == 1
              ? nmb_mmq_nvfp4_dense_vec(w_dev.get(), x_dev.get(), out_dev.get(),
                                        rows, tokens, cols, nullptr)
              : nmb_mmq_q8_0_dense_vec(w_dev.get(), x_dev.get(), out_dev.get(),
                                       rows, tokens, cols, nullptr);
      if (rc != 0) {
        std::printf("dense_vec %-4s rows=%d N=%d failed rc=%d\n", fmt, rows,
                    tokens, rc);
        ok = false;
        continue;
      }
      char name[96];
      std::snprintf(name, sizeof name, "dense_vec %s rows=%-3d N=%d", fmt, rows,
                    tokens);
      ok = Compare(Reference(w, x, tokens, expert_of), Download(&out_dev), rows,
                   tokens, name) &&
           ok;
    }
  }

  // MMQ tier across the tile boundaries.
  for (const int rows : {64, 256}) {
    for (const int tokens : {16, 32}) {
      auto bytes = MakeWeightBytes(kind, cols, rows, seed ^ 9);
      auto x =
          MakeValues(static_cast<std::size_t>(tokens) * cols, seed ^ 6, 1.0F);
      TensorRef w = MakeRef(&bytes, kind, cols, rows, 1);
      HipBuffer<std::uint8_t> w_dev(bytes.size());
      HipBuffer<float> x_dev(x.size());
      HipBuffer<float> out_dev(static_cast<std::size_t>(rows) * tokens);
      Upload(w_dev.get(), bytes.data(), bytes.size());
      Upload(&x_dev, x);
      std::vector<std::int32_t> expert_of(static_cast<std::size_t>(tokens), 0);
      const int rc =
          kind == 0
              ? nmb_mmq_iq2_xxs_dense(w_dev.get(), x_dev.get(), out_dev.get(),
                                      rows, tokens, cols, nullptr)
          : kind == 1
              ? nmb_mmq_nvfp4_dense(w_dev.get(), x_dev.get(), out_dev.get(),
                                    rows, tokens, cols, nullptr)
              : nmb_mmq_q8_0_dense(w_dev.get(), x_dev.get(), out_dev.get(),
                                   rows, tokens, cols, nullptr);
      if (rc != 0) {
        std::printf("dense %-4s rows=%d N=%d failed rc=%d\n", fmt, rows, tokens,
                    rc);
        ok = false;
        continue;
      }
      char name[96];
      std::snprintf(name, sizeof name, "dense(mmq) %s rows=%-3d N=%d", fmt,
                    rows, tokens);
      ok = Compare(Reference(w, x, tokens, expert_of), Download(&out_dev), rows,
                   tokens, name) &&
           ok;
    }
  }

  // Routed vector tier: one selected expert per token from a stack of four.
  for (const int tokens : {1, 3, 4}) {
    constexpr int kRows = 192;
    constexpr int kExperts = 4;
    auto bytes = MakeWeightBytes(kind, cols, kRows * kExperts, seed ^ 3);
    auto x =
        MakeValues(static_cast<std::size_t>(tokens) * cols, seed ^ 7, 1.0F);
    std::vector<std::int32_t> expert_of(static_cast<std::size_t>(tokens));
    for (std::int32_t& e : expert_of) {
      e = static_cast<std::int32_t>(NextRandom(&seed) % kExperts);
    }
    TensorRef w = MakeRef(&bytes, kind, cols, kRows, kExperts);
    HipBuffer<std::uint8_t> w_dev(bytes.size());
    HipBuffer<float> x_dev(x.size());
    HipBuffer<std::int32_t> ids_dev(expert_of.size());
    HipBuffer<float> out_dev(static_cast<std::size_t>(kRows) * tokens);
    Upload(w_dev.get(), bytes.data(), bytes.size());
    Upload(&x_dev, x);
    Upload(ids_dev.get(), expert_of.data(), ids_dev.bytes());
    const int weight_type =
        static_cast<int>(kind == 0   ? gufo::core::GgmlType::kIQ2_XXS
                         : kind == 1 ? gufo::core::GgmlType::kNVFP4
                                     : gufo::core::GgmlType::kQ8_0);
    const int rc = nmb_mmq_moe_vec(
        weight_type, w_dev.get(), x_dev.get(), ids_dev.get(), out_dev.get(),
        kRows, cols, tokens, kExperts, 1, nullptr, nullptr, nullptr);
    if (rc != 0) {
      std::printf("moe_vec %-4s N=%d failed rc=%d\n", fmt, tokens, rc);
      ok = false;
      continue;
    }
    char name[96];
    std::snprintf(name, sizeof name, "moe_vec %s N=%d experts=4", fmt, tokens);
    ok = Compare(Reference(w, x, tokens, expert_of), Download(&out_dev), kRows,
                 tokens, name) &&
         ok;
  }

  // Routed MMQ tier: several experts per token from a stack of four. Q8_0
  // raw already has upstream coverage and no model expert uses it.
  if (kind != 2) {
    for (const int tokens : {16, 32}) {
      for (const int used : {1, 3}) {
        constexpr int kRows = 192;
        constexpr int kExperts = 4;
        auto bytes = MakeWeightBytes(kind, cols, kRows * kExperts, seed ^ 11);
        auto x = MakeValues(static_cast<std::size_t>(tokens) * cols, seed ^ 12,
                            1.0F);
        std::vector<std::int32_t> expert_of(static_cast<std::size_t>(tokens) *
                                            used);
        for (std::int32_t& e : expert_of) {
          e = static_cast<std::int32_t>(NextRandom(&seed) % kExperts);
        }
        TensorRef w = MakeRef(&bytes, kind, cols, kRows, kExperts);
        HipBuffer<std::uint8_t> w_dev(bytes.size());
        HipBuffer<float> x_dev(x.size());
        HipBuffer<std::int32_t> ids_dev(expert_of.size());
        HipBuffer<float> out_dev(static_cast<std::size_t>(kRows) * tokens *
                                 used);
        Upload(w_dev.get(), bytes.data(), bytes.size());
        Upload(&x_dev, x);
        Upload(ids_dev.get(), expert_of.data(), ids_dev.bytes());
        const int rc =
            kind == 0
                ? nmb_mmq_iq2_xxs_moe_raw(w_dev.get(), x_dev.get(),
                                          ids_dev.get(), out_dev.get(), kRows,
                                          cols, tokens, kExperts, used, nullptr)
                : nmb_mmq_nvfp4_moe_raw(w_dev.get(), x_dev.get(), ids_dev.get(),
                                        out_dev.get(), kRows, cols, tokens,
                                        kExperts, used, nullptr);
        if (rc != 0) {
          std::printf("moe_raw %-4s N=%d used=%d failed rc=%d\n", fmt, tokens,
                      used, rc);
          ok = false;
          continue;
        }
        const std::vector<float> gpu = Download(&out_dev);
        // reference[(t * used + u) * kRows + r] =
        //     dot(dequant(expert_of[t * used + u], row r), x[t])
        std::vector<float> reference(gpu.size(), 0.0F);
        std::vector<float> row(cols);
        double scale = 0.0;
        double worst = 0.0;
        for (int t = 0; t < tokens; ++t) {
          for (int u = 0; u < used; ++u) {
            const std::size_t pair = static_cast<std::size_t>(t) * used + u;
            for (int r = 0; r < kRows; ++r) {
              DequantizeRow(w, static_cast<std::uint64_t>(expert_of[pair]),
                            static_cast<std::uint64_t>(r), row.data());
              double sum = 0.0;
              for (int c = 0; c < cols; ++c) {
                sum += static_cast<double>(row[c]) *
                       static_cast<double>(
                           x[static_cast<std::size_t>(t) * cols + c]);
              }
              const std::size_t i = pair * kRows + r;
              reference[i] = static_cast<float>(sum);
              scale = std::max(scale, std::abs(sum));
            }
          }
        }
        for (std::size_t i = 0; i < gpu.size(); ++i) {
          worst = std::max(worst, std::abs(static_cast<double>(reference[i]) -
                                           static_cast<double>(gpu[i])));
        }
        const double rel = worst / std::max(scale, 1e-30);
        char name[96];
        std::snprintf(name, sizeof name, "moe_raw %s N=%-2d used=%d experts=4",
                      fmt, tokens, used);
        const bool raw_ok = rel < kTolerance;
        std::printf("%-44s %s  rel/scale=%.3e\n", name,
                    raw_ok ? "PASS" : "FAIL", rel);
        ok = raw_ok && ok;
      }
    }
  }
  return ok;
}

}  // namespace

int main() {
  if (nmb_mmq_init(0) != 0) {
    std::fprintf(stderr, "nmb_mmq_init failed\n");
    return 1;
  }
  bool ok = true;
  ok = RunFormat(0, "IQ2_XXS") && ok;
  ok = RunFormat(1, "NVFP4") && ok;
  ok = RunFormat(2, "Q8_0") && ok;
  std::printf("rocm_mmq: %s\n", ok ? "ALL PASS" : "FAILURES");
  return ok ? 0 : 1;
}

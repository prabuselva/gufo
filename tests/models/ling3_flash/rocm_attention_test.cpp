// Absorbed MLA decode attention checks: synthetic latent caches through the
// split-KV partial+combine kernels against a host reference that mirrors the
// scalar oracle's MlaAttention softmax and latent-weighted sum in double.
// No model artifact is required.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/attention.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace q = gufo::models::ling3_flash::rocm;
namespace t = gufo::tests::ling3_flash;

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvRank = 512;
constexpr std::uint32_t kRot = 64;
constexpr std::uint32_t kTile = 1024;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cout << name << " FAILED (tolerance " << tolerance << ")\n";
    return false;
  }
  return true;
}

bool TestDecode(std::uint32_t n_pos, std::uint32_t seed) {
  const std::size_t q_dim = kKvRank + kRot;
  const auto q_abs = t::MakeValues(kHeads * q_dim, seed, 1.0F);
  const auto kv =
      t::MakeValues(static_cast<std::size_t>(n_pos) * kKvRank, seed + 1U, 1.0F);
  const auto kpe =
      t::MakeValues(static_cast<std::size_t>(n_pos) * kRot, seed + 2U, 1.0F);
  const float scale = 1.0F / std::sqrt(192.0F);

  // Host reference: exact softmax over all positions, latent sum in double.
  std::vector<float> ref(static_cast<std::size_t>(kHeads) * kKvRank);
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const float* qa = q_abs.data() + static_cast<std::size_t>(h) * q_dim;
    std::vector<double> scores(n_pos);
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::uint32_t p = 0; p < n_pos; ++p) {
      double acc = 0.0;
      const float* kv_row = kv.data() + static_cast<std::size_t>(p) * kKvRank;
      for (std::uint32_t i = 0; i < kKvRank; ++i) {
        acc += static_cast<double>(qa[i]) * kv_row[i];
      }
      const float* pe_row = kpe.data() + static_cast<std::size_t>(p) * kRot;
      for (std::uint32_t i = 0; i < kRot; ++i) {
        acc += static_cast<double>(qa[kKvRank + i]) * pe_row[i];
      }
      scores[p] = acc * static_cast<double>(scale);
      max_score = std::max(max_score, scores[p]);
    }
    double sum = 0.0;
    for (double& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    float* dst = ref.data() + static_cast<std::size_t>(h) * kKvRank;
    for (std::uint32_t r = 0; r < kKvRank; ++r) {
      double acc = 0.0;
      for (std::uint32_t p = 0; p < n_pos; ++p) {
        acc +=
            (scores[p] / sum) * kv[static_cast<std::size_t>(p) * kKvRank + r];
      }
      dst[r] = static_cast<float>(acc);
    }
  }

  const std::uint32_t tiles = q::MlaDecodeTiles(n_pos, kTile);
  t::HipBuffer<float> d_q(kHeads * q_dim),
      d_kv(static_cast<std::size_t>(n_pos) * kKvRank),
      d_kpe(static_cast<std::size_t>(n_pos) * kRot),
      d_ws(static_cast<std::size_t>(tiles) * kHeads * (kKvRank + 2U)),
      d_out(static_cast<std::size_t>(kHeads) * kKvRank);
  t::Upload(&d_q, q_abs);
  t::Upload(&d_kv, kv);
  t::Upload(&d_kpe, kpe);
  q::MlaDecodePartial(d_q.get(), d_kv.get(), d_kpe.get(), n_pos, kTile, scale,
                      d_ws.get(), kHeads, kKvRank, kRot, nullptr);
  q::MlaDecodeCombine(d_ws.get(), tiles, kHeads, kKvRank, d_out.get(), nullptr);
  char name[64];
  std::snprintf(name, sizeof(name), "MlaDecode n_pos=%u tiles=%u", n_pos,
                tiles);
  return Check(name, t::WorstRelative(ref, t::Download(&d_out, ref.size())),
               1e-4);
}

// Chunked prefill attention vs a host fp64 causal softmax over the caches.
// The kernel accumulates scores and the online softmax in fp32 (the decode
// kernels sum in fp64), so the tolerance is looser than the decode check;
// the worst observed error on gfx1151 over these cases is 4.1e-4.
bool TestPrefill(std::uint32_t start_pos, std::uint32_t tokens,
                 std::uint32_t seed) {
  const std::uint32_t n_pos = start_pos + tokens;
  const auto q_abs = t::MakeValues(
      static_cast<std::size_t>(tokens) * kHeads * kKvRank, seed, 1.0F);
  const auto q_rope = t::MakeValues(
      static_cast<std::size_t>(tokens) * kHeads * kRot, seed + 1U, 1.0F);
  const auto kv =
      t::MakeValues(static_cast<std::size_t>(n_pos) * kKvRank, seed + 2U, 1.0F);
  const auto kpe =
      t::MakeValues(static_cast<std::size_t>(n_pos) * kRot, seed + 3U, 1.0F);
  const float scale = 1.0F / std::sqrt(192.0F);

  std::vector<float> ref(static_cast<std::size_t>(tokens) * kHeads * kKvRank);
  for (std::uint32_t t_i = 0; t_i < tokens; ++t_i) {
    const std::uint32_t visible = start_pos + t_i + 1U;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t qo =
          (static_cast<std::size_t>(t_i) * kHeads + h) * kKvRank;
      const std::size_t ro =
          (static_cast<std::size_t>(t_i) * kHeads + h) * kRot;
      const float* qa = q_abs.data() + qo;
      const float* qr = q_rope.data() + ro;
      std::vector<double> scores(visible);
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::uint32_t p = 0; p < visible; ++p) {
        double acc = 0.0;
        const float* kv_row = kv.data() + static_cast<std::size_t>(p) * kKvRank;
        for (std::uint32_t i = 0; i < kKvRank; ++i) {
          acc += static_cast<double>(qa[i]) * kv_row[i];
        }
        const float* pe_row = kpe.data() + static_cast<std::size_t>(p) * kRot;
        for (std::uint32_t i = 0; i < kRot; ++i) {
          acc += static_cast<double>(qr[i]) * pe_row[i];
        }
        scores[p] = acc * static_cast<double>(scale);
        max_score = std::max(max_score, scores[p]);
      }
      double sum = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
      }
      float* dst = ref.data() + qo;
      for (std::uint32_t r = 0; r < kKvRank; ++r) {
        double acc = 0.0;
        for (std::uint32_t p = 0; p < visible; ++p) {
          acc +=
              (scores[p] / sum) * kv[static_cast<std::size_t>(p) * kKvRank + r];
        }
        dst[r] = static_cast<float>(acc);
      }
    }
  }

  t::HipBuffer<float> d_qa(q_abs.size()), d_qr(q_rope.size()), d_kv(kv.size()),
      d_kpe(kpe.size()), d_out(ref.size());
  t::Upload(&d_qa, q_abs);
  t::Upload(&d_qr, q_rope);
  t::Upload(&d_kv, kv);
  t::Upload(&d_kpe, kpe);
  q::MlaPrefill(d_qa.get(), d_qr.get(), d_kv.get(), d_kpe.get(), d_out.get(),
                start_pos, tokens, kHeads, kKvRank, kRot, scale, nullptr);
  char name[64];
  std::snprintf(name, sizeof(name), "MlaPrefill start=%u tokens=%u", start_pos,
                tokens);
  return Check(name, t::WorstRelative(ref, t::Download(&d_out, ref.size())),
               5e-4);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = TestDecode(1U, 0xAAAA0001U) && ok;
    ok = TestDecode(7U, 0xAAAA0101U) && ok;
    ok = TestDecode(kTile, 0xAAAA0201U) && ok;
    ok = TestDecode(kTile + 1U, 0xAAAA0301U) && ok;
    ok = TestDecode(3000U, 0xAAAA0401U) && ok;
    ok = TestPrefill(0U, 1U, 0xBB000001U) && ok;
    ok = TestPrefill(0U, 37U, 0xBB000101U) && ok;
    ok = TestPrefill(5U, 32U, 0xBB000201U) && ok;
    ok = TestPrefill(130U, 64U, 0xBB000301U) && ok;
    ok = TestPrefill(300U, 33U, 0xBB000401U) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
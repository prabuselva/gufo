// Standalone A/B for the Ling3 MLA decode partial: scalar kernel vs the
// fused four-head group kernel, both checked against the fp64 host oracle.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/attention.hip.cpp"

namespace q = gufo::models::ling3_flash::rocm;

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvRank = 512;
constexpr std::uint32_t kRot = 64;
constexpr std::uint32_t kTile = 256;
constexpr std::uint32_t kThreads = 256;

void CheckCuda(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

float* Dev(std::size_t n) {
  float* p = nullptr;
  CheckCuda(hipMalloc(&p, n * sizeof(float)), "hipMalloc");
  return p;
}

void Oracle(const std::vector<float>& q_abs, const std::vector<float>& kv,
            const std::vector<float>& kpe, std::uint32_t n_pos, float scale,
            std::vector<float>* out) {
  out->assign(static_cast<std::size_t>(kHeads) * kKvRank, 0.0F);
  const std::size_t q_dim = kKvRank + kRot;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const float* qa = q_abs.data() + static_cast<std::size_t>(h) * q_dim;
    std::vector<double> scores(n_pos);
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::uint32_t p = 0; p < n_pos; ++p) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kKvRank; ++i) {
        acc += static_cast<double>(qa[i]) *
               kv[static_cast<std::size_t>(p) * kKvRank + i];
      }
      for (std::uint32_t i = 0; i < kRot; ++i) {
        acc += static_cast<double>(qa[kKvRank + i]) *
               kpe[static_cast<std::size_t>(p) * kRot + i];
      }
      scores[p] = acc * static_cast<double>(scale);
      max_score = std::max(max_score, scores[p]);
    }
    double sum = 0.0;
    for (double& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    float* dst = out->data() + static_cast<std::size_t>(h) * kKvRank;
    for (std::uint32_t r = 0; r < kKvRank; ++r) {
      double acc = 0.0;
      for (std::uint32_t p = 0; p < n_pos; ++p) {
        acc += (scores[p] / sum) *
               kv[static_cast<std::size_t>(p) * kKvRank + r];
      }
      dst[r] = static_cast<float>(acc);
    }
  }
}

double WorstRelative(const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = std::fabs(ref[i]);
    const double e = std::fabs(static_cast<double>(ref[i]) - got[i]);
    const double rel = r > 1e-6 ? e / r : e / 1e-6;
    if (rel > worst) worst = rel;
  }
  return worst;
}

struct Buffers {
  float* q;
  float* kv;
  float* kpe;
  float* ws;
  float* out;
  std::size_t ws_bytes;
};

Buffers MakeBuffers(std::uint32_t n_pos, const std::vector<float>& q_abs,
                    const std::vector<float>& kv,
                    const std::vector<float>& kpe) {
  Buffers b;
  b.q = Dev(q_abs.size());
  b.kv = Dev(kv.size());
  b.kpe = Dev(kpe.size());
  const std::size_t tiles = q::MlaDecodeTiles(n_pos, kTile);
  b.ws = Dev(tiles * kHeads * (kKvRank + 2U));
  b.out = Dev(static_cast<std::size_t>(kHeads) * kKvRank);
  CheckCuda(hipMemcpy(b.q, q_abs.data(), q_abs.size() * 4,
                      hipMemcpyHostToDevice),
            "copy q");
  CheckCuda(hipMemcpy(b.kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice),
            "copy kv");
  CheckCuda(hipMemcpy(b.kpe, kpe.data(), kpe.size() * 4,
                      hipMemcpyHostToDevice),
            "copy kpe");
  return b;
}

void RunVariant(bool fused, const Buffers& b, std::uint32_t n_pos) {
  const std::uint32_t tiles = q::MlaDecodeTiles(n_pos, kTile);
  const float scale = 1.0F / std::sqrt(192.0F);
  if (fused) {
    q::MlaDecodePartialGroupKernel<<<dim3(kHeads / q::kDecodeGroup, tiles),
                                     kThreads>>>(
        b.q, b.kv, b.kpe, n_pos, kTile, scale, b.ws, kHeads, kKvRank, kRot);
  } else {
    q::MlaDecodePartialKernel<<<dim3(tiles, kHeads), kThreads>>>(
        b.q, b.kv, b.kpe, n_pos, kTile, scale, b.ws, kHeads, kKvRank, kRot);
  }
  q::MlaDecodeCombineKernel<<<kHeads, kThreads>>>(b.ws, tiles, kHeads, kKvRank,
                                                  b.out);
}

double TimeRun(bool fused, const Buffers& b, std::uint32_t n_pos, int reps) {
  hipEvent_t e0, e1;
  CheckCuda(hipEventCreate(&e0), "event");
  CheckCuda(hipEventCreate(&e1), "event");
  RunVariant(fused, b, n_pos);
  CheckCuda(hipDeviceSynchronize(), "warm");
  CheckCuda(hipEventRecord(e0), "rec0");
  for (int i = 0; i < reps; ++i) {
    RunVariant(fused, b, n_pos);
  }
  CheckCuda(hipEventRecord(e1), "rec1");
  CheckCuda(hipEventSynchronize(e1), "sync");
  float ms = 0.0F;
  CheckCuda(hipEventElapsedTime(&ms, e0, e1), "elapsed");
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return static_cast<double>(ms) * 1000.0 / reps;
}

std::vector<float> Random(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

}  // namespace

int main() {
  bool ok = true;
  std::printf("correctness:\n");
  for (std::uint32_t n : {std::uint32_t{7}, std::uint32_t{16},
                          std::uint32_t{33}, std::uint32_t{1025},
                          std::uint32_t{2050}}) {
    const std::size_t q_dim = kKvRank + kRot;
    auto q_abs = Random(kHeads * q_dim, 11 + n);
    auto kv = Random(static_cast<std::size_t>(n) * kKvRank, 21 + n);
    auto kpe = Random(static_cast<std::size_t>(n) * kRot, 31 + n);
    std::vector<float> ref;
    Oracle(q_abs, kv, kpe, n, 1.0F / std::sqrt(192.0F), &ref);
    for (bool fused : {false, true}) {
      Buffers b = MakeBuffers(n, q_abs, kv, kpe);
      RunVariant(fused, b, n);
      CheckCuda(hipDeviceSynchronize(), "run");
      std::vector<float> got(ref.size());
      CheckCuda(hipMemcpy(got.data(), b.out, got.size() * 4,
                          hipMemcpyDeviceToHost),
                "out");
      const double err = WorstRelative(ref, got);
      const bool pass = err <= 1e-4;
      ok = ok && pass;
      std::printf("  n=%5u %-5s worst %.3e %s\n", n, fused ? "fused" : "old",
                  err, pass ? "OK" : "FAIL");
      hipFree(b.q);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.ws);
      hipFree(b.out);
    }
  }
  std::printf("timing (us per MLA decode call, all 32 heads):\n");
  for (std::uint32_t n : {std::uint32_t{1024}, std::uint32_t{3072},
                          std::uint32_t{9216}, std::uint32_t{17408}}) {
    const std::size_t q_dim = kKvRank + kRot;
    auto q_abs = Random(kHeads * q_dim, 77 + n);
    auto kv = Random(static_cast<std::size_t>(n) * kKvRank, 88 + n);
    auto kpe = Random(static_cast<std::size_t>(n) * kRot, 99 + n);
    Buffers b = MakeBuffers(n, q_abs, kv, kpe);
    const double old_us = TimeRun(false, b, n, 20);
    const double new_us = TimeRun(true, b, n, 200);
    std::printf("  n=%6u old %10.1f  fused %9.1f  speedup %6.1fx\n", n, old_us,
                new_us, old_us / new_us);
    hipFree(b.q);
    hipFree(b.kv);
    hipFree(b.kpe);
    hipFree(b.ws);
    hipFree(b.out);
  }
  return ok ? 0 : 1;
}

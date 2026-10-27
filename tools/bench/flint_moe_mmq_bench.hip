// Flint MoE routed-MMQ launch sweep (gfx1151).
//
// Flint 40B-A2B MoE layers: 256 experts, expert_ffn 512, hidden 2048,
// top-6, plus one down GEMM over the 6 slots as slots x 1 expert. The
// production path calls qfn_mmq_q8_0_moe_raw with the vendored default
// launch rules; deepseek_v4 showed that qfn/ds4_mmq_set_routed_max_expert_rows
// (trim empty column tiles) and .._set_routed_tile_cols (shrink the 80-column
// tile for thin buckets) decide a large part of the id-tier rate.
//
// For each (shape, batch) this times: default, max-rows bound, and the bound
// with forced column-tile widths, on uniform-random routing (all experts
// live). Reports time, effective TFLOP/s over the executed MACs and the
// distinct-expert weight GB/s once-per-read floor.
//
// Build (host ROCm, no nix):
//   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. \
//     tools/bench/flint_moe_mmq_bench.hip \
//     build/release/src/models/qwen38_flash_next/libgufo_qwen38_flash_next_mmq.a \
//     -o /tmp/flint_moe_mmq_bench
#include <src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#define CHECK(call)                                                     \
  do {                                                                  \
    hipError_t err_ = (call);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "HIP %s -> %s\n", #call,                     \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

struct Shape {
  const char* name;
  int M;   // rows of W per expert (output dim, K-contiguous rows)
  int K;   // hidden
  int used;  // experts per token (down runs slots x 1)
  int slots_per_tok;  // multiplier applied to ids/out rows (down: 6)
};

constexpr int kExperts = 256;

double TimeLaunch(const std::function<void()>& fn, int reps) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  fn();  // warm
  fn();
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) fn();
  CHECK(hipEventRecord(b));
  CHECK(hipEventSynchronize(b));
  float ms = 0.f;
  CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return ms / reps;
}

}  // namespace

#include <functional>

int main(int argc, char** argv) {
  const int batches[] = {128, 512, 2048};
  if (argc > 1) std::printf("args ignored; sweep is fixed\n");

  std::mt19937 rng(1234);
  CHECK(hipSetDevice(0));
  if (qfn_mmq_init(0) != 0) {
    std::fprintf(stderr, "qfn_mmq_init failed\n");
    return 1;
  }

  Shape shapes[] = {
      {"gate_up (M=512,K=2048,used=6)", 512, 2048, 6, 1},
      {"down    (M=2048,K=512,slots x1)", 2048, 512, 1, 6},
  };

  for (const Shape& s : shapes) {
    // 256 experts x [K x M] Q8_0: bytes = experts * M * (K/32*(32+2))
    const size_t w_bytes =
        (size_t)kExperts * s.M * ((size_t)s.K / 32 * (32 + 2));
    void* W = nullptr;
    CHECK(hipMalloc(&W, w_bytes));
    {
      std::vector<unsigned char> pat(w_bytes);
      std::uniform_int_distribution<int> d(-20, 20);
      for (size_t i = 0; i < w_bytes; ++i)
        pat[i] = static_cast<unsigned char>(128 + (i % 3 == 0 ? d(rng) % 16 : 0));
      CHECK(hipMemcpy(W, pat.data(), w_bytes, hipMemcpyHostToDevice));
    }
    const int n_tok = 2048;  // allocate worst case
    float* X = nullptr;
    float* OUT = nullptr;
    int32_t* IDS = nullptr;
    CHECK(hipMalloc(&X, (size_t)n_tok * s.slots_per_tok * s.K * 4));
    CHECK(hipMalloc(&OUT, (size_t)n_tok * s.slots_per_tok * 6 * s.M * 4));
    CHECK(hipMalloc(&IDS, (size_t)n_tok * s.slots_per_tok * 6 * 4));
    CHECK(hipMemset(X, 0x3c, (size_t)n_tok * s.slots_per_tok * s.K * 4));

    for (const int B : batches) {
      // Uniform routing for `used` slots per token (down uses used=1 with
      // slot-indexed ids over 6*B slots).
      const int rows = B * s.slots_per_tok;
      std::vector<int32_t> ids((size_t)rows * s.used);
      std::vector<unsigned int> counts(kExperts, 0);
      for (int r = 0; r < rows; ++r) {
        for (int u = 0; u < s.used; ++u) {
          const int32_t e = (int32_t)((rng() ^ (r * 2654435761u)) % kExperts);
          ids[(size_t)r * s.used + u] = e;
          ++counts[e];
        }
      }
      CHECK(hipMemcpy(IDS, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
      unsigned int max_c = 0, live = 0;
      size_t live_bytes = 0;
      for (int e = 0; e < kExperts; ++e) {
        max_c = std::max(max_c, counts[e]);
        if (counts[e]) { ++live; live_bytes += (size_t)s.M * ((size_t)s.K / 32 * 34); }
      }
      const double flops =
          2.0 * rows * s.used * (double)s.M * s.K;
      auto run_once = [&](int max_rows, int tile_cols) {
        qfn_mmq_set_routed_max_expert_rows(max_rows);
        qfn_mmq_set_routed_tile_cols(tile_cols);
        const int rc = qfn_mmq_q8_0_moe_raw(W, X, IDS, OUT, s.M, s.K, rows,
                                            kExperts, s.used, 0);
        if (rc != 0) std::fprintf(stderr, "moe_raw rc=%d\n", rc);
      };
      const struct { const char* tag; int max_rows; int cols; } cfgs[] = {
          {"default       ", 0, 0},
          {"maxrows       ", (int)max_c, 0},
          {"maxrows c16   ", (int)max_c, 16},
          {"maxrows c48   ", (int)max_c, 48},
          {"maxrows c80   ", (int)max_c, 80},
          {"countsel c?   ", (int)max_c,
           (int)qfn_mmq_routed_tile_cols_for_counts(counts.data(), kExperts)},
      };
      std::printf("\n%s B=%d (slots=%d, live=%u/256, max bucket=%u)\n",
                  s.name, B, rows, live, max_c);
      for (const auto& c : cfgs) {
        const double ms = TimeLaunch(
            [&] { run_once(c.max_rows, c.cols); }, 5);
        const double tfs = flops / (ms * 1e-3) / 1e12;
        const double gbs = live_bytes / (ms * 1e-3) / 1e9;
        std::printf("  %s cols%-2d %8.3f ms  %6.2f TF/s  %7.1f GB/s(experts-once)\n",
                    c.tag, c.cols, ms, tfs, gbs);
      }
    }
    CHECK(hipFree(IDS));
    CHECK(hipFree(OUT));
    CHECK(hipFree(X));
    CHECK(hipFree(W));
  }
  return 0;
}

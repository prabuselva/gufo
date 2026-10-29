// Tile-configuration matrix for the Flint dense W8A8 prefill tier at the
// exact production shapes (attn_q 4096x2048, attn_out 2048x4096) over the
// whole-prompt chunk sizes. Baseline `shipped` is the production route
// LaunchBatchedQuantGEMMPreQuantized picks today: the wave64 WKQuantA8
// 128x128 2x2 schedule qualified on square Qwen shapes, never re-tuned for
// Flint's skewed m/k. Candidates are the wave64 schedules from
// flint_w8a8_wave64_cfgs.def (native wave64 TU) and the wave32 W8A8Blocked
// schedules the shared launcher would use without the wave64 gate. Every
// schedule keeps the per-element int32 K-ascending accumulation order, so
// all candidates must be bitwise identical to the baseline on every shape.
//
//   tools/bench/build.sh flint_w8a8_bench
//   /tmp/flint_w8a8_bench [reps] [batch ...]
//
// Synthetic Q8_0 weights/activations are only a payload: route numerics are
// adjudicated by the flint engine/forward gates at adoption time, and the
// per-candidate `mismatch` column must read 0 for any timing to matter.
// C9: `attn_kv` (m=512, k=2048 - the W8A8-ineligible k/v projections) adds
// an `mmq dense` baseline line: `flint_mmq_q8_0_dense` timing plus its
// relative deviation from the shipped W8A8 output (both are quantized
// approximations, so bitwise comparison does not apply here; the W8A8-vs-
// true-quality contract is the flint forward gate, the same one C7m passed
// for the q projection).
#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

extern "C" {
int flint_mmq_init(int device);
int flint_mmq_q8_0_dense(const void* W_q8_0, const float* X_f32,
                         float* out_f32, int M, int N, int K,
                         hipStream_t stream);
}

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hip"

#define CHECK(call)                                                     \
  do {                                                                  \
    const hipError_t err__ = (call);                                    \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                   hipGetErrorString(err__), __FILE__, __LINE__);       \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace hipk = gufo::hip;
namespace core = gufo::core;

namespace {

using Wave64Fn = void (*)(const void*, const void*, float*, std::size_t,
                          std::size_t, std::size_t, hipStream_t);

#define X(id, BM, BN, BK, WM, WN)                                \
  extern "C" void flint_w8a8_v64_##id(const void*, const void*,  \
                                      float*, std::size_t,       \
                                      std::size_t, std::size_t,  \
                                      hipStream_t);
#include "flint_w8a8_wave64_cfgs.def"
#undef X

#define X(id, BM, BN, BK, WM, WN)                                \
  {                                                              \
    flint_w8a8_v64_##id, "v64 " #BM "x" #BN " bk" #BK            \
                         " " #WM "x" #WN                         \
  },
const struct {
  Wave64Fn fn;
  const char* name;
} kWave64Cfgs[] = {
#include "flint_w8a8_wave64_cfgs.def"
#undef X
}
;

// Wave32 candidates from the shared launcher's own kernels (this TU is
// compiled wave32, so the w32 WMMA builtins are legal here).
template <int BM, int BN, int BK, int WM, int WN>
void LaunchW8A8W32(const void* w, const void* x_tiled, float* y,
                   std::size_t batch, std::size_t m, std::size_t k) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  hipk::W8A8BlockedWmmaGEMMKernel<BM, BN, BK, WM, WN>
      <<<grid, 256>>>(w, x_tiled, y, batch, m, k);
}

template <int BM, int BN, int BK, int WM, int WN>
void LaunchKQuantW32(const void* w, const void* x_tiled, float* y,
                     std::size_t batch, std::size_t m, std::size_t k) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  hipk::WKQuantA8BlockedWmmaGEMMKernel<BM, BN, BK, WM, WN, core::GgmlType::
                                            kQ8_0,
                                        32>
      <<<grid, 256>>>(w, x_tiled, y, batch, m, k);
}

__global__ void CompareBits(const float* reference, const float* candidate,
                            std::size_t count, unsigned* mismatches) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count &&
      __float_as_uint(reference[i]) != __float_as_uint(candidate[i]))
    atomicAdd(mismatches, 1U);
}

// ggml block_q8_0 layout: {__half d; int8_t qs[32];}, 34 bytes
// (src/models/qwen/hip/quant_ops.hpp). A 36-byte float-scale block makes
// every weight read misaligned inside the WMMA kernels.
struct Q8Block {
  __half d;
  std::int8_t qs[32];
};

static_assert(sizeof(Q8Block) == 34, "block_q8_0 must be packed 34 bytes");

std::vector<Q8Block> MakeQ8Weights(std::size_t rows, std::size_t cols,
                                   std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<Q8Block> out(rows * cols / 32);
  for (std::size_t blk = 0; blk < out.size(); ++blk) {
    float amax = 0.f;
    float v[32];
    for (float& e : v) {
      e = dist(rng) * 0.05f;
      amax = std::max(amax, std::fabs(e));
    }
    const float d = amax / 127.f;
    const float scale = d == 0.f ? 1.f : d;
    out[blk].d = __float2half(scale);
    for (int i = 0; i < 32; ++i) {
      out[blk].qs[i] = static_cast<std::int8_t>(std::lround(v[i] / scale));
    }
  }
  return out;
}

float TimeIt(const std::function<void()>& fn, int reps) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  fn();
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) {
    fn();
  }
  CHECK(hipEventRecord(b));
  CHECK(hipEventSynchronize(b));
  float ms = 0.f;
  CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return ms / reps;
}

void RunShape(const char* label, std::size_t m, std::size_t k,
              std::size_t batch, int reps, std::mt19937& rng) {
  auto w = MakeQ8Weights(m, k, rng);
  std::vector<float> x(batch * k);
  std::uniform_real_distribution<float> xd(-3.f, 3.f);
  for (float& e : x) e = xd(rng);

  void* dw;
  float *dx, *y_ref, *y_cand;
  unsigned* d_mismatch;
  // Zeroed guard tail: any candidate that speculates a read past the last
  // row reads zeros instead of faulting an unmapped page; the bitwise
  // column still gates correctness.
  constexpr std::size_t kGuard = 256u * 1024u;
  const auto alloc_guarded = [&](void** ptr, std::size_t bytes) {
    CHECK(hipMalloc(ptr, bytes + kGuard));
    CHECK(hipMemset(*ptr, 0, bytes + kGuard));
  };
  alloc_guarded(&dw, w.size() * sizeof(Q8Block));
  CHECK(hipMemcpy(dw, w.data(), w.size() * sizeof(Q8Block),
                  hipMemcpyHostToDevice));
  alloc_guarded(reinterpret_cast<void**>(&dx), x.size() * sizeof(float));
  CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  alloc_guarded(reinterpret_cast<void**>(&y_ref), batch * m * sizeof(float));
  alloc_guarded(reinterpret_cast<void**>(&y_cand), batch * m * sizeof(float));
  CHECK(hipMalloc(&d_mismatch, sizeof(unsigned)));
  const std::size_t q8_bytes = hipk::QuantizedActivationBytes(batch, k);
  void* q8;
  alloc_guarded(&q8, q8_bytes);

  // The shipped route (baseline) and every candidate share this tile.
  CHECK(hipMemset(y_ref, 0, batch * m * sizeof(float)));
  hipk::LaunchQuantizeActivationQ8_1FromFp32(dx, q8, batch, k, nullptr);
  CHECK(hipDeviceSynchronize());

  const double flops = 2.0 * batch * m * k;
  std::printf("\n%s m=%zu k=%zu batch=%zu  (%.1f GFLOP/call)\n", label, m, k,
              batch, flops / 1e9);

  const double ms_ship =
      TimeIt(
          [&] {
            hipk::LaunchBatchedQuantGEMMPreQuantized(
                core::GgmlType::kQ8_0, dw, q8, y_ref, batch, m, k, nullptr);
          },
          reps);
  std::printf("  %-24s %8.3f ms %7.1f TF/s   (baseline)\n", "shipped",
              ms_ship, flops / ms_ship / 1e9);

  const char* keep = std::getenv("FLINT_W8A8_KEEP");
  const auto run_candidate = [&](const char* name,
                                 const std::function<void(float*)>& launch) {
    if (keep != nullptr && std::strstr(name, keep) == nullptr) {
      return;
    }
    std::printf("  .. running %s\n", name);
    CHECK(hipMemset(y_cand, 0, batch * m * sizeof(float)));
    launch(y_cand);
    const hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
      std::fprintf(stderr, "launch error on %s: %s\n", name,
                   hipGetErrorString(launch_err));
      std::exit(2);
    }
    CHECK(hipDeviceSynchronize());
    unsigned mismatch = 0;
    CHECK(hipMemset(d_mismatch, 0, sizeof(unsigned)));
    const std::size_t count = batch * m;
    CompareBits<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        y_ref, y_cand, count, d_mismatch);
    CHECK(hipMemcpy(&mismatch, d_mismatch, sizeof(unsigned),
                    hipMemcpyDeviceToHost));
    const double ms = TimeIt([&] { launch(y_cand); }, reps);
    std::printf("  %-24s %8.3f ms %7.1f TF/s  %+7.1f%%  mismatch=%u\n", name,
                ms, flops / ms / 1e9, (ms_ship / ms - 1.0) * 100.0,
                mismatch);
    // Unbuffered progress: the last printed line is exactly the candidate
    // where a faulting config would abort the run.
    std::fflush(stdout);
  };

  for (const auto& cfg : kWave64Cfgs) {
    run_candidate(cfg.name, [&](float* y) {
      cfg.fn(dw, q8, y, batch, m, k, nullptr);
    });
  }
  run_candidate("w32 blk 4x2 bk2", [&, dw, q8, batch, m, k](float* y) {
    LaunchW8A8W32<128, 128, 2, 4, 2>(dw, q8, y, batch, m, k);
  });
  run_candidate("w32 blk bn64 bk2", [&, dw, q8, batch, m, k](float* y) {
    LaunchW8A8W32<128, 64, 2, 4, 2>(dw, q8, y, batch, m, k);
  });
  run_candidate("w32 blk 2x4 bk2", [&, dw, q8, batch, m, k](float* y) {
    LaunchW8A8W32<128, 128, 2, 2, 4>(dw, q8, y, batch, m, k);
  });
  run_candidate("w32 blk 64m bk4", [&, dw, q8, batch, m, k](float* y) {
    LaunchW8A8W32<64, 128, 4, 4, 2>(dw, q8, y, batch, m, k);
  });
  run_candidate("w32 kq 4x2 bk2", [&, dw, q8, batch, m, k](float* y) {
    LaunchKQuantW32<128, 128, 2, 4, 2>(dw, q8, y, batch, m, k);
  });
  run_candidate("w32 kq 2x4 bk2", [&, dw, q8, batch, m, k](float* y) {
    LaunchKQuantW32<128, 128, 2, 2, 4>(dw, q8, y, batch, m, k);
  });

  // C9: production fallback for this shape is the mmq tier; report its
  // timing and how far it deviates from the shipped W8A8 result.
  if (m <= 1024U) {
    CHECK(hipMemset(y_cand, 0, batch * m * sizeof(float)));
    const int rc = flint_mmq_q8_0_dense(dw, dx, y_cand, static_cast<int>(m),
                                        static_cast<int>(batch),
                                        static_cast<int>(k), nullptr);
    if (rc != 0) {
      std::fprintf(stderr, "flint_mmq_q8_0_dense rc=%d\n", rc);
      std::exit(2);
    }
    CHECK(hipDeviceSynchronize());
    const std::size_t count = batch * m;
    std::vector<float> h_ref(count);
    std::vector<float> h_mmq(count);
    CHECK(hipMemcpy(h_ref.data(), y_ref, count * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h_mmq.data(), y_cand, count * sizeof(float),
                    hipMemcpyDeviceToHost));
    double max_abs = 0.0;
    double scale = 1e-30;
    double sq = 0.0;
    std::size_t nan_ref = 0;
    std::size_t nan_mmq = 0;
    std::size_t first_nan = static_cast<std::size_t>(-1);
    std::size_t min_nan_row = static_cast<std::size_t>(-1);
    std::size_t max_nan_row = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (std::isnan(h_ref[i])) {
        ++nan_ref;
        if (first_nan == static_cast<std::size_t>(-1)) first_nan = i;
        const std::size_t row = i % m;
        min_nan_row = std::min(min_nan_row, row);
        max_nan_row = std::max(max_nan_row, row);
      }
      if (std::isnan(h_mmq[i])) ++nan_mmq;
      const double d = std::fabs(h_ref[i] - h_mmq[i]);
      max_abs = std::max(max_abs, d);
      scale = std::max(scale, std::fabs(static_cast<double>(h_ref[i])));
      sq += d * d;
    }
    const double ms = TimeIt(
        [&] {
          (void)flint_mmq_q8_0_dense(dw, dx, y_cand, static_cast<int>(m),
                                     static_cast<int>(batch),
                                     static_cast<int>(k), nullptr);
        },
        reps);
    std::printf(
        "  %-24s %8.3f ms %7.1f TF/s  %+7.1f%%  "
        "maxrel=%.3e relL2=%.3e nan_ref=%zu nan_mmq=%zu "
        "first=%zu(tok %zu row %zu) rows[%zu..%zu]\n",
        "mmq dense", ms, flops / ms / 1e9, (ms_ship / ms - 1.0) * 100.0,
        max_abs / scale, std::sqrt(sq / static_cast<double>(count)) / scale,
        nan_ref, nan_mmq, first_nan, first_nan / m, first_nan % m, min_nan_row,
        max_nan_row);
  }

  CHECK(hipFree(dw));
  CHECK(hipFree(dx));
  CHECK(hipFree(y_ref));
  CHECK(hipFree(y_cand));
  CHECK(hipFree(q8));
  CHECK(hipFree(d_mismatch));
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  std::vector<std::size_t> batches;
  if (argc > 2) {
    for (int i = 2; i < argc; ++i) batches.push_back(std::atol(argv[i]));
  } else {
    batches = {512, 2048, 4096};
  }
  std::mt19937 rng(1234);
  if (flint_mmq_init(0) != 0) {
    std::fprintf(stderr, "flint_mmq_init failed\n");
    return 2;
  }
  const char* shapes = std::getenv("FLINT_W8A8_SHAPES");
  const auto want = [&](const char* label) {
    return shapes == nullptr || std::strstr(shapes, label) != nullptr;
  };
  for (std::size_t batch : batches) {
    if (want("attn_q")) RunShape("attn_q  ", 4096, 2048, batch, reps, rng);
    if (want("attn_kv")) RunShape("attn_kv ", 512, 2048, batch, reps, rng);
    if (want("attn_out")) RunShape("attn_out", 2048, 4096, batch, reps, rng);
    if (want("shexp_gu")) RunShape("shexp_gu", 1024, 2048, batch, reps, rng);
    if (want("shexp_dn")) RunShape("shexp_dn", 2048, 1024, batch, reps, rng);
    if (want("ffn_gu")) RunShape("ffn_gu  ", 6144, 2048, batch, reps, rng);
    if (want("ffn_dn")) RunShape("ffn_dn  ", 2048, 6144, batch, reps, rng);
  }
  return 0;
}

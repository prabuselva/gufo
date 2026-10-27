// A/B harness for the Flint dense Q8_0 tier: the shipped mmq Gemm path
// (quantize_mmq_q8_1 + mul_mat_q per call) against the shared qwen W8A8
// int8-WMMA route (tiled Q8_1 activation quantize +
// LaunchBatchedQuantGEMMPreQuantized, wave64 at batch >= 96).
// Shapes and dtypes match the Flint artifacts: hidden 2048, shared-expert
// ff 1024, attn q [4096x2048], attn_out [2048x4096], k/v [512x2048].
//
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -DENGINE_ENABLE_HIP \
//     -I . -I src/models/flint/kernels/rocm/mmq \
//     -x hip tools/bench/flint_w8a8_bench.hip -x none \
//     build/gpu-test/src/models/flint/libgufo_flint_kernels.a \
//     build/gpu-test/src/models/flint/libgufo_flint_mmq.a \
//     build/gpu-test/libgufo_core.a \
//     -lhipblas -lrocblas -lhsa-runtime64 -o /tmp/flint_w8a8_bench
//
// Usage: /tmp/flint_w8a8_bench [reps]
// Timing harness only: the f32 sanity columns are meaningless (they feed
// synthetic payloads through the mmq q8_1 padding contract, which expects
// ggml's exact row/K-block layout); route numerics are adjudicated by the
// flint engine/forward gates. Timings are valid (reproduced 3x).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "src/models/flint/kernels/rocm/linear.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace flint = gufo::models::flint::rocm;
namespace core = gufo::core;
namespace hipk = gufo::hip;

#define CHECK(call)                                                     \
  do {                                                                  \
    const hipError_t err__ = (call);                                    \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                   hipGetErrorString(err__), __FILE__, __LINE__);       \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

struct Q8Block {
  float d;
  std::int8_t qs[32];
};

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
    out[blk].d = d == 0.f ? 1.f : d;
    for (int i = 0; i < 32; ++i) {
      out[blk].qs[i] = static_cast<std::int8_t>(
          std::lround(v[i] / out[blk].d));
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

struct Shape {
  const char* name;
  std::size_t m;
  std::size_t k;
};

// Tile schedules for the wave64 kernel (WaveSize=64 => WM*WN == 4, BM %
// (16*WM)==0, BN % (16*WN)==0). The shipped route uses <128,128>; smaller
// BM keeps every row tile and more blocks resident when m <= 1024.
template <int BM, int BN, int BK, int WM, int WN>
void LaunchCfg(const void* w, const void* x_tiled, float* out,
               std::size_t batch, std::size_t m, std::size_t k,
               hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<BM, BN, BK, WM, WN,
                                            core::GgmlType::kQ8_0, 64>
      <<<grid, WM * WN * 64, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  const std::size_t batch = argc > 2 ? std::atol(argv[2]) : 2048;
  std::mt19937 rng(1234);
  const Shape shapes[] = {
      {"attn_q    ", 4096, 2048}, {"attn_out  ", 2048, 4096},
      {"shexp_gate", 1024, 2048}, {"shexp_up  ", 1024, 2048},
      {"shexp_down", 2048, 1024}, {"attn_k    ", 512, 2048},
  };
  std::printf("batch=%zu reps=%d\n", batch, reps);
  for (const Shape& s : shapes) {
    auto w = MakeQ8Weights(s.m, s.k, rng);
    std::vector<float> x(batch * s.k);
    std::uniform_real_distribution<float> xd(-3.f, 3.f);
    for (float& e : x) {
      e = xd(rng);
    }
    void* dw;
    float *dx, *y1, *y2;
    CHECK(hipMalloc(&dw, w.size() * sizeof(Q8Block)));
    CHECK(hipMemcpy(dw, w.data(), w.size() * sizeof(Q8Block),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
    CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&y1, batch * s.m * sizeof(float)));
    CHECK(hipMalloc(&y2, batch * s.m * sizeof(float)));
    const std::size_t q8_bytes = hipk::Q8ActTiledBytes(batch, s.k);
    void* q8;
    CHECK(hipMalloc(&q8, q8_bytes));

    const std::size_t row_bytes = (s.k / 32) * sizeof(Q8Block);
    const double flops = 2.0 * batch * s.m * s.k;
    const double ms_mmq =
        TimeIt([&] {
          flint::Gemm(dw, core::GgmlType::kQ8_0,
                      static_cast<std::uint32_t>(s.m),
                      static_cast<std::uint32_t>(s.k), row_bytes, dx, y1,
                      static_cast<std::uint32_t>(batch), nullptr);
        }, reps);
    // Quantize-activation cost alone (paid once per distinct input at
    // integration; q/k/v and gate/up share it).
    const double ms_q =
        TimeIt([&] {
          hipk::LaunchQuantizeActivationQ8_1FromFp32(dx, q8, batch, s.k,
                                                     nullptr);
        }, reps);
    const double ms_w8 =
        TimeIt([&] {
          hipk::LaunchQuantizeActivationQ8_1FromFp32(dx, q8, batch, s.k,
                                                     nullptr);
          hipk::LaunchBatchedQuantGEMMPreQuantized(
              core::GgmlType::kQ8_0, dw, q8, y2, batch, s.m, s.k, nullptr);
        }, reps);
    // GEMM-only schedules sharing the prepared activation tile.
    const double cfg_a =
        TimeIt([&] { LaunchCfg<128, 128, 2, 2, 2>(dw, q8, y2, batch, s.m,
                                                  s.k, nullptr); },
               reps);
    const double cfg_b =
        TimeIt([&] { LaunchCfg<64, 128, 2, 2, 2>(dw, q8, y2, batch, s.m,
                                                 s.k, nullptr); },
               reps);
    const double cfg_c =
        TimeIt([&] { LaunchCfg<32, 128, 2, 2, 2>(dw, q8, y2, batch, s.m,
                                                 s.k, nullptr); },
               reps);
    const double cfg_d =
        TimeIt([&] { LaunchCfg<16, 64, 2, 1, 4>(dw, q8, y2, batch, s.m, s.k,
                                                nullptr); },
               reps);

    // Numerical comparison against mmq as reference (both are int8-act).
    // Re-run the shipped schedule once so y2 holds a schedule whose output
    // layout is known-good, not the last config swept.
    hipk::LaunchQuantizeActivationQ8_1FromFp32(dx, q8, batch, s.k, nullptr);
    CHECK(hipDeviceSynchronize());
    LaunchCfg<128, 128, 2, 2, 2>(dw, q8, y2, batch, s.m, s.k, nullptr);
    CHECK(hipDeviceSynchronize());
    std::vector<float> o1(batch * s.m), o2(batch * s.m);
    CHECK(hipMemcpy(o1.data(), y1, o1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(o2.data(), y2, o2.size() * 4, hipMemcpyDeviceToHost));
    // Host fp32 reference for the first three elements under both candidate
    // out indexings: token-major y[idx] = x[idx / m] . w[idx % m] vs
    // row-major y[idx] = x[idx % batch] . w[idx / batch].
    auto Dot = [&](const float* xr, const std::uint8_t* wr) {
      double sacc = 0.0;
      for (std::size_t b = 0; b < s.k / 32; ++b) {
        float dq;
        std::memcpy(&dq, wr + b * 36, 4);
        for (int q = 0; q < 32; ++q) {
          sacc += static_cast<double>(dq) *
                  static_cast<int>(
                      static_cast<signed char>(wr[b * 36 + 4 + q]));
        }
      }
      double acc = 0.0;
      for (std::size_t j = 0; j < s.k; ++j) acc += static_cast<double>(xr[j]) * sacc;
      return acc;
    };
    const auto* wbytes = reinterpret_cast<const std::uint8_t*>(w.data());
    auto RefTok = [&](std::size_t p) {
      return Dot(&x[(p / s.m) * s.k], wbytes + (p % s.m) * (s.k / 32) * 36);
    };
    auto RefRow = [&](std::size_t p) {
      return Dot(&x[p % batch], wbytes + (p / batch) * (s.k / 32) * 36);
    };
    std::printf(
        "  [diag] idx0 o1=%.2f o2=%.2f refTok=%.2f refRow=%.2f | "
        "idx1 o1=%.2f o2=%.2f refTok=%.2f refRow=%.2f\n",
        o1[0], o2[0], RefTok(0), RefRow(0), o1[1], o2[1], RefTok(1),
        RefRow(1));
    double se = 0.0, ref = 0.0;
    std::size_t nan1 = 0, nan2 = 0;
    for (std::size_t i = 0; i < o1.size(); ++i) {
      if (!std::isfinite(o1[i])) ++nan1;
      if (!std::isfinite(o2[i])) ++nan2;
      const double d = o2[i] - o1[i];
      se += d * d;
      ref += static_cast<double>(o1[i]) * o1[i];
    }
    std::printf(
        "%s m=%5zu k=%5zu mmq %7.3f(%5.1fTF) q=%.3f w8full %7.3f(%5.1fTF) "
        "cfg128x128 %7.3f cfg64x128 %7.3f cfg32x128 %7.3f cfg16x64 %7.3f  "
        "rms=%.5f nnan=%zu/%zu\n",
        s.name, s.m, s.k, ms_mmq, flops / (ms_mmq * 1e9), ms_q, ms_w8,
        flops / (ms_w8 * 1e9), cfg_a, cfg_b, cfg_c, cfg_d,
        std::sqrt(se / std::max(ref, 1.0)), nan2, o1.size());
    CHECK(hipFree(dw));
    CHECK(hipFree(dx));
    CHECK(hipFree(y1));
    CHECK(hipFree(y2));
    CHECK(hipFree(q8));
  }
  return 0;
}

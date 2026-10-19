// NVFP4 feasibility probe for gfx1151 (Nemotron Bolt ssm_in/ssm_out weights).
//
// Golden-compares the nemotron-private vendored MMQ GEMM and MMVQ GEMV NVFP4
// paths (E2M1 nibbles + one UE4M3 sub-scale per 16 elements, 64 elements per
// block) against a scalar CPU reference, then reports weight bandwidth.
//
// The probe links the nemotron-private quant-matmul library (vendored mmq,
// `nmb_` prefix; see STATUS.md Phase 3 until it exists):
//   cmake --build --preset release --target gufo_nemotron_h_moe_mmq
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. \
//       src/models/nemotron_h_moe/tools/nvfp4_probe.hip \
//       -Lbuild/release/src/models/nemotron_h_moe \
//       -lgufo_nemotron_h_moe_mmq -o /tmp/nvfp4_probe
//
// First hardware evidence (2026-10-18) was collected against an equivalent
// temporary additive instantiation in the deepseek mmq library; that patch
// was reverted per the model-private golden rule. Keep this file building
// against the vendored copy only.
//
// The GPU result quantizes the F32 activations to Q8_1, so a Q8_1-shaped
// deviation from the exact reference is expected; the thresholds below are
// set relative to that floor, not to zero.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

// ABI of the vendored nemotron mmq library (kernels/rocm/mmq, Phase 3).
extern "C" int nmb_mmq_init(int device);
extern "C" int nmb_mmq_nvfp4_dense(const void* W_nvfp4, const float* X_f32,
                                   float* out_f32, int M, int N, int K,
                                   hipStream_t stream);
extern "C" int nmb_mmq_nvfp4_dense_vec(const void* W_nvfp4, const float* X_f32,
                                       float* out_f32, int M, int N, int K,
                                       hipStream_t stream);

// Host-provided folded-Q8_1 activation cache hook (owned by the model
// backend in production). This probe runs without it: returning 0 routes the
// aligned entries through their normal quantize prelude.
extern "C" int nmb_hip_q8_fold_take_q81(const void*, uint64_t, const void**) {
  return 0;
}

#define CU(x)                                                        \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    if (e_ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                \
                   hipGetErrorString(e_), __FILE__, __LINE__);       \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

namespace {

// kvalues_fp4 (doubled E2M1 values), shared by MXFP4/NVFP4 upstream.
const int kValuesFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8,
                            -12};

constexpr int kQk = 64;       // elements per NVFP4 block
constexpr int kQkSub = 16;    // sub-block size for per-group UE4M3 scales

// Replica of ggml_ue4m3_to_fp32(): value * 0.5 folds the doubled E2M1 table.
float Ue4M3ToFloat(std::uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  const float raw = exp == 0
                        ? std::ldexp(static_cast<float>(man), -9)
                        : std::ldexp(1.0f + static_cast<float>(man) / 8.0f,
                                     exp - 7);
  return raw * 0.5f;
}

// Weight tensor [M, K] as M*K/64 blocks of d[4] + qs[32].
std::vector<std::uint8_t> MakeNvfp4Weights(int m, int k,
                                           std::mt19937_64* rng) {
  std::uniform_int_distribution<int> exp_dist(3, 9);
  std::uniform_int_distribution<int> man_dist(0, 7);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::vector<std::uint8_t> w(
      static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * 36 / kQk);
  const int blocks_per_row = k / kQk;
  for (int row = 0; row < m; ++row) {
    for (int b = 0; b < blocks_per_row; ++b) {
      std::uint8_t* blk =
          w.data() + (static_cast<std::size_t>(row) * blocks_per_row + b) * 36;
      for (int s = 0; s < kQk / kQkSub; ++s) {
        int enc = 0x80;
        while (enc == 0x80 || enc == 0xFF) {  // keep nonzero, non-NaN.
          enc = (exp_dist(*rng) << 3) | man_dist(*rng);
        }
        blk[s] = static_cast<std::uint8_t>(enc);
      }
      for (int q = 0; q < kQk / 2; ++q) {
        blk[4 + q] = static_cast<std::uint8_t>(byte_dist(*rng));
      }
    }
  }
  return w;
}

// out_exact[t * M + r] (m innermost) = dot(row r, token t) in double.
void ReferenceGemm(const std::vector<std::uint8_t>& w,
                   const std::vector<float>& x, int m, int n, int k,
                   std::vector<double>* out) {
  out->assign(static_cast<std::size_t>(m) * n, 0.0);
  const int blocks_per_row = k / kQk;
  for (int r = 0; r < m; ++r) {
    for (int b = 0; b < blocks_per_row; ++b) {
      const std::uint8_t* blk =
          w.data() + (static_cast<std::size_t>(r) * blocks_per_row + b) * 36;
      const int k0 = b * kQk;
      for (int s = 0; s < kQk / kQkSub; ++s) {
        const float d = Ue4M3ToFloat(blk[s]);
        for (int j = 0; j < kQkSub / 2; ++j) {
          const std::uint8_t byte = blk[4 + s * (kQkSub / 2) + j];
          const double v0 = kValuesFp4[byte & 0x0F] * static_cast<double>(d);
          const double v1 = kValuesFp4[byte >> 4] * static_cast<double>(d);
          const int k00 = k0 + s * kQkSub + j;
          const int k01 = k00 + kQkSub / 2;
          for (int t = 0; t < n; ++t) {
            (*out)[static_cast<std::size_t>(t) * m + r] +=
                v0 * x[static_cast<std::size_t>(t) * k + k00] +
                v1 * x[static_cast<std::size_t>(t) * k + k01];
          }
        }
      }
    }
  }
}

struct CaseResult {
  bool ok = false;
  double max_abs = 0.0;
  double max_rel_vs_scale = 0.0;
};

// Reports the error under whichever of the two output layouts matches best.
CaseResult CheckOutput(const std::vector<float>& gpu,
                       const std::vector<double>& ref, int m, int n) {
  double scale = 0.0;
  for (const double v : ref) {
    scale = std::max(scale, std::abs(v));
  }
  double best_abs = 0.0;
  for (const bool m_innermost : {true, false}) {
    double max_abs = 0.0;
    for (int r = 0; r < m; ++r) {
      for (int t = 0; t < n; ++t) {
        const std::size_t idx =
            m_innermost ? static_cast<std::size_t>(t) * m + r
                        : static_cast<std::size_t>(r) * n + t;
        max_abs = std::max(max_abs,
                           std::abs(static_cast<double>(gpu[idx]) - ref[idx]));
      }
    }
    if (max_abs < best_abs || best_abs == 0.0) {
      best_abs = max_abs;
    }
  }
  CaseResult res;
  res.max_abs = best_abs;
  res.max_rel_vs_scale = best_abs / std::max(scale, 1e-30);
  res.ok = res.max_rel_vs_scale < 0.03;  // Q8_1 activation-quant floor.
  return res;
}

struct Harness {
  void* w_dev = nullptr;
  float* x_dev = nullptr;
  float* out_dev = nullptr;
};

Harness AllocDevice(std::size_t w_bytes, std::size_t x_elems,
                    std::size_t out_elems) {
  Harness h;
  CU(hipMalloc(&h.w_dev, w_bytes));
  CU(hipMalloc(reinterpret_cast<void**>(&h.x_dev), x_elems * sizeof(float)));
  CU(hipMalloc(reinterpret_cast<void**>(&h.out_dev),
               out_elems * sizeof(float)));
  return h;
}

bool RunOne(const char* name, int m, int n, int k, bool use_vec,
            std::mt19937* rng) {
  std::mt19937_64 wrng(0xB017u ^ (m * 131 + n * 7 + k));
  const std::vector<std::uint8_t> w = MakeNvfp4Weights(m, k, &wrng);
  std::vector<float> x(static_cast<std::size_t>(n) * k);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  for (float& v : x) {
    v = x_dist(*rng);
  }
  std::vector<double> ref;
  ReferenceGemm(w, x, m, n, k, &ref);

  Harness h = AllocDevice(w.size(), x.size(), static_cast<std::size_t>(m) * n);
  CU(hipMemcpy(h.w_dev, w.data(), w.size(), hipMemcpyHostToDevice));
  CU(hipMemcpy(h.x_dev, x.data(), x.size() * sizeof(float),
               hipMemcpyHostToDevice));
  const int rc = use_vec ? nmb_mmq_nvfp4_dense_vec(h.w_dev, h.x_dev, h.out_dev,
                                                   m, n, k, nullptr)
                         : nmb_mmq_nvfp4_dense(h.w_dev, h.x_dev, h.out_dev, m,
                                               n, k, nullptr);
  std::vector<float> gpu(static_cast<std::size_t>(m) * n);
  CU(hipMemcpy(gpu.data(), h.out_dev, gpu.size() * sizeof(float),
               hipMemcpyDeviceToHost));
  CU(hipFree(h.w_dev));
  CU(hipFree(h.x_dev));
  CU(hipFree(h.out_dev));
  if (rc != 0) {
    std::printf("%-30s LAUNCH FAILED rc=%d\n", name, rc);
    return false;
  }
  const CaseResult res = CheckOutput(gpu, ref, m, n);
  std::printf("%-30s %s  max_abs=%.4g  rel/scale=%.2e\n", name,
              res.ok ? "PASS" : "FAIL", res.max_abs, res.max_rel_vs_scale);
  return res.ok;
}

void RunTiming(int m, int n, int k, std::mt19937* rng) {
  std::mt19937_64 wrng(0x5EEDu);
  const std::vector<std::uint8_t> w = MakeNvfp4Weights(m, k, &wrng);
  std::vector<float> x(static_cast<std::size_t>(n) * k);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  for (float& v : x) {
    v = x_dist(*rng);
  }
  Harness h = AllocDevice(w.size(), x.size(), static_cast<std::size_t>(m) * n);
  CU(hipMemcpy(h.w_dev, w.data(), w.size(), hipMemcpyHostToDevice));
  CU(hipMemcpy(h.x_dev, x.data(), x.size() * sizeof(float),
               hipMemcpyHostToDevice));
  for (int i = 0; i < 5; ++i) {
    nmb_mmq_nvfp4_dense(h.w_dev, h.x_dev, h.out_dev, m, n, k, nullptr);
  }
  CU(hipDeviceSynchronize());
  hipEvent_t t0;
  hipEvent_t t1;
  CU(hipEventCreate(&t0));
  CU(hipEventCreate(&t1));
  constexpr int kIters = 50;
  CU(hipEventRecord(t0));
  for (int i = 0; i < kIters; ++i) {
    nmb_mmq_nvfp4_dense(h.w_dev, h.x_dev, h.out_dev, m, n, k, nullptr);
  }
  CU(hipEventRecord(t1));
  CU(hipEventSynchronize(t1));
  float ms = 0.0f;
  CU(hipEventElapsedTime(&ms, t0, t1));
  ms /= kIters;
  const double w_bytes = static_cast<double>(w.size());
  const double x_bytes = static_cast<double>(x.size()) * sizeof(float);
  const double gb = (w_bytes + x_bytes) / 1e9;
  std::printf(
      "timed   M=%d N=%d K=%d  %.3f ms  %.1f GB/s (w+x)\n", m, n, k, ms,
      gb / (ms / 1e3));
  CU(hipEventDestroy(t0));
  CU(hipEventDestroy(t1));
  CU(hipFree(h.w_dev));
  CU(hipFree(h.x_dev));
  CU(hipFree(h.out_dev));
}

}  // namespace

int main() {
  if (nmb_mmq_init(0) != 0) {
    std::fprintf(stderr, "nmb_mmq_init failed\n");
    return 1;
  }
  std::mt19937 rng(42);
  bool ok = true;
  // GEMV (mmvq) shapes: decode-side ssm_in [4096 -> 18560] style skinny N.
  ok &= RunOne("vec  M=1024 N=1  K=1024", 1024, 1, 1024, true, &rng);
  ok &= RunOne("vec  M=1024 N=3  K=1024", 1024, 3, 1024, true, &rng);
  ok &= RunOne("vec  M=1024 N=8  K=256", 1024, 8, 256, true, &rng);
  // GEMM (mmq) shapes: chunked prefill depth.
  ok &= RunOne("gemm M=1024 N=16 K=1024", 1024, 16, 1024, false, &rng);
  ok &= RunOne("gemm M=1024 N=32 K=256", 1024, 32, 256, false, &rng);
  ok &= RunOne("gemm M=4096 N=64 K=1024", 4096, 64, 1024, false, &rng);
  ok &= RunOne("gemm M=256  N=256 K=1024", 256, 256, 1024, false, &rng);
  RunTiming(18560, 512, 4096, &rng);  // ssm_in prefill shape.
  RunTiming(4096, 512, 10240, &rng);  // ssm_out prefill shape.
  std::printf("%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}

// Decode-tier probe: production f32-activation GEMVs vs the vendored
// ggml mmvq (Q8_1-activation) tiers at the real t=1 Nemotron Bolt shapes.
// Build (production static libs keep the exact release codegen):
//   hipcc -x hip --offload-arch=gfx1151 -O3 -DNDEBUG \
//     -I<repo> -I<repo>/src/models/nemotron_h_moe/kernels/rocm/mmq \
//     -DGGML_HIP_NO_VMM -DENGINE_ENABLE_HIP=1 \
//     tools/bench/nem_decode_mmvq_bench.hip \
//     build/release/src/models/nemotron_h_moe/libgufo_nemotron_h_moe_mmq.a \
//     build/release/src/models/nemotron_h_moe/libgufo_nemotron_h_moe_kernels.a \
//     -lamdhip64 -o /tmp/nem_decode_mmvq_bench
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nmb_mmq.h"
#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"

#define CU(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::printf("hip err %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

namespace {

std::uint32_t g_seed = 12345;
double Rand() {
  g_seed = g_seed * 1664525U + 1013904223U;
  return (static_cast<double>(g_seed >> 8) / 8388608.0) - 1.0;
}

void* MakeRandomWeights(std::size_t bytes, int kind, std::size_t block_bytes,
                        std::size_t block_elems) {
  std::vector<std::uint8_t> h(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    h[i] = static_cast<std::uint8_t>(std::rand() & 0xFF);
  }
  const std::size_t scale_head = kind == 1 ? 4 : 2;
  for (std::size_t i = 0; i + block_bytes <= bytes; i += block_bytes) {
    for (std::size_t j = 0; j < scale_head; ++j) {
      if (kind == 1) {
        h[i + j] &= 0x7EU;  // finite UE4M3 sub-scales
      } else {
        h[i + j] = (j == 0) ? static_cast<std::uint8_t>(0x30 | (h[i] & 1U))
                            : static_cast<std::uint8_t>(h[i + j] & 0x7FU);
      }
    }
    (void)block_elems;
  }
  void* d = nullptr;
  CU(hipMalloc(&d, bytes));
  CU(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
  return d;
}

float* MakeRandomRows(std::size_t elems) {
  std::vector<float> h(elems);
  for (float& v : h) v = static_cast<float>(Rand());
  float* d = nullptr;
  CU(hipMalloc(&d, elems * sizeof(float)));
  CU(hipMemcpy(d, h.data(), elems * sizeof(float), hipMemcpyHostToDevice));
  return d;
}

double RelErr(float* a, float* b, std::size_t n) {
  std::vector<float> ha(n), hb(n);
  CU(hipMemcpy(ha.data(), a, n * sizeof(float), hipMemcpyDeviceToHost));
  CU(hipMemcpy(hb.data(), b, n * sizeof(float), hipMemcpyDeviceToHost));
  // Same contract metric as rocm_mmq_test::Compare: worst absolute error
  // normalized by the largest reference magnitude (per-element relative
  // error punishes Q8_1 activation noise on cancelling dots).
  double scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    scale = std::max(scale, std::fabs(static_cast<double>(hb[i])));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double rel = std::fabs(static_cast<double>(ha[i]) - hb[i]);
    if (rel > worst) worst = rel;
  }
  return worst / std::max(scale, 1e-30);
}

template <typename Fn>
double TimeUs(Fn&& fn, int iters) {
  for (int i = 0; i < 3; ++i) fn();
  CU(hipDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  CU(hipDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

void Report(const char* name, double prod_us, double cand_us, double rel) {
  std::printf("%-34s prod=%8.1fus cand=%8.1fus gain=%5.2fx rel=%.3e %s\n",
              name, prod_us, cand_us, prod_us / cand_us, rel,
              cand_us < prod_us * 0.98 ? "WIN" : "");
}

}  // namespace

int main() {
  std::srand(7);
  CU(hipSetDevice(0));
  if (nmb_mmq_init(0) != 0) { std::printf("nmb_mmq_init failed\n"); return 1; }

  // ---- NVFP4 dense t=1: ssm_in (18560 x 4096) and ssm_out (4096 x 8192)
  {
    struct { const char* n; int m, k; } shapes[] = {{"nvfp4 ssm_in", 18560, 4096},
                                                    {"nvfp4 ssm_out", 4096, 8192}};
    for (auto& s : shapes) {
      const std::size_t wbytes =
          static_cast<std::size_t>(s.m) * (s.k / 64) * 36;
      void* w = MakeRandomWeights(wbytes, 1, 36, 64);
      float* x = MakeRandomRows(s.k);
      float* o1 = nullptr; float* o2 = nullptr;
      CU(hipMalloc(&o1, s.m * sizeof(float)));
      CU(hipMalloc(&o2, s.m * sizeof(float)));
      double a = TimeUs([&] {
        nbl_gemv_nvfp4_f32(x, w, o1, s.m, s.k, 1, nullptr);
      }, 30);
      nbl_gemv_nvfp4_f32(x, w, o1, s.m, s.k, 1, nullptr);
      nbl_gemv_f32(kNblGemvNvfp4, w, x, o1, s.m, s.k, 1, nullptr);
      double b = TimeUs([&] {
        nmb_mmq_nvfp4_dense_vec(w, x, o2, s.m, 1, s.k, nullptr);
      }, 30);
      CU(hipDeviceSynchronize());
      Report(s.n, a, b, RelErr(o2, o1, s.m));
      hipFree(w); hipFree(x); hipFree(o1); hipFree(o2);
    }
  }

  // ---- Q8_0 dense t=1: qkv (65536x1 shape seen) latent/attn_out sizes
  {
    struct { const char* n; int m, k; } shapes[] = {{"q8 attn_out", 4096, 4096},
                                                    {"q8 latent", 1024, 4096},
                                                    {"q8 latent_up", 4096, 1024},
                                                    {"q8 logits", 131072, 4096}};
    for (auto& s : shapes) {
      const std::size_t wbytes =
          static_cast<std::size_t>(s.m) * (s.k / 32) * 34;
      void* w = MakeRandomWeights(wbytes, 0, 34, 32);
      float* x = MakeRandomRows(s.k);
      float* o1 = nullptr; float* o2 = nullptr;
      CU(hipMalloc(&o1, s.m * sizeof(float)));
      CU(hipMalloc(&o2, s.m * sizeof(float)));
      double a = TimeUs([&] {
        nbl_gemv_f32(kNblGemvQ8_0, w, x, o1, s.m, s.k, 1, nullptr);
      }, 30);
      nbl_gemv_f32(kNblGemvQ8_0, w, x, o1, s.m, s.k, 1, nullptr);
      double b = TimeUs([&] {
        nmb_mmq_q8_0_dense_vec(w, x, o2, s.m, 1, s.k, nullptr);
      }, 30);
      CU(hipDeviceSynchronize());
      Report(s.n, a, b, RelErr(o2, o1, s.m));
      hipFree(w); hipFree(x); hipFree(o1); hipFree(o2);
    }
  }

  // ---- IQ2_XXS routed MoE t=1 slots=8 (up: ff @ k=1024, down: 1024 @ ff).
  {
    const int E = 368, topk = 8;
    struct { const char* n; int m, k; int shared; } cases[] = {
        {"iq2 moe up ff1536", 1536, 1024, 0},
        {"iq2 moe up ff2048", 2048, 1024, 0},
        {"iq2 moe up ff2560", 2560, 1024, 0},
        {"iq2 moe down ff1536", 1024, 1536, 0},
        {"iq2 moe down ff2048", 1024, 2048, 0},
        {"iq2 moe down ff2560", 1024, 2560, 0},
        {"iq2 shared up g5376", 5376, 1024, 1},
        {"iq2 shared down g5376", 1024, 5376, 1}};
    std::int32_t ids_h[8];
    for (int i = 0; i < topk; ++i) ids_h[i] = static_cast<int>(Rand() * E * 0 + std::rand() % E);
    std::int32_t* ids = nullptr;
    CU(hipMalloc(&ids, topk * sizeof(std::int32_t)));
    for (auto& s : cases) {
      const int slots = s.shared ? 1 : topk;
      const std::size_t per_expert =
          static_cast<std::size_t>(s.m) * (s.k / 256) * 66;
      void* w = MakeRandomWeights(per_expert * E, 0, 66, 256);
      float* x = MakeRandomRows(static_cast<std::size_t>(slots) * s.k);
      float* o1 = nullptr; float* o2 = nullptr;
      CU(hipMalloc(&o1, static_cast<std::size_t>(slots) * s.m * sizeof(float)));
      CU(hipMalloc(&o2, static_cast<std::size_t>(slots) * s.m * sizeof(float)));
      for (int i = 0; i < topk; ++i) ids_h[i] = std::rand() % E;
      CU(hipMemcpy(ids, ids_h, slots * sizeof(std::int32_t), hipMemcpyHostToDevice));
      double a = TimeUs([&] {
        nbl_gemv_moe_f32(kNblGemvIq2Xxs, w, x, ids, o1, s.m, s.k, slots,
                         s.k, 1, nullptr);
      }, 30);
      nbl_gemv_moe_f32(kNblGemvIq2Xxs, w, x, ids, o1, s.m, s.k, slots, s.k, 1,
                       nullptr);
      double b = TimeUs([&] {
        nmb_mmq_iq2_xxs_moe_raw(w, x, ids, o2, s.m, s.k, 1, E, slots, nullptr);
      }, 30);
      CU(hipDeviceSynchronize());
      const double rel = RelErr(o2, o1, static_cast<std::size_t>(slots) * s.m);
      Report(s.n, a, b, rel);
      hipFree(w); hipFree(x); hipFree(o1); hipFree(o2);
    }
    hipFree(ids);
  }
  std::printf("done\n");
  return 0;
}

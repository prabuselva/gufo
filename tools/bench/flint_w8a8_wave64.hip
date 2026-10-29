// Wave64 tile-config launchers for the Flint dense W8A8 tuning bench.
// Compiled with -mwavefrontsize64 (see tools/bench/build.sh) exactly like
// src/models/qwen/hip/kernels/prefill_quant_wave64.hip, whose shipped
// TryLaunchQuantPrefillWave64 also links into the bench as the baseline.
// The configuration list lives in flint_w8a8_wave64_cfgs.def.
#define ENGINE_ENABLE_HIP 1
#include <cstddef>

#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_wave64.hip"

namespace {

template <int BM, int BN, int BK, int WM, int WN>
void LaunchCfg(const void* w, const void* x_tiled, float* y,
               std::size_t batch, std::size_t m, std::size_t k,
               hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<BM, BN, BK, WM, WN,
                                            gufo::core::GgmlType::kQ8_0, 64>
      <<<grid, WM * WN * 64, 0, stream>>>(w, x_tiled, y, batch, m, k);
}

}  // namespace

#define X(id, BM, BN, BK, WM, WN)                                        \
  extern "C" void flint_w8a8_v64_##id(const void* w, const void* x,      \
                                      float* y, std::size_t batch,       \
                                      std::size_t m, std::size_t k,      \
                                      hipStream_t stream) {              \
    LaunchCfg<BM, BN, BK, WM, WN>(w, x, y, batch, m, k, stream);         \
  }
#include "flint_w8a8_wave64_cfgs.def"
#undef X

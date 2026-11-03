#pragma once
// Staging-side declarations for the Q8 prefill fast paths that live in
// src/families/qwen/kernels/prefill_quant_gemm.hip through
// staging/core/quant/prefill_wave4_launchers_staging.inl. Only staging model
// kernels dispatch through them today (flint C8b).

#if defined(ENGINE_ENABLE_HIP)
#include <cstddef>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// Zeroes the uninitialized tail of the last 16-token tile of a Q8_1
/// activation buffer. No-op when batch % 16 == 0. Lets a fused producer
/// write the tiled layout directly and still match
/// LaunchQuantizeActivationQ8_1FromFp32 bit-for-bit.
void LaunchZeroQ8ActTailFor(void* q8_1_out, std::size_t batch, std::size_t k,
                            hipStream_t stream = nullptr);

/// Wave-32 alternative for the pre-quantized route: the Q8_0 weight matrix
/// consumed by WKQuantA8BlockedWmmaGEMMKernel<128,128,2,4,2,Q8_0,32>, the
/// same tile the PreQuantized fallback uses when the wave-64 gate declines.
/// Bit-identical output to LaunchBatchedQuantGEMMPreQuantized on every
/// overlapping shape; models that measured this tile faster on specific
/// shapes dispatch through it (flint C8b,
/// docs/models/flint/artifacts/flint-c8b-w8a8-config-matrix.log). Returns
/// false for non-Q8_0 weights and batch < 96; the caller then uses
/// LaunchBatchedQuantGEMMPreQuantized.
[[nodiscard]] bool TryLaunchQuantPrefillWave32(core::GgmlType type,
                                               const void* w,
                                               const void* q8_1_x, float* y,
                                               std::size_t batch, std::size_t m,
                                               std::size_t k,
                                               hipStream_t stream = nullptr);


}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

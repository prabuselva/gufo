#include "src/models/qwen36_a3b/kernels/rocm/gemm.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>

#include "qfn_mmq.h"
#include "src/models/qwen36_a3b/kernels/rocm/dense_f16_gemm.hpp"

namespace gufo::models::qwen36_a3b::rocm {
namespace {

// Dense F16 GEMM route bounds (gfx1151; see docs/models/gemma4/EXPERIMENTS.md).
// rows >= 2048 keeps the router (256 rows) and the narrow KV and shared-expert
// gate/up projections (512 rows) off the F16 path; cols <= 4096 is the measured
// winning ceiling (the widest dense projection here is 4096, so it never
// excludes a real shape but keeps the rule consistent with the other trunk).
// batch >= 96 matches the wide prefill tiles the kernel was tuned and measured
// on.
constexpr std::uint32_t kDenseF16MinBatch = 96;
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 4096;

// Process-wide hipBLAS handle. The executor drives a single stream, so one
// handle reused across calls is safe; the stream is rebound on every call.
hipblasHandle_t BlasHandle() {
  static hipblasHandle_t handle = [] {
    hipblasHandle_t h = nullptr;
    (void)hipblasCreate(&h);
    return h;
  }();
  return handle;
}

// Lazily grown device scratch holding the BF16 narrowing of the activations.
// Only the BF16 GEMM path touches it and only one stream is live, so a single
// growable buffer is race-free. Sized in elements.
hip_bfloat16* Bf16Scratch(std::size_t elems) {
  static hip_bfloat16* buf = nullptr;
  static std::size_t cap = 0;
  if (elems > cap) {
    if (buf != nullptr) {
      (void)hipFree(buf);
    }
    cap = elems < (1u << 16) ? (1u << 16) : elems;
    (void)hipMalloc(&buf, cap * sizeof(hip_bfloat16));
  }
  return buf;
}

__global__ void NarrowBf16(const float* __restrict__ x,
                           hip_bfloat16* __restrict__ out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = hip_bfloat16(x[i]);
  }
}

// Lazily grown device scratch holding the F16 narrowing of the activations,
// mirroring Bf16Scratch. Only the dense F16 GEMM route touches it and only one
// stream is live, so a single growable buffer is race-free. Sized in elements.
__half* HalfScratch(std::size_t elems) {
  static __half* buf = nullptr;
  static std::size_t cap = 0;
  if (elems > cap) {
    if (buf != nullptr) {
      (void)hipFree(buf);
    }
    cap = elems < (1u << 16) ? (1u << 16) : elems;
    (void)hipMalloc(&buf, cap * sizeof(__half));
  }
  return buf;
}

__global__ void NarrowHalf(const float* __restrict__ x,
                           __half* __restrict__ out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __float2half_rn(x[i]);
  }
}

// Native grouped-MoE fallback for the unquantized (F32/BF16) expert encodings
// that the mmq routed path does not cover. One block per (token, slot) output
// vector; each thread strides the output rows and reduces over the input. The
// expert row base is selected from `ids` on device, so no host round-trip is
// needed. Weights stay in their native encoding.
template<typename T>
__global__ void MoeVecFallback(const T* __restrict__ w,
                               const float* __restrict__ x,
                               const std::int32_t* __restrict__ ids,
                               float* __restrict__ out, std::uint32_t rows,
                               std::uint32_t cols, std::uint32_t used,
                               std::size_t expert_stride) {
  const std::uint32_t pair = blockIdx.x;
  const std::int32_t e = ids[pair];
  const std::uint32_t token = pair / used;
  const T* __restrict__ wrow = w + static_cast<std::size_t>(e) * expert_stride;
  const float* __restrict__ xrow = x + static_cast<std::size_t>(token) * cols;
  float* __restrict__ orow = out + static_cast<std::size_t>(pair) * rows;
  for (std::uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
    const T* __restrict__ wr = wrow + static_cast<std::size_t>(r) * cols;
    float acc = 0.0f;
    for (std::uint32_t k = 0; k < cols; ++k) {
      acc += static_cast<float>(wr[k]) * xrow[k];
    }
    orow[r] = acc;
  }
}

}  // namespace

void Gemm(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          std::uint32_t batch, hipStream_t stream) {
  (void)row_bytes;
  switch (type) {
    case GemvType::kQ8_0:
      // Wide prefill batches take the F16 WMMA dense GEMM wherever its lower
      // arithmetic intensity makes halved activation traffic and no
      // FP32->Q8_1 requantize outweigh int8's 2x tensor throughput (1.32-1.59x
      // measured on the sibling trunk, docs/models/gemma4/EXPERIMENTS.md). The
      // weights stay Q8_0 and are dequantized to F16 in LDS; only the
      // activations are narrowed. Any shape outside the measured winning
      // window, or one the kernel rejects, falls through to the int8 mmq path.
      if (batch >= kDenseF16MinBatch && rows >= kDenseF16MinRows &&
          cols <= kDenseF16MaxCols) {
        const std::size_t n = static_cast<std::size_t>(batch) * cols;
        __half* xh = HalfScratch(n);
        const std::size_t block = 256;
        const std::size_t grid = (n + block - 1) / block;
        NarrowHalf<<<grid, block, 0, stream>>>(x, xh, n);
        if (DenseF16Gemm(base, xh, out, batch, rows, cols, stream)) {
          break;
        }
      }
      (void)qfn_mmq_q8_0_dense(base, x, out, static_cast<int>(rows),
                               static_cast<int>(batch), static_cast<int>(cols),
                               stream);
      break;
    case GemvType::kF32: {
      const float alpha = 1.0F;
      const float beta = 0.0F;
      (void)hipblasSetStream(BlasHandle(), stream);
      (void)hipblasSgemm(
          BlasHandle(), HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(rows),
          static_cast<int>(batch), static_cast<int>(cols), &alpha,
          static_cast<const float*>(base), static_cast<int>(cols), x,
          static_cast<int>(cols), &beta, out, static_cast<int>(rows));
      break;
    }
    case GemvType::kBF16: {
      const std::size_t n = static_cast<std::size_t>(batch) * cols;
      hip_bfloat16* xb = Bf16Scratch(n);
      const std::size_t block = 256;
      const std::size_t grid = (n + block - 1) / block;
      NarrowBf16<<<grid, block, 0, stream>>>(x, xb, n);
      const float alpha = 1.0F;
      const float beta = 0.0F;
      (void)hipblasSetStream(BlasHandle(), stream);
      (void)hipblasGemmEx(
          BlasHandle(), HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(rows),
          static_cast<int>(batch), static_cast<int>(cols), &alpha, base,
          HIP_R_16BF, static_cast<int>(cols), xb, HIP_R_16BF,
          static_cast<int>(cols), &beta, out, HIP_R_32F, static_cast<int>(rows),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
      break;
    }
    case GemvType::kQ4_K:
    case GemvType::kQ5_K:
      // Routed-expert encodings only. The dense GEMM tier never sees a K-quant
      // tensor: ValidateTypes restricts every non-expert tensor to Q8_0/F32/
      // BF16, and the routed experts go through the WMMA route, not this tier.
      break;
  }
}

void GemmMoe(const void* base, GemvType type, std::uint32_t rows,
             std::uint32_t cols, std::size_t row_bytes, const float* x,
             const std::int32_t* ids, float* out, std::uint32_t n_tokens,
             std::uint32_t n_experts, std::uint32_t n_expert_used,
             hipStream_t stream) {
  (void)row_bytes;
  if (type == GemvType::kQ8_0) {
    (void)qfn_mmq_q8_0_moe_raw(
        base, x, ids, out, static_cast<int>(rows), static_cast<int>(cols),
        static_cast<int>(n_tokens), static_cast<int>(n_experts),
        static_cast<int>(n_expert_used), stream);
    return;
  }
  const std::uint32_t pairs = n_tokens * n_expert_used;
  const std::size_t expert_stride = static_cast<std::size_t>(rows) * cols;
  const dim3 grid(pairs);
  const dim3 block(256);
  if (type == GemvType::kF32) {
    MoeVecFallback<float><<<grid, block, 0, stream>>>(
        static_cast<const float*>(base), x, ids, out, rows, cols, n_expert_used,
        expert_stride);
  } else {
    MoeVecFallback<hip_bfloat16><<<grid, block, 0, stream>>>(
        static_cast<const hip_bfloat16*>(base), x, ids, out, rows, cols,
        n_expert_used, expert_stride);
  }
}

}  // namespace gufo::models::qwen36_a3b::rocm
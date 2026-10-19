// Fused elementwise kernels for the Nemotron 3.5 Bolt GPU engine. All
// activations are f32; the semantics mirror the scalar oracle in
// src/models/nemotron_h_moe/reference.cpp exactly.
#ifndef GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_OPS_H_
#define GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_OPS_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w
int nbl_rms_norm(const float* x, const float* w, float* out, int rows,
                 int hidden, float eps, hipStream_t stream);

/// x[i] *= s
int nbl_scale(float* x, float s, int64_t n, hipStream_t stream);

/// res[i] += add[i]
int nbl_add(float* res, const float* add, int64_t n, hipStream_t stream);

/// dst[i] = max(src[i], 0)^2
int nbl_relu_sqr(const float* src, float* dst, int64_t n, hipStream_t stream);

/// Dequantize token row `row` of a Q8_0 matrix [rows x cols] to f32.
int nbl_dequant_row_q8_0(const void* w, int64_t row, int cols, float* out,
                         hipStream_t stream);

/// One block per token. scores = sigmoid(logits) + bias; ids receives the
/// indices of the highest `top_k` scores (ties resolve to the lower index);
/// weights receives sigmoid(logits[ids]) renormalized by
/// 1 / max(sum, 6.103515625e-5) (when norm) and multiplied by scale.
/// ids/weights are [tokens][top_k].
int nbl_router(const float* logits, const float* bias, int n_experts, int top_k,
               int norm, float scale, int n_tokens, int32_t* ids,
               float* weights, hipStream_t stream);

/// dst[t][j] = sum_u w[t][u] * src[(t * top_k + u) * m + j]
int nbl_moe_combine(float* dst, const float* src, const float* w, int m,
                    int top_k, int n_tokens, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_OPS_H_

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

/// Same as nbl_rms_norm where row r scales with w[(r % groups) * hidden].
/// x/out are [rows][hidden], w is [groups][hidden].
int nbl_rms_norm_grouped(const float* x, const float* w, float* out, int rows,
                         int hidden, int groups, float eps, hipStream_t stream);

/// F32 dense GEMM: out[t][n_out] = dot(x[t][K], w[n_out][K]).
int nbl_gemm_f32(const float* x, const float* w, float* out, int n_tokens,
                 int n_out, int k, hipStream_t stream);

/// Dequantize token row `row` of a Q8_0 matrix [rows x cols] to f32.
int nbl_dequant_row_q8_0(const void* w, int64_t row, int cols, float* out,
                         hipStream_t stream);

/// Weight encodings accepted by the F32-activation GEMV family.
enum NblGemvType {
  kNblGemvF32 = 0,
  kNblGemvQ8_0 = 1,
  kNblGemvIq2Xxs = 2,
  kNblGemvNvfp4 = 3
};

/// F32-activation GEMV on dequantized weights of the given encoding:
/// out[n] = dot(x[k], w[n]). Activations stay exact f32, matching the
/// oracle; Q8_1-quantized activations are far too lossy end to end.
int nbl_gemv_f32(int type, const void* w, const float* x, float* out, int n_out,
                 int k, hipStream_t stream);

/// Same as nbl_gemv_f32 over gathered experts: expert for slot t is ids[t],
/// out[t * n_out + n] = dot(x[t * x_stride ..], w_expert[expert][n]);
/// `slots` slots; x_stride is the per-slot element stride of x (0 shares one
/// vector across slots).
int nbl_gemv_moe_f32(int type, const void* w, const float* x,
                     const int32_t* ids, float* out, int n_out, int k,
                     int slots, int64_t x_stride, hipStream_t stream);

/// F32-activation GEMV over ggml NVFP4 weights [n_out][k/64]:
/// out[n] = dot(x[K], w[n]). Keeps activations exact (no Q8_1 rounding):
/// mamba projections amplify weight-only noise through conv/dt.
int nbl_gemv_nvfp4_f32(const float* x, const void* w, float* out, int n_out,
                       int k, hipStream_t stream);

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

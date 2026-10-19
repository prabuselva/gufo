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

/// Batched form: out[t] = row ids[t] of the Q8_0 matrix, ids device array of
/// `tokens` rows.
int nbl_dequant_rows_q8_0(const void* w, const int32_t* ids, int tokens,
                          int cols, float* out, hipStream_t stream);

/// Weight encodings accepted by the F32-activation GEMV family.
enum NblGemvType {
  kNblGemvF32 = 0,
  kNblGemvQ8_0 = 1,
  kNblGemvIq2Xxs = 2,
  kNblGemvNvfp4 = 3
};

/// F32-activation GEMV on dequantized weights of the given encoding:
/// out[t][n] = dot(x[t][k], w[n]) for `tokens` rows of x. Activations stay
/// exact f32, matching the oracle; Q8_1-quantized activations are far too
/// lossy end to end.
int nbl_gemv_f32(int type, const void* w, const float* x, float* out, int n_out,
                 int k, int tokens, hipStream_t stream);

/// Same as nbl_gemv_f32 over gathered experts: expert for slot s of token t is
/// ids[t * slots + s], out[(t * slots + s) * n_out + n] =
/// dot(x[t, s, ..], w_expert[expert][n]); `slots` slots per token; x_stride is
/// the per-slot element stride of x (0 shares one k-vector across slots of a
/// token; tokens are separated by k in that case, by slots * x_stride else).
int nbl_gemv_moe_f32(int type, const void* w, const float* x,
                     const int32_t* ids, float* out, int n_out, int k,
                     int slots, int64_t x_stride, int tokens,
                     hipStream_t stream);

/// Token-tile width of the row-batched GEMV kernels: nbl_expert_index `nt`
/// and every nbl_gemv_rows_* caller must use this value.
#define NBL_ROW_NT 4

/// Routed-expert token index. `ids[s]` is the expert selected by flat slot
/// s in [0, slots). Writes `list` (rank -> slot, grouped by expert) and a
/// flattened tile-table `desc` of ceil(count[e]/nt) entries per expert:
/// desc[3*i] = expert, desc[3*i+1] = list begin rank, desc[3*i+2] = tile
/// length (1..nt); `n_desc` receives the tile count. Position order inside
/// one expert follows atomic scatter order; every slot's outputs are
/// position independent, so results are run-invariant.
int nbl_expert_index(const int32_t* ids, int n_experts, int slots, int nt,
                     int32_t* list, int32_t* desc, int32_t* n_desc,
                     hipStream_t stream);

/// Row-batched GEMV over `tokens` rows with weights read once per 4-token
/// tile: same chunked weight chain and per-output ascending accumulation as
/// nbl_gemv_f32, but each block owns 256 output rows and reuses one
/// activation tile. Dense form: out[t][n] = dot(x[t*x_stride], w[n]).
/// Gated by the caller to prefill-size token counts; decode keeps the
/// (row, token)-grid GEMV for wave coverage.
int nbl_gemv_rows_f32(int type, const void* w, const float* x, float* out,
                      int n_out, int k, int64_t x_stride, int tokens,
                      hipStream_t stream);

/// Row-batched form of nbl_gemv_moe_f32 over the expert-sorted list:
/// out[list[r]][n] = dot(x[ list[r]/xdiv ], w[expert[n tiles]]...) with
/// desc/list from nbl_expert_index and the 1D launch bound
/// grid_tiles <= slots/nt + n_experts. xdiv is top_k when the per-token
/// activation is shared across its slots, 0 when each slot row is used.
int nbl_gemv_rows_moe(int type, const void* w, const float* x,
                      const int32_t* list, const int32_t* desc,
                      const int32_t* n_desc, int grid_tiles, int xdiv,
                      float* out, int n_out, int k, hipStream_t stream);

/// F32-activation GEMV over ggml NVFP4 weights [n_out][k/64]:
/// out[t][n] = dot(x[t][k], w[n]). Keeps activations exact (no Q8_1
/// rounding): mamba projections amplify weight-only noise through conv/dt.
int nbl_gemv_nvfp4_f32(const float* x, const void* w, float* out, int n_out,
                       int k, int tokens, hipStream_t stream);

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

// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {
int nmb_mmq_init(int device);

void nmb_mmq_set_routed_max_expert_rows(int rows);

void nmb_mmq_set_routed_tile_cols(int cols);

int nmb_mmq_routed_tile_cols_for_counts(const unsigned int* counts,
                                        int n_experts);

int nmb_mmq_q8_0_dense(const void* W_q8_0, const float* X_f32, float* out_f32,
                       int M, int N, int K, hipStream_t stream);

int nmb_mmq_nvfp4_dense(const void* W_nvfp4, const float* X_f32, float* out_f32,
                        int M, int N, int K, hipStream_t stream);

int nmb_mmq_iq2_xxs_dense(const void* W_iq2_xxs, const float* X_f32,
                          float* out_f32, int M, int N, int K,
                          hipStream_t stream);

// Staging for N_token rows of a shared [rows, K] activation: quantize once
// into nmb_mmq_dense_q8_1_bytes(rows, K) and feed any *_dense_preq entry.
// Dense weight types all use the same activation layout, so a pre-quantized
// buffer is bit-identical to the per-call quantize of the *_dense entries.
size_t nmb_mmq_dense_q8_1_bytes(int rows, int K);

int nmb_mmq_dense_q8_1_quantize(const float* X_f32, void* X_q8, int rows, int K,
                                hipStream_t stream);

int nmb_mmq_q8_0_dense_preq(const void* W_q8_0, const void* X_q8,
                            float* out_f32, int M, int N, int K,
                            hipStream_t stream);

int nmb_mmq_nvfp4_dense_preq(const void* W_nvfp4, const void* X_q8,
                             float* out_f32, int M, int N, int K,
                             hipStream_t stream);

int nmb_mmq_iq2_xxs_dense_preq(const void* W_iq2_xxs, const void* X_q8,
                               float* out_f32, int M, int N, int K,
                               hipStream_t stream);

int nmb_mmq_nvfp4_dense_vec(const void* W_nvfp4, const float* X_f32,
                            float* out_f32, int M, int N, int K,
                            hipStream_t stream);

int nmb_mmq_iq2_xxs_dense_vec(const void* W_iq2_xxs, const float* X_f32,
                              float* out_f32, int M, int N, int K,
                              hipStream_t stream);

int nmb_mmq_q8_0_dense_vec(const void* W_q8_0, const float* X_f32,
                           float* out_f32, int M, int N, int K,
                           hipStream_t stream);

int nmb_mmq_q4_K_moe_pair_unique(const void* W_a, const void* W_b,
                                 const float* X_f32, const int32_t* ids,
                                 float* out_a, float* out_b, int M, int K,
                                 int n_tokens, int n_experts, int n_expert_used,
                                 hipStream_t stream);

// Routed vector projection for GGML Q4_K/Q5_K/Q5_1/Q8_0. Optional paired
// weights use the same shape, routing and quantized input; both outputs are
// [n_tokens][n_expert_used][M]. W_b and out_b must be supplied together.
int nmb_mmq_moe_vec(int weight_type, const void* W, const float* X_f32,
                    const int32_t* ids, float* out, int M, int K, int n_tokens,
                    int n_experts, int n_expert_used, hipStream_t stream,
                    const void* W_b = nullptr, float* out_b = nullptr);

// Q4_K/Q5_K/Q8_0 gate/up for 1–8 tokens, with the SwiGLU result in out.
// Q4_K also accepts up to 64 independent rows, grouped by expert.
int nmb_mmq_moe_gated_vec(int weight_type, const void* gate, const void* up,
                          const float* x, const int32_t* ids, float* out, int m,
                          int k, int tokens, int experts, int experts_used,
                          hipStream_t stream);

// nmb_mmq_moe_gated_vec over x already in nmb_mmq_quantize_q8_1's layout.
int nmb_mmq_moe_gated_vec_preq(int weight_type, const void* gate,
                               const void* up, const void* x_q8,
                               const int32_t* ids, float* out, int m, int k,
                               int tokens, int experts, int experts_used,
                               hipStream_t stream);

size_t nmb_mmq_q8_1_bytes(int N, int K);

int nmb_mmq_quantize_q8_1(const float* X_f32, void* X_q8, int N, int K,
                          hipStream_t stream);

// Exact decode arithmetic for 1–8 rows, or up to 32 ungated rows.
// Wide projections (M >= 8192, K = 2560) accept up to 48 ungated rows.
// The 320×10240 HC projection also accepts larger multiples of eight.
// Only N output rows are written.
int nmb_mmq_q8_0_dense_vec_preq(const void* W_q8_0, const void* W_gate,
                                const void* X_q8, float* out_f32, int M, int N,
                                int K, hipStream_t stream);

int nmb_mmq_build_ids_maps(const int32_t* ids, int32_t* ids_src1,
                           int32_t* ids_dst, int32_t* expert_bounds,
                           int n_experts, int n_tokens, int n_expert_used,
                           int nchannels_y, int si1, int sis1,
                           hipStream_t stream);

int nmb_mmq_q8_0_moe_raw(const void* W, const float* X_f32, const int32_t* ids,
                         float* out, int M, int K, int n_tokens, int n_experts,
                         int n_expert_used, hipStream_t stream);

int nmb_mmq_q4_K_moe_raw(const void* W, const float* X_f32, const int32_t* ids,
                         float* out, int M, int K, int n_tokens, int n_experts,
                         int n_expert_used, hipStream_t stream);

int nmb_mmq_q5_1_moe_raw(const void* W, const float* X_f32, const int32_t* ids,
                         float* out, int M, int K, int n_tokens, int n_experts,
                         int n_expert_used, hipStream_t stream);

int nmb_mmq_q5_K_moe_raw(const void* W, const float* X_f32, const int32_t* ids,
                         float* out, int M, int K, int n_tokens, int n_experts,
                         int n_expert_used, hipStream_t stream);

int nmb_mmq_iq2_xxs_moe_raw(const void* W, const float* X_f32,
                            const int32_t* ids, float* out, int M, int K,
                            int n_tokens, int n_experts, int n_expert_used,
                            hipStream_t stream);

// nmb_mmq_iq2_xxs_moe_raw with SqReLU (x > 0 ? x * x : 0) folded into the
// activation quantize: X_f32 holds the pre-activation rows.
int nmb_mmq_iq2_xxs_moe_raw_sqract(const void* W, const float* X_f32,
                                   const int32_t* ids, float* out, int M, int K,
                                   int n_tokens, int n_experts,
                                   int n_expert_used, hipStream_t stream);

int nmb_mmq_nvfp4_moe_raw(const void* W, const float* X_f32, const int32_t* ids,
                          float* out, int M, int K, int n_tokens, int n_experts,
                          int n_expert_used, hipStream_t stream);

// K-quant entries dispatched by GGML weight type code (12/13/14 =
// Q4_K/Q5_K/Q6_K, 23 = IQ4_XS). Anything else, or a K that is not a whole
// weight block, returns -1 so the caller keeps the gemv tier.
int nmb_mmq_kq_dense(int weight_type, const void* W, const float* X_f32,
                     float* out, int M, int N, int K, hipStream_t stream);
// X_q8 is the shared D4 staging buffer; it is used only when the weight type
// reads the D4 layout (see mmq_get_q8_1_ds_layout). X_f32 is the matching
// fp32 twin, quantized internally per type for DS4-layout weights (Q4_K,
// Q5_K). Returns -1 when neither source can be used for the type.
int nmb_mmq_kq_dense_preq(int weight_type, const void* W, const void* X_q8,
                          const float* X_f32, float* out, int M, int N, int K,
                          hipStream_t stream);
int nmb_mmq_kq_dense_vec(int weight_type, const void* W, const float* X_f32,
                         float* out, int M, int N, int K, hipStream_t stream);
int nmb_mmq_kq_moe_raw(int weight_type, const void* W, const float* X_f32,
                       const int32_t* ids, float* out, int M, int K,
                       int n_tokens, int n_experts, int n_expert_used,
                       hipStream_t stream);
int nmb_mmq_kq_moe_vec(int weight_type, const void* W, const float* X_f32,
                       const int32_t* ids, float* out, int M, int K,
                       int n_tokens, int n_experts, int n_expert_used,
                       hipStream_t stream);
}

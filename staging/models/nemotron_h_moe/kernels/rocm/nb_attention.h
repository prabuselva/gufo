// GPU attention kernels for Nemotron Bolt (NoPE GQA): KV append and
// flash-style causal scaled-dot-product attention versus running state.
#ifndef GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_ATTENTION_H_
#define GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_ATTENTION_H_

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Appends `tokens` freshly projected rows to the layer KV cache. k_src and
// v_src are [tokens][*dim]; the cache holds [capacity][*dim] rows with k in
// the first kvdim floats and v in the next kvdim floats per row.
int nbl_attn_append(const float* k_src, const float* v_src, float* kv, int pos0,
                    int tokens, int kvdim, hipStream_t stream);

// Causal NoPE attention over the appended cache. q is [tokens][heads*hd];
// out is [tokens][heads*hdv]. Row t attends positions [0, pos0 + t].
// kv rows are [capacity][2*kvdim]; kv head g = h * kvh / heads owns
// k = row[0, kvdim) floats g*hd. and v = row[kvdim, 2kvdim) floats g*hdv. (hd
// and hdv are equal in the checkpoint; the stride math keeps them distinct.)
int nbl_attn(const float* q, const float* kv, int pos0, int tokens, float* out,
             int heads, int hd, int hdv, int kvh, int kvdim, float scale,
             hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_ATTENTION_H_

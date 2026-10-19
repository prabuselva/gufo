// Mamba-2 sequential state-space scan for the Nemotron 3.5 Bolt GPU engine.
// One kernel runs the whole recurrent part of a chunk: causal depthwise conv
// + SiLU over the xBC channels, per-head Euler-SSM update with the state
// held in registers across the token loop, and the z gate; the grouped
// RMSNorm runs separately (nbl_rms_norm rows = tokens * groups). It covers
// decode (T = 1) and prefill with identical semantics to
// reference.cpp StepMamba.
#ifndef GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_MAMBA2_H_
#define GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_MAMBA2_H_

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/// zxbc: [T][proj] scaled in-projection output
///   (z | nx | B*C | dt). cw [xbc][taps] taps-fast, cb [xbc], a/d/dtb [heads].
/// conv_state [(taps-1)][xbc] and ssm_state [heads][hd][state] persist and
/// are updated in place. Writes y [T][inner] WITHOUT the grouped norm.
/// Requires state % 4 == 0 (state/4 <= 64), inner % 64 == 0, 2 <= taps <= 4.
int nbl_mamba_scan(const float* zxbc, const float* cw, const float* cb,
                   const float* a, const float* d, const float* dtb,
                   float* conv_state, float* ssm_state, float* y, int inner,
                   int heads, int hd, int state, int groups, int taps, int xbc,
                   int proj, int tokens, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_NB_MAMBA2_H_

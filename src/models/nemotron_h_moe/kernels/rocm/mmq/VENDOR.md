# Quantized HIP kernels

Copied from the Qwen3.8-Flash-Next vendored tree
(`src/models/qwen38_flash_next/kernels/rocm/mmq`) and renamed `qfn_` to
`nmb_` so both libraries link into one binary. Adapted from
`ggml-org/llama.cpp` at `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see
`LICENSE`); the upstream source directory is `ggml/src/ggml-cuda`, built with
HIP for AMD Strix Halo (`gfx1151`).

`nmb_mmq.h` is the public adapter. It supplies Q8_0, NVFP4 and IQ2_XXS dense
projections (MMQ tier plus an MMVQ vector tier for decode-length batches),
and Q4_K, Q5_K, Q5_1 and Q8_0 routed expert projections, including the paired
Q4_K gate/up path. `nmb_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

Gufo additions over the upstream pin: the `vec_dot_nvfp4_q8_1` and
`vec_dot_iq2_xxs_q8_1` dots are wired into the MMVQ dispatch
(`get_vec_dot_q_hip`, `get_vdr_mmvq`, `mul_mat_vec_moe_dispatch`) and the
dense-vec entries route the single-expert ids case through the same routed
kernel, so both tiers share one dot-product order per format.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix.

When importing an upstream fix, preserve the `nmb_` namespace, adapter and
HIP-only dispatch, update the pin, and rerun
`src/models/nemotron_h_moe/tools/nvfp4_probe.hip` plus the nemotron ROCm
operator tests under `tests/models/nemotron_h_moe/`.

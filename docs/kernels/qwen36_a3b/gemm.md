# Qwen3.6-35B-A3B — Prefill GEMM Tier (`gemm.hip.cpp`)

Source: `src/models/qwen36_a3b/kernels/rocm/gemm.hip.cpp` (151 lines).
Public API: `gemm.hpp`. Batched projections `out[t][r] = Σ_k W[r][k] ·
x[t][k]` for the prefill path. The file is mostly **dispatch**: the heavy
compute lives either in the shared mmq library or in hipBLAS. Weights
always stay in their native encoding.

## Tier dispatch

| Encoding | Path |
| --- | --- |
| Q8_0 | `qfn_mmq_q8_0_dense` / `qfn_mmq_q8_0_moe_raw` — the shared MMQ tensor-core path from `src/models/qwen38_flash_next/kernels/rocm/mmq/` (CMake target `gufo_qwen38_flash_next_mmq`, header dir exported; documented with that model) |
| F32 | `hipblasSgemm` (W stored row-major → `OP_T · OP_N`) |
| BF16 | `NarrowBf16` of the activations + `hipblasGemmEx` (BF16 in, F32 out, `COMPUTE_32F`) |
| MoE, non-Q8_0 | `MoeVecFallback<T>` (correctness path, not perf) |

The production prefill MoE does **not** go through `GemmMoe` for Q8_0
when the F16 WMMA tier accepts the shape — see
[routed-f16-moe.md](routed-f16-moe.md); `GemmMoe` is the fallback the
executor calls when `RoutedF16Gemm` returns false.

## Kernels

### `NarrowBf16` :42

Elementwise F32 → BF16 of the activation batch (`batch·cols`), block
256, grid-stride. Writes `Bf16Scratch`.

### `MoeVecFallback<T>` :57

Native grouped-MoE for the F32/BF16 expert stacks the mmq routed path
does not cover. One block per (token, slot) output vector (`grid =
n_tokens·n_expert_used`), block 256; each thread strides output rows
and runs a serial float dot over `cols`. Expert row base selected from
`ids` on device (no host round-trip). Naive bandwidth behavior (each
thread walks a full row serially) — it exists so unquantized artifacts
stay correct, not fast.

## Support machinery

- `BlasHandle` :17 — process-wide hipBLAS handle (the executor drives a
  single stream; the stream is rebound on every call).
- `Bf16Scratch` :29 — lazily grown static device buffer (min 64 K
  elements), race-free because one stream is live.

## Launchers

### `Gemm` :80

Dense projection. Q8_0 → mmq dense; F32 → SGEMM; BF16 → narrow +
GemmEx. `row_bytes` is unused (the encodings are stride-derivable from
`cols`).

### `GemmMoe` :123

Grouped MoE projection: `out[(t·used + s)][r] = Σ_k W[ids[t·used+s]][r][k]
· x[t][k]`. Q8_0 → `qfn_mmq_q8_0_moe_raw`; else `MoeVecFallback<T>`
with `expert_stride = rows·cols`.

## Numerics

- The mmq path's accumulation order is pinned by
  `qwen36_a3b_rocm_gemv_test` (it exercises the GEMM encodings against
  the scalar oracle at batch > 1).
- hipBLAS paths are only used for F32/BF16 artifacts; the shipped
  Q8_K_XL artifact exercises mmq + the routed F16 tier.

## Optimization notes

- `MoeVecFallback` is O(rows·cols) per block with zero tiling; if a
  BF16-expert artifact ever becomes a target, port it to the routed F16
  kernel (it already accepts `WeightType::kBF16`) instead of optimizing
  this one.
- The BF16 path narrows activations on every call; a fused
  narrow-into-mmq or keeping activations BF16 end-to-end is unexplored.
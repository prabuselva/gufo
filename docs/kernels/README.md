# Gufo Kernel Reference

Per-kernel documentation for the HIP kernels Gufo runs on Strix Halo
(gfx1151). Each document pins, for every `__global__` kernel in a source
file: what it computes, the launcher and dispatch rule, the launch
configuration, the algorithm and memory-access pattern, the numerical
contract against the scalar oracle, and the measured performance notes and
open optimization ideas.

These documents are the working set for kernel optimization work under
`.agents/skills/optimize-kernel/SKILL.md`. Read the model's
`docs/models/<model>/OPTIMIZATIONS.md` for the stage profile that says
*which* kernels matter; read the kernel document here for *how* the kernel
is built.

## Model indexes

| Model | Sources | Documents |
| --- | --- | --- |
| [Qwen3.6-35B-A3B](qwen36_a3b/README.md) | `src/models/qwen36_a3b/kernels/rocm/` | [fused ops](qwen36_a3b/fused-ops.md), [GEMV](qwen36_a3b/gemv.md), [GEMM tier](qwen36_a3b/gemm.md), [routed F16 MoE](qwen36_a3b/routed-f16-moe.md) |
| [Gemma-4-26B-A4B](gemma4/README.md) | `src/models/gemma4/kernels/rocm/` | [fused ops](gemma4/fused-ops.md), [GEMV](gemma4/gemv.md), [GEMM tier](gemma4/gemm.md), [routed F16 MoE](gemma4/routed-f16-moe.md) |

## Hardware baseline (gfx1151)

- Wavefront size 32. `warpSize`-generic code is still queried once and
  cached where it matters (`GdnDeltaLoop`).
- Measured streaming peak ~240 GB/s
  (`tools/bench/gfx1151_peak.hip`). Memory-bound kernels are judged
  against this ceiling; the decode GEMV tier reaches 231 GB/s (96 %).
- Matrix cores: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (F16 in,
  F32 accumulate) — the only WMMA form used.
- Standalone kernel benches: `tools/bench/build.sh` (preserves compiler
  flags); ISA inspection: `tools/prof/isa_mix.py`; pipeline/wall profiles:
  `tools/prof/prof.py`.

## Conventions

- Activations are row-major `float32` unless the document says otherwise.
- Every kernel has a scalar oracle in the model's `reference.cpp` (or the
  test file); the oracle error budget is recorded per kernel. A kernel
  rewrite must reproduce the oracle to the recorded error, not merely
  "close".
- Production paths keep quality: fast-math intrinsics (`__expf`) are only
  used where a document records that the oracle does not pin the path
  (e.g. the routed F16 MoE epilogue); decode numerics use exact `expf`.
- Launchers are the public API (`kernels.hpp`, `gemv.hpp`, `gemm.hpp`,
  `routed_f16.hpp`); `__global__` kernels are file-static except
  `ExpertCountsKernel`.
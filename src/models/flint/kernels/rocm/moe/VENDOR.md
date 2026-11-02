# Flint grouped MoE prefill route (Gemma-4 fork)

Fork of the Gemma-4 prefill MoE route: `moe_prefill.hpp` /
`moe_prefill.hip.cpp` copied verbatim (namespace-only rename) from
`src/models/gemma4/kernels/rocm/moe_prefill.*`, and
`routed_compact.hpp` / `routed_compact.hip.cpp` extracted device-side
builders from the same directory's `routed_f16.hip.cpp`
(`RoutedPadBoundsKernel`, `RoutedScatterKernel`, `BuildRoutedTilesKernel`,
narrowing) — the 16-padded bucket layout, `expert | tile << 16` tile maps
and F32→F16 activation narrowing that `LaunchRoutedHalfGemm` consumes.
Gemma-4's route was itself "ported from the Qwen3.8-Flash-Next routed
GEMM" (see its `routed_f16.hpp` header); the lineage is Gufo-internal.

Flint owns this copy to A/B the Q8_0 grouped route (128-row buckets, WMMA
f16 fragments, weights dequantized after the LDS read) against the
mmq-routed entry points at Flint shapes without touching the Gemma-4
production kernels. Flint uses the GEMMs per tensor (`geglu=false`): its
gate and up weights are separate tensors with a SiLU-Mul activation, so
activation fusion stays out; `routed_compact` replaces the per-layer
counts→host→mmq-launch-bound round trip with pure device-side compaction.

Forked at the working tree of 2026-10-27 (Gemma-4 commits `9ef6ecc4`,
`def99473`). Any upstream fix lands here first for Flint evaluation; do
not back-propagate without rerunning the Flint gates.

# Ling 3.0 Flash — ROCm kernel sources

Sources: `src/models/ling3_flash/kernels/rocm/`. The scalar oracles live in
`src/models/ling3_flash/reference.cpp` / `cpu_ops.cpp`; every kernel is
compared against them by the `ling3_rocm_*` tests
([../../../tests/models/ling3_flash](../../../tests/models/ling3_flash)).

| Source | Contents | Used by |
| --- | --- | --- |
| `gemv.hip.cpp` | Quantized GEMV tier: decodes Q4_K/Q5_K/Q6_K/Q8_0 trunk rows and the Q4_0 MTP rows on the fly, one warp-group per output row, split-K accumulations for narrow matrices | trunk and MTP projections |
| `kernels.hip.cpp` | Fused single-token operators between projections: RMSNorm, KDA short conv + sigmoid gate + delta-rule step (recurrent state, no KV), RoPE halves for MLA, sigmoid group-limited top-8-of-4x8 routing, SwiGLU with per-expert clamps | KDA/MLA/MoE layers |
| `attention.hip.cpp` | Split-KV absorbed-MLA decode attention over the latent+rope key (rank 512 + 64), value head 128 | MLA layers (every 6th trunk layer, MTP block) |

Conventions: fp32 activations, `wave32`, `expf`/`sigmoid` kept faithful (no
`-ffast-math`; `-fno-finite-math-only`) so oracle error budgets stay tight.
The executor (`kernels/rocm/executor.cpp`) streams the sharded artifact to
the GPU once; sessions own KV plus KDA recurrent state and share the scratch.

Per-kernel launch configurations and measured ceilings are recorded during
optimization work following `.agents/skills/optimize-kernel/SKILL.md`; run
`tools/bench/build.sh` benches and `tools/prof/prof.py` before editing.

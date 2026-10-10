# Ling 3.0 Flash quality

Gufo-path numerical checks pass; official target-model parity is not yet
established. Tests: `tests/models/ling3_flash` via CTest labels
`ling3_flash.*`; the GPU suites locate the pinned artifact through
`GUFO_LING3_FLASH_GGUF`.

## Retained checks

| Check | Retained result |
| --- | --- |
| `ling3_flash.rocm_attention` | Passed on the 16-warp fused prefill build (Oct 11, 2026); worst prefill absolute error 9.7e-5 vs fp64 oracle (fixed contract 5e-4), decode 2.4e-6. |
| `ling3_flash.rocm_kda_prefill` / `rocm_forward` / `rocm_inference` | 3/3 passed on the `9b1434ca` kernel change (gpu-test build, assertions on). |
| Fused-form kernel change (old → new) | Harness random-input gate: maxabs 4e-9 vs the previous fused form (compiler contraction only; relative noise on near-zero outputs reaches ~1e-7 — treat relative metrics with |ref|≈0 as meaningless). |
| Register/spill audit | `MlaPrefillFusedKernel` release fatbin (Release flags + `-UNDEBUG`): 118 VGPR, 0 spill, 0 SGPR spill, 36864 B LDS. Prior spill state (`-DNDEBUG`, 143 VGPR / 316 B per lane) is the root cause of the fixed 6× prefill slowdown; see [experiments](EXPERIMENTS.md). |
| Microbench replay | `tools/bench/ling3_mla_prefill_bench.hip` `replay` reproduces engine-event timings for all prefill chunk depths; `cand` gates kernel variants (bitdiff, maxabs, nan) before timing. |

## Benchmark method

`gufo bench` cells run one rep per row on an exclusive GPU via
`tools/bench/gpu_exclusive.sh` (server stopped and VRAM released before the
run, respawned and `/v1/models`-confirmed after); timings are therefore
directly comparable across binaries with identical build presets.
`± 0.00` reflects a single rep, not repeated-run variance; historical spread
is ~2% for pp and tg cells.

## Gaps

- Token-level continuation replay (functional suite) has not been rerun for
  the prefill attention changes; run the affected
  [`tests/functional`](../../../tests/functional/README.md) profile before
  claiming end-to-end equivalence.
- No reference-project quality parity yet (logit comparison against the
  upstream implementation).

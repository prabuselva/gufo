# Gemma-4-26B-A4B — Optimizations

Retained production optimizations only; A/B evidence lives in
[EXPERIMENTS.md](EXPERIMENTS.md). Hardware ceilings:
`tools/bench/gfx1151_peak.hip`. Profile with `tools/prof/prof.py` and
`tools/prof/isa_mix.py`.

## Kernel tiers (planned, M5)

| Workload | Tier |
| --- | --- |
| Decode projections (batch 1) | `gemv` Q8_0/BF16/F32, one warp per row |
| Prefill dense projections | Q8_0 → `qfn_mmq_q8_0_dense` (`gufo_qwen38_flash_next_mmq`); BF16 → hipBLAS GemmEx; F32 → SGEMM |
| Prefill routed MoE | `routed_f16` WMMA (dequant-after-LDS, fused gate+up `kPair`), Q8_0 + BF16 experts |
| Decode routed MoE | grouped GEMV over the compacted expert buckets |
| Attention | WMMA flash prefill + decode, head_dim 256 (window 1024) / 512, F16 KV mirror |

## Roofline

Decode moves ~27.6 GB of resident weights per token at greedy batch 1
(~3.1 GB active/token at top-8 of 128 experts + shared FFN + attention).
Against the ~240 GB/s streaming peak (`gfx1151_peak.hip`) the greedy roofline
is ~75–80 t/s before MTP; MTP amortizes the reads over k+1 verified tokens.
Measured values land here per phase.

## Retained wins

(to be filled as M5–M11 land)
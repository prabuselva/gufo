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

### vec4 Q8_0 GEMV (decode)

Decode projections and the fused-qkv GEMV load four int8 codes per lane (one
`uint32` at byte `4·lane` of a 128-byte group of four Q8_0 blocks) instead of
one byte per lane, applying each block's own scale to its four codes. Used for
dense Q8_0 when `cols % 128 == 0` and for the fused-qkv `Multi4` when every
projection has `nb % 4 == 0` (all gemma4 projections qualify). Matched
exclusive decode pp128/tg256: **38.26 → 39.91 t/s (+4.3%)** with a
bit-identical `output_sha256`. Grouped MoE GEMV stays scalar — its 1e-4
contract cannot absorb the vec4 reorder (measured 2.19e-4). See
[EXPERIMENTS.md](EXPERIMENTS.md).

### float4 RMSNorm (decode + prefill)

`RmsNormKernel` and `FusedAddRmsNormKernel` take a `dim % 4 == 0` fast path:
16-byte `float4` loads/stores and a `float` per-thread sum-of-squares (the
block reduce stays in `double` over the ≤1024 partials). The scalar `double`
path remains the fallback for `dim % 4 != 0`; every gemma4 norm dim (2816, 512,
256) is a multiple of 4. Standalone kernel: decode shape (rows=1, dim=2816)
**6.43 → 3.22 µs (−50%)**, prefill shape (rows=512) **48.9 → 19.6 µs (−60%)**.
Matched exclusive decode pp128/tg256: **39.98 → 41.75 t/s (+4.4%)**, pp128
+1.0%. Worst-relative error vs the `double` oracle is **1.55e-7** (inside the
1e-5 norm contract) and the greedy `output_sha256` is bit-identical to the
baseline. See [EXPERIMENTS.md](EXPERIMENTS.md).
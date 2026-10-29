# Flint Preview 40B-A2B — Experiments

Retained and rejected prefill decisions, one line each. Full evidence,
commands and matched numbers live in the numbered entries of
[STATUS.md](STATUS.md).

## Retained (shipped defaults)

- C7a/C7g WMMA attention replaces the SIMT prologue path (config sweep in
  `tools/bench/flint_attn_wmma_bench`).
- C7b/C7c token-tiled dense prefill tiers: Q8 GEMV tile + mmq tensor-core
  dispatch for batched Q8 projections.
- C7d/C7h prefill chunk 256 → 4096: fewer launches and per-chunk MoE
  fixed cost; pp8192 1657 → 1707 → (with C7l) 1930.
- C7e token-tiled bf16 router GEMV (3.8–4.3x the plain per-row GEMV).
- C7f routed mmq launch bounds derived from per-expert counts.
- C7i/C7k flint_mmq fork (mechanical rename, `VENDOR.md` provenance) so
  the Q8_0 pipeline evolves without touching other models' kernels.
- C7l grouped Q8_0 MoE prefill route (Gemma-4 `moe_prefill` fork):
  gateup +77 %, down +120 % TF vs the mmq route; pp2048 +20 %, pp8192
  +13 %, pp16384 +9 %.
- C7m W8A8 int8-WMMA dense prefill route (shared qwen tier) for eligible
  projections with pre-quantized activation sidecars.
- C7n vectorized elementwise kernels + pipelined router tile GEMV.
- C7o binary16 routed down-projection slot rows.
- C7p fused grouped gate/up launch with SiLU epilogue.
- C7q fused SwiGLU gate/up epilogue for the dense FFNs.
- C7u activation-quantize four Q8_1 blocks per wavefront (float4 loads).
- C7v fused RMSNorm + (residual add) + Q8_1 quantize, gated
  `rows >= 3072 && cols % 256 == 0`: −80 ms per pp8192 at kernel level,
  bit-identical by construction; e2e win below the bench noise floor,
  zero regressions.

## Rejected / investigated, no adoptable lever

- C7j fused-iteration LDS staging in the mmq Q8_0 pipeline: 97.6 KB
  blocks cut residency 3 waves → 1; control −47 % — barrier win (+22 %)
  cannot pay for it (inert knob left off).
- Router GEMV variants: LDS-staged x tiles with per-chunk barriers
  −18..−25 %, 64-token tiles −300 %, plain `__ldg` −4 %
  (`tools/bench/flint_gemv_bf16_tile_bench.hip`).
- C7r attention correction-chain rework — investigated, no adoptable
  change.
- C7s gate/up prefetch ring — investigated, no adoptable lever.
- C7t attention long-context scheduling — investigated, no adoptable
  change; the WMMA partition and read masks stay.
- C7w prefill chunk 4096 → 8192: the grouped route is not per-chunk
  weight-read floor-bound; pp8192 flat (−0.3 %), pp16384 −0.8 % across
  all matched rounds; 4096 stays.

## Known open levers (measured cost, not yet attacked)

- Decode at depth ≤ 16 K trails llama.cpp (41.7–42.4 vs 43.7–47.7 tg);
  the C7 campaign was prefill-only.
- Dense `mul_mat_q` re-encodes its activation per call on the small-col
  projections (~85 ms per pp8192 profile); a shared pre-encoded sidecar
  in the mmq layout is untried.
- QkEpilogue ~111 ms per pp8192 profile: not yet analyzed.

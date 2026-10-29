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
- C8a multi-row Bf16 router tile (`GemvBatchBf16TileRKernel<8,8>`,
  dispatch gate `rows % 64 == 0`): one warp owns 8 rows × 8 tokens so a
  loaded x k-step feeds 8 accumulator chains; per-token k-ascending
  chains keep it bitwise identical to the per-token GEMV
  (`tools/bench/flint_gemv_tile2_bench.hip`, mismatches=0 on
  256×2048×{256,1024,2048,8192}). Router kernel 0.96 → 0.41 ms at the
  pp2048 shape (2.4×); pp2048 2677.90 → 2778.63/2760.90 (+3.7 %/+3.1 %),
  pp8192 2159.85 → 2246.51 (+4.0 %), tg unchanged, all three flint gate
  tests bitwise clean in the same session.
- C8b wave-32 128×128 4×2 K-quant tile (`TryLaunchQuantPrefillWave32`) for
  flint `Project()` shapes `batch >= 2048 && (rows >= 4096 || (batch >=
  4096 && cols >= 4096 && rows >= 2048))`: a full tile-config matrix,
  16 configs over the six dense projection shapes at batch
  {512, 2048, 4096} (`tools/bench/flint_w8a8_bench.hip`, wave-64 configs
  through a native `-mwavefrontsize64` launcher TU), every candidate
  bitwise identical to the shipped output. The wave-32 tile wins
  +3.5/+5.5 % (attn_q, dense ffn gate/up at batch 2048) and
  +4.4/+5.8/+2.2 % (attn_q, attn_out, ffn gate/up at 4096); outside the
  zone wave-64 wins by 3–19 %, so the gate keeps the shared launcher and
  every decode/small batch on it. E2E 2776.90/2790.32 pp2048 and
  2238.33/2248.36 pp8192 — an e2e effect below the matched-run spread
  (the shape wins shrink once the projections interleave with the rest
  of the forward), zero regressions; kept as default. The matrix run also
  localized a bench-only allocation bug (the tiled Q8_1 buffer was sized
  without the activation-sum sidecar; production uses
  `QuantizedActivationBytes`, which includes it).
  Artifacts `flint-c8b-w8a8-config-matrix.log`, `flint-c8b-adoption.log`.
- C9 int8 W8A8 attention K/V prefill projections: `W8A8Eligible` extended
  with `rows >= 512 && batch >= 1024`, and `kv8_ok` reuses attn_q's
  pre-quantized `qa8` sidecar for attn_k/attn_v. At the 512×2048 kv shape
  the shipped tier beats mmq by 52/30/54 % at batch {1024,2048,4096}
  (0.222 → 0.155 ms at 2048), while mmq wins +53 % at batch
  512, hence the batch gate; maxrel≈1.0e-04, relL2≈5e-6 against the bench's
  exact same-quantization reference (`tools/bench/flint_w8a8_bench.hip`).
  E2E pp2048 2842.47/2851.07 (+2.3/+2.6 %), pp8192 2282.92 (+1.6 %),
  pp16384 1703.15 (+4.6 % — every 4096-token chunk dispatches it), tg
  unchanged; three gate tests clean. Outputs are byte-identical below
  batch 1024 (850-token probe control), so every bitwise contract keeps
  its range. At 2116 tokens the greedy trajectory flips at a 0.23-margin
  near-tie top-1 with the same magnitude as a *layer-0-only* variant —
  whose chain matches full-C9 exactly — i.e. the int8 accumulation-order
  class already shipped since C7m, nothing new. The same bench run fixed
  a packed-weight-layout bug in the bench itself (36-byte float-scale
  blocks against the kernels' 34-byte `block_q8_0`) that had corrupted
  earlier runs with deterministic NaNs; kernel timings were unaffected.
  Artifacts `flint-c9-kv-w8a8-matrix.log`, `flint-c9-kv-w8a8-adoption.log`.

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

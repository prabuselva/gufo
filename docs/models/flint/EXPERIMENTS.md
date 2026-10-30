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
- C10 QkEpilogue 8-token walk: the single-token epilogue ran at 54 % of the
  DRAM model (launch/block overhead + per-token `powf` and cos/sin). One
  block now walks `kTokensPerBlock = 8` tokens with the block-uniform
  gamma/frequency/row-base setup hoisted out of the token loop and
  `sincosf` replacing the cosf/sinf pair; the per-token reduce tree, FP
  order and rounding are untouched, and
  `tools/bench/flint_qk_bench.hip` audits both bit-identical whole-buffer
  output against the shipped kernel at every probed shape (B 1-class..4096)
  and sincosf-vs-cosf/sinf equality over the full rope domain (theta
  1e4..5e6, pos < 131072, 0 mismatches). Kernel level 1.44x (rope) / 1.12x
  (NoPE) at B=2048 → 230 GB/s of the modelled traffic; the pp2048 profile
  confirms 575.0 → 366.6 µs/call (3.6 % → 2.33 %). Matched e2e +0.8/+1.0/+1.9 %
  over the same-morning pre-C10 control (2836.80/2276.36/1692.00 →
  2858.40/2298.76/1723.65), tg unchanged; three gate tests clean and
  bitwise-safe by construction. Artifact `flint-c10-qk-epilogue.log`.
- C11 WMMA attention `<8, 2>` query blocks: the attention kernel runs at
  18-25 % of the KV streaming roofline (3.2 ms windowed-513 / 6.5 ms full
  per layer at pp2048 against ~0.58/~1.13 ms minima), so the shipped
  `<8, 1>` block shape was re-swept wider after generalizing the 256
  -thread staging assumption to `kThreads` (guarded `store_keys`/
  `store_values`, byte-identical at 256). `<8, 2>` (512 threads, 16 row
  blocks, 54.5 KB LDS) is bitwise-identical to `<2, 4>` on both measured
  shapes in `tools/bench/flint_attn_wmma_bench` and beat `<8, 1>` by
  9 % windowed / 7 % full (3.37 → 3.08 and 8.50 → 7.94 ms). A
  timing-only probe clone in the same bench showed the three in-loop
  `__syncthreads` cost just 8-12 % (windowed 3.08 → 2.82, full 7.94 →
  6.95 with barriers removed — invalid numbers, diagnostic only), which
  rules out double-buffered staging as the missing term. Production
  dispatch `Launch<8, 2> || Launch<2, 4>`; gates 3/3 bitwise. Matched
  same-afternoon e2e: pp8192 2311.78 → 2403.31 (+4.0 %) and pp16384
  1712.85 → 1797.72 (+5.0 %), both single deterministic runs; pp2048
  2887.75 → 2893.14 (+0.2 %) inside its ±7.2 r2 noise — the global
  layers that grow with context carry the gain. Artifact
  `flint-c11-attn-qb2.log`.

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
- C11 barrier double-buffering for attention staging — rejected on the
  measured stall share: removing all three in-loop barriers (invalid
  timing-only probe) buys only 8-12 % of kernel time, so an LDS-doubling
  residency trade cannot pay for itself.
- C11 attention prefetch depth 2 (bit-identical K/V double-ahead staging
  in a bench clone of the shipped kernel) — rejected:
  bitwise-clean but -17 % slower (windowed 3.05 to 3.58 ms, full 7.83 to
  9.25 ms); the staging latency is not the off-roofline term.
- C12 gate/up weight prefetch ring in `RoutedHalfDownKernel` — rejected.
  A bit-identical ring-depth knob (1..4 LDS row buffers plus register
  slots, `tools/bench/flint_moe_down_probe.hip`) is -21 %/-25 % faster
  than the shipped depth-1 pipeline at the probe's token-heavy
  [gate|up] shapes, but the same rings measured end-to-end give no win
  anywhere (-0.9 % to +0.1 % across ring depths 2 and 3 at
  pp2048..pp16384 even though the geglu kernel total in a pp2048
  profile improves 2.8 % at ring 2: the probe shapes are token-heavy
  compared to production geglu tiles, and the ring's LDS growth (28.6
  to 38.9/49.2 KB) trades residency the shipped shapes need. Confirms
  C7s: geglu stays at ring depth 1.
- C7w prefill chunk 4096 → 8192: the grouped route is not per-chunk
  weight-read floor-bound; pp8192 flat (−0.3 %), pp16384 −0.8 % across
  all matched rounds; 4096 stays.
- C13 gate/up token tiles 128 → 96/64/32 in `RoutedHalfDownKernel` —
  rejected. A dual tile map (96-row tiles for geglu, 128-row for down)
  is bit-identical (probe bit-diff 0 at every shape; gates 3/3) and the
  probe sweeps `gateup w+up` from 12.052 → 4.645/5.294/6.592 ms at chunk
  4096 (−61/−56/−45 %), but matched e2e gives nothing: pp2048
  2893.14 → 2892.61, pp8192 2403.31 → 2385.39, pp16384 1797.72 →
  1789.13, and a profile confirms the 96-row kernel *is* resident with a
  slightly worse production average (1.728 vs 1.65-1.70 ms). Production
  geglu launches ~256 live tiles at ~12-row buckets with a 3824-block
  grid — far past any residency limit and already streaming the sparse
  weight rows at the ~190-200 GB/s roofline the down kernel shares; the
  probe grid is small enough that its 128-row staging hurts residency,
  which is what the sweep measured instead (probe absolutes are ~7x off
  production for this kernel). Second recurrence of the C12
  probe-versus-production lesson: token-tile sizing and prefetch depth
  are both closed for `RoutedHalfDownKernel`. Artifact
  `flint-c13-geglu-tile-rejected.log`.

## Known open levers (measured cost, not yet attacked)

- Decode at depth ≤ 16 K trails llama.cpp (41.7–42.4 vs 43.7–47.7 tg);
  the C7 campaign was prefill-only.
- Dense `mul_mat_q` re-encodes its activation per call on the small-col
  projections (~85 ms per pp8192 profile); a shared pre-encoded sidecar
  in the mmq layout is untried.
- The grouped route's geglu and down kernels stream sparse weight rows
  at the ~190-200 GB/s roofline at pp2048 (C13); only changing the data
  movement contract, not the tile schedule, can move them — e.g. folding
  `SumExperts` into the down epilogue, which needs the quality-evidence
  class like C9.

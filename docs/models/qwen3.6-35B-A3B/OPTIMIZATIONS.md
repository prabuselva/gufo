# Qwen3.6-35B-A3B — Prefill & Decode Optimizations (gfx1151)

Running log of the kernel work on `src/models/qwen36_a3b` on Strix Halo
(gfx1151, measured streaming peak ~240 GB/s via `tools/bench/gfx1151_peak.hip`).
All numbers are for the Q8_K_XL artifact (Q8_0 + F32 + BF16 weights, kept
quantized on device). Geometry: hidden 2048, 40 layers (30 GDN + 10 GQA),
16 q-heads / 2 kv-heads / head_dim 256, ssm 16 k-heads / 32 v-heads / head_dim
128, 256 experts top-8 with expert_ff 512, vocab 248320.

Headline: **prefill 6480 ms → 1630 ms**, **decode 14.1 tps → 33.3 tps**.
Target: ≥60 tps (roofline below).

## Prefill (6480 → 1630 ms)

Landed in `47c27ce` (batched GEMM prefill path).

1. **Batched GEMM prompt path** — the whole prompt is processed as batches
   instead of per-token GEMV loops; linear projections go through the WMMA
   GEMM (`kernels/rocm/gemm.hip.cpp`) with the (expert, row-tile) map for MoE.
2. **Tiled prefill attention** (`AttentionPrefillTiled`) — WMMA tiles over
   the KV block replace the naive per-query kernel
   (`AttentionPrefillNaive` kept as reference).
3. **`GdnDeltaLoopKernel`** — single-warp-per-row sequential delta-rule loop
   (`kRows × 8` blocks) replaces the shared-memory parallel scan for the
   linear-attention prefill; matches the reference recurrence exactly.
4. **`SplitQGateKernel`** — fuses the Q/gate split of the fused QKV projection
   into one pass.
5. **Prefill GDN pipeline** — `GdnConvPrefillKernel`,
   `GdnNormQkPrefillKernel`, `GdnHistoryUpdateKernel`,
   `GdnOutNormPrefillKernel` keep the conv/norm/state-update chain in
   batched kernels.

## Decode (14.1 → 33.3 tps)

Landed in `49b6ee3`.

1. **Flash-decoding attention** (`AttentionDecodeSplitKernel` +
   `AttentionDecodeCombineKernel`, `kernels.hip.cpp:496/580`).
   Dispatched for `head_dim == 256`: `splits = ceil(n_kv / 64)`, capped at 32.
   Grid `heads × splits`, block 256; each wave owns one KV token, lane `l`
   holds `q[l + 32m]`; online max/sum/acc per wave, 8-wave LDS fold writes
   partials `[head][split][head_dim+2]`; the combine kernel folds splits and
   applies the sigmoid output gate. The old single-block
   `AttentionDecodeKernel` (one block per head, serial over KV) remains the
   fallback. **attn_core: 1.28 ms → 0.090 ms.**
2. **Grouped MoE decode GEMV** (`GemvGroupedQ8_0`, `GemvGroupedDense<T>`,
   launcher `GemvGrouped`, `gemv.hip.cpp:59/93/243`).
   Block 128; wave `w` owns the (slot, row) pair `blockIdx.x * 4 + w`;
   `row = w + ids[slot] * expert_stride_blocks + r * nblocks`. One launch per
   projection over all 8 selected experts (gate/up/down), plus one fused
   SwiGLU over `used × expert_ff`. Expert ids stay on device — the old
   device-to-host readback fallback loop is deleted; the launcher covers
   Q8_0/F32/BF16. **moe gate/up/down: 0.058–0.062 ms each, uniform across
   all 1280 calls.**
3. **GEMV rewrite** — 4 rows per block with the diagonal-loop fix in
   `GemvQ8_0` (`gemv.hip.cpp:34`); byte-per-lane Q8_0 loads.
   Standalone bench (`tools/bench/lmhead_bench.hip`, 248320×2048 Q8_0):
   **2.336 ms = 231 GB/s ≈ 96 % of measured peak.** Variants tried and
   rejected: misaligned-uint32 vec4 loads 214 GB/s, vec2 227 GB/s,
   vec4 + LDS-staged x 120 GB/s (occupancy loss). The GEMV kernel itself is
   not the decode bottleneck.
4. **`GdnDeltaDecodeKernel`** (`kernels.hip.cpp:361`, dispatched for
   ssm `head_dim == 128`). One warp per (head, row): grid
   `(head_dim/4, v_heads)`, block 128, warp `w` takes row `blockIdx.x * 4 + w`
   of head `blockIdx.y` (k-head = head mod k_heads). `q`/`k` rows are read as
   `float4` per lane; state row is a `float4` stream. Decay
   `expf(a·softplus(alpha+dt))`, beta gate `sigmoid(beta)`. Reductions use a
   double-precision butterfly (`WarpReduceSumD`, all-lane `shfl_xor`).
   Numerics: `s + delta*k` is rounded to float before the q-dot so the oracle
   error is bit-identical to the reference path (attn 9.89e-5 / 1.83e-4 on the
   two test cases, state 1.9e-5). **lin_delta: 0.041 ms × 30 layers.**
5. **MoE debug-log removal** — per-(layer, chunk) routing prints removed from
   the hot decode path.

## Decode profile at 33.3 tps

`profile.hpp` `StageProfiler` (per-mark `hipDeviceSynchronize`; inflates the
total by ~250 ms over 32 steps, so real step time is below the profiled
963 ms / 32 ≈ 30 ms). Per-stage totals over 32 steps:

| stage | ms | per step |
| --- | --- | --- |
| lin_gemm_in | 160 | 5.00 |
| output (lm_head) | 141 | 4.41 |
| moe down / up / gate | 80 / 78 / 75 | 2.50 / 2.44 / 2.34 |
| moe_shared | 64 | 2.00 |
| lin_gemm_out | 54 | 1.69 |
| moe_router | 51 | 1.59 |
| attn_gemm_qkv | 46 | 1.44 |
| lin_delta | 40 | 1.25 |
| attn_core | 29 | 0.91 |
| add | 25 | 0.78 |
| attn_gemm_out | 22 | 0.69 |
| attn / ffn | 21 / 21 | 0.66 |
| moe_epilogue | 16 | 0.50 |
| lin_conv | 12 | 0.38 |
| lin_outnorm | 11 | 0.34 |
| lin_normqk | 10 | 0.31 |
| attn_rope_norm | 7.6 | 0.24 |

## Roofline and open work

Active weight bytes per token ≈ 3.07 GB (8/256 experts + shared + attention +
lm_head) → **12.8 ms/step ≈ 78 tps ceiling** at 240 GB/s; ≥60 tps is inside
reach.

Open items, in priority order:

1. **lm_head gap — explained, rerun pending** — the `output` stage ran from
    `Mark("output")` to the *next step's* `Mark("embed")`, so it absorbed the
    engine's synchronous 1 MB logits D2H + host sampling between steps
    (engine.cpp DecodeStep). `Mark("sample")` now closes the stage right after
    the lm_head GEMV; the next profile run should show `output` ≈ 2.3 ms
    (matching the standalone bench) and a new `sample` row for the readback.
2. **moe_shared / router consolidation** — 2.00 + 1.59 ms/step of small
    GEMVs foldable into the grouped launches. Both are launch-bound, not
    bandwidth-bound: the router GEMV moves 21 MB/step (≈0.09 ms roofline) and
    the shared expert 134 MB/step (≈0.56 ms); the rest is per-kernel latency
    across 40 layers.
3. **MTP speculative decoding** — `draft_proposed=0` today; enabling the MTP
   block is the multiplier that takes 33 → 60+ tps.

## Verification

- Oracle tests (run directly, MB-scale):
  `qwen36_a3b_rocm_gemv_test`, `qwen36_a3b_rocm_gdn_test`,
  `qwen36_a3b_rocm_attention_test` — all pass; GdnDelta errors match the
  pre-refactor reference bit-for-bit.
- Forward test (user-run, loads the 40 GB model): trunk step0 0.0107703,
  step1 0.00451447, step2 0.00276079, mtp 0.00246238.
- Profiler runner: `test_build_profiler.sh`.
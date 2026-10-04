# Qwen3.6-35B-A3B — Prefill & Decode Optimizations (gfx1151)

Running log of the kernel work on `src/models/qwen36_a3b` on Strix Halo
(gfx1151, measured streaming peak ~240 GB/s via `tools/bench/gfx1151_peak.hip`).
All numbers are for the Q8_K_XL artifact (Q8_0 + F32 + BF16 weights, kept
quantized on device). Geometry: hidden 2048, 40 layers (30 GDN + 10 GQA),
16 q-heads / 2 kv-heads / head_dim 256, ssm 16 k-heads / 32 v-heads / head_dim
128, 256 experts top-8 with expert_ff 512, vocab 248320.

Headline: **prefill 6480 ms → 1630 ms**, **decode 14.1 tps → 60.8 tps**
(MTP k = 2). Target: ≥60 tps — met.

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

## Prefill attention WMMA + row-split GDN (Phase 1/2a)

Landed in `fc428317`. Targets the depth-dependent prefill collapse (the 10
full-attention layers are O(n²); the 30 GDN layers and MoE are O(n)).

1. **WMMA flash prefill attention** (`WmmaCausalAttentionKernel`,
   `kernels.hip.cpp:1046`) — tensor-core `wmma_f32_16x16x16_f16_w32`, online
   softmax, double-buffered K/V into LDS, grid `(query_tiles, 8 head_pairs)`,
   32 query rows × 2 heads/block. Replaces the scalar `AttentionPrefillTiled`
   (kept as oracle; parity in `qwen36_a3b_rocm_attention_test`).
2. **FP16 KV mirror** (`KvCacheWriteF16`) — written alongside every fp32 KV
   write so the WMMA kernel reads half the bytes; the MTP draft caches gained
   their own mirrors later (see the draft-skip section below).
3. **Row-split GDN delta loop** (`GdnDeltaLoopRowSplitKernel`,
   `kernels.hip.cpp:~1624`) — prefill-only (`snap==nullptr`): 4 lanes per
   128-wide state row, 8 row-groups/wave, register-resident fp32 state, DPP
   butterfly reductions (`RowXorAdd`). The snapshot/verify path keeps the
   serial `GdnDeltaLoopKernel`; parity in `qwen36_a3b_rocm_gdn_test`.

Raw `Executor::Prefill` sweep (chunk=2048, UD-Q8_K_XL, `GUFO_QWEN36_A3B_BENCH`):

| n | t/s | slimsami | ratio |
| ---: | ---: | ---: | ---: |
| 512    | 1,420 | 1,475 | 96 % |
| 2,048  | 1,882 | 2,311 | 81 % |
| 8,192  | 1,693 | 2,300 | 74 % |
| 16,384 | 1,414 | 2,168 | 65 % |
| 65,536 |   771 |   —   | — |
| 100,000|   567 |   —   | — |

100K lifted ~2.3× (250 → 567). Still collapses vs slimsami's flat ~2.2K. The
residual is the attention KV traffic: `kWmmaHeads=2` splits the 8-head GQA
group across 4 blocks, each re-reading the same KV head's prefix — a 4×
redundant read that makes long-context attention memory-bound. **Next:** load
the K/V tile once per KV group (all `heads_per_kv` query heads per block) to
cut KV traffic ~4×; profile at 100K first. See
`optimization_missing_report.md` §8.

## Prefill attention LDS transpose (packed V + bank swizzle)

Landed on top of `fc428317`. Targets the WMMA kernel's V transpose into LDS.

1. **Packed V transpose** (`WmmaCausalAttentionKernel`, `kernels.hip.cpp:1046`)
   — the V load remaps so lane `L` holds dims `[wave*32 + (L%4)*8, +8)` of keys
   `2*(L/4)` and `2*(L/4)+1`; the two keys land in adjacent V^T columns and
   transpose to one packed dword write instead of 16 half writes.
2. **Bank swizzle** (`WmmaVtRow<kVtStride,kVtSwizzle>`, `kVtSwizzle=16`) — a
   16-half pad every 8 dims spreads the transpose writes and the PV fragment
   reads over all 32 LDS banks (without it the four lane groups collide 4-way).

A standalone microbenchmark (`tools/bench/attn_causal_bench.hip`, 16q/2kv,
b=2048 s=98304) put the pipelined kernel at 1.69 ms vs 1.94 ms for the shipped
prefetch-only mapping — a 1.15× kernel gain that implied ~10 % at 100K. It did
**not** survive the full path: a matched same-session interleaved A/B
(`tools/bench/gpu_exclusive.sh`, 2 reps, UD-Q8_K_XL) measured only **+0.6 %
@4096, +0.9 % @16384, +0.8 % @100K** (candidate ahead in 6/6 paired runs). The
kernel is LDS-transpose-bound in isolation but **global-KV-bandwidth-bound** in
the real 100K path — the packed/swizzle change cuts LDS conflicts, not the 4×
redundant global KV re-read, so the end-to-end effect is marginal. Kept as a
strict, parity-safe improvement (attention parity holds across multi-tile,
causal-tail and deep-prefix cases). The real 100K lever remains the 4× KV
re-read above. See `EXPERIMENTS.md`.

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

## Decode profile at 34.2 tps

`profile.hpp` `StageProfiler` (per-mark `hipDeviceSynchronize`; profiled
total 938 ms / 32 ≈ 29 ms matches the real step time). Per-stage ms per
step, measured after the gate/up pair fusion and before the `GemvMulti`
fusion (re-profile pending):

| stage | per step |
| --- | --- |
| lin_gemm_in | 5.02 |
| moe gateup (routed, fused) | 4.18 |
| moe down (routed) | 2.50 |
| output (lm_head) | 2.37 |
| sample (logits D2H + host sampler) | 2.04 |
| moe_shared | 1.76 |
| lin_gemm_out | 1.68 |
| moe_router | 1.58 |
| attn_gemm_qkv | 1.45 |
| lin_delta | 1.24 |
| attn_core | 0.90 |
| add | 0.79 |
| attn_gemm_out | 0.69 |
| attn / ffn | 0.65 |
| moe_epilogue | 0.51 |
| lin_conv | 0.38 |
| lin_outnorm | 0.33 |
| lin_normqk | 0.32 |
| attn_rope_norm | 0.24 |

## MTP speculative decode (33.3 → 55.2 tps)

Landed in `966b3db8`..`14ebbdb0`.

1. **MTP draft block** — the model's MTP layer is bound in `Model::Load`
   (`6e51aa51`) and run by `Executor::MtpForward`: one fused
   (previous-hidden, embedding) step that writes its own KV/GDN state at
   `mtp_position_` and projects through the shared head. `MtpStep` feeds
   the main-path hidden; `MtpAdvance` keeps the draft state aligned after a
   committed token. `DecodeStep` drafts one token, verifies it with the
   main model, and serves it over HTTP (`1447ab01`).
2. **Decode-tier verify** (`14ebbdb0`) — the two-row verify pass reuses the
   decode kernels instead of the prefill GEMM path: `Gemv2` (two activation
   rows per GEMV launch), `RmsNormRows`, batched GDN with per-row state
   snapshots for rollback, and `AttentionDecode2` — the flash-decoding
   split kernel generalized so row `o` attends `[0, n_kv0 + o)` with a
   per-row causal mask inside the shared online-softmax loop.
   **attn_core stays 0.089 ms/call for a verify row pair; a full
   draft+verify round costs 35.4 ms for 1.97 tokens = 55.2 tps** at
   96.9 % draft acceptance.

Profile per 16 rounds (565.9 ms total, k = 1): moe_gateup 117.8,
lin_gemm_in 76.5, moe_routed_down 68.9, moe_epilogue 54.4, output 38.4,
attn_core 17.0, mtp 3.7 ms. The round is still one weight-streaming pass
per stage, so a rejected draft costs a full step (~18 ms for the
correction token) and the win scales with acceptance, not kernel time.

## k-draft speculation and sampling support

Generalizes the fixed one-token loop to `k` drafts
(`--draft-tokens` max, `--min-draft-tokens` floor, adaptive between
them: full accept +1, full reject −1; equal values pin `k` for sweeps).

- `GemvRows` / `AttentionDecodeRows` are the `Rows`-templated versions of
  the two-row kernels (`switch (tokens)` dispatch, 2–5 rows). `k` is
  capped at 4 (5 verify rows) because the split kernel's LDS is
  `Rows × 8 × 256 × 4 B` — 40 KB fits, 8 rows would exceed the 64 KB
  budget. CLI values above 4 are clamped.
- Draft chain: `MtpStep(t0)` then `MtpDraft(d_i)` feeds the *unnormalized*
  MTP output hidden (`mtp_chain_`, copied before the in-place shared
  head norm) as the previous-hidden input of the next draft.
- Verify: `Executor::Verify(t0, drafts, k)` runs `k+1` rows through the
  batch path into `verify_logits_` / `verify_h_`; `RollbackVerify(keep)`
  restores the GDN state/history snapshot row `keep − 1` and rewinds
  `position_`/`mtp_position_` so the next step re-drafts from the
  accepted prefix. Stale KV beyond `position_` is never read.
- Temperature > 0 uses Leviathan speculative sampling: the draft
  distribution `q` is captured per draft token, a draft is kept with
  probability `min(1, p(d)/q(d))`, and on rejection the correction token
  is drawn from `max(0, p − q)` (`SamplerState::SampleResidual`).
  Speculation stays disabled while penalties are active.
- Stop tokens inside an accepted block truncate the commit at the stop.

Served sweep (greedy, pinned `k`): **k = 2 → 60.8 tps** (90.6 %
acceptance, 128-token completion), k = 3 → 56–58 tps, k = 4 (CLI 6
clamped) → 49 tps. Acceptance falls faster than the verify rows amortize
beyond k = 2; the adaptive default should settle there.

## MTP draft skip-by-default + f16 KV mirror (prefill collapse fix)

Plain (non-speculative) prefill was running the MTP draft layer through the
scalar oracle attention (`AttentionPrefillTiled`, 19.3 % of pp16384 GPU time):
`MtpPrefillChunk` passed null f16 mirrors to `AttentionBatch`, so the draft fell
off the WMMA path, and the draft block ran unconditionally on `model.has_mtp()`
even when the session would never decode with MTP. This — not the head-major
coalescing gap — was the dominant cause of the depth-dependent collapse; the
earlier "4× redundant read" note in the Phase 1/2a section was the wrong lever.

Fix (matches the reference, which gates the draft on the speculative session
mode):

1. **Skip by default** — `Executor::mtp_enabled_` (default `false`); `Prefill`
   only calls `MtpPrefillChunk` when it is set. `Session::SetMtpEnabled` exposes
   it and `DecodeStep` also requires it, so a skipped-draft session can never
   read a stale cache. The flag must be set before the first `Sync` (the draft
   cache is filled during prefill and cannot be rebuilt lazily). Serve, the
   speculative bench `tg` loop and interactive chat enable it; plain bench `pp`
   leaves it off.
2. **f16 KV mirror for the draft** — `mtp_k_cache_f16_`/`mtp_v_cache_f16_`,
   written by `AttentionBatch`/`Attention` alongside the fp32 draft cache, so
   when MTP *is* enabled the draft uses the same `WmmaCausalAttentionKernel` as
   the trunk (supersedes "MTP caches stay fp32" above).

Exclusive-GPU `gufo bench` (same machine/session as BENCHMARKS.md): pp16384
1515 → 1883 t/s (+24 %), pp8192 1778 → 2040 (+15 %), curve flattened (peak
2172). MTP acceptance is bit-identical to the pre-fix scalar draft (n=1 56/64,
n=2 63/90) with an identical output hash — the WMMA draft reproduces the oracle
exactly. The residual 84 % at 16384 vs the reference is the head-major packed-KV
coalescing gap (see BENCHMARKS.md), the next lever.

## Roofline and open work

Active weight bytes per token ≈ 3.07 GB (8/256 experts + shared + attention +
lm_head) → **12.8 ms/step ≈ 78 tps ceiling** at 240 GB/s; ≥60 tps is inside
reach.

Open items, in priority order:

1. **lm_head gap — resolved** — the old `output` stage ran from
    `Mark("output")` to the *next step's* `Mark("embed")`, absorbing the
    engine's synchronous 1 MB logits D2H + host sampling. With
    `Mark("sample")` the GEMV measures 2.37 ms/step — at roofline — and
    the readback shows as its own `sample` row (2.04 ms/step).
2. **`sample` stage** — 2.04 ms/step for a 1 MB logits D2H plus host
    argmax/sampling. A pinned async D2H (or GPU-side argmax) removes
    most of it; it is the largest non-kernel cost.
3. **Same-`x` projection fusion — done** — `GemvMulti` (bit-exact per
    oracle test) folds the Q8_0 projections sharing one activation row
    into a single launch and runs the F32 side vectors (`ssm_alpha`,
    `ssm_beta`, `ffn_gate_inp_shexp` when stored as F32) as plain `Gemv`:
    `lin_gemm_in` fuses qkv+gate, `attn_gemm_qkv` is 3-in-1,
    `moe_shared` fuses gate+up. The first revision rejected any group
    containing a non-Q8_0 tensor, so `lin_gemm_in`/`moe_shared` silently
    fell back and the profile was unchanged; the subset split restores
    the fusion. Cold-L2 bench (`tools/bench/moe_gemv_bench.hip`):
    lin_gemm_in −10.8 % (211 GB/s), shared expert 0.0197 → 0.0096 ms
    warm. Together with `GemvGroupedPair` (routed gate+up) this removes
    140 launches/step. Router+TopK fusion was rejected: it would
    reorder the router dot product and break token parity. vec4 int8
    loads measured only +9.5 % cold and are not bit-exact — rejected.
    The next lever is hipGraph capture (see open item 5).
4. **MTP speculative decoding — done** — see the two sections above;
   k = 1 serves 55.2 tps, the k-draft generalization is in tree.
5. **hipGraph capture of the decode step** — the remaining launch-gap
   lever; blockers are the scalar `pos` baked into the Attention cache
   pointer and `n_kv` argument.

## Verification

- Oracle tests (run directly, MB-scale):
  `qwen36_a3b_rocm_gemv_test` (includes bit-exact checks for
  `GemvQ8_0Multi4` / `GemvGroupedQ8_0Pair` vs the separate launches, and
  `GemvRows` for 2–5 activation rows across Q8_0/F32/BF16),
  `qwen36_a3b_rocm_gdn_test`, `qwen36_a3b_rocm_attention_test` (covers
  `AttentionDecodeRows` for 2–5 rows over starts straddling the 64-token
  split boundary and the 32-split cap, including fully masked
  row/split blocks) — all pass; GdnDelta errors match the pre-refactor
  reference bit-for-bit.
- Forward test (user-run, loads the 40 GB model): trunk step0 0.0107703,
  step1 0.00451447, step2 0.00276079, mtp 0.00246238.
- Profiler runner: `test_build_profiler.sh`.
# Qwen3.6-35B-A3B — Missing Prefill Optimizations vs gufo_slimsami

Branch under repair: `feat/support_qwen36_35b_a3b` (`src/models/qwen36_a3b`).
Reference: `../gufo_slimsami` branch `feat/qwen35moe-35b-a3b`
(`src/models/qwen`, the shared Flash-Next kernel stack).

## 1. Symptom

| Context | before | after Phase 1+2a | gufo_slimsami (prefill) |
| ---: | ---: | ---: | ---: |
| 2K   | —      | 1,882 t/s | 2,311 t/s |
| 8K   | ~1,400 | 1,693 t/s | 2,300 t/s |
| 16K  | —      | 1,414 t/s | 2,168 t/s |
| 100K | ~250   |   567 t/s | ~1,500–2,000 t/s |

Phase 1 (WMMA attention + FP16 KV) and Phase 2a (row-split GDN) lifted 100K
~2.3× (250 → 567 t/s). Throughput still *falls* with depth on this branch
(1,882 → 567); slimsami stays flat (2,311 → 2,168 across 2K–16K). The
remaining gap is now isolated to the prefill attention kernel's KV traffic —
see §8.

## 2. Quantitative root cause

Fit `time(n) = a·n + b·n²` to the two measured points (8K → 1400 t/s,
100K → 250 t/s):

- linear-only rate `1/a ≈ 2,380 t/s` — this equals slimsami's *flat* rate, so
  the O(n) pipeline (MoE, dense projections, GDN) already matches slimsami;
- the quadratic term `b·n²` is ≈ 90 % of the 100K step time.

Only the 10 full-attention layers are O(n²). Therefore the collapse is
entirely the **prefill attention kernel**.

Attention FLOPs (causal, 10 layers, 16 heads, head_dim 256):

- @100K ≈ 4.1e14 FLOP. At the scalar kernel's ~1 TFLOP/s ≈ 410 s → ~220 t/s
  (matches the observed 250). At slimsami's ~20 TFLOP/s WMMA ≈ 20 s.

## 3. Root causes (ranked)

1. **Prefill attention has no tensor cores.**
   `kernels/rocm/kernels.hip.cpp:887` `AttentionPrefillTiled` scores with a
   scalar `for d: dot += qh[d]*sK[k][d]` loop (line 949), 32 query rows/block,
   re-streaming the whole FP32 KV prefix per query tile. No WMMA, no prefetch.
2. **KV cache is FP32.** `executor.hpp:280` `std::vector<float*> k_cache_`.
   Attention reads 2× the bytes slimsami does; WMMA needs FP16 operands.
3. **Unfused attention prep.** `RmsNormRows×2 + Rope + KV memcpy` are separate
   passes (`executor.cpp:865-880`).
4. **GDN prefill is a serial single-warp loop.** `kernels.hip.cpp:1096`
   `GdnDeltaLoopKernel`, grid (16,32)=512 warps, each looping `tokens`
   sequentially. O(n) but low parallelism → raises the linear floor.
5. **BF16/F32 experts use a scalar fallback.** `gemm.hip.cpp:57`
   `MoeVecFallback` (per-token serial dot). Only affects non-Q8 layers.

## 4. Already present (do not re-do)

- Chunked prefill (fixed 2048) — `executor.cpp:923`.
- Routed F16 WMMA MoE + compaction + tile map — `executor.cpp:699`,
  `routed_f16.hip.cpp` (== slimsami's main MoE win).
- hipBLAS dense GEMM for F32/BF16 projections — `gemm.hip.cpp`.
- Decode: flash-decoding, grouped MoE GEMV, MTP k-draft — fine.

## 5. Reference implementations to port (slimsami)

- `src/models/qwen/hip/kernels/attention_wmma.hip` — WMMA flash prefill attn
  (`LaunchWmmaAttentionShape<kQueryHeads,kKvHeads>`, `PackTiledAttentionKvKernel`).
- `src/models/qwen/hip/detail/attention_policy.hpp` — head-shape gate
  (16Q/2KV, head_dim 256) + FP16-KV requirement.
- `src/models/qwen/hip/ops/attention.hpp` — FP16 KV cache API, fused
  QK-norm+RoPE+KV-write (`opt-c010`).
- `src/models/qwen/hip/kernels/ssm_row_split.hip` — parallel GDN recurrence
  (`LaunchBatchedSSMConvRecurrenceRowSplit`, `BatchedDeltaNetRowSplitKernel`).

## 6. Plan

Strategy: port slimsami kernels into the isolated `src/models/qwen36_a3b`
tree; keep the existing scalar kernels as the oracle/reference and assert
parity in tests.

Phase 1 (critical, fixes the collapse) — **DONE** (commit `fc428317`):

  a. **Done** FP16 KV cache planes for the 10 attention layers
     (`KvCacheWriteF16`, paired-write at every fp32 KV write).
  b. **Done** `AttentionPrefillTiled` → WMMA tensor-core flash attention
     (`WmmaCausalAttentionKernel`, online softmax, double-buffered K/V,
     32 query rows × 2 heads/block).
  c. Kept the scalar kernel as the oracle; parity asserted in
     `qwen36_a3b_rocm_attention_test`.
  d. **Not yet** fused QK-norm + RoPE + KV-write (still separate passes).

Phase 2 (linear floor):

  e. **Done** row-split GDN delta loop for prefill
     (`GdnDeltaLoopRowSplitKernel`, 4 lanes/128-wide state row, DPP butterfly
     reductions); the snapshot path keeps the serial kernel. Parity in
     `qwen36_a3b_rocm_gdn_test`.
  f. **N/A — already present.** The grouped WMMA expert GEMM exists and is the
     active prefill route for this model: `MoeBatch` dispatches Q8_0/BF16
     experts to `RoutedF16Gemm` (`routed_f16.hip.cpp`, WMMA
     `wmma_f32_16x16x16_f16_w32`, decodes Q8_0/BF16/Q4_K/Q5_K/Q5_1). The
     model's routed experts are Q8_0 (118) + BF16 (5), all covered.
     `MoeVecFallback` is only reached for F32/unsupported encodings.

Phase 3 (the remaining collapse) — see §8:

  g. **Share the KV tile across the full GQA group** in
     `WmmaCausalAttentionKernel` (currently `kWmmaHeads=2` → 4× redundant KV
     read per KV head). This is the next fix.
  h. Expose `--prefill-chunk` (512/1024/2048); sweep.

## 7. Verification

- `qwen36_a3b_rocm_attention_test` — extend to WMMA-vs-scalar parity over the
  16Q/2KV shape (FP16-KV tolerance).
- `qwen36_a3b_rocm_gdn_test` — row-split vs serial bit-parity.
- Forward test trunk/mtp errors; then `gufo bench --n-prompt 512..16384` and a
  100K depth row.
- **Target:** flat ≥1.5K t/s at 100K (vs 250 today), no decode regression.

## 8. Results after Phase 1 + 2a (measured) and the remaining gap

Raw `Executor::Prefill` sweep (chunk=2048, UD-Q8_K_XL, 3 reps), via
`GUFO_QWEN36_A3B_BENCH` in `qwen36_a3b_rocm_forward_test`. slimsami numbers are
its HTTP `pp` for the same artifact (`gufo_slimsami`
`docs/models/qwen3.6-35b-a3b/BENCHMARKS.md`); harnesses differ (raw executor vs
serve) but the *shape* is the signal.

| n | this branch | slimsami | ratio |
| ---: | ---: | ---: | ---: |
| 512   | 1,420 | 1,475 | 96 % |
| 2,048 | 1,882 | 2,311 | 81 % |
| 8,192 | 1,693 | 2,300 | 74 % |
| 16,384| 1,414 | 2,168 | 65 % |
| 65,536|   771 |   —   | — |
| 100,000|  567 |   —   | — |

The ratio degrades monotonically with depth (96 % → 65 % → collapse), so the
residual gap is the depth-dependent term: the 10 full-attention layers.

**Root cause — GQA KV re-read in `WmmaCausalAttentionKernel`.** The grid is
`(query_tiles, 8 head_pairs)` with `kWmmaHeads=2` query heads per block
(`kernels.hip.cpp:1021,1082-1086`). With 16 q / 2 kv heads, `heads_per_kv = 8`,
so `kv_head = head_pair / 4`: four `head_pair` blocks share one KV head and each
loads that head's full FP16 KV prefix independently — a **4× redundant KV
read**. At long context attention is memory-bound (KV bytes re-read scale as
`n²/32 × 4`), so this redundancy drives the collapse.

slimsami instead groups all `heads_per_kv` query heads per KV head and reads the
KV prefix **once per group** (`attention_batched.hip:368-428`,
`grouped_attention_min_batch = 2048`), keeping its curve flat.

**Fix (Phase 3, next):** cover one KV head and all 8 of its query heads in a
block (raise `kWmmaHeads` toward `heads_per_kv`, shrinking `kWmmaQueryRows` to
fit the LDS/register budget) so the K/V tile is loaded once and reused across
the whole GQA group — cutting attention KV traffic ~4×. Expected to lift the
long-context rate toward slimsami's flat ~1.5–2K.

**Before implementing:** profile at 100K (`tools/prof/prof.py`) to confirm
`attn_core` dominates the step and to measure the lift; also check whether any
non-attention stage (GDN, MoE) trails slimsami at short context (the 81 % at 2K
is consistent with the same KV redundancy already biting at prefix≈2K, but a
profile will attribute it).
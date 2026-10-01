# Qwen3.6-35B-A3B — Kernel Experiments

Concise retained/rejected log for `src/models/qwen36_a3b` on gfx1151. Current
production speeds live in `OPTIMIZATIONS.md`; quality evidence in the model's
tests. Every row is a matched same-session comparison through
`tools/bench/gpu_exclusive.sh` (exclusive VRAM), not a cross-run guess.

## Prefill attention — WMMA kernel data movement

Shape for microbenchmarks: 16 q-heads / 2 kv-heads, head_dim 256, b=2048,
start_pos=98304 (the 100K last chunk). Microbench `tools/bench/attn_causal_bench.hip`.

| change | scope | result | decision |
| --- | --- | --- | --- |
| Packed V transpose + bank swizzle (`kVtSwizzle=16`) | microbench | 1.94 → 1.69 ms (1.15×) | — |
| Packed V transpose + bank swizzle | end-to-end prefill | +0.6 % @4096, +0.9 % @16384, +0.8 % @100K (6/6 paired) | **retained** |
| Prefetch depth 2 / depth 3 | microbench | 2.25 / 2.70 ms (regress) | rejected |
| 32 keys / 16 waves tile | microbench | 3.22 ms (regress) | rejected |
| 3 blocks/CU | microbench | 1.745 ms (vs 1.693 at 2) | rejected |
| `kVtPad=0` (no V^T pad) | microbench | 1.993 ms (regress) | rejected |

**Key finding:** the WMMA attention kernel is LDS-transpose-bound in isolation
but **global-KV-bandwidth-bound** in the real 100K path. Ablations on the
production tile put the global K/V load at 31–50 % of kernel time; packed-V and
the swizzle only reduce LDS transpose conflicts, so a 1.15× microbench gain
collapses to ~0.8 % end-to-end. The dominant 100K cost is the **4× redundant
global KV re-read**: the grid `(query_tiles, 8 head_pairs)` with `kWmmaHeads=2`
streams each KV head's prefix from HBM once per head-pair block. The next lever
is one block covering a full KV group (all `heads_per_kv` query heads) to cut
that traffic ~4× — not further LDS-transpose tuning.

## Prefill attention — opt-in sliding-window + sink sparsity

Qwen3.6-35B-A3B ships dense Gated Attention (no trained QSA/indexer weights in
the GGUF), so a lossless long-context speedup is impossible; the only lever is a
**training-free, opt-in** restriction of each query to the first `sink` keys and
the last `window` keys (StreamingLLM-style). The WMMA kernel is templated on
`kSparse`; the dense instantiation (`window == 0`) is byte-for-byte the previous
kernel, and the sparse instantiation never loads, scores or multiplies the
dropped middle key tiles. The softmax mask drops boundary-tile keys outside the
window, so the retained probabilities (and therefore PV) stay exact.

Off by default. Enable with `--attn-window <tokens>` (and optional
`--attn-sink <tokens>`, default 0) on `gufo serve llm` and `gufo prompt`; the
`GUFO_QWEN36_ATTN_WINDOW` / `GUFO_QWEN36_ATTN_SINK` environment variables remain
as a fallback and are ignored when the flag is set. Correctness is asserted
against a masked CPU reference in `qwen36_a3b_rocm_attention_test` (worst rel
err ~3.5e-4, same as dense FP16). Matched same-session A/B
(`GUFO_QWEN36_A3B_BENCH`):

| context | dense | window=2048 sink=4 | window=4096 sink=4 |
| --- | --- | --- | --- |
| 4 096 | 1915.8 tok/s | 1941.5 (+1.3 %) | 1916.6 (+0.0 %) |
| 16 384 | 1482.5 tok/s | 1590.9 (+7.3 %) | 1563.9 (+5.5 %) |
| 100 000 | 582.9 tok/s | 752.1 (+29.0 %) | 747.2 (+28.2 %) |

**Decision: retained as opt-in, default off.** The gain scales with context
length (the dropped middle range only exists once context > window) and is
bounded because only the 1-in-4 full-attention layers benefit and each still
keeps a 2048-token window. It is lossy: quality must be validated per workload
before enabling, and a missing-model skip is not a quality pass.

The kernel-level effect can be isolated with `tools/bench/attn_causal_bench.hip`
(`-w <window> -n <sink>`), which adds a `pipe: sparse w/sink` variant beside a
`pipe: dense full (key0)` baseline over the same key range; run it through
`tools/bench/gpu_exclusive.sh` for exclusive access. The sparse variant is
restricted to prefetch depth 1 (the double-buffer the shipped kernel uses)
because the ring prefetch assumes contiguous key tiles.
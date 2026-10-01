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
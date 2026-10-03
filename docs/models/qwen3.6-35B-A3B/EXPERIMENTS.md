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
| Head-major packed KV (repack + pre-transposed V, ported from reference `attention_wmma.hip`) | end-to-end prefill | −4 % @32K, −6 % @64K, −9 % @100K (interleaved A/B; bit-exact, 0/4.2 M mismatches) | **rejected** |

**Key finding (corrected for the packed production kernel):** the earlier
"global-KV-bandwidth-bound / 4× redundant re-read is dominant" note was measured
on the **unpacked** kernel, where the global K/V load really is ~49 % of time
(`ablate: no K/V global loads` 1.707 ms vs unpacked 3.345 ms). Once packed-V +
swizzle + prefetch shipped, the bottleneck moved: a fresh phase ablation of the
**production packed tile** at the 100 K chunk (16 q / 2 kv, head_dim 256,
b=2048, start_pos=98304, `attn_causal_bench -DKQUERY_HEADS=16 -DKKV_HEADS=2`)
shows the kernel is **WMMA-compute-bound**, not KV-bound:

| ablation (packed `pipe: all three` = 1.811 ms) | ms | % of kernel |
| --- | --- | --- |
| no S wmma | 1.224 | 32.4 % |
| no PV wmma | 1.398 | 22.8 % |
| no V transpose | 1.632 | 9.9 % |
| no K/V global load | 1.717 | **5.2 %** |
| no softmax | 1.793 | 1.0 % |

So the two matrix multiplies are ~55 % of the kernel and the global KV traffic
is only ~5 %. This **prices out the full-KV-group lever**: even deleting *all*
global KV traffic saves 5 %, and covering a full KV group forces `kWaves=16`,
which breaks the packed-V mapping (it assumes 8 waves × 32 dims = 256) and drops
back to the unpacked path — a ~45 % loss (unpacked 3.345 vs packed 1.811) to
chase a ≤5 % traffic cut. Rejected without building it.

The reference's head-major packed-KV fast path was ported bit-exactly and
measured to test whether it closes the long-context gap. It does not: the
repack reads the whole `[0, context_end)` prefix and writes a head-major copy on
every prefill chunk, so it *adds* global KV traffic (the very term that
dominates the *unpacked* kernel) to save an LDS transpose our packed kernel
already performs cheaply. The loss grows with context (−4 % → −9 %), the
opposite of the O(n²) win the reference sees only because its unpacked path is
the slow one. Reverted; the packed WMMA kernel stays the default.

The only remaining structural lever is eliminating the V LDS transpose (~10 %
ceiling, the `no V transpose` row) by storing the f16 V mirror **pre-transposed
into the WMMA fragment layout** — the reference's design, and unlike the repack
it is amortized (transpose each token once at KV-write time, O(1)/token, not
O(context)/chunk). It is *not* pursued here: a simple `[dim][position]` mirror
makes each PV fragment lane read 16 contiguous halves from 32 different dim rows
(32 × 32 B transactions, uncoalesced), which can erase the transpose saving, and
a coalesced variant must bake the 16 × 16 fragment tiling into the cache layout
— fragile, tiling-coupled, and touching `KvCacheWriteF16`, the dense + sparse
attention reads, the draft mirror and their tests. With the ceiling at ~10 % of
attention and the last comparable change regressing end-to-end, its expected
value does not justify that risk; the attention core is left at its packed
default and the residual long-context gap is WMMA matmul efficiency (~34 % of
peak), not data movement.

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

## Host readback — busy-wait vs blocking-sync (process-global)

Every synchronous device→host readback blocks the host on HIP's default signal
wait, which on this ROCm build is a userspace **busy-wait** (`rocr::core::
BusyWaitSignal`). It pegs one core at 100 % for the whole wait and lags the
whole machine. The hot sites are the per-layer MoE expert-count readback in
`kernels/rocm/executor.cpp` (`MoeBatch`) and the per-step logits readback in
`engine.cpp` (`DecodeStep`); the same mechanism pegs every model, not just this
one.

The per-event `hipEventBlockingSync` flag is **ignored** by this `libamdhip64`,
so a pinned-buffer + blocking-event readback still spins. Only the device-level
`hipDeviceScheduleBlockingSync`, set before any context exists, parks the thread
on an OS event. It is set once in `src/cli/main.cpp` before `run()`, so it
applies to every model the `gufo` binary loads. Standalone wait probe (~0.6 s of
GPU work, exclusive GPU):

| wait mechanism | cores during wait |
| --- | --- |
| `hipDeviceSynchronize` (default) | 0.998 (spin) |
| `hipEventBlockingSync` event (default) | 0.997 (spin — flag ignored) |
| `hipDeviceScheduleBlockingSync` | 0.043 (sleeps) |

Matched same-session A/B (`gufo prompt`, 1500 tokens, seed 42, greedy,
window=2048 sink=4, exclusive GPU):

| build | decode CPU | decode tok/s | output |
| --- | --- | --- | --- |
| baseline | 1.02 cores | 53.54 | — |
| blocking-sync | 0.17 cores | 53.22 (−0.6 %) | byte-identical |

**Decision: retained, always-on.** The residual 0.17 cores is the CPU sampler
(softmax over the 151 K vocab), not a spin. Pinned readback buffers were also
tested and gave no further gain (0.18 vs 0.17 cores), so they were dropped in
favour of the single device flag.

## Decode fusion 1 — routed MoE gate+up+SwiGLU in one launch

The routed-expert path launched the grouped gate and up GEMVs
(`GemvGroupedPair`) and then a separate `Swiglu` pass that re-read both
intermediates from global memory and wrote the product back — one extra kernel
launch and one extra round-trip of the `[8 x 512]` activations per layer, 40
times per decode step. `GemvGroupedSwiglu` folds the epilogue into the pair
kernel: each wave already holds the gate and up dot products in registers, so
it writes `silu(gate) * up` directly and the standalone pass disappears.

The dots reuse the exact `GemvGroupedQ8_0Pair` loop, reduction order and
lane-0 write, and the epilogue `DSiluF(gate) * up` reproduces the standalone
`SwigluKernel` (`DSilu(gate[i]) * up[i]`) bit-for-bit — the float→global→float
round-trip the old path took is exact, so removing it changes nothing. The
oracle test `qwen36_a3b.rocm_gemv` compares the fused launch against
`GemvGroupedPair` + `Swiglu` on synthetic Q8_0 experts and reports
`GemvGroupedQ8_0Swiglu bit-exact: yes`.

Matched same-session A/B (`gufo serve`, no-MTP, seed 42, 128 tokens, greedy,
exclusive GPU, `NOPROFILE`):

| build | decode tok/s | output |
| --- | --- | --- |
| baseline (pair + Swiglu) | 43.66 | reference |
| fused gate+up+Swiglu | 44.18, 44.93 | byte-identical head |

**Decision: retained, always-on.** The fused launch is the default in
`Executor::Moe`; the pair+Swiglu path remains only as the fallback when the
weights are not Q8_0. The gain is small (~1–3 %) because the removed pass is
one of many latency-bound stages, but it is bit-identical and strictly removes
a launch and a global round-trip, so it is a pure win and the template for the
remaining decode fusions.
## Decode fusion 2 — attention QK-norm + RoPE + KV write in one launch

The GQA decode front-end ran the raw `q_gate` projection through a deinterleave
(`SplitQGate`), then two per-head RMSNorms (q and k), two partial RoPE passes,
two async copies into the KV cache and an FP16 mirror write — eight launches
across the ten attention layers, each a tiny latency-bound kernel that
round-tripped the head vectors through global memory. `FusedQKNormRoPEKvWrite`
does the deinterleave, both RMSNorms, both RoPEs and the KV write (FP32 planes
plus the FP16 mirror) in a single launch, keeping the normed vector in shared
memory until it is rotated so no intermediate ever leaves the block.

Bit-identity is by construction: the RMSNorm reuses the exact `RmsNormKernel`
reduction (double `BlockReduceSum`, block 256, `scale = rsqrtf(sum/dim + eps)`),
the normed value is held as `float` before rotation to match the old
global round-trip, RoPE reuses the identical `powf`/`cosf`/`sinf` factors, and
the FP16 mirror uses `__float2half_rn`. The oracle test
`qwen36_a3b.rocm_attention` compares the fused launch against the unfused
`SplitQGate` + `RmsNormRows` + `Rope` + KV-write chain and reports q, gate, k,
k_cache, v_cache and the FP16 mirror all `bit-exact: yes`.

Matched same-session A/B (`gufo serve`, no-MTP, seed 42, 128 tokens, greedy,
exclusive GPU, `NOPROFILE`), on top of fusion 1:

| build | decode tok/s | output |
| --- | --- | --- |
| fusion 1 only | 44.18, 44.93 | reference |
| fusion 1 + fusion 2 | 44.17, 44.51 | byte-identical head |

**Decision: retained, always-on.** The throughput delta is inside the ±0.5–0.8
tok/s run-to-run noise at 128 tokens, so this fusion is not a measured win on
its own; it is kept because it is bit-identical and strictly removes seven
launches and their global round-trips per attention layer (70 per decode step),
which cannot regress and lowers launch pressure for the larger fusions that
follow. The unfused chain remains only as the fallback when the fused launch's
preconditions are not met.

## Decode fusion 3 — GDN conv + q/k RMSNorm in one launch

The Gated DeltaNet decode front-end launched the depthwise causal conv
(`GdnConv`) and then a separate per-head RMSNorm of the q and k groups
(`GdnNormQk`) — two latency-bound kernels over the 30 linear-attention layers,
with the convolved q/k written to global memory only to be read straight back by
the norm. `GdnConvNormQk` gives each block one head and each thread one channel
of that head: the conv is the same per-channel tap dot plus in-place history
shift, and the q/k blocks reduce the head's convolved values with the identical
`BlockReduceSum` and `blockDim == head_dim` as `GdnNormQk`, so `qn`/`kn` are
bit-identical while the convolved q/k never leaves the block. The value block
still writes the convolved value the delta recurrence consumes.

The oracle test `qwen36_a3b.rocm_gdn` compares the fused launch against
`GdnConv` + `GdnNormQk` on the model's 16-key/32-value/128-wide geometry and
reports qn, kn, the value half of convolved and the advanced history all
`bit-exact: yes`; the existing conv/norm/delta/prefill cases still pass.

Matched same-session A/B (`gufo serve`, no-MTP, seed 42, 128 tokens, greedy,
exclusive GPU, `NOPROFILE`), on top of fusions 1 and 2:

| build | decode tok/s | output |
| --- | --- | --- |
| fusions 1 + 2 | 44.17, 44.51 | reference |
| fusions 1 + 2 + 3 | 44.54, 44.70 | byte-identical head |

**Decision: retained, always-on.** The fused launch is the default in
`Executor::LinearAttention`; the `GdnConv` + `GdnNormQk` chain remains only as
the fallback when `head_dim > 1024` or the channel layout differs. The throughput
delta is at the top of the run-to-run band but still inside noise at 128 tokens;
it is kept because it is bit-identical and strictly removes one launch and the
convolved q/k round-trip per linear layer (30 per decode step).

## Decode fusion 4 — residual add + RMSNorm in one launch

Every `Add` in `Executor::Step` is immediately followed by an `RmsNormRows` over
the same residual stream: the attention residual add feeds the post-attention
norm, and the FFN residual add feeds the next layer's attention norm (or the
final output norm). Each pair was two latency-bound launches over `hidden_size`
— the add wrote `x` to global memory only for the norm to read it straight back.
`FusedAddRmsNorm` forms `x += addend` as a float in the same order, accumulates
the squares as `double`, reduces with the identical `BlockReduceSum` and
`block == min(dim, 256)` as `RmsNormKernel`, and writes the scaled output — so
both the updated residual and the normed vector are bit-identical while the
residual never leaves the block. The loop hoists layer 0's attention norm (no
preceding add) and folds the last layer's FFN add into the output norm.

The oracle test `qwen36_a3b.rocm_attention` compares the fused launch against
`Add` + `RmsNormRows` at dims 128/256/1024/2048/3072/4096 (straddling the
256-thread block boundary) and reports both `x` and `out` `bit-exact: yes`. The
end-to-end `qwen36_a3b.rocm_inference` CPU/GPU greedy check still matches token
for token.

Matched same-session A/B (`gufo serve`, no-MTP, seed 42, **256 tokens** for a
lower-noise band, greedy, exclusive GPU, `NOPROFILE`):

| build | decode tok/s | output |
| --- | --- | --- |
| baseline (HEAD, no fusions) | 43.57 | reference |
| fusions 1 + 2 + 3 | 44.51 | byte-identical head |
| fusions 1 + 2 + 3 + 4 | 44.62 | byte-identical head |

**Decision: retained, always-on.** `FusedAddRmsNorm` is the default in
`Executor::Step`; it is bit-identical and removes two launches per layer (80 per
decode step). The cumulative effect of all four fusions is **+2.4 %** over the
unfused baseline at 256 tokens, byte-identical output throughout. The individual
fusions each land near the low end of the run-to-run band: the decode step is
not launch-overhead bound but bound by the latency of the small glue kernels and
the near-peak GEMVs (measured: lm_head 231 GB/s = 97 % of the 238 GB/s DRAM
ceiling). The clean elementwise-epilogue fusions are now exhausted — the
remaining stages (delta recurrence, out-norm, router top-k, MoE epilogue) need
cross-block or global reductions, not epilogues.

## Decode GEMV bandwidth — already at the DRAM ceiling (rejected)

The four fusions moved throughput only +2.4 %, so the decode step was profiled
directly (`GUFO_QWEN36_PROFILE=1`, spec off, 256 tokens): GPU-busy time equals
wall time (~25 ms/token), i.e. the step is GPU-bound with no idle to reclaim —
HIP graphs and further launch fusion cannot help. ~83 % of GPU time is Q8_0
GEMV, so the GEMV kernel itself was benchmarked against the real shapes with a
bit-exactness check (`tools/bench/lmhead_bench.hip`, exclusive GPU):

| shape | bytes | current | unroll8 | rows8 | rows2 | vec4 |
| --- | --- | --- | --- | --- | --- | --- |
| lm_head 248320x2048 | 540 MB (cold) | 232.6 GB/s | 234.2 | 232.4 | 233.4 | 221.7 |
| ssm_qkv 8192x2048 | 17.8 MB (warm) | 480 GB/s | 531 | 517 | 508 | 328 |

The 540 MB `lm_head` is far larger than the 32 MiB MALL, so it streams cold and
pins the true DRAM ceiling at **~233 GB/s**; every variant lands within noise of
it and `vec4` is strictly worse. The 480 GB/s on `ssm_qkv` is a cache artifact —
17.8 MB fits in MALL and the bench re-reads it warm, whereas in the model each
layer's weights are cold (30 x 17.8 MB >> cache). All bit-exact variants report
`exact`.

**Decision: rejected.** The Q8_0 GEMV already saturates DRAM bandwidth for cold
streaming reads; there is no kernel-level bandwidth win. A batch-1 greedy decode
can only go faster by reading fewer bytes/token (smaller quant, changes quality)
or reading the weights for more than one token (speculative decoding).

## MTP is the throughput lever — production already exceeds the reference

The 44 tok/s figure used throughout the fusion work was measured with
`--speculative off` (greedy), which is **not** the production path. The model
ships an MTP block and `run_rocm_qwen36_35b_a3b_mmproj.sh` serves it with
`--speculative mtp --min-draft-tokens 2 --draft-tokens 2`. Matched same-session
A/B (seed 42, greedy, 256 tokens, exclusive GPU, `NOPROFILE`):

| config | decode tok/s | acceptance |
| --- | --- | --- |
| `--speculative off` (greedy) | 44.86 | — |
| `--speculative mtp` (default draft) | 58.73 | — |
| `--speculative mtp --draft-tokens 2` (production) | **65.93** | 84 % (147/175) |

MTP reads the weights — including the 540 MB `lm_head` — once for k+1 verified
tokens, which is exactly the lever a bandwidth-bound batch-1 decode needs. At
84 % acceptance the production config reaches **65.9 tok/s**, above the ~60
tok/s slimsami reference.

**Decision: no further decode-kernel work required for the serving target.**
The apparent gap to ~60 tok/s was a benchmarking artifact (greedy vs MTP): the
production MTP serving path meets the target. This is *not* a claim that the
decode kernel matches the reference — a strict like-for-like `gufo bench`
comparison (same flags, same 16-token prefix, greedy) still shows real gaps to
the reference on greedy decode, long-context prefill and MTP n=2, root-caused in
[BENCHMARKS.md](BENCHMARKS.md) "Reading the numbers".

## Prefill fusion — GDN conv + q/k RMSNorm in one launch

The decode fusion 3 above had no prefill counterpart: `Executor::LinearAttentionBatch`
ran the chunked depthwise causal conv (`GdnConvPrefill`) and then a separate
per-head RMSNorm of the q and k groups (`GdnNormQkPrefill`) — two launches over
the 30 linear-attention layers per prefill chunk, with the convolved q/k written
to `pf_convolved_` only to be read straight back by the norm. Profiling the
16k prefill put the GDN front-end at ~17 % of kernel time against the reference's
~11 %, so this round-trip was the lowest-risk slice of that gap.

`GdnConvNormQkPrefill` gives each block one (token, head) and each thread one
channel: the conv is the same per-channel tap dot (halo from the qkv rows for
`s >= 0`, from the history for `s < 0`) plus `DSilu`, and the q/k blocks reduce
the head's convolved values with the identical `BlockReduceSum` and
`blockDim == head_dim` as `GdnNormQkPrefill`, so `qn`/`kn` are bit-identical
while the convolved q/k never leaves the block. The value block still writes the
convolved value the delta recurrence consumes; the history is left for
`GdnHistoryUpdate` to advance, exactly as before.

The oracle test `qwen36_a3b.rocm_gdn` (`TestFusedConvNormQkPrefill`) compares the
fused launch against `GdnConvPrefill` + `GdnNormQkPrefill` on the model's
16-key/32-value/128-wide geometry at 1/2/5/17/64 tokens and reports qn, kn and
the value half of convolved all `bit-exact: yes`; the existing prefill-vs-decode
checks still pass within 1e-4.

Interleaved same-session A/B (`gufo bench`, depths 2048/8192/16384, 2 passes ×
3 repetitions, exclusive GPU, baseline and candidate run back-to-back per pass to
cancel drift):

| depth | baseline (pass 1, 2) | candidate (pass 1, 2) | gain |
| ---: | ---: | ---: | ---: |
| pp2048 | 2163.4, 2171.6 | 2194.4, 2192.1 | +1.2 % |
| pp8192 | 2086.2, 2084.6 | 2114.7, 2102.5 | +1.1 % |
| pp16384 | 1924.7, 1922.9 | 1940.0, 1938.4 | +0.8 % |

**Decision: retained, always-on.** `GdnConvNormQkPrefill` is the default in
`Executor::LinearAttentionBatch`. The candidate wins all six paired comparisons,
and at pp16384 — where the baseline is tight (±2.5) — the +15 t/s gain is far
outside the run-to-run band. It is bit-identical and removes one launch plus the
convolved q/k round-trip per linear layer per chunk. The remaining GDN gap is now
the delta recurrence itself, which is latency-bound (sequential over tokens) and
needs a chunked reformulation rather than an epilogue.

## Why the reference fork is faster at long context: a whole-pipeline profile diff

Question: `gufo_slimsami` reaches pp102400 = 1331 t/s against our 1114 t/s
(1.19x), and the gap widens with depth (88 % at 32k, 85 % at 65k, 84 % at 100k).
Earlier rounds assumed the gap lived in the WMMA attention kernel. To test that
without guessing, both binaries were profiled over an identical exclusive-GPU
`bench --n-prompt 102400 --n-gen 1 --repetitions 1` run with
`tools/prof/prof.py run --stages qwen`, then compared by kernel category (the
two forks name kernels differently, so the shared stage map only matches the
reference's names). The measured throughput was re-confirmed without the
profiler first (1113.67 vs 1331.15 t/s), so the profile is not distorting it.

Whole-run totals (both include the warm-up and timed passes; attention work is
N^2/2 and therefore chunk-independent, so the totals are directly comparable):

| metric | ours | reference |
| --- | ---: | ---: |
| kernel-time sum | 175.6 s | 151.4 s |
| wall span | 200.8 s | 151.8 s |
| idle in span | 25.2 s (12.6 %) | 0.43 s (0.3 %) |
| dispatches | 372 140 | 75 500 |
| attention kernel | 85.2 s | 67.4 s |
| GDN (linear-attn) kernels | 17.1 s | 9.8 s |

Three compounding causes, largest first:

1. **Attention kernel — head-major packed path (ours +26 %).** The reference
   runs `WmmaCausalAttention<16,2,32,16,true>` (`kHeadMajor=true`) as its
   default for chunks >= 1024. It pre-packs the KV prefix into a WMMA-fragment
   ordered buffer (`PackAttentionHeads`, only 0.4 % of time) so the kernel loads
   V *directly as WMMA fragments* — no LDS transpose and no per-16-key barrier
   (`attention_wmma.hip:266-279`). Our build runs the non-head-major packed-V
   kernel that still does the strided global read + LDS transpose every tile.
   Same N^2/2 work, 85.2 s vs 67.4 s. This is the head-major path ported and
   reverted earlier this session; the revert measured a regression, so a
   faithful re-port must reproduce the reference's fragment-ordered pack and its
   direct `LoadFrag` V read, not a generic head-major layout.

2. **GDN recurrence and epilogues (ours +74 %).** Ours spends 17.1 s on the 31
   linear layers (`GdnDeltaLoopRowSplit` 9.5 s + `GdnOutNormPrefill` 3.9 s +
   `GdnConvNormQkPrefill` 3.8 s) against the reference's 9.8 s
   (`BatchedDeltaNetRowSplit` 5.7 s + `BatchedSSMConv` 1.9 s +
   `BatchedFusedSSMPostNormGate` 1.7 s + `DeltaNetPrepKq` 0.6 s). The reference
   fuses post-norm+gate+quantize and qk-norm+rope+kv-write into single batched
   kernels; ours runs more, smaller, less-fused launches. The delta recurrence
   itself is also slower per unit work because ours uses ~2x smaller prefill
   chunks, doubling the per-chunk state load/store overhead of the sequential
   recurrence.

3. **Dispatch / launch-bound idle (ours 372k dispatches, 12.6 % idle vs 75k,
   0.3 %).** Ours launches ~5x more kernels. The single largest contributor is a
   per-token embedding loop (`executor.cpp:1001-1005`) that fires one 32-thread
   `EmbedRowQ8_0` per token — 204 817 launches (about 2x the 102 400 tokens over
   two passes) — where the reference issues one `LaunchBatchedEmbeddingLookup`.
   Ours also caps `prefill_chunk_` at 2048 (`executor.cpp:222`, to bound the
   MoE scratch near 230 MB) while the reference uses ~4096, halving its
   per-chunk dispatch count.

**Conclusion.** The long-context gap is not one magic kernel. It is (a) the
head-major attention fast path, (b) broader GDN fusion/batching, and (c) larger
chunks plus batched embedding that remove launch-bound idle. Ranked by expected
throughput impact and risk: head-major attention is the only lever large enough
to close most of the 19 % gap but is the fragile, tiling-coupled path already
reverted once; batched embedding is the safest dispatch win (removes ~55 % of
our launches, numerically identical) but mostly attacks idle rather than the
timed-pass GPU busy time; the GDN recurrence needs the chunked reformulation
already noted above.

### Attempted lever: batched prefill embedding (neutral, not retained as a win)

Cause 3 above names the per-token embedding loop (`executor.cpp:1001-1005`,
204 817 `EmbedRowQ8_0` launches over a 100k prefill). A batched `EmbedRows`
(grid over `rows x q8_blocks`, token ids copied once to a device buffer
`pf_tokens_`) replaces the loop with one launch. The per-element arithmetic is
unchanged, so each row is bit-identical; a greedy `prompt` over a 3200-token
(2-chunk) prefix produced byte-identical generated text on the baseline and the
candidate (only the model-load timing line differed).

Interleaved same-session A/B (baseline = fused-GDN binary, candidate = +batched
embedding; 3 passes, exclusive GPU):

| depth | baseline median | candidate median | delta |
| ---: | ---: | ---: | ---: |
| pp32768 | 1698 | 1722 | +1.4 % (within noise) |
| pp102400 | 1110 | 1112 | +0.2 % (within noise) |

**Decision: throughput-neutral, not committed as a performance win.** Embedding
is ~0.2 % of prefill GPU time, so removing its launches cannot move the timed
pass; the 12.6 % idle is not embedding-launch-bound (it is dominated by the
large host-side gaps around model setup, not the per-token kernels). The change
is correct and strictly reduces dispatches, but it does not close the long-context
gap, which lives in the attention kernel (cause 1) and the GDN recurrence
(cause 2). Reverted to keep the working tree focused on levers that move the
measured number.

## Short-context profile diff (pp16384): the gap is GDN, not attention

Established the short-context gap (interleaved, exclusive GPU, 3 passes, ours =
fused-GDN binary, ref = gufo_slimsami):

| depth | ours median | ref median | ratio |
| ---: | ---: | ---: | ---: |
| pp2048 | 2174 | 2387 | 91.1 % |
| pp8192 | 2121 | 2344 | 90.5 % |
| pp16384 | 1936 | 2205 | 87.8 % |
| pp102400 | 1110 | 1331 | 84 % |

A ~9 % gap exists even at pp2048 where attention is negligible, so most of the
gap is **context-independent per-token work**, not the attention quadratic.
Profiled both binaries at pp16384 (`prof.py run --stages '' --top 30`, DBs in
`/tmp/opencode/prof_ours16k`, `/tmp/opencode/prof_ref16k`). Per-stage totals:

| stage | ours | ref | delta |
| --- | ---: | ---: | ---: |
| GDN (delta loop + conv + out-norm + qk) | 2813 ms | 1571 ms | **+1242 ms (+79 %)** |
| Attention | 2310 ms | 1763 ms | +547 ms (context-dependent) |
| MoE experts (RoutedF16GEMM + grouped + epilogue) | ~4668 ms | ~4727 ms | ~equal |

The GDN delta (~1242 ms) is essentially the whole 16K prefill gap (~1.03 s);
MoE is already on par. GDN is linear in tokens, so this gap is present at every
depth and is the ~9 % floor. Two sub-gaps:

- **Core recurrence**: `GdnDeltaLoopRowSplitKernel` 1541 ms (510 launches,
  grid 2x32) vs ref `BatchedDeltaNetRowSplitKernel` 899 ms (240 launches, grid
  1x32). Same RowSplit algorithm, ours +71 %. Ref batches multiple chunks per
  launch (fewer, larger grids).
- **Epilogue**: ours `GdnOutNormPrefill` 647 + `GdnConvNormQkPrefill` 621 =
  1268 ms (2 kernels) vs ref `BatchedSSMConv` 308 +
  `BatchedFusedSSMPostNormGateQuantizeQ8_1` 265 + `BatchedDeltaNetPrepKq` 91 =
  664 ms (3 batched/fused kernels). Ours +604 ms.

**Lever: GDN.** Attack the core-recurrence efficiency first (biggest single
sub-gap, +642 ms, context-independent), then the epilogue fusion. Test at
pp2048/pp8192/pp16384 before escalating.

## GDN prep kernels: hoist decay/beta/q·k out of the serial recurrence

The row-split recurrence (`GdnDeltaLoopRowSplitKernel`) recomputed, for every
token, three transcendentals (`expf`/softplus for the decay, sigmoid for beta)
and a full 4-lane `RowXorAdd` reduction for `k·q` — all redundantly, because all
256 threads of every block that shares a head computed the same head scalars, and
the `k·q` dot does not depend on the state row. These sat in the issue slots of
the serial token loop, which is latency-bound on the state recurrence.

Added two prep kernels that run once per chunk before the loop:

- `GdnPrepAlphaBetaKernel` — `alpha_pre[t,h] = expf(a[h]*softplus(alpha[t,h]+dt[h]))`,
  `beta_pre[t,h] = sigmoid(beta[t,h])`, one thread per (token, value head).
- `GdnPrepKqKernel` — `kq_pre[t,kh] = (q·k)*1/sqrt(d)`, four cooperating lanes per
  (token, key head) accumulating the same eight `float4` in the same order and
  reducing with the same `RowXorAdd` butterfly as the loop, so the stored value is
  bit-identical to the in-loop `kq`.

The loop now reads `decay`, `b`, `kq` as three scalars. The reduction *count* for
`u`/`p` is unchanged (still two `RowXorAdd`); the win is removing one `RowXorAdd`
plus four transcendentals from the serial loop's critical path.

**Bit-exact.** The prep expressions and the `k·q` accumulation order/reduction
match the removed in-loop code operand-for-operand. The GDN unit test
(`qwen36_a3b.rocm_gdn`, prefill-vs-decode and the snapshot verify path) passes,
and a greedy `prompt` over a 3200-token (2-chunk) prefix is byte-identical to the
fused-GDN baseline (180 bytes, only the model-load timing line differs).

**Throughput (interleaved, exclusive GPU).** Baseline = fused-GDN binary,
candidate = +prep kernels. Short context (3 passes, `-r 3`), then long-context
confirmation:

| depth | baseline | candidate | delta |
| ---: | ---: | ---: | ---: |
| pp2048 | 2187.8 | 2262.8 | **+3.4 %** |
| pp8192 | 2115.8 | 2177.4 | **+2.9 %** |
| pp16384 | 1945.1 | 1999.8 | **+2.8 %** |
| pp32768 | 1712.0 | 1742.5 | +1.8 % |
| pp65536 | 1363.5 | 1379.3 | +1.2 % (3 passes) |
| pp102400 | 1108.6 | 1126.0 | +1.6 % |

Candidate beat baseline in every cell of both short-context passes, far above the
per-run stddev. The gain shrinks with depth as expected: the per-token saving is
constant (the loop runs per chunk, capped at 2048) while attention's quadratic
share grows. A single-pass pp65536 first read −1.5 %; three interleaved passes
resolved it to +1.2 % (candidate stable 1385/1377/1376).

The short-context A/B above was measured with the (neutral) batched-embedding
change still present. After reverting embedding to per-token `EmbedRow`, prep
alone re-measured **+1.96 % (pp8192)** and **+1.95 % (pp16384)**, still bit-exact —
so the recurrence win stands on its own.

**Decision: retained as the default.** Bit-exact, no env switch, positive at every
depth. The next GDN sub-gap (core-recurrence launch batching, epilogue fusion)
remains open.

## GDN recurrence: prefetch the critical-path scalars one token ahead

After the prep kernels the loop reads `decay`, `b`, `kq` as scalars, but it still
loaded them (and the convolved `v` row) fresh at the top of every token iteration
and used them immediately, so four global-load latencies sat on the serial
recurrence's critical path. The loop is latency-bound, not compute-bound: at
~1600 cycles/token for ~100 FMA, the exposed load latency dominates. The `k`/`q`
vectors were already double-buffered (prefetched for `t+1`); the scalars were not.

Double-buffered `decay`, `b`, `kq` and `vcur[r]` exactly like `k`/`q`: load the
current token's before the loop, prefetch `t+1`'s inside the existing prefetch
block, swap at the bottom. Pure load scheduling — the arithmetic, its order, and
the two `RowXorAdd` reductions are untouched, so the result is bit-identical.

**Bit-exact.** The GDN unit test passes and a greedy `prompt` over the 3200-token
(2-chunk) prefix is byte-identical to the prep baseline (204 bytes, only the
model-load timing line differs).

**Kernel time (profiler, pp16384, exclusive GPU).** The target kernel drops
directly, with the attention kernel unchanged:

| kernel | baseline (prep) | +prefetch | delta |
| --- | ---: | ---: | ---: |
| `GdnDeltaLoopRowSplitKernel` mean | 2217.7 µs | 1801.4 µs | **−18.8 %** |
| `GdnDeltaLoopRowSplitKernel` total | 1729.8 ms | 1405.1 ms | −324.7 ms |
| `WmmaCausalAttentionKernel` mean | 13272 µs | 13384 µs | noise |

**Throughput (interleaved, exclusive GPU, 3 passes, `-r 3`).** Baseline = prep
binary, candidate = +scalar prefetch. The kernel is 6.8 % of GPU-busy and ~30 % of
the span is idle, so the −18.8 % kernel win lands as a smaller end-to-end gain:

| depth | baseline | candidate | delta |
| ---: | ---: | ---: | ---: |
| pp2048 | 2247.4 | 2265.8 | **+0.8 %** |
| pp8192 | 2161.9 | 2171.3 | **+0.4 %** |
| pp16384 | 1970.4 | 1984.9 | **+0.7 %** |

Candidate beat baseline in 2 of 3 passes at every depth and the mean is positive
everywhere; the effect is modest relative to run-to-run spread but the change is
provably not a regression (it only removes exposed latency, adding four register
copies per iteration).

**Decision: retained as the default.** Bit-exact, no env switch, positive at every
depth. The remaining GDN sub-gap is the recurrence's low parallelism (64 blocks:
32 value heads × 2 row halves) — raising it needs a chunked-scan reformulation
that changes the summation order, so it is not a bit-exact lever.

## Dense projections on the matrix cores: F16 WMMA (rejected — not numerics-preserving)

After the GDN wins the residual prefill gap vs the reference widens with depth
(pp2048 +4.3 %, pp8192 +6.9 %, pp16384 +9.9 %), the signature of per-token dense
compute. A pp16384 profile puts `mul_mat_q` at 23.4 % of GPU-busy — the single
biggest kernel — while the routed experts already run on the matrix cores.

**Premise correction (verified after the fact).** This record originally framed
`mul_mat_q` as the dp4a tier and the reference's int8 WMMA GEMM as the missing
piece. That is wrong. On gfx1151 `__GFX11__` maps to `AMD_WMMA_AVAILABLE` (not
`AMD_MFMA_AVAILABLE`), and `mul_mat_q_process_tile` selects `vec_dot_mma` under
that macro (`mmq.hpp:3422-3428`); the int8 `mma` overload then emits
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (`mma.hpp:1008-1013`). So our dense
Q8_0 projections **already run int8×int8→int32 on the WMMA matrix cores** — the
same hardware path as the reference's `W8A8BlockedWmmaGEMMKernel`. The gap is
therefore a *kernel-tuning* difference (the generic `mul_mat_q` tile/warp
selection vs a purpose-built GEMM), not a dp4a-vs-matrix-core difference. The
F16 experiment below is reinterpreted accordingly: it swapped the tuned
`RoutedF16GEMMKernel` in for `mul_mat_q`, and its +4 % is that tuning delta.

**Lever tried (bounded).** Feed a dense Q8_0 projection through the existing
matrix-core `RoutedF16GEMMKernel` by treating the token batch as a single
64-row-tiled bucket: identity `rows_in`/`rows_out` over `[0, tokens)`, pad bounds
`{0, tokens}`, one tile per `ceil(tokens/64)` packed as `j<<16`, activations
narrowed F32→F16, F32 output. Wired into the attention/linear-attention `proj`
dispatch and the shared expert (router left on dp4a for exact top-k; reduction
dims not a multiple of 64 and `rows < 64` fall back to the dp4a GEMM). The kernel
already masks rows past `bucket_rows` on both load and store, and with `expert=0`
the weight pointer is the plain `[rows][cols]` block_q8_0 base, so the mapping is
correct.

**Throughput (interleaved, exclusive GPU, 3 passes, `-r 3`).** Baseline = the
committed dp4a tree, candidate = +F16 WMMA dense.

| depth | baseline | candidate | delta |
| ---: | ---: | ---: | ---: |
| pp2048 | 2287.2 | 2314.0 | **+1.2 %** |
| pp8192 | 2198.0 | 2256.5 | **+2.7 %** |
| pp16384 | 2011.3 | 2091.2 | **+4.0 %** |

The lever is real and grows with depth — a better-tuned matrix-core kernel for
the dense projections closes roughly half the 16K gap to the reference.

**Quality (greedy, exclusive GPU).** The 3200-token `long` prompt is byte-identical
at 64 and 256 tokens, but the harder arithmetic prompt diverges by token ~15:
baseline `Step-by-Step**` → candidate `Step-by-Step Solution**`, and
`Train B leaves at 10` → `Train B leaves at 1`. The F16 route dequantizes Q8_0
weights and narrows F32 activations to F16, so `wmma_f32_16x16x16_f16` products
differ from the baseline int8×int8→int32 WMMA path; on borderline tokens the
argmax flips.

**Decision: rejected.** The optimize-kernel contract is to preserve the product
and accumulation sequence; F16×F16→F32 cannot. Because our baseline is *already*
int8 WMMA, the bit-exact way to recover this +4 % is **not** a port — it is to
tune the existing int8 WMMA `mul_mat_q` for the dense prefill shapes (M≈2048–4096,
K=2048, N=tokens): the column-tile width (`mmq_x`), `mmq_y`, warp count and the
`ncols_grid_max`/`mmq_x_request` knobs the struct already exposes. That keeps the
int8×int8→int32 products exact. The F16 experiment is kept here as evidence of the
tuning headroom, not as a candidate.

*Update:* the `mmq_x` width knob was tested (see the next record) and is neutral —
the default already uses the widest tile. The remaining headroom is in the kernel's
warp/LDS/epilogue structure, not in the exposed tile-geometry knobs.

**Ceiling check (`gfx1151_peak`, exclusive GPU, 2.90 GHz, 20 CUs).** int8 WMMA
is not slower than float WMMA on this part, so the +4 % cannot be an int8-vs-f16
hardware gap — it is tuning:

| path | sustained |
| --- | ---: |
| WMMA int8 | 56.99 TOPS |
| WMMA fp16 | 55.19 TFLOPS |
| WMMA bf16 | 55.20 TFLOPS |
| VALU fp32 fma | 23.81 TFLOPS |

The dense `mul_mat_q` therefore has the same matrix-core ceiling available to it
as the F16 kernel; the headroom is in tile/warp selection, not the instruction.

## Shared-input activation quantize dedup (neutral — launch-overhead-bound)

The pp16384 profile shows `quantize_mmq_q8_1` at 3.6 % of GPU-busy over 3960
launches. The linear-attention input projections (`ssm_qkv`, `ssm_gate`,
`ssm_alpha`, `ssm_beta`) all read the same activation `x`, and the attention
projections (`attn_q`, `attn_k`, `attn_v`) all read the same `x`. Each dense
`mul_mat_q` call quantizes `x` independently, so the shared quantize is issued
4× (linear) and 3× (attention) per layer instead of once.

**Lever tried.** Factor the quantize and the GEMM out of `qfn_mmq_dense_impl`
into `mmq_quantize_src1` + `qfn_mmq_dense_gemm<type>`, and add
`qfn_mmq_q8_0_dense_group` (quantize `x` once into one `block_q8_1_mmq` scratch,
then run one MMQ GEMM per weight). Exposed as `GemmGroupQ8_0` and dispatched from
`LinearAttentionBatch`/`AttentionBatch` when `tokens > kMaxVerifyRows`, every
member is Q8_0, and the input widths match; otherwise fall back to the
per-projection `proj`.

**Bit-exactness (greedy, exclusive GPU).** The quantized activation depends only
on `(tokens, K)`, not on any weight's output width, so the group's single buffer
is byte-identical to each per-call buffer and every GEMM reads the same bytes.
Both the 3200-token `long` prompt (1051 B) and the harder `hard` prompt (717 B)
are byte-identical at 256 tokens greedy.

**Throughput (interleaved B,N,B,N, exclusive GPU, `-r 3`).**

| depth | baseline | candidate | delta |
| ---: | ---: | ---: | ---: |
| pp2048  | 2272.4 | 2251.0 | −0.9 % (R1 −1.9 %, R2 +0.05 %) |
| pp8192  | 2190.0 | 2163.1 | −1.2 % |
| pp16384 | 2011.7 | 2011.1 | −0.03 % (wash) |

**Decision: not retained.** Bit-exact and strictly less work (≈5 fewer quantize
launches per layer per chunk), but the saving is launch-overhead-bound: a
`[2048×2048]` quantize is ~0.1 ms and the removed launches are ~µs each, so the
total is well under the ±1 % run-to-run floor. pp16384 is a wash and pp2048 flips
sign between rounds. The change was reverted; the lever is recorded so the
quantize-dedup idea is not retried expecting a win. The dense-GEMM headroom is in
`mul_mat_q` tile tuning (see the F16 record above), not in the quantize.

## Dense `mul_mat_q` column-tile width: `mmq_x` 80 → 128 (neutral — default already optimal)

The F16 record points at tile tuning as the bit-exact way to recover the dense
+4 %. The default selection loop in `mul_mat_q_case` picks the *widest* usable
`mmq_x` (fewest column tiles), so dense currently runs at the host cap of **80**
(`get_mmq_x_max_host`). That cap's comment says it was swept only for the
**IQ2_XXS routed-MoE** workload, and the device supports up to **128**
(`get_mmq_x_max_device`). Dense token columns are contiguous and padding-free, so
a wider tile could amortize the per-tile dequant better than the scattered routed
buckets that set the 80 cap.

**Lever tried.** Let a caller *request* a width up to the device max while keeping
the default loop bounded to the host cap (so routed/MoE defaults are untouched):
widen the `mmq_x_request` gate to `get_mmq_x_max_device()` and set
`args.mmq_x_request = 128` in `qfn_mmq_dense_impl`.

**Bit-exactness (greedy, exclusive GPU).** Non-split-K (`use_stream_k=false`)
means each output element's K-reduction is one warp's fixed template loop,
independent of the grid tile width. Both the 3200-token `long` prompt (1051 B) and
the `hard` prompt (717 B) are byte-identical at 256 tokens greedy.

**Throughput (interleaved B,N,B,N, exclusive GPU, `-r 3`).**

| depth | baseline (x=80) | candidate (x=128) | delta |
| ---: | ---: | ---: | ---: |
| pp2048  | 2259.6 | 2267.2 | +0.34 % (R1 +0.71 %, R2 −0.03 %) |
| pp8192  | 2184.9 | 2169.8 | −0.69 % (R1 −1.52 %, R2 +0.15 %) |
| pp16384 | 2010.3 | 1998.7 | −0.58 % (R1 −1.12 %, R2 −0.03 %) |

**Decision: not retained.** Bit-exact but neutral-to-slightly-negative, exactly as
the cap comment predicted for the routed sweep (64/96/112/128 all tie 80 within
noise) — and now confirmed for the dense Q8_0 shapes too. The default already uses
the widest tile, so there is no width left to win. The dense-GEMM headroom is
therefore **not** a knob tweak: recovering the F16 kernel's +4 % bit-exactly needs a
purpose-built int8 WMMA dense kernel (its own warp/LDS/epilogue structure), not a
retune of the generic `mul_mat_q` tile geometry. Reverted.

## Routed MoE fused gate+up (`kPair`): one launch, bit-exact (retained)

The routed MoE gate and up projections each launched a separate
`RoutedF16GEMMKernel` over the same 64-row tile map and the same F16 activation
rows: the gate wrote F32 to `pf_gate_`, then the up kernel read it back and applied
SwiGLU. `RoutedF16GEMMKernel` already carried an un-dispatched `kPair` path (a 5th
template param plus a `w_up` argument) that decodes the gate rows from `w` and the
matching up rows from `w_up` in one launch and applies SwiGLU in the epilogue — it
was simply never wired up. The pp16384 profile puts `RoutedF16GEMMKernel` at 21 %
of GPU-busy, so this is the routed-side lever.

**Lever.** Dispatch `kPair`. Add `LaunchRoutedGatedF16<BN>` + `RoutedGatedF16Gemm`
(`grid.x = ceil(m/64)`, `BN` 64 or 128, gate as `w`, up as `w_up`, F16 out only).
The executor builds a *second* paired tile map after the 64-row map — one entry per
`pair_rows`-row tile, `pair_rows = 128` when it cuts the launch to at most three
quarters of the 64-row tile count — and, when `wmma && tokens >= 1024 &&
gate_type == up_type`, replaces the two launches with one fused call. The down
projection is unchanged (it reads the same F16 activation rows indexed by slot).

**Bit-exactness (greedy, exclusive GPU).** The fused epilogue reproduces the
separate up epilogue's `up * SiluF(gate)` exactly (`silu = gate * SigmoidF(gate)`,
then `up * silu`); the K accumulation order is unchanged (same `BK = 2` WMMA loop —
`BN` only changes the token-tile count, not a given element's reduction), and the
in-register gate accumulator equals the F32 `pf_gate_` the separate path round
trips. `-ffast-math` is off for this target, so nothing reassociates. Both the
3200-token `long` prompt (1051 B) and the harder `hard` prompt (717 B) are
byte-identical at 256 tokens greedy.

**Throughput (interleaved B,N,B,N, exclusive GPU; `-r 3` at ≤16K, `-r 1` at ≥32K).**

| depth | baseline | fused | delta |
| ---: | ---: | ---: | ---: |
| pp2048   | 2268.8 | 2286.7 | **+0.79 %** |
| pp8192   | 2175.4 | 2201.2 | **+1.19 %** |
| pp16384  | 2010.2 | 2042.2 | **+1.59 %** |
| pp32768  | 1767.1 | 1792.1 | **+1.41 %** |
| pp65536  | 1379.5 | 1399.5 | **+1.46 %** |
| pp102400 | 1125.0 | 1143.9 | **+1.68 %** |

**Decision: retained.** Bit-exact and positive at every depth, strongest at long
context (pp16384 +1.6 % both rounds, tight error bars). The fused kernel reads `x`
once, halves the gate+up launches, and drops the gate's F32 write/read round trip.
It is now the default for chunks of ≥1024 tokens whose gate and up share an
encoding; smaller chunks and mismatched encodings keep the two-launch path.

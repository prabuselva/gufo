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

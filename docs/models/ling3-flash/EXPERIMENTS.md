# Ling 3.0 Flash experiments

| Experiment | Decision / evidence |
| --- | --- |
| Fused MLA prefill kernel (Q-tile + online softmax + PV in one pass, 32×256 exec tile) | Retained (`473c946b`); up to 2.3× prefill vs staged path, suite worst prefill error 9.7e-5 (contract 5e-4). |
| `-UNDEBUG` on `gufo_ling3_flash_kernels` | Retained (`473c946b`): `MlaPrefillFusedKernel` spilled 143 VGPR / 316 B per lane (DRAM scratch) under `-DNDEBUG` only; allocator artifact, asserts absent from the kernel. A/B with `-Rpass-analysis=kernel-resource-usage`; release fatbin now 191 VGPR / 0 spill. |
| KV row staged into registers (`v[KvSlice]`) shared by QK dot and PV update | Retained (`9b1434ca`): harness gain 1.02–1.05× alone; halves shared-memory traffic per (query, key) pair. Not bit-identical to the previous form (compiler contraction moves results ≤ ~4 ulp, maxabs 4e-9 in harness); suite passes. |
| Fused kernel widened to 16 warps, `WarpQ` 4→2, Q-tile held at 32 | Retained (`9b1434ca`): harness gains 1.25–1.43× (best at depth), e2e pp +4.2/+8.4/+13.0/+22.4% at pp2048/4096/8192/16384; release build 118 VGPR, 0 spill, occupancy 12 waves/SIMD. Root lever: 36 KB LDS made 8 warps the resident maximum; the kernel is latency-bound (~2.2 TF/s fp32), not LDS-bound. |
| 16 warps with Q-tile 64 (`WarpQ` 4) | Rejected: 1.27–1.36× — worse than tile-32 at 16K depth (805→846 µs) with equal short-context gains. |
| Shipped-form 8-warp with KV register reuse only | Superseded by the combined 16-warp change (1.02–1.05× alone). |
| WMMA/fp16 attention scores | Deferred: tolerance risk against the 5e-4 prefill contract; not attempted. |
| Live-token-tile skip extended to every `RoutedF16GEMMKernel` instantiation (was `kPair`/`kQ5`/`kQ8 BN≥48` only; plain Q4_K/Q5_K/Q6_K paths ran WMMA for dead token tiles of short expert buckets) | Retained (working tree): `tools/bench/ling3_routed_bench.hip` (tokens=1024, α=0.09, fixed seed) down launch 2444→1950 µs (−20.2%, both run orders in matched exclusive sessions), layer −6.5%; outputs bit-identical (Fnv sums `0848f30e6e00c75d`/`5ab1d584de7338af` unchanged, oracle ratios unchanged). Side effect: Q4_K BN64 down instantiation 172→111 VGPR, 0 spill, occupancy 12. e2e exclusive ABBA pp +3.7/+2.9/+2.8/+1.6% at pp2048/4096/8192/16384. |
| Compile-time stage ablation of the post-skip Q4_K bench kernels (temporary `#if` guards removing one pipeline stage at a time: WMMA→scalar fold, nibble decode→register fill, activation-frag LDS loads→const, activation/weight global loads→const; one exclusive session, base re-measured 3×: gate 3324–3421 µs, down 1937–1962 µs) | Measurement only (guards reverted). Recovery vs base — gate: weight global loads −17%, WMMA −14%, weight decode −7%, activation global −5%, activation-frag LDS loads ~0%; down: WMMA −12%, weight global −8%, decode −7%, activation global −2%, frag LDS ~0%. No single dominant stage; the ≥60% residual is the single-buffered per-K-block `__syncthreads`/LDS round-trip chain, so further decode micro-tuning cannot pay — a pipeline restructure (double-buffered LDS with one barrier per stage, or global→LDS-direct weight loads) is the next real lever. Hardware context (`gfx1151_peak`): WMMA fp16 55.25 TFLOPS, VALU fp32 23.95 TFLOPS, DRAM read 240.7 GB/s; gate runs at 34% of the WMMA bound. |
| Double-buffered LDS KB pipeline in `RoutedF16GEMMKernel` (two parity planes, one `__syncthreads` per stage instead of two, register prefetch kept at depth 1 with the next stage's global load issued after the commit; instantiations whose doubled plan exceeds the 64 KB static-LDS limit keep the old loop) | Rejected: bit-identical (same Fnv sums) but slower in exclusive ABBA — gate 3371→3726 µs (+10.5%), down 1952→2199 µs (+12.7%). The doubled plan halves resident workgroups (Q4_K BN64 down: LDS 13440→26880 B, occupancy 12→8 waves/SIMD, 111→118 VGPR), and this kernel hides its barrier latency through wave occupancy, so the halved occupancy costs more than the removed barrier. Single-buffered + occupancy is the local optimum; restructurings that keep the LDS footprint (e.g. more work per wave to shrink sync frequency) remain open. |
| Hardware counters per dispatch (`rocprofv3`, exclusive sessions) | Measurement only. Issue-throughput bound: gate VALU ≈171 M instructions ≈57% of peak issue, LDS ≈9%; waits split WAIT_CNT 61% / BARRIER 18%; `SQC_DCACHE_MISSES` ≈1k per dispatch — weight streams are L1/L2-resident, so bytes-moved and DRAM-swizzle levers are dead. |
| ISA mix of the shipped Q4_K BN64 kernels (`hipcc --cuda-device-only -S`, `tools/prof/isa_mix.py`) | Measurement only. Per whole kernel, down 3739 / gate 1758 instructions: WMMA ≈1%, packed decode 4–6%, 64-bit address math (paired `v_add_co`, `v_mad_u64_u32`, `v_lshlrev_b64`) ≈15%, f32 scale products (`v_mul_f32` 119/64, `v_fma_f32` 96/48), `s_delay_alu` fillers 17%/14%. Every remaining gain must remove issued instructions — pipeline retiming and scheduling hints are exhausted. |
| Loop-body ISA decomposition of the KB steady loop (back-edge span analysis of the same listings) | Measurement only. The KB loop itself (down 336 instr: 16 WMMA, 96 packed decode, ~34 LDS loads, 4 global loads; gate 374 with the same shape) contains no 64-bit pointer math and no f32 scale products — the kernel-level address/f32 counts live entirely in the once-per-block prologue and peeled epilogue chains (~900-2900 instructions vs 40-80 × 350 of loop work). A buffer-load/addressing rework therefore has near-zero expected steady-state gain and was dropped before implementation. Remaining in-loop headroom: decode chain (ablation ≤7%) and WMMA staging only. |
| `#pragma unroll 2` on the KB loop | Rejected: bit-identical (same Fnv sums), resources unchanged (114/108 VGPR, occupancy 12, no spill), but exclusive ABBA gate 3362→3404 µs (+1.2%), down 1956→1979 µs (+1.1%). Schedule slack is already covered by occupancy. |
| Decode pass profile attribution (`tools/prof/prof.py run --stages ling3`, pp2048/tg64, exclusive; pass-204 kernel table via `prof_results.db`) | Measurement only. Decode is GEMV-tier + fixed-overhead: per token GemvGroupedQ4K 5.8 ms, GemvQ8_0 5.5, GemvQ6K 3.7, GemvQ4K 3.7, **RouterTopKKernel 2.7 ms (~9%, 40× 67.8 µs)**, GemvF32 1.7, KdaDelta 1.6; ~1911 dispatches/token, 31.2 ms kernel time vs ~13 ms DRAM roofline. |
| Parallel `RouterTopKKernel` top-k (block-wide `BlockArgMax` (max value, lowest-index) rounds replacing the thread-0 serial group-select and expert argmax-removal scans; winner thread writes the `-inf` removal, thread-0 keeps the single-precision order-sensitive weight writes and double renorm sum) | Retained (`967de4ba`): bit-identical to the serial kernel in 112/112 randomized A/B cases (old/new extracted from git, `n_exp` 64–512, `groups` 1–64, random + adversarial-tie logits, 1 and 3 token rows; harness `/tmp/opencode/router_ab.hip`) and `ling3_flash.rocm_ops` contract pass (`RouterTopKRows` reports bit-identical, weights worst error 9.8e-8). Kernel micro (exclusive): 64.0→16.6 µs/call (−74%). The old cost was thread-0-only: `used×n_exp` + `groups_used×groups` dependent LDS scans ≈5100 ops ≈50 µs of single-thread latency chain. e2e exclusive ABBA (full recipe, `/tmp/opencode/gufo_bench_rrfull.log`): tg128 +8.2%, tg1024 +7.6% (29.24→31.47), pp +2.0…+3.4% (`RouterTopKRows` shares the kernel). |
| vec4 Q8_0 dense GEMV ported from gemma4 (`GemvQ8_0Vec4`, gated `cols % 128 == 0` in `Gemv()`) | Retained (working tree): quality-contract not bit-identity; `ling3_flash_rocm_gemv_test` GemvQ8_0 worst error 2.3e-5 (tol 2e-3, `kCols=2560` exercises the vec4 path). Kernel micro (exclusive, 5000-launch, `/tmp/opencode/gufo_bench_v4micro.log`): −30% on all five dense shapes (2560×2560 15.76→10.98 µs, 6144×2560 34.06→21.72). e2e exclusive: short tg256 ABBA inconclusive (+0.6%, overlapping); dedicated tg1024 BAAB ×3 (+1.0%, separated, `gufo_bench_v4tg1k`); matched full-recipe BAAB (`gufo_bench_v4full`) tg128 +2.9%, tg1024 +3.2%, both fully separated, pp cells within noise (pp16384 noisy). Realized gain ≪ the 5.5 ms/token Q8_0 share × 30%, so decode wall time now sits in the MoE/other-quant tiers and launch gaps, not Q8_0 bytes. |
| Wide Q4_K grouped walk: one `uint32` weight load per lane per super-block (128 B/warp/instr, scale pair unpacked once, `Q4ScaleMin` ×2 instead of ×8 per lane) | Rejected after the `packed >> (4*i)` → `>> (8*i)` nibble bug was fixed (one-hot element-map probe `/tmp/opencode/gq4map`: all 256 elements map to the scalar-rule byte/nibble; wide-vs-scalar maxdiff 1.2–2.4e-4 in `/tmp/opencode/gq4w`): exclusive micro is *slower* — 768×2560 47.3→49.8 µs/call (+5.3%), 2560×768 53.9→55.2 (+2.5%). The scalar `qs[lane]` walk is already a fully-coalesced 32 B/warp/instr stream at ~70–78% of the 240.7 GB/s read peak and the wide x-loads (two 32-element windows per lane, 8×4 B instead of 2×16 B vector gathers) forfeit more than the weight-load count saves; kernel reverted to the shipped walk. |

| Decode-only profile (`tools/prof/prof.py run --stages=`, pp8/tg256 `--speculative mtp`, exclusive) | Measurement only. Decode is ≥94% GPU-busy; top kernels GemvGroupedQ4K 24.0%, GemvQ6K 8.2%, KdaDeltaKernel 6.3%, MlaDecodePartialGroup 4.1% (119 µs/call). **`__amd_rocclr_copyBuffer` 2.3% at 94582 calls × 1.63 µs ≈ 92 D2D copies per decode forward** — the MLA append cluster (per MLA site: 2 cache-row copies + 32 per-head 256 B q-tail copies). |
| HIP graph capture/replay of the MTP decode step (8-file executor+attention change, capture once per shape, replay per token) | Rejected: capture burst only ~130 ms/run, so steady replay overhead made tg1024 spec −1.1…−1.7% vs the GPU-bound HEAD (≥94% busy — graphs only cut launch gaps). Fully reverted; `--draft-tokens 3/5/7` sweep also flat (36.56/36.46/36.72), so accept-policy tuning is not the lever either. |
| `MlaDecodePartialGroup` double-buffered LDS staging (prefetch next chunk to a second 45 KB plane, `GroupDoubleBufKernel` in `tools/bench/ling3_mla_decode_bench.hip`) | Rejected: bit-identical (`bits=yes`) but 0.85× (n=1024) down to 0.48× at depth — 45 KB LDS forces 1 block/CU and the prefetch delays the current chunk's compute on the LSU; no spills. |
| Direct-write MLA cache rows: `RmsNorm` outputs the latent straight into the `kv_cache` position row, new `RopeTo` (separate src/dst + strides, per-thread mapping identical to `Rope`) writes the rotated kpe row and each head's q rope tail into `q_abs + kv_rank`, removing all 34 per-site `hipMemcpyAsync` appends at both decode sites (`MlaLayer` step, `MlaBatch` verify) | Retained (working tree): bit-identical by construction (same ops per thread; writes are disjoint from the gemv outputs, so launch order is irrelevant). 67 of the 92 per-forward copies removed. ling3 ctest 10/12 with the HEAD-identical failure signature (spec parity 49/48, accepted 21/48; pre-existing verify/rollback-bit-parity + gemv FAILs). Matched BAAB tg1024 exclusive (`gufo_bench_ropebaab`/`gufo_bench_plain`): spec `--speculative mtp` 36.22/36.22 → 36.91/37.33 (+2.5%); plain 31.80/31.53 → 32.07/32.15 (+1.4%). This clears the ~37 t/s spec target. |

Open items: MTP speculative decode reaches 37.1 t/s (tg1024) after the
MLA copy removal, meeting the ~37 t/s target; plain decode is 32.1. The
Q8_0 result shows
kernel-level savings do not convert 1:1 at this operating point (~1911
dispatches/token, ~31 ms kernel time vs ~13 ms DRAM roofline, ~12% pass
gaps). Wide-load widening of `GemvGroupedQ4K` was measured and rejected
above (weights already coalesced and near the DRAM read ceiling; GroupedQ6K
runs *below* its byte floor via L2 reuse, so byte-count levers are spent
there too), so the remaining levers are structural — row pairing in the
grouped tiers (2 rows/warp to halve dispatches and share `ids`/scale
prologues), launch-count reduction (~1911 dispatches/token, 3.6 ms idle),
and GemvF32 wg32→128 (1.69 ms/token at 75× 22.5 µs).
`GemvRows` (multi-token draft verify) still uses only the scalar Q8_0 path.
On the
prefill side, after the
live-token-tile skip the stage-ablation split above shows no dominant term
(weight global loads and WMMA each recover ≤17%, the rest being pipeline
latency), and the double-buffered LDS variant was rejected because the
occupancy it sacrifices outweighs the removed barrier; the KB loop body is
already lean (no address math, no f32 scale products), so the only remaining
in-loop lever is the packed decode chain (≤7% ablation ceiling), which is not
bit-identical to shorten. The
gate launch (paired path, 1.7× the down MACs) is now the larger kernel.
Prefill still decays 833→402 t/s from 2K→16K depth, driven by the same
attention kernel. `#pragma unroll 4` on the key loop
now warns (`loop not unrolled`) in the 16-warp form; harmless, revisit if the
loop is touched again.
Reference refresh (llama.cpp b11557, ROCm + Vulkan, October 12,
`BENCHMARKS.md`): the 2K→16K decay is *not* a hardware ceiling — llama.cpp
holds 659→507 (ROCm) / 619→504 (Vulkan) over the same depth range where Gufo
falls 840→402, putting Gufo 20% behind at pp16384 after its +27%/+36% lead
at pp2048. On decode, llama.cpp's Vulkan path runs 41.4–41.6 t/s cold
against Gufo's 33.6 (tg128) / 32.4 (tg1024); their tg1024 is no slower than
tg128, and their pp2048+tg blends imply rate-consistent cold-level decode at
depth (b11557 reports only blended pair rates), while Gufo loses a further ~4–6% preparing a
2048-token frontier before timed generation (`--n-prompt 0 --n-gen 128,1024
-d 2048`: 31.40/31.35). The decode-frontier depth cost (`MlaDecodePartial`
scan shape, `KdaDeltaKernel` state readout) is the concrete follow-up; the
prefill decay remains the biggest known gap to the reference.

`__expf` online softmax — ACCEPTED (October 15, `docs/build_optimized_`
`kernels/STATUS.md` lever 5.1a). Both `::expf` calls in the fused MLA
prefill softmax (`MlaPrefillFusedKernel`) became `__expf`. Kernel probe at
depth (`tools/bench/ling3_mla_prefill_bench.hip`, tok=2048 start=14336):
836.7 → 705.7 ms (1.19×) at 3.5e-8 maxabs vs the exact-clone probe —
~4 orders under the 5e-4 error contract. End-to-end paired exclusive
windows (control = HEAD `0554f35f` without the change, alternating reps):
pp2048 +3.0%, pp4096 +5.0%, pp8192 +8.4%, pp16384 +10.9% (grid refreshed
in `BENCHMARKS.md`; 16K gap to llama ROCm −20.7% → −12.0%); gates
`ling3_flash.rocm_attention/forward/inference/quality` pass (the
`rocm_mtp_accept` failure reproduces identically on the control binary —
pre-existing at HEAD, tracked under STATUS 5.6). The same probe pins QK
dots + softmax at ≈61% of kernel time (PV-only bound 2.57×) and a
×24-scaled fp64-oracle run rules out single-rounded fp16 QK scores at
real activation magnitudes (fp16 ULP on a ±1e4 raw score ≈0.4 nats after
scaling), so the WMMA follow-up (STATUS 5.1b) must accumulate the 3-term
hi/lo split (`Qh·Kh + Qh·Kl + Ql·Kh`, fp32 accumulate) — this supersedes
the earlier blanket "WMMA/fp16 attention scores" rejection for the score
path.

WMMA hi/lo MLA score kernel — REJECTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.1b, reverted). The
3-term hi/lo-split score route the 5.1a oracle sketch called for was
built end-to-end (hi/lo fp16 slab pack on the append path,
`MlaPrefillWmmaKernel` — 8-wave blocks, 32 queries × 512 dims,
16-key tiles, online softmax, `p == 0` PV skip) and passed the
numerics gates (`ling3_flash.rocm_attention` 5e-4 contract on the
full Q4_K_M artifact). The standalone `wmma2` bench win
(1.66×/1.77× vs shipped fused) was a synthetic-data artifact: at
×24-scaled random activations the online softmax flushes almost every
`exp(score − max)` to zero, so both kernels skip nearly all P·V and
the timing compares QK alone. Paired end-to-end exclusive depths all
regressed vs the 5.1a control (tok/s): pp2048 541.6 vs 831.7, pp4096
329.7 vs 739.4, pp8192 188.3 vs 576.0, pp16384 103.3 vs 392.5
(`gufo_bench_ta51.log`). Paired dispatch profiles
(`prof51_{cand,ctrl}/prof_results.db`): WMMA averaged 1364.5 ms per
dispatch vs fused 220.1 ms; at one identical shallow shape fused
cost 13.66 ms and WMMA 242 ms. ISA dumps ruled out codegen (the
release-build kernel matches the fast bench build instruction-for-
instruction; flag bisection found only a 1-register spill toggled by
`-DNDEBUG`). The decisive probe was bench `replay` mode at production
cadence (tokens=1024, rows=4096): deep chunk 71.5 ms fused vs 47.6 ms
wmma at ×24, flipping to 74.0 ms vs 562 ms (7.6×) at unit scale —
matching production dispatch times, because real MLA activations
never fire the skip. Mechanism: the fused kernel stages the V
dimension slice into registers for the QK dot, so the PV update
adds ~2.5 ms total; a tensor-core score kernel must fetch V
separately — 256 scattered `float4` global loads per lane per
16-key tile — and stays LSU-issue-bound (~15× the fused PV). Fixing
PV with `v_wmma` costs 5–7× the QK tensor work plus hi/lo P
rounding, capping the route at parity. Ruling: the blanket rejection
of WMMA/fp16 attention scores stands for this kernel family — a
tensor-core MLA route is only viable if it keeps PV-for-free (each
lane's V slice remains a register byproduct of the score dots).

Decode-frontier Pass-A row broadcast — ACCEPTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.2a). The prefill-decay
follow-up (deep-dive entry above) first needed attribution. Dispatch
profiles at frontier depths {0, 2048, 8192} (`bench --n-prompt 8
--n-gen 128 -d $D`, decode windows split from the second-rep prep
prefill, `/tmp/opencode/prof52_d*`): the depth cost is entirely
`MlaDecodePartialGroupKernel` (79.4 → 332.8 → 1126.0 µs per call,
grid 8×{1,9,33}); `KdaDeltaKernel` is depth-flat (41.8–43.4 µs ×
35 calls/token) and the apparent `KdaDeltaLoopRowSplit` growth is prefill
router contamination (its only call site is the prefill executor,
`executor.cpp:1037`). The kernel's Pass A had assigned one warp per
(staged row, head), so each 256-key tile chunk re-read every row from
shared memory four times — LDS-issue-bound at ≈34 µs/tile. The fix
scores one row against all four group queries per warp, holding each
lane's stripe of the four binary32 query vectors in registers
(predicated `float4` loads, ≤5/lane): one shared-memory `float4` chain
per row, exact same term order per score (bit-identical), 34 → 102
VGPR with no spills and scratch 0. Matched BAAB pair, run twice in one
exclusive window (control = the 5.1a build, `gufo_bench_ba52a.log`):
tg128 d0 33.11/33.32 → 33.37/33.52 (cold neutral, launch-bound), d2048
30.33/30.39 → 31.08/31.17 (+2.5 %), d8192 23.47/23.59 → 25.69/25.75
(**+9.4 %**); depth-shared prefill piggybacks (`pp8 @ d8192`
67.56/68.42 → 72.82/72.42, +7.0 %). Quality gates (`rocm_attention`
incl. the 5e-4 contract, `rocm_forward`, `rocm_inference`,
`rocm_quality`) pass in the same session (`gufo_bench_q52a.log`, rc 8 =
the two pre-existing failures only). The frontier gap is narrowed, not
closed (−7.0 % vs cold at d2048), and cold decode remains launch-bound
(46–79 % idle in the decode span) — tracked as STATUS 5.2/5.4.

Decode staging double-buffer — REJECTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.2b). After 5.2a, a
re-profile at d8192 (`/tmp/opencode/prof52b_d8192`) put
`MlaDecodePartialGroupKernel` at 896.7 µs per call (from 1126.0), ≈27 µs
per 256-key tile, and the per-chunk cadence looked like one exposed
global-load latency per 8-key staging step. The candidate staged
consecutive chunks into alternating shared-memory halves so chunk c+1's
global loads were in flight during chunk c's Pass A dots and Pass B
latent mix, replacing the two `__syncthreads()` per chunk with one
(contents bit-identical; `rocm_attention` passed on the candidate). It
regressed consistently in a matched BAAB pair, twice, exclusive
(control = the 5.2a build, `gufo_bench_ba52b.log`): tg128 d0
34.19/33.62 → 33.64/34.10 (neutral), d2048 32.42/31.98 → 30.52/30.72
(−4.5 %), d8192 27.98/27.30 → **24.18/23.33 (−15.0 %)**, prefill
neutral. The doubled LDS (≈25 → 46 kB) leaves room for one 256-thread
block per CU instead of two: the kernel is LDS-issue/barrier-bound, not
exposed-load-bound, so removing inter-block overlap costs more than the
staging latency hides, and the d2048 loss carries the predicted tail
(72 blocks can no longer co-reside over 54 CUs). Reverted to the 5.2a
form; the remaining depth cost belongs to vectorizing Pass B's scalar
re-reads or widening the head group, not to staging pipelining.

Decode Pass-B direct-cache latent mix — REJECTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.2c). The follow-up to
the staging double-buffer rejection tested the other halving of shared
traffic: pass B's latent mix reads the KV columns straight from cache
(fully coalesced along the latent index, per-thread accumulation chains
and values identical to the staged copy) instead of re-staging every
chunk into shared memory, with one extra barrier after the score
exponentials to publish scores and metadata. `rocm_attention` passed and
global traffic halved, but the matched BAAB pair, twice, exclusive
(control = the 5.2a build, `gufo_bench_ba52c.log`) was net neutral:
tg128 d0 34.32/34.11 → 34.11/33.98, d2048 32.35/32.59 → 32.40/32.44,
d8192 27.99/28.19 → 28.17/28.43 (≤ +1 %), while `pp8 @ d2048` fell
107.95/107.96 → 105.91/107.47. With 5.2b this brackets the kernel:
at 2-blocks/CU occupancy the staging sweeps are neither latency- nor
bandwidth-exposed, so decode-frontier work moves on to checking the
`kDecodeGroup` 8 register tile or to the launch-side levers (5.4), and
5.2 stays at its 5.2a gain (−7 % frontier residual at d2048).

F32 GEMV tier — four rows per block — ACCEPTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.4a). The decode-window
dispatch table (`/tmp/opencode/prof52b_d8192`) attributes 1.67 ms/token
to `GemvF32`: the F32 router logit GEMV (grid=16384 single-wave blocks,
40/token, 34.6 µs) and `ssm_beta` (grid=1024, 35/token, 8.4 µs). Both
launched one output row per 32-thread block; the tier now mirrors the
quantized launches — `<<<(rows+3)/4, 128>>>` with row `blockIdx.x*4 +
warp` and a wave-local lane — so each row's strided dot and `__shfl_down`
reduction chain run unchanged and results are bit-identical (the
`rocm_gemv` failure set is exactly the parked `GemvRowsQ8` baseline with
`GemvRowsF32 bit-exact yes`; the full `ling3_flash` suite shows only the
two known parked fails, `gufo_bench_gemv54`/`gufo_bench_suite54`). The
matched BAAB pair, twice, exclusive (control = the 5.2a build,
`gufo_bench_ba54.log`): tg128 d0 33.71/33.83 → 33.80/34.10 (+0.5 %),
d2048 32.02/31.99 → 32.60/32.65 (+1.9 %), d8192 27.50/27.75 →
27.91/27.91 (+1.0 %), `pp8` within noise at all depths: the saving is
block scheduling on thousands of tiny GEMVs, which shows up per token
rather than in throughput-bound prefill. The grouped Q4_K/Q6_K row
pairing (≈ 60 % of decode GEMV time, needs paired shared ids) remains
the open 5.4 piece.

Grouped Q4_K/Q6_K tiers — two rows per wave — ACCEPTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.4b). The routed MoE and
per-head projections walked one `(slot, row)` pair per wave: decode's
grouped launches are thousands of short row dots (a handful of
256-element super-blocks each), where block scheduling and load latency,
not weight bandwidth, dominated. The two kernels are now templated on
rows-per-wave and the launcher folds two *consecutive output rows of the
same slot* per wave (`<2>` whenever `rows` is even, which it always is
for `expert_ff`, `hidden_size`, `kv_rank` and `v_dim`; odd counts keep
the old `<1>` walk). Each row keeps the exact super-block/j accumulation
chain — the x values are loaded once and shared, the two rows only
interleave their independent weight-stream decodes — so every output is
bit-identical to the single-pair launch: the gemv test's grouped worst
errors match the previous run byte-for-byte and the full `ling3_flash`
suite fails only the two known parked tests
(`gufo_bench_suite54b`). Matched BAAB pair, twice, exclusive (control =
the 5.2a build, cumulative with 5.4a, `gufo_bench_ba54b.log`): tg128 d0
33.39/33.64 → 34.88/35.26 (+4.7 %), d2048 32.36/31.74 → 33.51/33.15
(+3.9 %), d8192 27.98/27.31 → 28.92/28.53 (+3.9 %), `pp8` within noise.
Decode is now +4.7 % over the pre-5.4 build at cold context and +3.9 %
at both frontier depths from the two launch-side levers together; the
same fold for the MTP-draft grouped tiers (Q4_0/Q8_0, 5.4c below) and
lever 5.5's router/elementwise fusion are the follow-ups.

Grouped Q4_0/Q8_0 tiers — two rows per wave — ACCEPTED (October 16,
`docs/build_optimized_kernels/STATUS.md` lever 5.4c). The artifact's MTP
draft block keeps Q4_0 rows, so the remaining un-folded grouped tiers
run once per drafted token. `GemvGroupedQ4_0`/`GemvGroupedQ8_0` are now
templated exactly like the 5.4b pair — two consecutive output rows per
wave with independent per-row `(d*q)*x` chains, hoisted shared x values
and unchanged block order, `<2>` when `rows` is even — so outputs are
bit-identical: the fold is a bit-exact PASS in the standalone microbench
(`gufo_bench_mofold1.log`), and the exclusive suite fails only the two
parked gates with a byte-identical failure block, the grouped worst
errors unchanged at 8.52457e-06 / 3.35717e-06
(`gufo_bench_mo54suite.log`); `--compare` against a control build shows
zero scratch increase and the new instantiations scratch-free. Cold-L2
down-shape replay (2048×512, 8 experts): 185.6 → 201.6 GB/s (**+8.6
%**). In model the fold touches only the single MTP draft layer, so a
matched BAAB tg pair with `--speculative mtp` is neutral within noise
(tg1024 40.51/40.56 → 40.50/40.49, `gufo_bench_mo54tgm.log`) — landed
as a strictly-faster kernel and draft-heavy serving head-room, not as a
measured model gain.

Head group 4→8 for decode MLA (`kDecodeGroup`) — REJECTED (October 15,
`docs/build_optimized_kernels/STATUS.md` lever 5.2d). The last frontier
lever from 5.2 was checked directly: doubling the group to eight heads
halves the head grid and the cross-group KV re-staging, and the kernel
built clean (189 VGPR, no spills — the register tile was never the
blocker), but residency fell from 12 to 8 waves per CU.
`rocm_attention` passed — the group split changes no accumulation order
— and the matched BAAB pair, twice, exclusive (control = the accepted
5.4b build, `gufo_bench_ba52d.log`) split by depth: tg128 d8192
28.23/28.62 → 29.17/29.46 (**+2.3 %**), where 33 tile-blocks amortize
over the 54 CUs and halved staging traffic pays, but d2048
33.74/33.10 → 32.69/33.10 (−1.5 ±1.6 %) and, decisively, cold d0
35.31/35.09 → 34.79/34.80 (−1.1 % on *both* passes with σ ≤ 0.1): with
only four blocks in the head dimension a launch has neither
inter-block nor adequate wave overlap. Reverted to group 4. Combined
with 5.2b (occupancy via LDS) and 5.2c (traffic removal), the decode
kernel's frontier cost is now bracketed from three directions: after
5.2a's row broadcast the remaining depth cost is structural tile work,
not something staging width, staging locality or group width can
recover. 5.2 closes at its 5.2a gain.

Binary16 router retype + WMMA prefill route (`5.3a`) — ACCEPTED (October
15, `docs/build_optimized_kernels/STATUS.md` lever 5.3a). pp-attribution
on the 5.4b control showed the F32 router `GemvGroupedDense<float>` chain
consuming 486 ms of the 2514 ms timed pp2048 pass (19.3 %, 80 dispatches
over 512 × 2560 expert rows). The qwen36 4.1b port removes it: after the
upload staging flush, `Uploader::ConvertRouters` copies every resident F32
router into a fresh binary16 buffer and frees the original (−113 MB),
failing the load outright if any value leaves the binary16 range; prefill
projects the chunk's router logits through the existing `DenseF16Gemm`
tier (the kF16 dequant is a word `bit_cast` of the kBF16 fetch path), and
decode keeps its scalar chain via new `GemvF16`/`EmbedRowF16` kernels on
half rows, halving the per-token router footprints too. The new WMMA
instantiations land at 95–103 VGPR with 0 spill (`--compare` vs the 5.4b
control: no kernel worse); the exclusive `ling3_flash` suite shows only
the two known parked fails and `rocm_forward`/`rocm_inference`/
`rocm_quality` pass — no expert-selection flips. Matched BAAB pair,
twice, exclusive (control = the 5.4b build, `gufo_bench_ba53a.log`):
pp2048 874.10/872.64 → 1077.77/1078.65 (**+23.3 %/+23.6 %**), pp8192
631.22/630.66 → 730.76/727.66 (**+15.8 %/+15.4 %**), cold tg128 d0
35.35/35.57 → 35.91/36.04 (+1.6 %/+1.3 %). The −440 ms/pass at pp2048
matches the attribution, and the gain dwarfs the donor's +0.9 % because
Gufo still ran this projection as expert-grouped GEMV. The grouped
`k_b`/`v_b` MLA GEMV (345 ms) is the next prefill target.

16K depth-decay attribution (measurement-only, October 15). The
remaining −2.3 % pp16384 cell was attributed with rocpd windows of timed
pp2048 and pp16384 runs on the same 5.3a build
(`gufo_bench_prof53b.log`). The MLA prefill kernel is 15.6 % of device
time at 2048 but 61.5 % at 16384 — attention launches average 17.5 ms at
2048 while the deepest chunk launches run 370 ms at the end of a 16K
pass — and every other 16K kernel (routed WMMA GEMM 19.0 %, absorbed
`k_b`/`v_b` grouped GEMV 6.1 %, the KDA chain) stays linear.
`MlaPrefillFusedKernel` is pure fp32 VALU: scalar `fmaf` over the shared
512-wide latent plus 64 rope dims in 16-key tiles with 16-warp blocks,
no tensor cores. A causal 16384 pass through absorbed MLA costs
2 · (576+512) · 32 · N(N+1)/2 ≈ 9.35 TFLOP per MLA layer, ≈ 65 TFLOP
over the seven MLA layers, so the measured ≈ 20.3 s of attention per pass
is ≈ 3.2 TF/s — 13 % of the measured 23.95 TFLOPS fp32 VALU peak and 6 %
of the 55.25 TFLOPS WMMA fp16 ceiling; per pair-lane it spends roughly
48 useful FMAs against a comparable number of shuffle, exp2, causal-mask
and staging ops, and fp32 shared-memory key staging halves the traffic
again. llama.cpp crosses the same quadratic in its WMMA tiers, which is
why both reference backends decay only 23 % across the 8× depth and the
crossover now lands exactly at 16K. No code changed here: the fix is the
planned gemma4 3.4a WMMA prefill-attention port, recorded as the
remaining item of lever 5.3 in `docs/build_optimized_kernels/STATUS.md`.

Head-bucket WMMA route for the absorbed MLA `k_b`/`v_b` (`5.3b`) —
ACCEPTED (October 16, `docs/build_optimized_kernels/STATUS.md` lever
5.3b). The 345 ms grouped GEMV left after 5.3a is the two per-head
projections, whose GGML types (read with the `gguf` Python package:
`attn_k_b.weight` Q8_0 (128, 512, 32), `attn_v_b.weight` Q6_K
(512, 128, 32)) both already exist as routed-F16 WMMA
instantiations, and whose routing is trivial: every token visits
every head, so the histograms are uniform. The heads become virtual
experts — one device-built `ExpertCounts` + `RoutedCompact` map per
MLA layer serves both GEMMs: the slot-indexed scatter rows address
the [slot][nope] query plane narrowed by the new
`NarrowRowsStripped` and the [slot][kv_rank] absorbed query plane for
`k_b`, and the token-indexed gather rows feed the token-major
attention latent to `v_b`, whose output scatters back by slot. A
uniform histogram makes the tile list closed-form, `[e | (j << 16)]`,
so a 12-instruction `HeadTiles` kernel removes the only host fetch.
The rope tail of the query row is untouched (the attention kernel
consumes only the stripped nope half) and decode/verify keep their
bit-identical `GemvGrouped` chains: `rocm_forward`,
`rocm_inference` and `rocm_quality` pass with only the two parked
fails (`gufo_bench_53bq.log`), and `--compare` vs the rebuilt 5.3a
control shows no kernel worse and the two additions scratch-0.
Matched BAAB pair, twice, exclusive (`gufo_bench_53bpair2.log`):
pp2048 1083.27/1082.12 → 1240.14/1225.72 (**+14.5 %/+13.3 %**),
pp8192 731.88/729.66 → 804.45/797.21 (**+9.9 %/+9.3 %**), fully
separated. The realized −231 ms/pass at pp2048 is about two thirds of
the 345 ms attribution — per-head buckets run small m/k (512×128 and
128×512 on padded 64-row tiles), so the tier beats the scalar rate
without saturating it; at 16384 the realized −2.1 s/pass exactly
recovers the recorded 6.1 % linear GEMV share (33.0 s → 30.9 s).
Dedicated decode BAABs are neutral — cold d0 tg128 35.96/35.95 →
35.89/36.01 (`gufo_bench_53btg.log`), d8192 28.54/28.79 →
28.54/29.11 (`gufo_bench_53btgd2.log`); the −0.8 % in-pipeline tg128
after pp8192 is pipeline context, not the untouched decode path. The
standard grid refresh (`gufo_bench_53bgrid.log`: pp2048 1248.54,
pp4096 1065.16, pp8192 798.14, pp16384 529.77) moves the reference
crossing past 16K — Gufo now leads the 16K cell too (+4.4 % vs the
ROCm reference). The quadratic `MlaPrefillFusedKernel` remains the
depth bottleneck (entry above).

WMMA hi/lo MLA prefill attention (`5.3c`) — ACCEPTED (October 16,
`docs/build_optimized_kernels/STATUS.md` lever 5.3c). Reopens the 5.1
WMMA attention ruling by removing its root cause instead of repeating
its design: the 5.1 attempt fetched each `(row, dim-group)` thread's V
values as 256 scattered global loads per lane per 16-key tile and died
LSU-issue-bound (7.6× slower than fused at unit-scale activations,
`gufo_bench_rp51.log`). `WmmaMlaPrefill` keeps every dot — QK *and*
P·V — on `v_wmma_f32_16x16_16_f16_w32` and stages the K and transposed
V blocks into the *same* LDS tile, filled by the coalesced float2
loads the K region already performs: zero extra global traffic, one
4-barrier staging block, no per-lane V fetch. Q and K enter as hi/lo
`__half` pairs (5.1a contract) over the 512 latent + 64 rope dims, one
wave-pair per 32 rows, online softmax, P re-staged as its own hi/lo
pair (hi plane + largest-remainder residual). Dispatch: `MlaPrefill`
routes `kv_rank == 512, rotary == 64`, even heads ≥ 2 to
`WmmaMlaPrefill<2U>`; everything else keeps fused.

The quality contract nearly killed it before the timing did: ctest
failed at 5.6e-4/6.2e-4 against the 5e-4 element-wise criterion
(0/37, 5/32) and a bisect with a ctest-metric mode in the bench
(`wmma` gate prints both `WorstRelative` worsts with the 1e-3 floor
from `hip_test.hpp`) showed the failing elements sit at `|ref| ≈ 1e-3`
(absolute budget ≈ 4e-7), candidate 7.2e-7 vs fused 6e-8 — a real
12× accuracy gap, not floor inflation. Null probes, each *bit-identical*
in the metric: the 4th QK term `ql·kl`, the 4th PV term `pl·vl`, fp64
`o_acc` rescale, fp64 `running_sum`, bypassing the wmma C-operand
chain into fp32 adds, and full P+V triple-plane (hi+lo+residual)
representation — the representation and the accumulation width were
not the source. The decisive probe (per-`(row,key)` scalar fp32 score
dot replacing the wmma sum) halved the metric to 3.798e-4: the score
accumulator chain was the fingerprint, specifically the tiny hi·lo
correction products being rounded away as they entered the large
|Σ| ≈ 22 hi·hi running sum across 30 serial `Wmma` steps. The fix is
to stop mixing operand classes: `acc_hh`, `acc_hl`, `acc_lh`
accumulate separately and meet as `(hh + hl) + lh` on the way to
`s_lds` — one rounding step each, corrections never quantized against
a partial sum 4 orders larger. Bench worst fell to 2.45e-4/2.91e-4/
2.04e-4/2.84e-4 on 7/32/64/512-token shapes (fused 5.7e-5–9.3e-5),
and ctest passes all five prefill shapes at the unmodified 5e-4
contract; VGPR 237, spill 0, LDS 49408, `expf` softmax (the `__expf`
swap probed neutral to a bit). Standalone `cand` 1.44× vs shipped
fused at tok2048/ctx16384 unit scale — the regime the 5.1 replay
showed is the production one. Re-adding the register-hungry correctness
scaffolds later (PV-side 3-chain `o_acc` split alone costs +72 VGPR on
9 tiles and spills) is unnecessary given that margin at this error
level — the shipped kernel is the measured minimum.

Matched BAAB pair, twice, exclusive, control = the 5.3b build
bd269b0b (`gufo_bench_53cpair.log`, `gufo_bench_53c16k.log`): pp2048
1253.04/1258.16 → 1330.17/1333.09 (**+6.2 %/+5.9 %**), pp8192
803.16/805.29 → **952.95/954.40** (**+18.7 %/+18.5 %**), pp16384
533.04/533.61 → **668.54/668.67** (**+25.4 %/+25.3 %**), fully
separated, and tg128 neutral (36.12–36.24 both sides) — the gain
grows with the quadratic the fused kernel was paying, exactly the
depth-decay signature this lever existed to fix.
`--compare` vs the 5.3b control: 0 worse, 1 one-side-only — the new
kernel stages its WMMA fragments in 288 B of private scratch with zero
spills (allowance added to `tools/ci/kernel-resources.json`), the same
by-design pattern as the `AttentionPrefillWmma` donor.

Router GEMV/fused top-k selection in one launch (STATUS.md lever 5.5a):
`RouterTopKFusedF16Kernel` extends the f16 router logits GEMV grid
(128×128, 512 experts × 2560 cols) so the last wave performs the
existing greedy `RouterSelectRow` selection from global logits after a
ticket barrier, saving the separate 512-thread `RouterTopKKernel`
launch per MoE layer per token. Selection correctness needed two
findings: plain `__threadfence()` + ticket + plain global loads is
sufficient on gfx1151 (L1 is invalidation-coherent; atomics and
`__ldcg`/`__stcg` unneeded — the latter have no float overloads in
ROCm 10), and the winner-retirement test must be `threadIdx.x ==
best_e % blockDim.x`, not `== best_e` — the standalone `<<<1,512>>>`
launch only hides the identity because blockDim equals `n_exp` (at 128
threads winners were never retired: all 8 ids repeated the first
winner with flat 2.5/8 weights). Bit-exact ids/weights over 900
randomized rounds across three shapes
(`tools/bench/ling3_router_fused_bench.hip`, retained with the
rejection), full-suite parity with HEAD `6af0f594` on the three parked
gates (`rocm_gemv`, `rocm_inference` step-1 greedy parity `cpu 220 gpu
2538` at top-2 gap 0.0990877 — reproduced 6/6 on a HEAD control binary
and in the HEAD full-gate baseline, so pre-existing, not lever-caused —
and `rocm_mtp_accept`), `--compare` 0 worse / 0 better / 126 same, and
the standalone launch-pair wall −0.15…−0.66 µs (~2 %) per router pair.

Matched BAAB pair, twice, exclusive, control = the HEAD release binary
(`gufo_bench_ba55a.log`): tg128 d0 36.09 → 35.84 (**−0.7 %**), d2048
34.28 → 34.00 (**−0.8 %**), d8192 29.39 → 29.48 (+0.3 %, noise), pp8
neutral — the candidate lost every d0/d2048 pair. The router pair is
≈36.8 µs of the 27.8 ms/token decode step (0.14 %), and the launches
were already pipelined behind GPU work, so the savings ceiling was
below noise while fusion serializes the selection into a 128-thread
tail block after the GEMV. Rejected and reverted; any revisit must run
the selection alongside, not after, the GEMV tail — and should target
the elementwise fusions (Add→RmsNorm, Swiglu-in-down-GEMV) first,
which touch comparable launch counts without extending the critical
path.

Residual Add fused with the consuming RmsNorm (STATUS.md lever 5.5b):
the decode attribution profiled 92 `RmsNormKernel` launches (8.28 µs
avg) plus 160 `AddKernel` launches (1.30 µs avg) per token, and every
decode/draft norm consumes an immediately preceding full-row residual
`Add` (`Step()` layer tail → next-layer norm, MTP draft head), so one
fused launch removes ~94 dispatches/token. `AddRmsNormKernel` keeps the
exact grid-stride double sum, the exact `BlockSumDouble` and the exact
scale chain, writing `res_out = res + x` and then `y = res_out * scale
* w` — bit-exact by construction against the separate pair across 300
randomized rounds × n ∈ {2560, 2048, 2551} incl. a ragged tail
(`tools/bench/ling3_add_rmsnorm_bench.hip`, retained), with the fused
launch wall −1.6…−2.4 µs per pair under contention. Gate parity held:
the full suite failed exactly the three parked gates with byte-identical
signatures (`divergence at step 1: cpu 220 gpu 2538` at gap 0.0990877;
25/32 draft agreement, the number in every HEAD-era log), and
kernel-resources `--compare` showed 0 worse with the fused kernel
scratch-free (127 kernels vs 126, no new allowance). Matched BAAB ×2
plus an `--speculative mtp` tg1024 pair, exclusive, control = the HEAD
release binary (`gufo_bench_ba55b.log`): tg128 d0 36.05 → 36.06, d2048
34.26 → 34.22, d8192 29.41 → 29.43, spec d0 40.19 → 40.27 and d2048
37.03 → 36.96 — every cell ≤ 0.2 % — while pp8 ran +0.0/−0.3/−0.9 %
across depths. An exclusive decode profile of the fused binary
(`/tmp/opencode/prof55b_fused`) explains it: `AddRmsNormKernel`
averages 10.78 µs, more than the separate pair's 9.58 µs, because the
1×256 block serializes the second read+write phase behind its own
barrier while the tiny `Add` ran its 1.3 µs hidden inside the
pipelined stream. Rejected and reverted: at this node the launch
budget is already behind the GPU and per-kernel cost, not launch count,
is what a fused latency-bound single-block kernel can only pay, not
save; future elementwise work must reduce the latency-bound kernels'
own duration (e.g. wider blocks or removing reads), not their dispatch
count.

Bit-exact RmsNorm latency trims (STATUS.md lever 5.5c, microbench
only): staging `x` into LDS in the accumulate pass (and additionally
prefetching `w` into registers) preserves the exact double
accumulation order and is bit-exact over 300 randomized rounds ×
{2560, 2048, 2551} × 96 slots (`tools/bench/ling3_rms_norm_bench.hip`),
but exclusive per-launch time only falls 7.37 → 6.98 µs (LDS+reg-w
7.03): × 92/token = +0.13 %, under the BAAB noise floor. The 7.37 µs
for a 20 KiB row is the fixed single-CU cost of the `<<<1,256>>>`
contract (a 10-block `Add` moves 30 KiB in 1.3 µs), and exact ordering
pins that launch shape. Rejected without an in-model run; the decode
elementwise budget is only movable by changing the reduction contract,
which the model does not — next lever is the parked rocm_mtp_accept
gate under STATUS 5.6.

Repairing the parked MTP-accept gate (STATUS.md lever 5.6a): the gate
bisect started from two suspects — batched `Verify` rows differing from
sequential `Step`, and draft agreement stuck at 25/32. The first suspect
was a phantom: every `rows 0/4` / `126692-float row delta` log came from
stale or experiment-modified pre-session binaries; a clean control build
of HEAD verifies bit-exactly (rows 4/4, 22/22 steps across the bisect,
3/3 deterministic runs). The real kernel bug was in `GemvRows`: the
Q8_0 Vec4 multi-row path accumulated its dot product in a different
order than the per-row `GemvQ8_0Vec4`, so draft rows diverged as soon as
a column block spanned more than one accumulation round. The
`GemvRowsQ8_0Vec4` kernel now mirrors the per-row accumulation order and
`rocm_gemv` rows 2–5 are bit-exact. The residual phase-2 parity failure
was structural, not numerical: the oracle built its reference sequence
with the batched WMMA `Prefill`, while every production spec session
feeds the MTP draft cache sequentially per token (`Step` + `MtpStep`);
oracle and spec bits therefore differed from token 3 regardless of
kernels. The oracle now replays the production sequential feed, which
also makes draft-chain non-perturbation by the interleaved `MtpStep`
part of the enforced contract. Evidence: `rocm_mtp_accept` passes
standalone (29.10 s) and the exclusive ling3 suite is 11/12 (only the
parked `rocm_inference` cpu-vs-gpu step-1 divergence fails);
kernel-resources `--compare` against a freshly built HEAD release
reports 0 worse with 4 one-side kernels (the new rows variants).
Exclusive BAAB ×2 plus matched `--speculative mtp` pairs
(`gufo_bench_q8baab`): plain pp/tg cells all ±0.4 % with mixed signs at
d0/2048/8192, while spec tg1024 ran 39.67 → 40.34 (+1.7 %) and spec
tg1024 @ d2048 37.04 → 37.50 (+1.2 %), each consistent across both
pairs — the multi-row verify GEMV is in the draft loop's hot path, so
the exactness fix is also a small speculative-decode win, and it is
kept.

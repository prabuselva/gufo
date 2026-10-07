# Gemma-4 26B-A4B: Gufo vs franzmoca (`feat/gemma`) kernel-by-kernel

This document compares the ROCm/HIP inference path of this repository against
the franzmoca fork (`feat/gemma`) for Gemma-4 26B-A4B on Strix Halo
(`gfx1151`), and explains where the remaining prefill gap comes from.

It is a *diff* document, not a benchmark: the retained speed numbers live in
[BENCHMARKS.md](BENCHMARKS.md) (this repo) and in
`gufo_franzmoca/docs/models/gemma-4-26b-a4b/BENCHMARKS.md`. The optimization
evidence for each franzmoca lever is in
`gufo_franzmoca/docs/models/gemma-4-26b-a4b/EXPERIMENTS.md` and
`gufo_franzmoca/docs/models/gemma-4-31b/EXPERIMENTS.md`.

> **Status (2026-10-07).** This document was written when this repo had *no*
> FP16 dense GEMM. That is no longer true: commit `cdbf18a3` added a
> model-private **dense-F16 WMMA route** (`DenseF16Gemm`) for the wide-batch
> Q8_0 projections, lifting prefill ~11 % (pp2048 1749 → 1944 t/s). The dense
> GEMM is therefore **no longer the dominant gap**. Every code claim below was
> re-checked against both trees on 2026-10-07; the franzmoca numbers are quoted
> from its own `BENCHMARKS.md` (its authoritative source) and the Gufo numbers
> from [BENCHMARKS.md](BENCHMARKS.md), re-confirmed on the current build with
> `tools/bench/gpu_exclusive.sh` (pp2048 1923 ± 3, tg128 41.96 ± 0.02).

## 1. Headline

| Path | This repo (Gufo) | franzmoca | Gap |
|---|---|---|---|
| Prefill, matched d0 (pp2048, no prefix) | 1944 t/s | 3427.92 t/s | **~1.76×** |
| Prefill, this repo one-shot depth | 1522 t/s (pp16384) | — (see caveat) | — |
| Prefill, franzmoca chunked depth | — | 2439.11 t/s (2048-chunk on 16 K prefix) | — |
| Decode AR (tg128, no MTP) | 41.96 t/s | 45.72 t/s | ~1.09× |
| Decode MTP | 59.99 t/s (n=4) | 177.12 t/s (repetitive) | drafting-dependent |

> **Methodology caveat.** The two repos measure prefill differently, so only
> the **d0 row is a clean apples-to-apples comparison**. franzmoca's grid fixes
> a pp2048 chunk and varies a *cached prefix* depth (HTTP, greedy, thinking
> off); its "d16384" number is a 2048-token chunk attending over a 16 K cached
> prefix. This repo's grid varies `--n-prompt` with no prefix (`gufo bench`,
> exclusive GPU); its "pp16384" is a single 16 K prefill. A 16 K one-shot and a
> 2 K-on-16 K-prefix chunk have different attention costs and are **not**
> directly comparable — hence the split rows above. At matched d0 the gap is
> ~1.76×, down from ~1.96× before the dense-F16 route.

Decode is close (1.09×). The dominant remaining gap is **prefill**, and it is
now spread across the MoE scheduling, the activation pipeline and attention —
not the dense GEMM, which this repo now accelerates.

## 2. Activation dtype strategy (the change that enabled the rest)

- **This repo** carries **FP32 activations** between every op. The dense Q8_0
  projections now take a **binary16 WMMA route at wide batch** (`batch ≥ 96`,
  `rows ≥ 2048`, `cols ≤ 4096`): the activations are narrowed to FP16 on the
  fly and the Q8_0 weights decode to FP16 in LDS
  (`DenseF16Gemm`, `kernels/rocm/dense_f16_gemm.{hpp,hip.cpp}`, routed from the
  `kQ8_0` case in `gemm.hip.cpp`). Outside that window (decode, MTP, the
  `cols = 8192` full-attention output projection) it falls through to the int8
  MMQ dense path. The MoE path narrows to FP16 for the routed GEMMs
  (`RoutedF16Gemm`). **But producers still write FP32 rows** — there is no
  end-to-end `half_prefill_` pipeline; the narrowing happens per-GEMM, not in
  the norm/attention/GeGLU producers.
- **franzmoca** runs a **binary16 activation pipeline end-to-end**
  (`half_prefill_`). Producers — fused norms, attention, GeGLU — write a
  binary16 copy directly and **skip the FP32 row entirely** when no consumer
  needs it (`PostAttentionNorm` with `h_half`/`h2_half` and null `h`,
  `kernels.hpp:129-137`). Dense Q8_0 projections run the same binary16 WMMA
  GEMM (`DenseF16Gemm`).

On `gfx1151` the FP16 matrix cores (WMMA `m16n16k16`) sustain materially higher
throughput than the int8 MMQ path. This repo has captured the *GEMM* half of
that win; the remaining half — producers emitting binary16 and skipping the
FP32 row write — is still open and is what removes the ~120 MB/layer of FP32
traffic franzmoca avoids.

## 3. Kernel-by-kernel

### 3.1 Dense projection GEMM (attention Q/K/V/O, dense FFN, lm_head)

| | This repo | franzmoca |
|---|---|---|
| Q8_0 prefill | **model-private `DenseF16Gemm`** — binary16 WMMA, Q8_0→FP16 in LDS, FP16 act, at `batch≥96 && rows≥2048 && cols≤4096`; int8 `qfn_mmq_q8_0_dense` fallback otherwise (`gemm.hip.cpp` `kQ8_0`) | `qwen38_flash_next::rocm::DenseF16Gemm` — binary16 WMMA, Q8_0→FP16 in-kernel, FP16 act, **cross-model link** (`executor.cpp:442-444`) |
| Shape tuning | none (single `kRowGroup==1` path) | `HalfPlan(m,k)` picks `kStagedRowGroups8` / `kRowGroups8` / `kAuto` per shape (`executor.cpp:417-426`); dispatch gated `plan!=kAuto && batch≥1024` (`kernels.hip.cpp:5483`) |
| K-quant prefill | not used for dense | `LaunchBatchedQuantGEMMPreQuantized` — W8A8 int8 WMMA, pre-quantized Q8_1 rows |
| F16 weights | — | `UnquantizedF16Gemm` — FP16×FP16 WMMA |
| F32 / BF16 | hipBLAS SGEMM / GemmEx | hipBLASLt BF16 plans |

**Effect.** This repo now runs the same *class* of kernel as franzmoca for the
dense projections it routes, which is the ~11 % prefill lift (1749 → 1944 at
pp2048). Two differences remain:

1. **No `DenseF16Plan`/`HalfPlan`.** franzmoca's `HalfPlan` sends the
   full-attention output projection (`m==2816, k≥4096`) through a *staged*
   tile plan (`kStagedRowGroups8`, 2013→1759 µs) and the dense MLP down
   (`k==2112`) through `kRowGroups8`. This repo's copy has only the single
   row-group path, so the `cols = 8192` attention-output projection is **gated
   out** of the F16 route (measured 0.89× on the untuned path) and stays int8.
   Adding the staged plan is the top actionable dense-GEMM lever.
2. **Cross-model vs private.** franzmoca reaches the kernel by linking
   `qwen38_flash_next::rocm::DenseF16Gemm` from the gemma4 executor; this repo
   keeps a model-private copy (no cross-model coupling), at the cost of not
   inheriting franzmoca's later `DenseF16Plan` tuning.

### 3.2 Routed MoE prefill (128 experts, top-8) — **largest remaining lever**

| | This repo | franzmoca |
|---|---|---|
| Tile map build | **device→host→device**: `hipMemcpy` counts to host, build `tiles_host_` on CPU, upload back (`executor.cpp:651-668`) | `BuildRoutedTiles` — device-built tile maps, **no host sync** (`moe.hip.cpp:754/797`, called `executor.cpp:1019`) |
| Routing | `RouterTopK` + `ExpertCounts` + `RoutedCompact` (multiple launches) | `MoeRoute` — one-launch routing (`moe.hip.cpp:810`, `executor.cpp:953/970`); experts on a 2nd stream |
| gate/up GEMM | `RoutedF16Gemm` (FP16 act) (`routed_f16.hpp:58`) | `LaunchRoutedHalfGemm` with **fused GeGLU epilogue** |
| GeGLU | **separate kernel** `GegluF16`/`GegluF32` between gate_up and down (`fused.hpp:70-78`) | fused into gate/up epilogue — `gu` intermediate never stored |
| Mixture | `MoeEpilogue` separate | `MoeFinish` — fused mixture + norms |

**Effect.** The host round-trip in `MoeBatch` (`executor.cpp:651-668`) is a
**synchronization stall on every MoE layer** (30 layers/prefill): a blocking
D2H of the expert counts, a CPU tile-map build, and an H2D upload. It serializes
the GPU against the CPU once per layer and defeats stream overlap. franzmoca
removes it entirely with device-side tile maps. This is now the **single
largest remaining prefill lever** for this repo — the dense GEMM it used to sit
behind is closed.

### 3.3 Attention prefill (WMMA)

| | This repo (`attention.hip.cpp`) | franzmoca (`attention_wmma.hip.cpp`) |
|---|---|---|
| Occupancy | `__launch_bounds__(kWaves*32, 1)` → **1 block/WGP** (`attention.hip.cpp:418`) | `amdgpu_waves_per_eu(8)` for global / hd256 → **2 blocks/WGP** (`attention_wmma.hip.cpp:85`) |
| Next-tile load | none — K/V loaded from global at top of each tile, blocking | **register prefetch** `k_next`/`v_next` overlap next tile's global load with current compute (`:219-220`) |
| Barriers/tile | ~4–5 `__syncthreads` | **3 barriers/tile** (`:377-379`) |
| Global-layer keys | full K cache | **derived keys** (`kGlobal`): V + rotated K pairs, halves KV traffic; cross-half `permlanex16` Q rebuild (`:75-78`, `:175`) |
| Output | FP32 only | **binary16 `out_half`** copy for the next WMMA GEMM |

**Effect.** franzmoca's EXPERIMENTS report the global-attention kernel at
89→81.7 ms and sliding-window at 2.59→1.88 ms from these changes. The 2-blocks/WGP
occupancy plus next-tile prefetch hides the KV memory latency that this repo's
1-block kernel exposes. The FP32-only output also forces this repo to re-narrow
attention output before the O-projection, whereas franzmoca's `out_half` feeds
the binary16 GEMM directly.

### 3.4 Fused norms / epilogues

| | This repo | franzmoca |
|---|---|---|
| Post-attention norm | separate residual-add + RMSNorm | `PostAttentionNorm`: fused residual-add + dual RMSNorm + optional Q8_1 quantize + optional binary16 side-copies; **null `h` skips the FP32 row** (`kernels.hpp:129-137`) |
| Post-FFN norm | separate | `PostFeedForwardNorm` fused, optional Q8_1 (`kernels.hpp:142`) |
| GeGLU | `GegluF16` / `GegluF32` separate (`fused.hpp:70-78`) | `GeGluQuantize` / `GeGluPackedHalf` fused with quantize / binary16 (`kernels.hpp:153/165`) |

**Effect.** Each fused producer that writes binary16 and skips the FP32 row
removes a full-width read+write from the prefill memory budget. Across 30 layers
this is the ~120 MB/layer saving cited in franzmoca's EXPERIMENTS. This is the
producer-side half of §2 and is still open in this repo.

### 3.5 Decode GEMV (the smaller gap)

| | This repo | franzmoca |
|---|---|---|
| K-quant GEMV | scalar `GemvGrouped` / `GemvGroupedPair` (`gemv.hpp:48/56`) | `LaunchKQuantGemv` — **split-K**, 8-wave reduction (`gemv.hpp:20`) |
| binary16 GEMV | — | `LaunchHalfGemv` — width-invariant 1–16 rows (`gemv.hpp:31`) |
| Q4 repack | — | `RepackQ8_0AsQ4K` (`gemv.hpp:44`) |

**Effect.** Decode is memory-bandwidth-bound, so the gap is small (41.96 vs
45.72 t/s AR). franzmoca's split-K GEMV and width-invariant binary16 GEMV give
a modest but real edge, and the MTP drafting/calibration path (prompt lookup +
calibrated drafts) is what lifts repetitive decode to 177 t/s.

## 4. Remaining prefill gap attribution (measured, post dense-F16)

The residual ~1.67× at matched d0 is **distributed across the compute kernels,
not concentrated in one lever.** A `rocprofv3` profile of this repo's gemma4
prefill (pp2048, exclusive, `tools/prof/prof.py run`) gives the kernel-time
split below; franzmoca's advantage is that it attacks several of these at once
with fusion, not that any single one is a silver bullet.

| Kernel (gufo gemma4 pp2048) | % of kernel time | franzmoca's difference |
|---|---|---|
| `RoutedF16GEMM` (MoE experts) | **34.2%** | already F16 WMMA here; franzmoca fuses GeGLU into the gate/up epilogue (removes the separate `Geglu*` pass + `gu_half` round-trip) |
| `DenseF16GEMM` (Q8_0 projections) | 18.8% | already F16 WMMA here; franzmoca adds `HalfPlan` staged tiles |
| `Cijk…BSS…` `hipblasSgemm` (F32 projections) | 6.6% | **franzmoca runs these as W8A8 int8 on tensor cores**, not FP32 SGEMM |
| `AttentionPrefillWmma` | ~10% | 2 blocks/WGP + prefetch + `out_half` |
| `MoeEpilogue` | 5.7% | fused into the down-GEMM epilogue |
| `QknormRope`+`KvNormRope`+`RmsNorm` | ~10% | fused into adjacent kernels / GEMM epilogues |
| `NarrowHalf`/`NarrowKernel`/`NarrowBf16` | ~2% | avoided by an end-to-end binary16 pipeline |
| `GegluKernel`+`GegluSeparate` | ~1.2% | fused into the routed gate/up epilogue |

**Correction to the earlier draft:** the "MoE host round-trip" was listed as the
*largest* lever. It has now been ported (device-side `BuildRoutedTiles`,
`db867be0`) and is **throughput-neutral** (pp2048 1933 vs 1923–1944 baseline,
pp16384 1517 vs 1522, tg128 41.9 vs 42.0). It is a cleaner architecture (removes
a synchronous D2H per MoE layer) and a prerequisite for one-launch routing, but
it is not where the prefill gap lives. The gap is the sum of the rows above, so
closing it requires the fusion ports (fused-GeGLU expert GEMM, fused norms,
W8A8 F32 projections), each of which is a substantial kernel port.

## 5. Why the earlier "3000 t/s unreachable" conclusion was wrong

`EXPERIMENTS.md` here once reasoned from the roofline of the *then-current*
kernels: int8 MMQ dense GEMM, FP32 activation traffic, and a per-MoE-layer host
sync. Under that model the memory traffic and MMQ throughput genuinely capped
near 1750 t/s. franzmoca changes the model itself — FP16 matrix cores, binary16
activations that skip FP32 rows, device-side MoE scheduling — and the same
hardware delivers 3400+ t/s. The ceiling was a property of the kernels, not of
`gfx1151`. This repo's dense-F16 route confirms the diagnosis: moving just the
dense GEMM to binary16 WMMA already lifted prefill ~11 %, and the remaining gap
tracks the levers in §4.

## 6. Porting roadmap (re-prioritized from the §4 profile)

1. ~~**Binary16 dense GEMM.**~~ **DONE** (`cdbf18a3`): model-private
   `DenseF16Gemm` routes wide-batch Q8_0 dense projections through binary16 WMMA.
   *Follow-up:* add the `DenseF16Plan`/`HalfPlan` staged path so the
   `cols = 8192` attention-output projection can join the F16 route.
2. ~~**Remove the MoE host round-trip.**~~ **DONE** (`db867be0`): device-side
   `BuildRoutedTiles` + dead-tile capacity replaces the `hipMemcpy`
   counts→CPU→tile-map→GPU in `MoeBatch`. Quality-gated (`--validate-prefill`
   PASS). **Measured throughput-neutral** — kept for the cleaner architecture
   (no per-layer D2H sync) and as the prerequisite for one-launch routing, not
   for speed. The earlier "largest lever" claim was wrong; see §4.
3. **Fuse GeGLU into the routed gate/up epilogue** (targets the 34% MoE GEMM +
   the ~1.2% separate GeGLU + the `gu_half` write/read round-trip). Port
   franzmoca's `LaunchRoutedHalfGemm` fused epilogue. Highest remaining ROI but a
   large kernel port; validate quality independently.
4. **W8A8 int8 for the F32-weight projections** (targets the 6.6% `hipblasSgemm`
   FP32 path). franzmoca runs these on int8 tensor cores via
   `LaunchBatchedQuantGEMMPreQuantized`. Changes rounding — needs a quality gate.
5. **Fuse norms / epilogues** (`QknormRope`+`KvNormRope`+`RmsNorm` ~10%,
   `MoeEpilogue` 5.7%) into adjacent kernels / GEMM epilogues, and move to an
   end-to-end binary16 activation pipeline so producers skip the FP32 row
   (removes the ~2% `Narrow*` traffic).
6. **Attention occupancy + prefetch.** Raise to 2 blocks/WGP
   (`amdgpu_waves_per_eu`), add next-tile register prefetch, skip-unchanged
   rescale, and a binary16 `out_half` output; add derived keys for global layers.
7. **Decode:** split-K K-quant GEMV + width-invariant binary16 GEMV; revisit MTP
   drafting/calibration for the repetitive-decode number.

Steps 3–5 are the prefill story now (the profile shows the cost is spread across
the MoE GEMM, the F32 SGEMM, and the unfused norms/epilogues); 6 refines it; 7
closes the decode gap. Each franzmoca lever is documented with measured
before/after numbers in its `EXPERIMENTS.md` and can be validated independently
with `tools/bench/gpu_exclusive.sh` at ≤16K context first (see AGENTS.md).

## 7. franzmoca reference numbers on this hardware (validated)

Re-measured on this Strix Halo (gfx1151), gemma4 Q8_K_XL, exclusive, r2, via
`gufo bench` (CLI, `--speculative off`), matching the published franzmoca
numbers to ~6 % (the residual is HTTP-vs-CLI overhead and rep count):

| workload | franzmoca (this HW) | gufo (this HW) | ratio |
|---|---|---|---|
| pp2048 @ d0 | 3231.0 ± 17.4 | ~1933 | 1.67× |
| tg128 @ d0 | 47.87 ± 0.08 | ~41.9 | 1.14× |
| pp2048 @ d16384 | 2292.2 ± 3.6 | — | — |
| tg128 @ d16384 | 43.86 ± 0.01 | — | — |

The prefill gap (~1.67×) is what §4 attributes to the distributed kernel costs;
the decode gap (~1.14×) is the GEMV/MTP work in step 7.
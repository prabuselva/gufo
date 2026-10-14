# Build optimized kernels — status tracker

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked

Record-keeping (golden rule): every lever — accepted AND rejected — gets its
record appended here (implementation, quality evidence, gates, matched
exclusive-window timings, lesson) as part of the same commit, and the
pp2048 timed-pass gap budget is re-measured against the fork after every
accepted lever. The backlog below is never trusted from memory.

## Phase 1 — Quality harness (gates all later phases)

- [x] 1.1 Port oracle KL harness: `tests/models/gemma4/reference_probe.cpp`,
      `logit_file.hpp`, adapted to our `ReferenceModel` API; CMake target
      `gemma4_reference_probe` (EXCLUDE_FROM_ALL, `-O2`) builds and runs.
- [x] 1.2 Port `tools/gemma4/compare_logits.py` (as-is, same G4LG v1 format).
      `compare_stages.py` deferred: needs `LayerTrace` export we don't have.
- [x] 1.3 Oracle `Storage` modes `kFloat32 / kHalfKv / kQ8Activations` in
      `src/models/gemma4/reference.{hpp,cpp}` (fork `reference.hpp:23-28`);
      `RoundQ8` formula verified identical to fork `reference.cpp:78-91`.
- [x] 1.4 `--validate-prefill` real KL bulk-vs-exact check in gemma4
      `cli_runner.cpp` (`KlDivergence` = softmax/log-sum-exp, same family as
      `compare_logits.py`). Tiered gate: PASS ≤3.7e-5 nats with top-1,
      RECORDED ≤5e-2 (fork-recorded `bulk 0.027` baseline class), FAIL above
      or non-finite. Measured Q8_K_XL (`/tmp/opencode/gufo_bench_g4val2.log`):
      64 → 9.6e-4, 512 → 2.14e-2 (top-1 flips, fails at first version —
      recorded baseline), 2048 → 7.1e-8 PASS. Drift is a short-prompt bulk
      path property; 2.5/3.2 must bring 512 to PASS.
- [x] 1.5 `tests/functional/mtp_vs_ar.py`: per-depth AR-vs-MTP
      `output_sha256` from verbose bench tg traces; deterministic per run,
      recorded (never assumed) invariant. Q8_K_XL + MTP sidecar: hash_match
      False at depths 0/8192/16384 (`/tmp/opencode/mtp-vs-ar/`) and
      16384/32768 degenerate-trajectory artifact
      (`/tmp/opencode/mtp-vs-ar-degen/`, n_gen 64, drafted/accepted up to
      45/11 — near-zero draft acceptance alongside the hash divergence).
      3.6 must make `--gate` pass.
- [~] 1.6 Exit gate: runtime-vs-oracle KL harness reproduced on shared
      artifacts. `tests/models/gemma4/runtime_probe.cpp` (`gemma4_runtime_probe`,
      EXCLUDE_FROM_ALL) mirrors `gemma4_reference_probe` framing (row p =
      logits after feeding token p; `--bulk-final` dumps the bulk-prefill last
      row); compared with `tools/gemma4/compare_logits.py`. Q8_K_XL, 25-token
      chat prompt (`/tmp/opencode/kl16/`): bulk last row vs oracle `--half-kv`
      KL 6.5e-7, top-1 1/1 — fork decode gate (≤1e-4) class; the sequential
      teacher-forced chain drifts (KL mean 0.044, max 0.44 @ pos 16, top-1
      23/25; `--half-kv` oracle identical, so KV rounding is not the driver) —
      decode-chain numeric drift is real and the likely cause of the near-zero
      MTP acceptance in 1.5; 2.2/3.6 must fix it. Full-line coverage
      (prefill rows 3.7e-5, 1542-row bulk 0.027 / top-1 1489) needs the
      per-position bulk logits hook from 2.5/3.3.

## Phase 2 — Shared kernel core (`src/core/hip/{gemm,moe,attn}`)

- [x] 2.1 `gemm/dense_f16_gemm` + `DenseF16Plan` (staged row-groups-8 for
      K≥4096, row-groups-8 for K=2112; fork `kernels.hip.cpp:6227-6300`).
      Landed as `src/core/hip/gemm/dense_f16_gemm.{hpp,hip.cpp}` (target
      `gufo_core_hip_gemm`); planned launches gate on batch≥1024.
      A/B at batch=2048 (`/tmp/opencode/gufo_bench_f16ab.log`, median of 15):
      int8→f16 auto 1.32–1.60×; staged8 attn_out_swa 1.01×, attn_out_full
      1.36× over auto (and 1.41× over int8), rgroups8 ffn_down 1.05×;
      plan_err identical to auto (K-order invariant).
      e2e Q8_K_XL pp2048/tg128 neutral within noise: candidate
      1975.14/42.02 and repeat 1997.85/41.59 vs report baseline 1992/42.29
      (`/tmp/opencode/gufo_bench_g4pp2k{,b}.log`).
- [ ] 2.2 `gemm/wmma_gemv` width-invariant decode/verify GEMV + split-K
      `gemv_tasks` + `half_store`.
- [ ] 2.3 K-quant decode variants (Q4_0/Q5_0/Q4_K/Q5_K/Q6_K/Q8_0). Code
      audit (lever-5 era): present — `GemvType` {F32,Q8_0,BF16,F16,Q4_K,Q5_K}
      (`gemv.hpp:21`) + grouped decode GEMVs for Q8_0/Q4_K/Q5_K incl. Pair
      variants (`gemv.hip.cpp:134-266`); missing — Q4_0/Q5_0/Q6_K dense
      decode variants (such models exit `GemvTypeOf` at `executor.cpp:18-40`);
      fork ships Q4_0/Q5_0 experts+dense (`b459d349`, `22341ae7`).
- [~] 2.4 `moe/` routed prefill tiled GEMM: device tile maps, half I/O, GeGLU
      epilogue, 64-row down tiles, one-launch decode-width rows, exact top-8.
      Most of this is already in `routed_f16.hip.cpp` (device-built tile maps
      with dead-tile skip, fused GeGLU epilogue, half I/O, Q4_K/Q5_K/BF16),
      so the port reduced to targeted deltas. **Done: fused `MoeFinish`**
      (`fused.hip.cpp`) — combine + both expert RMSNorms + their add + post-FFN
      norm + residual add + layer-output scale in one launch per row, with
      register-held rows and bit-identical reduction orders (float4 float
      partials, double block reduce, per-slot double combine matching
      `MoeEpilogueKernel`). The executor chain drops 8 launches per layer
      (MoeEpilogue, 3×RmsNorm incl. a dead ffn_norm pass on the half route,
      2×D2D copy, 2×Add, Scale) and `NarrowActivations` on the routed input
      (`RmsNormRowsHalf` writes the half rows directly — 3.3's MoE share).
      Gated: `gemma4.rocm_kernels`/`rocm_forward` pass, `--validate-prefill 16`
      top1=YES kl=0.034 (= sequential-drift level). pp2048 **2104.02/42.26**
      and repeat 2095.98/42.30 vs 1992 (report) / 1975–1998 (candidate
      controls): **+5.3 %, decode neutral**
      (`/tmp/opencode/gufo_bench_gomoe{,val}.log`). pp8192 1919.22
      (`gufo_bench_gomoe8k.log`) — no matched pre-MoeFinish control at 8192
      (retained Q8_K_XL figure 1782.49 predates P2.1), inconclusive there,
       no regression signal. **Remaining fork delta:** BN=96 / two-tile-height
       route (`RoutedHalfKQuantKernel` kTile=96) — deferred behind 3.4/2.5.
       Code audit (lever-5 era): exact top-8 confirmed — `RouterTopKKernel`
       (`fused.hip.cpp:482`) is softmax + k argmax iterations, deterministic
       lowest-index tie-break; down tiles are instantiated at 96 and 128
       (`moe_prefill.hip.cpp:1232-1308`, superseding the "64-row" wording),
       and one-launch decode-width grouped GEMVs exist for Q8_0/Q4_K/Q5_K
       (`gemv.hip.cpp:134-266`). Real remaining delta is the fork's kTokTiles
       streaming issue rate/LDS per the 3.2f lesson, plus the other quant
       decodes tracked in 2.3.
- [ ] 2.5 `attn/` prefill WMMA template: waves_per_eu, next-tile prefetch,
      rescale-skip, window staging, binary16 output, derived-keys hook.
- [ ] 2.6 Fused half elementwise set, pinned reduction order.
- [ ] 2.7 `tools/ci/kernel-resources.json` budgets + clang-23 spill check.

## Phase 3 — gemma4 onto shared core

Measured pp2048 timed-pass budget (Q8_K_XL, `rocprofv3` via
`tools/prof/prof.py`, `/tmp/opencode/prof_g4pp2k`): kernel 1098.7 ms /
span 1117.7 ms, idle 1.7% — GPU-bound, not launch-bound. Split: routed MoE
`RoutedF16GEMM`+`MoeEpilogue`+GeGLU+router ≈ 482 ms (44 %) → 2.4; dense
`DenseF16GEMM` 219 ms (20 %); hipblas BF16 attn-out route
`Cijk..BSS` 24×3.35 ms + `Cijk..SB` SGEMM 120×0.24 ms ≈ 109 ms (10 %) → 3.2
(the 3.35 ms/call matches staged8's microbench, so lifting the window into
`DenseF16Gemm` plus BF16→binary16 upload also kills the `NarrowBf16` pass);
attention WMMA 98 ms (9 %) → 2.5/3.4; elementwise (QknormRope, RmsNorm±Half,
KvNormRopeWrite, Add, NarrowHalf/Scale) ≈ 146 ms (13 %) → 2.6/3.3; leftover
int8 `mul_mat_q` 21 ms elsewhere. Fork 3326 t/s ≈ 616 ms — the 2.4 MoE route
is the single biggest remaining lever (first payoff: fused `MoeFinish`,
+5.3 % pp2048, then the 3.2a BF16→binary16 dense retype, +3.1 % pp2048; the
fork's 96-row tile kernel landed in 3.2b and was timing-neutral — the true
timed-pass budget is dense GEMM 26 % first, MoE 44 % incl. `MoeFinish`
second). A/B evidence: bare
`amdgpu_waves_per_eu(8)` on `AttentionPrefillWmmaKernel` REGRESSED pp2048
1997.85 → 1955.54 (`/tmp/opencode/gufo_bench_g4w8.log`) and was reverted —
it only pays together with the fork's register-frugal 2.5 template
(value-register key staging, next-tile prefetch), not as a standalone hint.

- [x] 3.1 Plan selection (`HalfPlan` analogue) in gemma4 executor —
      `gemm.hip.cpp` `HalfPlan(m,k)` feeds both `DenseF16Gemm` call sites.
      Microbench shows `attn_out_full` (2816×8192) staged8 beats int8 1.41×,
      so lifting `kDenseF16MaxCols=4096` is worth adopting once e2e KL at 3.3
      verifies the wider window.
- [~] 3.2 Remove `hipblasSgemm`/NarrowBf16 route; BF16→binary16 at upload;
      reject `kF32` projections at load. **Stake corrected** by a GGUFReader
      dtype histogram of the Q8_K_XL artifact: only ONE of 30 layers stores
      BF16 (attn q/k/v/out + FFN/expert stacks) and the routers are F32
      (2816×128) — so the BF16 route ≈27 ms and SGEMM ≈7 ms per timed pass
      (~35 ms total, not the 109 ms attributed from the raw `Cijk..BSS`/
      `Cijk..SB` totals, which spread over ~3–4 iterations). QAT Q4_K_XL
      scales: decode, prefill, load ≈6 s.
      **3.2a done (fork-mirror)**: `GemvType::kF16` + device-side
      `Bf16ToHalfKernel` in `gemv.{hpp,hip.cpp}`; `Uploader::Half()`/
      `ConvertHalves()` in `device_model.cpp` retype every non-expert dense
      weight (attn q/k/v/out, ffn gate/up/down, embeddings, draft
      projections) at load and reject any value outside binary16 range;
      router/experts keep their types. `Gemm()` kF16 = `NarrowHalf`
      activations + `DenseF16Gemm(weights_f16=true)` inside the measuring
      window (native-F16 `kW16` tile mode added to the shared
      `dense_f16_gemm.hip.cpp`; only gemma4 binds that core) else interim
      `hipblasGemmEx HIP_R_16F` outside it. Gates: `gemma4.rocm_kernels`
      + `gemma4.rocm_forward` pass, prefill validate top-1 match,
      kl 0.0333 (improved from 0.03415 — binary16 vs BF16 rounding).
      Matched pp2048: 2159.92/2167.61 vs MoeFinish 2095.98/2104.02
      (**+3.1 %**), tg neutral (41.7-42.4); pp8192 1977.58 unqualified
      (no matched control). `/tmp/opencode/gufo_bench_{win2048,win2048b,win8192}.log`.
      The kF16 window then spanned 2x4096 cols so attn_out (2816x8192,
      `HalfPlan` staged8) left hipBLAS too: pp2048 2174.42/2156.22
      (noise-equivalent), pp8192 1994.80, gates and kl unchanged
      (`gufo_bench_cap2*.log`).
      Remaining: delete the now-dead kBF16/`NarrowBf16`/`Bf16Scratch` case
      in `Gemm()`, and fold the F32 routers into the half route so
      `hipblasSgemm`/`kF32` can be rejected at load (shared with 3.5).
      **3.2b moe_prefill port landed (timing-neutral, quality win)**: fork
      `moe_prefill.{hpp,hip.cpp}` (verbatim; the 96-token-tile
      `RoutedHalfDownKernel` family + `LaunchDenseHalfGemm`, unused so far)
      added to `gufo_gemma4_kernels`; `BuildRoutedTiles` builds the gate and
      down tile maps in one pass with per-map tile heights into one buffer
      (`2 × RoutedTileCapacity(...,48)`, down map at offset span), and
      `RoutedPadBoundsKernel` now clears each bucket's padding rows while the
      bounds are hot, replacing the two whole-buffer memsets in
      `RoutedCompact`. `MoeBatch` dispatches each stack to
      `LaunchRoutedHalfGemm` (96-row tiles) when its GGML type is natively
      decoded and `hidden%256==0 && expert_ff%64==0`
      (`PrefillFormatOf`/`OwnGateUpFormat`/`OwnDownFormat`), else the 64-row
      fused `RoutedF16Gemm`. **Artifact-fact correction** (GGUFReader): this
      Q8_K_XL stores BOTH expert stacks as Q8_0 for 29 of 30 layers and BF16
      (kBF16=30) in blk.29 — the old "gate_up Q4_K" note was wrong, and
      `expert_feed_forward_length` is **704** (2112 is the dense FFN). Gate
      runs geglu into `pf_act_half_`, down writes F32 `pf_expert_out_`; blk.29
      correctly falls back to `RoutedF16Gemm` (2 calls/layer-batch, ~6.4 ms).
      Gates `gemma4.rocm_kernels` + `gemma4.rocm_forward` pass exclusive
      (`gufo_bench_gate{k,f}.log`); validate kl **0.01008** tv 0.0122 top-1
      match — a 3.3× numeric improvement over 0.0333 (GeGLU/quant epilogue
      rounding site moves as designed). Timing: pp2048 2175.67/2198.23 vs
      3.2a 2159.92/2167.61/2174.42/2156.22 — flat within noise; pp8192
      1971.91 vs 1994.80 unqualified (`gufo_bench_port1.log`). rocprofv3 of
      the candidate (timed pass 207, 998 ms: `prof.py show
      /tmp/opencode/prof_port1/prof_results.db`): MoE GEMM pair 2232+1132 us
      vs the replaced pair ≈2600+1700 — ≈19 % faster, but the MoE *budget*
      was smaller than the old 482 ms note implied (it lumped `MoeFinish`
      into "44 %"): MoE total is now 438 ms (gate 248 + down 126 + MoeFinish
      47 + bf16-layer 18), while **dense `DenseF16GEMMKernel` at 256 ms
      (26 %) is the single biggest item**, then attention WMMA 100 ms,
      QknormRope 41 + RmsNorm-family 48 + MoeFinish 47 + Cijk 28 + `mul_mat_q`
      21 (`qfn_mmq` — caller to be identified; ≈29-layer shape, not lm_head
      budget arithmetic) and elementwise `NarrowHalf` 14 ms. The fork's
       remaining 3326-vs-2180 gap therefore lives in the dense/attention/
       elementwise fusion pipeline (fork h_half end-to-end, `LaunchDenseHalfGemm`
       dense projections, fused rope/norm), not the routed expert GEMMs.
       **3.2c `prefill_chunk` 512 → 2048 (biggest single lever; supersedes
       the attribution above — fork benchmark stake corrected again)**: the
       cited fork figure 3326 belongs to the fork's UD-Q8_K_XL AR table at
       d4096 (3324.04, `gufo_franzmoca/docs/models/gemma-4-26b-a4b/
       BENCHMARKS.md`); **on this same quant the fork's d0 (≈pp2048) is
       3760.40** — a real same-artifact gap, not a quant mismatch. Their
       EXPERIMENTS.md times routed GEMMs "per 2048 rows" per layer, while we
       chunked prefill through an unsettable default `prefill_chunk = 512`
       (`engine.hpp`, `executor.cpp` fallback — no CLI ever assigned it), so
       every MoE layer streams the touched expert weights **4×** and tiles
       four times at quarter-M. Both defaults raised to the fork's 2048; all
       buffers derive from `C`, no capacity fixes needed. Measured exclusive:
       pp2048 **2649.47** (+21.8 % vs 2175.67), pp8192 **2409.30** (+22.2 %
       vs 1971.91), tg 42.57/42.28 neutral; validate kl **0.0100832073**
       tv 0.0122322326 — bit-identical to 3.2b (chunking only changes weight
       re-stream count, not numerics). `gufo_bench_{pc2048,pc8192}.log`.
       Same-quant d0 gap vs fork now 3760 vs 2649 = **1.42×**; re-profile
       before picking the next lever — MoE's 4×-streamed share shrank.
       **3.2d producer fusion, first slice (small win, zero numerics
       change)**: the router's `MulInPlace(router_scale)` + `ScaleInPlace
       (1/sqrt(hidden))` fold into the norm store (`RmsNormKernel`
       `post_scale`/`post_k`, multiply order matching the standalone chain);
       the attention-input norm writes one binary16 row (`pf_qkv_half_`)
       shared by q/k/v via widened `GemmHalfIn` (now also `kF16` with the
       8192-col extension, `HalfInWindow` probe) — replacing three
       `NarrowHalf` passes and the FP32 row when all three shapes qualify
       (all 30 layers here). Measured exclusive: pp2048 2647.97 (flat vs
       2649.47), **pp8192 2439.35 (+1.2 % vs 2409.30)**, tg 42.2-42.5
       neutral; validate kl 0.0100832073 **bit-identical** (rounding path
       proven equal); `gemma4.rocm_kernels` + `rocm_forward` pass
       (`gufo_bench_fus1.log`). Lesson: the fork's ~120→36 ms "elementwise"
       delta is NOT its producer fusion (worth only ≈15 ms here) — the gap
       concentrates in `QknormRopeKernel` **722 µs/call** +
       `KvNormRopeWriteKernel` 528 µs/call (≈50 ms/prefill vs the fork's
       whole "QkvPost and norms ~36 ms"), i.e. rope-trig/block-shape bound
       (`sincosf` per element at attention.hip.cpp), next lever; the fork
       keeps identical full-precision trig, so its speed comes from work
       layout, not accuracy trade. Re-profiled budget per pp2048 prefill
       (`/tmp/opencode/prof_pc2k`, trace 3 prefills): MoE 182+27 `MoeFinish`
        +12 blk29-fallback, dense F16 ≈148, attention ≈59, rope+norms ≈81,
        NarrowHalf 14 (now gone), router SGEMM 15, `mul_mat_q` 12.
        **3.2e warp-wide QK/KV norm-rope kernels**: `QknormRopeWarpKernel` and
        `KvNormRopeWriteWarpKernel` give each warp ONE TOKEN and loop heads —
        the `(pos, inv_freq)` rope angles are computed once per token
        (shared by all 16 Q / 8 KV heads) into registers, NEOX pairs
        `(i, half+i)` and the double RMS sum stay in registers per head with a
        butterfly shuffle (`WarpAllReduceSumD`, no broadcast); block-shaped
        (128 threads = 4 tokens). Wrappers select it for `tokens ≥ 32` &&
        `head_dim % 64 == 0` (≤1024) and keep the block kernels otherwise —
        at decode width a single warp looping 16 heads serializes and tg fell
        42→38 until the `tokens ≥ 32` guard sent decode back to the per-
        (token, head) block kernels. The first warp-per-HEAD cut is what the
        timed profile (pass tables, `/tmp/opencode/prof_rope`) exposed as
        still-bound (`QknormRope` 36 ms + `KvNormRopeWrite` 28 ms/prefill,
        ≈55 GB/s vs ≈156 GB/s achieved by nearby norms). Final numbers
        (exclusive): pp2048 **2852.38/2874.42 (+8.5 % vs 2647.97)**, pp8192
        **2570.96 (+5.4 %)**, tg 42.3–42.8 neutral; validate KL **0.0100832073
        tv 0.0122322326 — bit-identical to the 3.2b baseline** (double RMS
        sums round to the same float scale; the intermediate 0.01195/0.01267
        KL prints came from the decode path using warp kernels, gone once
        decode reverted), both gates pass (`gufo_bench_rope{1,2,3}.log`).
        Same-quant d0 gap now 3760 vs 2874 = **1.31×**. Fresh per-pp2048
        timing-pass budget (before this cut): MoE `RoutedHalfDown` 164+95 +
        `MoeFinish` 42 + `RoutedF16` 17, DenseF16 ≈227 (18.5 %),
        attention 89, rope+norms ≈122 (now ≈90), router SGEMM 21,
        `mul_mat_q<8,80>` 18 — next levers: MoE gate/up GEMM (fork 123 vs
        our 259+finish) and attention (fork 46 vs 89).
        **3.2f 128-row Q8_0 MoE tiles**: `gu_tile(fmt)` sends Q8_0 gate/up
        and down through `RoutedHalfDownKernel<kQ8_0, 128>` (builder,
        capacity and launcher made tile-size generic; 128 is rejected for
        other formats so a mixed-format artifact cannot mis-execute). At
        pp2048 the 16-padded buckets sit at 112–160 rows so `tile=96` spent
        ~2 tiles/expert and 128 spends ~1.5, yet the kernel sum moved only
        551→523 ms/trace (`/tmp/opencode/prof_moe128` vs `prof_rope` diff):
        neighbouring y-tiles share an expert's weights concurrently and the
        128 MB L3 already absorbs the re-read, so tile-count halving is not
        DRAM halving. End-to-end pp2048 **2877.10/2875.09 (≈+0.1 %, within
        noise)**, pp8192 2581.04 (+0.4 %), tg 41.4–42.3, KL **bit-identical
        0.0100832073**; gates pass.         Kept as a small monotone win (−5 % MoE
        kernel time, fewer dead tiles), NOT as the gate/up lever — closing
        fork's 123 ms still needs the streaming-path itself (`kTokTiles`
        issue rate/LDS), lesson: check cache-level traffic before tile
        multiplicity.
        **3.2g skip-rescale port evaluated and rejected (reverted)**: fork
        `ac2a4d67` per-row-block `__all(scale == 1)` gate around the WMMA
        attention accumulator rescale was applied verbatim to
        `AttentionPrefillWmmaKernel` and measured twice exclusive.
        Gates passed, KL **bit-identical 0.0100832073**, but pp2048
        2857.15/2862.62 (vs 2877.10/2875.09) and pp8192 2539.89/2555.35 (vs
        2581.04) — consistently −0.4 %…−1.6 %. The fork's wave owns one row
        block so one ballot skips a uniform large tail; our waves share all
        `kRowBlocks`, so several per-tile ballots cost more than the mostly-
        dead multiplies saved. Lesson: branch-shaped wins do not port across
        wave mappings. The remaining fork attention deltas (`ccef541a`
        two-blocks/WGP with `waves_per_eu`, `c96e2714` next-tile K/V register
        prefetch + sliding staging, `4a1a1c3c` rotated-dims-only global-layer
        caches) are structural: porting them means their wave-per-(row-block,
        128-dim-slice) mapping and their cache layout, not their predicates.
        Prior isolated `waves_per_eu(8)` on our layout REGRESSED, so pair it
        with the two-block shape if attempted.
        **3.2h upload-time QKV and gate/up fusion (port of the fork's
        concat-with-views)**: `Fuse()` stores q,k,v and gate,up as one
        row-major weight matrix per layer and points the per-tensor views at
        natural-layout slices; decode needs no change. Prefill fuses exactly
        when the half-in window holds — ONE wide `GemmHalfIn` emits the whole
        `[tokens][qkv_rows]` (gate-first `[tokens][2*ffn]`) box and the
        norm/RoPE consumers read the column slices at the fused row pitch —
        and otherwise the parts run their own GEMMs into natural boxes of the
        same scratch. Quality: pp2048 validator kl/tv **bit-identical to the
        unfused reference** (0.0002145108 / 0.0001046539 RECORDED), short
        fallback (64 tokens, batch < `kDenseF16MinBatch` → per-view path)
        RECORDED 0.0002459717, `rocm_forward` passed, kernels test extended
        with pitched-K/V cache rows reloaded from the device and checked
        bit-exact against the natural launch. Timing (exclusive window):
        pp2048 2965.95/2959.31 (vs 2877.10/2875.09, **+2.9 %**), pp8192
        2660.27 (vs 2581.04, **+3.1 %**), vs fork 3760 the remaining gap is
        the known 3.4 attention/FFN window work. Two follow-ups: driving a
        fused wide box through the FP32 `Gemm` routes FAILS the validator
        (kl 0.7277, deterministic across the Q8_0/F16 fallbacks) while every
        other route is bit-exact — the fused gate above keeps that route out
        of production, its root cause is still open; and the natural K/V
        units of `rocm_kernels_test` compare against a reference loop that
        overwrites the downloaded host caches before `got_k`/`got_v` reads
        them, so that check is tautological at HEAD and needs a tolerance
        redo (the new pitched checks avoid it by comparing launches, not
        re-derived values).
- [ ] 3.3 End-to-end binary16 activations (remove `NarrowActivations`); KL
      verified, not assumed.
- [~] 3.4 Attention occupancy + derived keys for global layers; per-position
      KL at 16 K gate. Attention budget at pp2048 timed pass: 91.9 ms vs fork
      46 ms — the largest remaining gap (dense FFN GEMM per-shape tiles is
      next at ~−38 ms: fork attn-out "two-block 8×1" 2013→1759 µs).
      Superseded by the lever-5 re-budget below (attention 55.6 vs 51.9,
      router SGEMM now the largest gap).
      **3.4a two-blocks-per-WGP WMMA attention port (fork `ccef541a` +
      `c96e2714` wave mapping, `half_store.hpp`)**: the fork's
      `WmmaPrefillAttentionKernel` (wave owns one row block × one 128-dim
      slice: 64 Q + 64 O VGPRs, binary16 caches, `waves_per_eu` pinned,
      next-tile register prefetch, sliding staging through the ring slot —
      ours runs `ring=0` direct indexing over the full F16 cache) replaces
      the 8-wave `AttentionPrefillWmmaKernel` behind
      `LaunchWmmaPrefillAttention`; hd256 runs 2 heads × 4 query blocks (two
      blocks/WGP), hd512 4×1; geometries outside 256/512 or GQA not
      divisible by `kHeads` fall back to the scalar oracle. The fork's
      `kGlobal` derived-keys fast path is compiled but gated on
      `rope_pairs==64`, so it stays dormant until the rotated-dims-only
      cache lands (3.4b). Gates (`gufo_bench_attnport.log`):
      `gemma4.rocm_kernels` (incl. the AttentionPrefill-vs-oracle unit) +
      `rocm_forward` pass; validator kl **0.0000930581** tv 0.0000874606
      top-1 match — 2.3× better than 3.2h's 0.0002145108 (identical
      head-dim partial sums, exact-half operands); pp2048
      **3118.36/3114.87 (+5.2 % vs 2965.95/2959.31)**, pp8192 **2929.57
      (+10.1 % vs 2660.27)**. Same-quant d0 gap 3760 vs 3118 = **1.21×**.
      **Batched embedding lookup (fork `BatchedEmbeddingLookupKernel`
      analogue, now default)**: one launch per prefill chunk (`EmbedRows`, grid
      (col-chunks, rows), `sqrt(hidden)` folded into the store, host-validated
      token ids uploaded per chunk) replaces the per-token `EmbedRow` loop +
      `ScaleInPlace` in `Executor::Prefill`; chunk embedding is bit-identical
      by construction (validator kl 0.0000930581 tv 0.0000874606 top-1 match
      unchanged, kernels/forward pass). Timing must be paired in-window: the
      first bench after a window starts reads cold-low (this was misread as a
      −0.8 % regression until control binaries were alternated). Matched
      alternation pp2048: candidate 3114.13/3124.67/3132.15/3141.87 vs fresh
      control 3105.32/3104.83/3045.84 (candidate never loses the pair);
       paired pp8192 2935.91 vs control 2898.82 (+1.3 %). 2048→4 launches.
       **Fused residual-add into the RMSNorm store pass (`RmsNormAddRows`,
       fork `PostAttentionNormKernel` analogue, now default)**: one launch
       writes `norm(x)*g + addend` at the three post-attention/pre-FFN sites
       (prefill, decode `StepLayer`, MTP draft), removing one full-width
       `Add` pass per layer (~8.6 µs/pp-rep vs fork). Bit-identical to
       `RmsNormRows` + `Add` by construction, and pinned by a memcmp unit
       check in `gemma4.rocm_kernels`. Gates (`gufo_bench_lever2d.log`):
       kernels/forward pass; validator kl **0.0000930581** tv 0.0000874606
       top-1 match EXACT; paired pp2048 cand vs control +0.57/+0.42/+0.31 %
       (cand won all three pairs), pp8192 cand-first 2960.51 vs control
       2929.84 (+1.05 %, order bias against the candidate there).
        Lesson pinned by isolation harness: ROCm `__fadd_rn`/`__fmul_rn` are
        plain operator wrappers and do **not** block FMA contraction — the
        fused store `v*scale*g + a` silently fused into an FMA (23 % of
        elements 1 ulp off). `#pragma clang fp contract(off)` scoped to a
        nested block around only the store loops (sumsq keeps its contracted
        reduction) restores bit-exactness at −O2/−O3.
        **Three post-attention norms -> one launch (`RmsNormRowsTriple`,
        prefill, now default)**: the FFN-norm binary16 slot (F16-window
        only), the router input (null gamma + `router_scale`/1/sqrt(hidden)
        fold) and the pre-FFW binary16 slot all norm the same `pf_attn_out_`
        rows, so one launch shares the sumsq and runs the three store loops
        (each verbatim from `RmsNormKernelHalf`/`RmsNormKernel` post_scale,
        same block=min(dim,256)/grid=rows geometry); 3 launches/layer -> 1.
        Bit-identical per slot by construction, pinned by a three-output
        memcmp unit check (plus a router-only null-slot call) in
        `gemma4.rocm_kernels`. Gates (`gufo_bench_lever3.log`, `lever3b`):
        kernels/forward pass; validator kl **0.0000930581** tv 0.0000874606
        top-1 match EXACT; seven matched pp2048 pairs cand mean 3125.96 vs
        control 3084.95 (+1.33 %, cand won 5/7, pair deltas
        +2.09/+4.20/-0.12/+0.20/-0.57/+0.67/+2.93 %), pp8192 cand-first
        2974.88 vs control 2955.60 (+0.65 %).
        **QKV post-projection merge -> one launch (`QkvPost`, prefill warp
        path, now default)**: one `QkvPostWarpKernel` (grid=tokens, block 512,
        warp-per-head round-robin over the q heads then the kv heads) replaces
        the `QknormRope` + `KvNormRopeWrite` warp launches; sincos computed
        once per warp, verbatim load/reduce/store loops for both passes, and
        the launcher falls back to the two standalone kernels outside the
        warp gate (decode/draft untouched). Q and V are bit-identical to the
        standalone launches (pinned at zero tolerance); K diverges only on
        rope-store rounding ties (measured 2/81920 + 1/40960 elements, one
        half ulp each) because FMA contraction is context-dependent and
        neither contraction setting reproduces the standalone choice — the
        merged store uses `#pragma clang fp contract(off)` for revision
        stability and the unit check caps K at 1 half ulp. Quality: absolute
        accuracy pinned by `gemma4.rocm_forward` (exact argmax at every
        position, bounded logit error); the bulk-vs-sequential validator
        fingerprint moves kl 0.0000930581 -> 0.0004176470 tv 0.0000874606 ->
        0.0001160507 (deterministic, top-1 match, 120x below the recorded
        FAIL bound — the kl is the bulk-vs-exact drift difference class, so
        unbiased 1-ulp K flips legitimately re-randomize it; control
        reproduces the old fingerprint exactly in the same window). Gates
        (`gufo_bench_lever4.log`, `lever4b`): kernels/forward pass; seven
        matched pp2048 pairs cand mean 3157.72 vs control 3121.76 (+1.15 %,
        cand won 6/7, pair deltas +2.06/+0.64/-1.87/+4.29/+1.02/+0.68/+1.34
         %), pp8192 cand-first 2992.71 vs control 2965.48 (+0.92 %).
         **Lever-5 re-budget (pp2048 timed pass 709.5 vs fork 668.5 kernel
         ms; profiles `/tmp/opencode/prof_pp2k_lv5` pass @20562.6 ms and
         `/tmp/opencode/prof_fork` @21944.6 ms; fork source local at
         `~/projects/llm/halo_forks/gufo_franzmoca`, tip `22341ae7`)**.
         Ahead of fork: MoE GEMMs (incl. blk29 BF16 fallback) 261.6 vs
         271.1, MoeFinish 41.3 vs 44.8. Router cost is EQUAL — the fork also
         runs the identical `hipblasSgemm` `Cijk` 22.1/30 + TopK 2.3
         (`gufo_franzmoca executor.cpp:1037-1045`), confirming the 3.2 ~7 ms
         reading class and killing the "fused router" stake of this entry's
         first draft (lesson: diff BOTH sides of every gap line before
         ranking). Real deltas (ours − fork, est saving): norms 33.1 vs 15.2
         −17.9 — fold the `RmsNormKernelAdd` residual pass (10.4) into
         `RmsNormRowsTriple` as a 4th store loop (lever-2 fused-store form,
         bit-exact candidate) −9, plus reconcile `RmsNormKernelHalf` (7.3,
         pre-attention norm) against fork's single `PostAttentionNorm`
         (they run ONE norm kernel per layer while `MoeFinish` already owns
         the post-FFN norm — check for a redundant re-norm); `QkvPostWarp`
         29.6 vs fork `QkvPostKernel<256/512>` 19.7 (fork
         `kernels.hip.cpp:405`) −9.8; activation narrow 7.9 vs 0.4 — fork
         producers emit f16 directly (`111ad652`, `3f6ae2b7`, task 3.3)
         −7.2; small-row K-quant GEMMs on `mul_mat_q<8,80>`+quantize 19.3
         vs fork `DenseF16<64,64,2,2>` 13.9 — small-tile `HalfPlan` entry
         −5.4; attention 55.6 vs 51.9 −3.7; GeGLU-in-GEMM-epilogue like
         fork `18e8c220` ours separate 7.4 vs packed 5.7 −1.7; RouterTopK
          vs fork TopK −1.0.
          **Lever 5c (accepted, same session): templatized
          `QkvPostWarpKernel<HD>` (HD=256/512 constexpr register arrays +
          pairs → the `m` loops fully unroll; HD=0 keeps the runtime kernel
          for exotic head dims, launcher gates in `attention.hip.cpp`).
          Element and accumulation order unchanged (double `WarpAllReduceSumD`,
          `contract(off)` K store) → q/v bit-identical, validator fingerprint
          reproduces HEAD exactly (kl 0.0004176470 tv 0.0001160507,
          top-1 match) and `gemma4.rocm_kernels` + `gemma4.rocm_forward`
          pass. Gates (`gufo_bench_lever5c.log`): four matched pp2048 pairs,
          cand 3176.20 mean (3109.95/3147.71/3222.59/3224.54) vs control
          (77e33192 snapshot `gufo_ctrl_v5`) 3087.25 (3094.38/3067.59/
          3043.95/3143.06) — +2.9 %, cand won 4/4 pairs
          (+15.6/+80.1/+178.6/+81.5); pp8192 cand 3017.9 vs 2969.7 (+1.6 %).
          Kernel line: QkvPost 29.6 -> 16.8 ms/pass (`/tmp/opencode/
          prof_pp2k_lv5c`) vs fork 20.9 — line CLOSED, now ahead.
          **Budget re-measure + CRITICAL gap correction (same-session
          evidence supersedes all stale cross-session numbers)**:
          (a) Both sides re-profiled with identical windowing (fork rebuilt
          at tip `22341ae7`); cluster sums ours 1486.2/4 router-units vs
          fork 1470.8/2: earlier window-derived per-line comparisons
          (QkvPost 29.6->16.8 vs ~20, ...) stay valid as kernel lines.
          (b) RETRACTION of this entry's first draft: the fork stream
          overlap is `concurrent = moe && n <= kMaxGroupSlots(16)` —
          GROUPED DECODE only (`gufo_franzmoca executor.cpp:842-847`), its
          prefill runs serial too (measured union-busy overlap 45.7 ms per
          2-pass cluster, not pipelining); there is no 100 ms scheduling
          gap to port.
          (c) PAIRED same-session wall A/B (`gufo_bench_forkab.log`, 4
          alternating pp2048 pairs): fork 3250.0 mean (3279.88/3287.67/
          3271.74/3160.66) vs candidate `86d1191d` 3208.0 (3186.30/
          3197.37/3211.24/3237.06) — **gap now −1.3 %** (fork won 3/4,
          pair deltas −93.6/−90.3/−60.5/+76.4). The historical "fork
          3760" and "baseline 3157" were mismatched-condition numbers from
          earlier sessions and are WITHDRAWN as gap references (lesson:
          cross-session t/s is worthless for gap claims — same-session
          paired only, thermals and harness included).
          (d) Fork tg1 prints 16k-25k t/s on this model — physically
          impossible; their decode timer is async-mis-measured (ours 42.5
          is the honest baseline). Do not copy decode-timing comparisons
          from fork bench output.
          Remaining levers, still ranked by the per-line kernel gaps:
          norms fold class (Add 9.6 into `RmsNormRowsTriple` ordering
          study, pre-attn `RmsNormKernelHalf` 6.7 vs fork single
          `PostAttentionNorm` 14.9), f16 producers to delete NarrowHalf
          (6.4), small-row DenseF16/`HalfPlan` (ours `mul_mat_q<8,80>`
          16.0 vs fork `DenseF16<64,64>` 14.8), GeGLU epilogue (6.9 vs
          fork packed ~5). Decode-side expert-stream overlap is a tg lever
          worth prototyping but its fork payoff CANNOT be read from fork
          bench (see (d))
          **Lever 6 (accepted, same-session): attention epilogue F16 twin
          deletes the last `NarrowHalf` (task 3.3)** — o_proj was the only
          remaining `Gemm(kQ8_0)` DenseF16-window call fed FP32 rows
          (29.5 calls/pass = one per layer). The WMMA prefill kernel
          already carried a guarded `out_half` store (`attention_wmma.
          hip.cpp:501-509`, `HalfOf` = the pinned-RN twin contract);
          `AttentionPrefill` now forwards a caller pointer (and the scalar
          oracle twin-stores `v` identically), the executor allocates
          `pf_attn_half_` [chunk][heads*head_dim] and runs o_proj through
          `GemmHalfIn` inside exactly the window `Gemm` would pick, with the
          unchanged FP32 `Gemm` as the sole fallback (both attention paths
          always write FP32, so out-of-window shapes are bit-for-bit the
          old path). By construction the twin is the RN narrowing of the
          exact value stored, so the validator fingerprint reproduces HEAD
          (kl 0.0004176470 tv 0.0001160507, top-1 match) and
          `gemma4.rocm_kernels` (+2 new bit-exact HalfTwin equality checks)
          and `gemma4.rocm_forward` pass. Gates (`gufo_bench_lever6.log`):
          four matched pp2048 pairs vs control (`gufo_ctrl_v6`, = `86d1191d`
          build) 3122.44 (3070.49/3165.41/3181.13/3072.73) — **cand 3188.69
          (3207.59/3189.10/3224.35/3133.73), +2.12 %, won 4/4**
          (+137.1/+23.7/+43.2/+61.0); validator-run pair +0.2 %. pp8192
          first window −0.5 % (2/2 negative) but the follow-up
          (`gufo_bench_lever6_8k.log`) is neutral: 3 pairs +13.3/−14.3/+8.8;
          pooled 5-pair mean −0.16 % — NEUTRAL at 8192, win scales at
          2048. Kernel line: NarrowHalf 6.4 -> 0 ms/pass; attention gains
          an F16 store inside an already-bound output epilogue.
          **Gap re-measure vs fork — ORDER BIAS SUSPECTED (unresolved)**:
          direct pp2048 A/B same session (`gufo_bench_lever6_fork.log`,
          3 fork-FIRST pairs): fork 3276.22 (3291.42/3279.17/3258.07) vs
          lever-6 candidate 3185.45 (3210.50/3208.73/3137.12) = −2.8 %.
          ORDER-BIAS PROBES SAME SESSION (`gufo_bench_ordprobe.log`,
          2 fork-first + 2 cand-first pairs): fork 3290.40 vs cand 3172.02,
          cand lost all four order-paired deltas (−85.7/−109.8/−127.0/
          −151.0) = **−3.6 %**, order effect <=0.6 % — NOT positioning.
          Residual puzzle kept on record: within-own-binary windows stay
          self-consistent (lever-6 +2.1 % over `gufo_ctrl_v6`) yet absolute
          levels shift +-4 % between windows even for identical binaries,
          so only direct fork A/B numbers count for the gap. Honest gap
          now: **fork ahead by ~2.8-3.6 % on pp2048**. (forkab's −1.3 % is
          superseded as a gap reference: same ordering, lower fork
          absolute; treat older cross-window gap figures as noise until a
          dedicated multi-pair cross window re-establishes the level.)
          **Lever 7 (REJECTED): fused residual-add + triple-norm launch**.
          One kernel replacing `RmsNormKernelAdd` followed by
          `RmsNormKernelTriple`: the sum s = fl(fl(x*s1*g)+a) is Add's exact
          store (contract-off loop), the three slot loops run verbatim over
          a per-thread float4 stash (hidden 2816 -> 3 float4/thread, no
          spill; `RmsNormAddTripleRows` refuses out-of-shape dims without
          touching memory), so all four outputs are bit-identical to the
          two-stage chain (new `gemma4.rocm_kernels` check: 0 diffs on res
          fp32 + all three slots, incl. refusal gate; gates run in the
          exclusive window, `gufo_bench_lever7.log`). Bench: pp2048 paired
          vs rebuilt HEAD control 3242.73: **-0.57 % (won 1/4,
          +9.7/-26.2/-47.3/-9.5)**; pp8192 -0.59 % (0/2). Profile
          (`prof_pp2k_lv7`, timed pass): fused line 24.03 ms vs Add
          10.10 + Triple 15.35 = 25.45 -> only -1.4 ms of 693 and even
          that not visible on wall time; dispatches 616 -> 586. REJECTED:
          the norm pair is NOT DRAM-round-trip-bound (L2 absorbs the
          intermediate), so the traffic model that ranked this ahead of
          the fork's PostAttentionNorm (14.9 vs 25.5 ms/pass)
          overestimated - that fork line does fewer/different outputs and
          is not pure fusion headroom. Implementation reverted to HEAD.
          INCIDENT recorded (protocol): two full machine reboots occurred
          while `gemma4.rocm_forward` ran as a RESIDENT test next to the
          VRAM-full server - model-loading GPU tests must run inside the
          exclusive window like benchmarks, never resident; kernel-only
          resident tests remain safe.
          **Lever 8 (REJECTED): `-ffast-math` on the WMMA prefill
          attention**. Full attribution of the residual per-line fork
          leads: the attention kernel source is semantically identical
          (the lone text diff, a `key_stride` `size_t` cast in
          attention_wmma.hip.cpp, stores into a `std::uint32_t` array and
          truncates identically - it is not the edge), launchers/flags are
          doc-only diffs, same clang binary, same RelWithDebInfo; the
          fork compiles `gemma4_kernels` with `-O3 -ffast-math
          -fno-finite-math-only` while gufo omits fast-math by policy
          (scalar-oracle fidelity, see src/models/gemma4/CMakeLists.txt).
          Scoped the exception to attention_wmma.hip.cpp only (source
          COMPILE_OPTIONS emit after the target's; flag order verified in
          compile_commands). Gates: `rocm_kernels` and `rocm_forward`
          (tolerance + determinism) PASSED, but the prefill validator
          caught corruption - `top1_match=no kl=2.2732 tv=0.2665` vs the
          control's `kl=0.0004176` at 1024-token validate
          (`gufo_bench_lever8.log`). Confirmation window
          (`gufo_bench_lever8b.log`): the fork CLI parses
          `--validate-prefill` but never executes it, so the fork's own
          fast-math accuracy is unmeasured on our gate; paired pp2048 of
          the candidate vs control was within noise (avg 3226.7 vs
          3194.5). REJECTED: fast-math online softmax corrupts bulk
          prefill for a gain not visible on wall time; the CMake
          exception is reverted. The residual fork prefill gap IS this
          fast-math codegen class (attention ~11 % per call, norms/GeGLU
          class a few ms) - closed on quality grounds, not reproduced.
          **Lever 10/11 (REJECTED): gate-safe fast-math subset and the
          oracle re-gate of fast-math attention** - the fast-math class
          is now fully closed. Lever 10 applied fast-math only to the
          tolerance-gated TUs (`gemm`, `gemv`, `moe_prefill`,
          `routed_f16`, `vision`, plus the shared
          `gufo_core_hip_gemm`): every quality gate passed -
          `rocm_kernels` bit-exact PASS, `rocm_forward` oracle PASS
          (10.4 s real run), validator **status=PASS**
          (kl=3.0411e-5 - better bulk-vs-exactness than the control's
          recorded 4.176e-4 drift) - yet paired bench was neutral/negative:
          pp2048 +0.31/-0.14/+0.30 %, pp8192 **-1.50**/-0.34 %
          (`gufo_bench_lever10.log`). Not adopted (no proven win; the
          shared DenseF16 target would also have shifted
          qwen36/qwen38 prefill bits with no measured gain). Lever 11
          re-gated attention-TU-only fast-math: `rocm_forward` oracle
          PASSED (10.44 s) but the bulk-vs-exact validator reproduced
          lever-8's failure verbatim (kl=2.2732417830, tv=0.2664549,
          top1 no, `gufo_bench_lever11.log`). Gate lesson recorded:
          `--validate-prefill` is bulk-vs-SEQUENTIAL consistency on one
          binary, so its verdict is codegen-context dependent - the
          target-wide probe (lever 9) passed kl at 3.86e-5 with fast-math
          attention because every path shifted together; the oracle
          forward test alone is blind to the tiled-bulk corruption.
          Production bulk prefill diverging 2.27 nats from exact is
          disqualifying on its own. CONCLUSION: the fork's residual
          prefill edge is only reachable through fast-math attention
          codegen, which reproducibly breaks bulk-vs-exact accuracy; no
          quality-safe configuration retains the remaining ~3 %. Both
          experiment patches reverted; HEAD (no fast-math on gemma4)
          stands.
- [ ] 3.5 Router fused into one GEMM prologue; router-logit rel-err + exact
      top-8 gate. Low priority now: the fork runs the identical
      `hipblasSgemm` + `TopKKernel` (router SGEMM 22.1 ms on both sides —
      see lever-5 re-budget), so no fork parity win exists here yet.
- [ ] 3.6 MTP width-invariance via `wmma_gemv`; hash-equality gate vs AR.
- [ ] 3.7 Calibrated draft policy (+`draft_policy_bench.py` pattern).
- [ ] 3.8 Re-measure `BENCHMARKS.md`; QUALITY.md MTP framing fixed.

## Phase 4 — qwen36_a3b fan-out (nodes 1,2,3,5,6,7,8; GDN: producers only)

- [ ] 4.1 Shared-core dispatch + plan table; kill `kF32`/SGEMM tier.
- [ ] 4.2 Half producers + elementwise/router fusion.
- [ ] 4.3 MTP width-invariance + hash gate; draft policy.
- [ ] 4.4 Oracle KL suite + benchmarks.

## Phase 5 — ling3_flash fan-out (nodes 1,3,5,6,7,8; full-attn occupancy)

- [ ] 5.1 Plan-aware dense tile-GEMM + half producers (`executor.cpp:982+`).
- [ ] 5.2 Router/elementwise fusion; `kF32` GEMV mapping removed.
- [ ] 5.3 MTP width-invariance + hash gate; draft policy.
- [ ] 5.4 Oracle KL suite + benchmarks.

## Phase 6 — Close-out

- [ ] 6.1 Exclusive-window benchmark matrix per model (interleaved vs fork).
- [ ] 6.2 Docs: per-model optimization notes, AGENTS.md shared-kernel carve-out.

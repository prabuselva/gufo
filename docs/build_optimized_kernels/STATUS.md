# Build optimized kernels — status tracker

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked

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
- [ ] 2.3 K-quant decode variants (Q4_0/Q5_0/Q4_K/Q5_K/Q6_K/Q8_0).
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
- [ ] 3.3 End-to-end binary16 activations (remove `NarrowActivations`); KL
      verified, not assumed.
- [ ] 3.4 Attention occupancy + derived keys for global layers; per-position
      KL at 16 K gate. Attention budget at pp2048 timed pass: 91.9 ms vs fork
      46 ms — the largest remaining gap (dense FFN GEMM per-shape tiles is
      next at ~−38 ms: fork attn-out "two-block 8×1" 2013→1759 µs).
- [ ] 3.5 Router fused into one GEMM prologue; router-logit rel-err + exact
      top-8 gate.
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

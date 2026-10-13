# Port plan: fork gemma4 kernel optimizations → shared ROCm core → gemma4 / qwen36_a3b / ling3_flash

Source analysis: [docs/models/gemma4/OPTIMIZATION_DIFF_REPORT_v2.md](../models/gemma4/OPTIMIZATION_DIFF_REPORT_v2.md).
Donor: `fm/feat/gemma` (`gufo_franzmoca` @ `22341ae7`). Our Gemma 4 and the fork's
share no git history (merge-base `2b4daede` has no gemma4), so nothing is
cherry-pickable: the fork's *design* is ported once into a shared layer and
consumed by every MoE model.

## Strategy (settled 2026-10-12)

1. Build the tuned kernels **once** in a shared layer; model executors only
   choose plans/policies.
2. Home: **`src/core/hip/`** — new `gemm/`, `moe/`, `attn/` submodules under
   `gufo::core::hip`. Model dirs keep dispatch, shapes, plans, oracle and
   tests. This carves a deliberate exception to "model-private kernels"
   (`dense_f16_gemm` is already duplicated four times: qwen38_flash_next,
   gemma4, qwen36_a3b, ling3_flash); the fork set the same precedent by
   consuming `qwen38_flash_next::rocm::DenseF16Gemm` from its gemma4 executor
   (`executor.cpp:427-437,485-490`).
3. Scope: all kernel levers + MTP width-invariance correctness + calibrated
   draft policy. Fork session/snapshot/prompt-lookup serve behavior is deferred
   to a follow-up evaluation.

## Node-compatibility map (report §8 levers × our models)

| Lever | gemma4 site | qwen36_a3b site | ling3_flash site |
|---|---|---|---|
| 1 Shape-tuned dense F16 plan | `kernels/rocm/dense_f16_gemm.hpp:25` (no plan) | `dense_f16_gemm.*` + `gemm.hip.cpp:137` | `kernels.hip.cpp DenseF16Gemm` w/ device tile maps (`executor.cpp:985-997`) |
| 2 Kill F32/BF16 SGEMM route | `gemm.hip.cpp:145-176`, `weights.cpp:126-234` allow `kF32` | `gemm.hip.cpp` (sgemm), `executor.cpp:27,36,41` | no sgemm; `executor.cpp:110` maps `kF32`; weights allowlist |
| 3 End-to-end binary16 activations | `executor.cpp:679 NarrowActivations` | prefill FP32 rows (`kernels.hip.cpp`) | `executor.cpp:982` same pattern |
| 4 Attention occupancy/traffic | `attention.hip.cpp:418` 1 block/WGP | `kernels.hip.cpp:1258` `(256,2)` WMMA; GDN unaffected | `attention.hip.cpp` default bounds; SSM layers unaffected |
| 5 Fuse elementwise layers | `fused.hip.cpp:273-428` | `kernels.hip.cpp` tier | `kernels.hip.cpp` tier |
| 6 Router in one GEMM | `fused.hip.cpp:282` | MoE router (`kernels.hip.cpp`) | MoE router (`routed_f16`) |
| 7 Width-invariant MTP verify + split-K K-quant GEMV | no WMMA GEMV in any `gemv.hip.cpp` | MTP `executor.cpp:214-396` | MTP `executor.cpp:215-520` |
| 8 Calibrated draft policy | none (`--draft-policy` is DFlash2-only, `src/cli/bench/bench.cpp:1217`) | MTP chain | MTP chain |

## Phase 1 — Quality harness (lands first; gates everything after)

- Port `tests/models/gemma4/reference_probe.cpp`, `logit_file.hpp` and
  `tools/gemma4/compare_logits.py`, `compare_stages.py` from the fork; adapt
  to our `src/models/gemma4/reference.cpp`.
- Add oracle `Storage` modes `kFloat32 / kHalfKv / kQ8Activations`
  (fork `reference.hpp:23-28`) to all three model oracles so a lever's KL can
  be predicted before its kernel exists and our binary16-KV assumption is
  quantified.
- Make `--validate-prefill` real for gemma4: bulk-vs-exact prefill KL in
  `models/gemma4/cli_runner.cpp` (the flag no-ops at
  `src/cli/bench/bench.cpp:1277`); same for qwen36/ling3 runners.
- New `tests/functional/` gates: MTP-vs-autoregressive `output_sha256`
  equality at depths {0, 8K, 16K} (`cache`, `long-context`); record the
  16 K/32 K degenerate-trajectory artifact (§6.2) tracked, not gated.
- **Exit gate:** our harness reproduces the fork's published KL on the shared
  artifacts (decode `7.5e-7`–`8.8e-7`, prefill `3.7e-5`, bulk `0.027`,
  top-1 1489/1542).

## Phase 2 — Shared kernel core in `src/core/hip/`

New fork-faithful primitives (fork refs):

- `gemm/`: `dense_f16_gemm` + `DenseF16Plan{kAuto,kRowGroups8,
  kStagedRowGroups8}` (fork `qwen38_flash_next/kernels/rocm/kernels.hpp:214`,
  `kernels.hip.cpp:6227-6300`); `wmma_gemv` width-invariant decode/verify GEMV
  + `gemv_tasks` split-K + `half_store`; K-quant decode variants Q4_0/Q5_0/
  Q4_K/Q5_K/Q6_K/Q8_0 (fork `b459d349`, `22341ae7`, `e6da366c`, `244ea3e7`).
- `moe/`: routed prefill tiled GEMM with device-built tile maps, half in/out,
  GeGLU fused in the gate/up epilogue (fork `18e8c220`, `16c3fbbc`), 64-row
  down tiles (`92f5119d`), one-launch decode-width expert rows (`82d8e8c2`);
  exact top-8, no dropping, no cap.
- `attn/`: prefill WMMA template with `amdgpu_waves_per_eu` sizing (fork
  `attention_wmma.hip.cpp:84-85`; ours `attention.hip.cpp:418` requests one
  block per WGP), next-tile prefetch, rescale-skip on unchanged row max
  (`ac2a4d67`), sliding-window staging (`c96ff32f`), binary16 output.
- Fused elementwise with half producers and **pinned reduction order** (fork
  measured RMSNorm block-width alone moving KL `1.3e-6`→`2.7e-5`).
- Regenerate `tools/ci/kernel-resources.json` budgets; preserve clang-23
  spill-freedom (`3e29d157`).
- **Gate:** unit tests beside `src/core/hip` + operator KL vs oracle; first
  consumer = gemma4 dense path, then qwen38/qwen36/ling3 switch over.

## Phase 3 — gemma4 onto the shared core (target §3 cells → ~1.0–1.1×)

Executor dispatch changes in `src/models/gemma4/kernels/rocm/executor.cpp`;
every lever A/B'd at ≤16K first, confirmed at 32K/65K/100K:

1. `HalfPlan`-style plan selection (m==2816: staged-8 for K≥4096, row-groups-8
   for K==2112).
2. Kill `hipblasSgemm`/`NarrowBf16` (`gemm.hip.cpp:145-176`); BF16→binary16 at
   upload (fork `weights.cpp:48-51`); reject `kF32` projections at load
   (`weights.cpp:126-234`). Acceptance includes QAT Q4_K_XL: prefill scales,
   load drops 69.6 s → ~6 s via fused dequant-to-binary16 + split-K decode.
3. End-to-end binary16 residual: remove `NarrowActivations`
   (`executor.cpp:679`); verify KL, do not assume `3.7e-5`.
4. Attention: shared `attn/` template + derived keys for the 5 global layers;
   gate on per-position KL at 16 K, not max-abs.
5. Router fused into one GEMM prologue (drop `RouterTopKKernel` launches,
   `fused.hip.cpp:282`); gate: router logits ≤~1e-6 relative, exact top-8
   membership.
6. MTP width-invariance through shared `wmma_gemv` (fork `executor.cpp:510-
   514`). Gate: `--speculative mtp` output hash == autoregressive (fixes
   §6.1); recheck §6.2 long-context loop after 3–4.
7. Calibrated draft policy last (fork `draft_policy_bench.py`, `84c76b0c`,
   `27c19710`); generalize the DFlash-only `--draft-policy` plumbing.

Refresh `BENCHMARKS.md` (3–5 % stale, report §3) and rewrite the `QUALITY.md`
MTP-drift framing (defect, not expectation).

## Phase 4 — qwen36_a3b fan-out

Shared kernels; model dispatch: plan table for its shapes, remove the
`kF32`/SGEMM tier (`gemm.hip.cpp`, `executor.cpp:27,36,41`), half producers,
router + elementwise fusion, MTP width-invariance via `wmma_gemv`, draft
policy. GDN layers take producer/traffic changes only. Gates: own oracle KL,
existing rocm tests, MTP hash-invariance, matched pp/tg A/B.

## Phase 5 — ling3_flash fan-out

Plan-aware dense tile-GEMM (`executor.cpp:985-997`), half producers
(`executor.cpp:982`), router/elementwise fusion (`routed_f16`), MTP
width-invariance + split-K K-quant decode, full-attention-layer occupancy
(default bounds today); SSM recurrence untouched. Same gate set on its oracle.

## Phase 6 — Close-out

Exclusive-window benchmark matrix per model (`tools/bench/bench_bg.sh` +
`bench_wait.sh` protocol, binaries interleaved once for the final vs-fork
claim), optimization notes per model docs, AGENTS.md shared-kernel carve-out,
report §8 items marked landed with attribution deltas.

## Cross-cutting rules

- Every lever = one change, gated: oracle KL + top-1 counts, operator tests
  (`nix develop -c cmake --build --preset gpu-test --target <test>` +
  `ctest --preset gpu-full -R '^<name>$'`), `tools/ci/check-format.py`,
  functional suites for the affected model (all three only when a shared-core
  file changes).
- Timings: mandatory per-request/phase at 5 % and 3 ms, never averaged;
  correctness and expected prefill/cache work are strict.
- Do-not-port (report §10): no Q8_1 activations on any prefill path, no
  max-abs-only gates, no fine-tune parity claims without our own harness.
- Preserve local invariants through kernel replacement: `b996f776` grammar
  path (chat-template side) and `3e29d157` clang-23 spill-freedom
  (`tools/ci/kernel-resources.json`, re-checked for every new kernel).

## Branch/PR sequence

`feat/gemma4-kl-harness` (Ph. 1, independent, lands first) →
`feat/core-hip-kernels` (Ph. 2) → `perf/gemma4-shared-kernels` (Ph. 3) →
`perf/qwen36-shared-kernels` (Ph. 4) → `perf/ling3-shared-kernels` (Ph. 5).

Status tracking: [STATUS.md](STATUS.md).

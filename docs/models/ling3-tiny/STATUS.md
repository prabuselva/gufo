# Tiny enablement — status tracker

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked

Record-keeping (golden rule, same as
[docs/build_optimized_kernels/STATUS.md](../../build_optimized_kernels/STATUS.md)):
every gate result — pass AND fail — is appended here in the same commit,
with the exact command and artifact; kernel-level findings land in
[EXPERIMENTS.md](EXPERIMENTS.md). The backlog below is never trusted from
memory. Strategy and acceptance criteria: [PLAN.md](PLAN.md). Shape facts:
[README.md](README.md).

## Phase 0 — Audit (done 2026-10-18, planning input)

- [x] 0.1 Tiny metadata measured from `Ling-3.0-tiny-Q8_0.gguf` with the
      repo `venv` `gguf` lib: 24 blocks, no `nextn_predict_layers` key,
      hidden 1536, 16 heads, 128 experts 8-of, expert_ff 512, dense FFN
      4608 only in block 0, MLA absorbed shapes identical to Flash
      (192/128/512/64), same vocab/eos family. Full table in README.md.
- [x] 0.2 Block-kind census from tensor tables: MLA at 3, 7, 11, 15, 19, 23
      (stride 4, NOT the stride-6 rule hardcoded at `config.cpp:232`);
      KDA on the other 18; MoE on 1..23. Invalidated the earlier
      planning-time assumption of `(il+1)%6==0`.
- [x] 0.3 Kernel-constant grep audit: no Flash geometry literals in
      `src/models/ling3_flash/kernels/`; executor sizes buffers from config
      (`executor.cpp:312, 341-417`); model/weights layers are
      config-driven (`weights.cpp:198` nextn load is conditional).
- [x] 0.4 Identified the four blockers: config pin `config.cpp:217-228`,
      MLA stride `:232`, unconditional executor MTP path
      (`executor.cpp:205-208, 1396-1413, 1610-1613`), unverified
      small-shape kernels. Effort estimate: ~3 dev-days,
      contingency +1-2 days for routed-kernel tile fixes (EXPERIMENTS E3).

## Phase 1 — Config profiles and loader (CPU) — done 2026-10-18

- [x] 1.1 Exact pin replaced by an anonymous-namespace allow-list
      (`constexpr GeometryProfile kProfiles[]`, `config.cpp`); profile match
      failure → `unsupported Ling 3.0 kernel geometry`, `profile_name` set.
      A later iteration added `mla_q_lora`/`mla_q_split` after the Flash
      artifact proved the fused export carries no `q_lora_rank` key
      (EXPERIMENTS row "Query-projection export differs per artifact").
- [x] 1.2 Per-profile MLA stride (Flash 6 + MTP tail, Tiny 4, no tail) kept
      in the profile record and cross-checked against the parsed
      `head_count_kv` placement in the match loop (profile asserts observed
      placement; no derivation).
- [x] 1.3 `nextn_layers=0` accepted through parse + `require_trunk`; the
      swiglu-array coverage checks stay inert when the keys are absent, and
      consumers clamp via bounds-safe `Config::ClampExp/ClampShexp`
      (EXPERIMENTS row "Tiny lacks SwiGLU clamp arrays").
- [x] 1.4 Config unit test: Tiny fixture (measured table) parses to profile
      Tiny with split-q asserts; Flash fixture asserts fused (rank key
      absent, 256/128 deviations + Tiny-without-rank rejected); mixed
      geometries rejected. `ling3_flash.config` Passed.
- [x] 1.5 Weights allowlist: split `attn_q_a`/`attn_q_a_norm`/`attn_q_b`
      bound for Tiny profile (fused path kept for Flash), `ssm_beta` pin
      relaxed to the dense type set (Tiny ships Q8_0). Real-file binding:
      `GUFO_LING3_FLASH_GGUF=…ling3_flash_weights_test` passes for BOTH
      artifacts (`Ling 3.0 weight binding passed.`).
- [x] 1.6 Gate: `ctest --test-dir build/cpu-test -R '^ling3_flash'` →
      config + chat_template Passed, weights/reference Skipped (no env);
      both artifact gates executed manually incl. the oracle
      (`Ling 3.0 scalar reference passed.`); `check-format.py` PASS.

## Phase 2 — MTP-free executor — done 2026-10-18

- [x] 2.1 MTP weight-validation list gated on `nextn_layers > 0`
      (`MtpWeights::Bind` returns nullopt at nextn 0; upload passes `nullptr`).
- [x] 2.2 MTP buffers/helpers gated via `has_mtp_`; no
      `num_layers_all == num_layers + 1` assumptions left (engine gates on
      `nextn_layers`, `engine.cpp:56-57`). Low-rank-q buffers
      (`mla_q_latent_`, `pf_mla_q_lat_`) allocate only under `mla_q_split`
      so the fused profile performs no 0-byte allocations.
- [x] 2.3 CPU oracle + weights tests are profile-driven (MTP optional,
      split shapes); Flash outputs unchanged — both artifact gates pass.
- [x] 2.4 Device compile verified with `cmake --build --preset release
      --target gufo` (links clean; gpu-test preset builds the same sources,
      not duplicated).

## Phase 3 — Kernel shape audit and fixes

> 2026-10-18 status note: the split-/low-rank-q device paths (decode
> `Gemv` chain, batch `GemvRows` chain, WMMA prefill split) and the
> 128-expert / 16-head shapes execute correctly end-to-end (Tiny oracle +
> binding gates green, GPU greedy smoke coherent, Flash artifact gates
> unchanged). Dedicated standalone probes per item below were not needed so
> far; keep them for any future artifact whose shapes stress the tiles.

- [ ] 3.1 Routed Q4_K/GEMV walks with expert_ff 512 = 2 super-blocks/row
      (Flash 768 = 3): verify super-block loop genericity, then probe
      `tools/bench/ling3_routed_bench.hip` at tiny shapes for bit-exact vs
      fp32 oracle; fix + Fnv evidence if wrong. → EXPERIMENTS.
- [ ] 3.2 KDA 16-head / 2048-extent qkv+conv paths: grid/block shapes;
      tiny-shape probe vs `reference.cpp` CPU ops. → EXPERIMENTS.
- [ ] 3.3 MLA absorbed 16-head prefill/decode tiles (fused prefill kernel
      exec-tile 32×256 assumes how many heads?): probe + fix. → EXPERIMENTS.
- [ ] 3.4 Dense FFN 4608 tile maps (hidden 1536): `DenseF16Gemm` plan
      lookup + `Gemv*` shapes. → EXPERIMENTS.
- [ ] 3.5 Router / buckets with 128 experts: bucket-size caps,
      `RoutedCompactRows(cap*heads, ...)` sizing, exp_probs_b + norm +
      scale-2.5 path (Tiny-only flags!) — Flash must stay exact.
- [ ] 3.6 Gate: full Flash routed/attention quality suites unchanged
      (Fnv sums, oracle ratios per QUALITY.md).

## Phase 4 — Quality and E2E on hardware

- [x] 4.1 GPU load smoke: `gufo prompt` greedy run passed exclusive-GPU
      (load 2.7 s, coherent 64-token answer, exit 0; EXPERIMENTS row
      "Tiny GPU enablement smoke + pp/tg"). `gufo serve` loaded Tiny and
      served chat + tool completions for the full replay (2026-10-18
      14:15, report rows below).
- [x] 4.2 Greedy logit parity vs `reference.cpp` on fixed prompts
      (top-1 + KL, Flash-harness style). History: first run FAILed at
      every MLA boundary (prefill-through-17 logits relative 2.71) and
      led to a production prefill v_b gather-row fix — full trace in
      EXPERIMENTS row "Prefill MLA v_b contracted over token rows".
      Gate PASS after the fix (exclusive `parity2`, 2026-10-18 14:20,
      forward + quality rc=0): step logits worst relative 0.0024, replay
      0.00095; prefill-through 17/39/40 logits 0.0506/0.1125/0.0477,
      hidden 0.0412/0.0450/0.0277, argmax matches at every checkpoint.
- [x] 4.3 pp/tg benchmarked (superseding the pre-fix smoke): full grid on
       `Ling-3.0-tiny-Q8_0.gguf`, exclusive, `-r 2`/`-r 3`
       (pp2048 5361.05, pp4096 4454.50, pp8192 3266.39, pp16384 2020.17;
       cold tg128 107.13, tg1024 99.60 tok/s) — recorded in
       [BENCHMARKS.md](BENCHMARKS.md), log `gufo_bench_bgrid10.log`.
       The same window re-ran the Flash grid: every retained Flash cell at
       or above its `p6grid` value — no regression.
- [x] 4.4 Tool-call E2E replay (pattern of the Flash driver): 12/12 checks
      PASS on Tiny (and re-confirmed 12/12 on Flash) with thinking on,
      exclusive, 2026-10-18 14:15 — non-stream `finish_reason=tool_calls`,
      single `get_weather` call with parsed `location`, framing
      containment, tool-result followup answers without markers; stream
      id-then-name once, incremental argument deltas, framing containment.
      Exposed and fixed a shared serve bug (every constrained request was
      400 `reasoning end marker is empty` — see EXPERIMENTS row "Serve
      constrained requests 400"; fix covered by `text_model_runner_test`).
      Kept artifacts: `/tmp/opencode/replay_report_TINY.json`,
      `replay_report_FLASH.json`, `replay_driver_{TINY,FLASH}.out`,
      `replay_server_{TINY,FLASH}.log`, `gufo_bench_replay8.log`.
- [x] 4.5 Docs/labels: model README support text (Known gaps struck),
      [ling3-flash README](../ling3-flash/README.md) links Tiny; GPU/unit
      tests inherit the registered `ling3_flash` labels (no TESTING.md
      per-model registry; Flash launch precedent).
- [x] 4.6 Final (2026-10-18): `ctest build/cpu-test` **131/131 PASS**
      (external-model tests skip without fixtures);
      `gpu-full -R ling3` exclusive **13/13 PASS** including
      `weights`/`reference`/forward/inference/quality on the real Flash GGUF
      (log `/tmp/opencode/gufo_bench_ling3full2.log`); the same forward/quality
      pair re-passed with the env pointed at the Tiny file (parity2 run).
      tool-call replay re-verified post constraint-core fix: TINY 12/12,
      FLASH 12/12 (`gufo_bench_replay9.log`, `replay_report_{TINY,FLASH}.json`);
      `check-format.py` PASS. The sweep exposed `json_constraint_test` red since
      the gemma4 reasoning-grammar commit — root cause, fix and retuned
      expectations in EXPERIMENTS ("json_constraint_test red since b996f776").

## Result summary

Effort budget from PLAN: ~3 working days, contingency +1–2 d if Phase 3
needs tile specializations. 2026-10-18: Phases 1–2 complete plus the GPU
grid (pp2048 5361.05, cold tg128 107.13 t/s; [BENCHMARKS.md](BENCHMARKS.md)). The enabling discovery was that the
query projection export differs per artifact — fused on Flash with no
`q_lora_rank` metadata, low-rank split with a norm on Tiny — and that Tiny
carries neither MTP tensors nor SwiGLU clamp arrays; all three are now
profile-driven. Greedy parity first FAILed at MLA prefill boundaries and
uncovered a production v_b gather-row bug (latent for Flash too); after
the fix the forward/quality gate passes and serve tool-call replay is
12/12 on both Tiny and Flash, including a shared serve fix for the empty
reasoning-close marker. The final sweep (4.6) is green: cpu 131/131,
gpu-full ling3 13/13, replay re-passed on the rebuilt constrained-decoding
core which the sweep un-broke (see the EXPERIMENTS regression row).

# NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB — Implementation Status

Working memory lane for bringing `nemotron_h_moe` up in Gufo on gfx1151.
Update every phase transition; keep evidence pointers, not prose. Companion
files: [EXPERIMENTS.md](EXPERIMENTS.md), [BENCHMARKS.md](BENCHMARKS.md).

Branch: `feat/support_nvidia_nemotron_bolt`. Artifact:
`~/projects/llm/models/gguf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/`
(NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf ~23 GiB IQ2 mixed +
mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf CLIP/C-RADIO, `clip` arch,
projector `nemotron_omni`).

Legend: `[x]` done+verified · `[~]` in progress · `[ ]` todo · `[!]` blocked.

Tooling: inspect GGUFs with the repo venv `.venv/bin/python` (has `gguf`
0.19; `ReaderTensor` attrs are `.name/.shape/.tensor_type.name/.data`), run
test binaries/ctest with `LD_LIBRARY_PATH=/opt/rocm/lib` on this host (built
binaries carry no RPATH for ROCm libs), and run every GPU benchmark through
`tools/bench/gpu_exclusive.sh` or the `bench_bg.sh` + `bench_wait.sh` pair in
one tool call (AGENTS protocol).

## Golden rules (this model)

- **Control-token safety.** Never emit a raw model control token (e.g. the
  `
` endoftext / EOS literal) in edits or tool output: it clashes with
  the agent's own stop token and truncates generation (seen 2026-10-18 as a
  stray NUL byte in this file). Spell it as the ASCII placeholder `EOSLIT`
  while implementing; at the very end run a specialized Python script to bulk
  replace `EOSLIT` -> the real control sequence, then verify zero placeholders
  remain. Same rule for `im_start`/`im_end`/tool-call literals: keep them in
  placeholder form until that final pass.
- **Model-private code only.** All new kernels and quant support (NVFP4 GEMM,
  Mamba-2, ReLU2 MoE, IQ2_XXS) live under `src/models/nemotron_h_moe/`. Never
  modify existing model trees (`deepseek_v4_flash`, `qwen*`, ...) to add
  nemotron capability — vendor the needed subset instead (each model already
  keeps its own mmq copy; precedent `qwen38_flash_next/kernels/rocm/mmq/`).
  Additive core infra (type id in `GgmlType`, size cases) is allowed only when
  no existing model's behavior changes.

## Facts nailed from the GGUF (verified 2026-10-18, `file_type=19`, 843 tensors)

- 88 layers, all RMS `attn_norm` pre-norm + residual; exactly one job per layer:
  - 40 Mamba-2 layers (n_head_kv=0, n_ff=0): `ssm_in` 4096→18560 NVFP4
    (+`.scale`), conv1d×4 over 10240 (x‖B‖C) + f32 bias, `ssm_a`/`ssm_d`/
    `ssm_dt.bias` (1,128) f32 one-per-head ⇒ **heads 128, headdim 64,
    dstate 128** (B/C = 8 groups × 128; `ssm_norm` (1024,8) over
    heads×headdim), `ssm_out` 8192→4096 NVFP4 (+`.scale`).
  - 8 NoPE attention layers at 7,16,25,36,47,58,69,78 (llama.cpp
    `LLAMA_ROPE_TYPE_NONE` for `nemotron_h_moe`): full 4096 q, GQA kv
    `head_count_kv=2`, head_dim 128, Q8_0, kq_scale 1/√128.
  - 40 latent-MoE-only layers (no attention/mamba): router `ffn_gate_inp`
    4096→n_expert_layer (heterogeneous: 240×12, 432×10, 368×9, 304×6, 496×3)
    + `exp_probs_b`, sigmoid gating, top-k=9 per layer-array; experts IQ2_XXS
    with **no gate matrix**, ReLU² act (`LLM_FFN_RELU_SQR`), per-layer
    `n_ff_exp` from `expert_feed_forward_length` array (2048/1536/…);
    `expert_weights_norm=true`, `expert_weights_scale=5.0`; shared expert
    IQ2_XXS 4096↔5376 (up/down only); latent down/up Q8_0 4096↔1024
    (`moe_latent_size=1024`) around routed experts only.
- dtype ids in this file: 8=Q8_0 (1.0625 B/elem verified from offsets:
  attn, latent, embd, output), 16=IQ2_XXS (2.0625 bpw), 40=NVFP4; NVFP4 block
  = 16-elem UE4M3 scales + packed fp4 (vendored mmq `block_nvfp4`).
- d_in_proj = 2·8192 (x,z) + 2·8·128 (B,C) + 128 (dt) = 18560; conv on xBC
  (10240 channels, kernel 4).
- Tokenizer: `gpt2` BPE, `pre=pixtral`, vocab 131072, eos=11 (`<end_of_turn>`-family),
  add_bos=false; jinja chat template 12.5 KB with `enable_thinking` macro +
  tool calling.
- Memory semantics (llama.cpp hybrid filters): attention cache iff
  `!is_recr && n_ff==0`; recurrent state iff `is_recr && n_ff==0`; MoE-only
  layers hold **no cache**.
- Reference: llama.cpp `origin/master@ff5a23376` (re-fetched 2026-10-19 for
  the parity gate; includes Nemotron Bolt support on top of the earlier
  5aeac37a3) `src/models/nemotron-h.cpp` (trunk graph + hparams + tensors)
  and `nemotron-h-moe.cpp` (MTP drafter head only; LiLiCorr GGUF not
  downloaded). Mamba-2 base: `src/models/mamba-base.cpp`.

## Phase 0 — Analysis and reference setup

- [x] GGUF metadata + tensor-layout dump (`/tmp/opencode/dump_gguf_meta.py`,
  layer classification script; counts above).
- [x] mmproj metadata dump (CLIP, 32 blocks, embd 1280, patch 16, image 512,
  `nemotron_omni`, `scale_factor=2`, min/max pixels).
- [x] llama.cpp reference fetched (`.../llama.cpp/llama.cpp`, `origin/master`
  = 5aeac37a3); graph refs saved to `/tmp/opencode/ref_nemotron-h.cpp`,
  `ref_nemotron-h-moe.cpp`, `ref_mamba-base.cpp`, `ref_llama-model.cpp`.
- [x] Gufo integration map (dispatch sites, module template, ops inventory,
  test harness) — see Decision log 2026-10-18.

## Phase 1 — Core plumbing (CPU-visible)

- [x] Core type plumbing (additive, all existing tests green
  `ctest -R 'gguf|quant'` in cpu-test): NVFP4=40 in `GgmlType`
  (`src/core/gguf_reader.hpp`) + `ToString`, `QuantizedBlockElements`=64,
  `QuantizedRowBytes` 36 B/block (4×UE4M3 + 32 B fp4, matches vendored mmq
  `block_nvfp4`). CPU dequant **not** added to core per Golden rule — lives in
  the model folder (Phase 2 oracle).
- [x] `src/models/nemotron_h_moe/config.*`: heterogeneous arrays
  (`n_ff[i]`, `n_head_kv[i]`, `n_expert_used[i]`, `n_ff_exp[i]`), per-layer
  kind Attn|Mamba|Moe from arrays, no `rotary_dim` requirement (NoPE OK).
  Verified against the real 24GB GGUF (ad-hoc harness): 88 = 8 attn + 40
  mamba + 40 MoE; heads 128 (dt width), headdim 64, in_proj 18560 and conv
  10240 match the tensor table exactly. Registered as `gufo_nemotron_h_moe`
  static lib (top-level `add_subdirectory`, unconditional like ling3_0).
- [x] `weights.*`: tensor binder (ling3_0 pattern) binding all 88 blocks and
  validating shape/format; whitelist F32/Q8_0/IQ2_XXS(66B/256)/NVFP4(36B/64).
  Per-layer expert counts read from router tensor rows (240..496, metadata
  only stores the max). NVFP4 trunk `.scale` tensors ARE required and bound
  (`ssm_in.scale`, `ssm_out.scale`). Corrected 2026-10-19: `ffn_latent_up/
  down.weight` are standard `[output rows × reduction cols]` row-major like
  every other projection (the earlier "stored output-first, must transpose"
  note was wrong); bound and consumed as plain mat-vecs, verified by the
  real-GGUF CPU oracle.
- [x] `tokenizer`: GPT-2 BPE + `pre=pixtral` pre-tokenization (model-private,
  gemma4 precedent); staged-EOS contract — `Encode()` never emits the raw
  EOS literal, `<GUFO_EOT_STAGE>` sentinel + `FinalizeEndOfTurnText()` swap.
  `nemotron_h_moe.tokenizer` ctest green.
- [ ] `chat_template` + control tokens: parse/validate the GGUF jinja
  semantics (enable_thinking, <|im_start|>/EOSLIT, tool
  <tool_call> JSON); model-private constants file (AGENTS rule: do not
  reuse Qwen constants).

## Phase 2 — CPU reference oracle (`reference.cpp`, EXCLUDE_FROM_ALL)

- [x] Forward per layer-kind with dequant-to-f32 matvec (cpu_ops); Mamba-2
  recurrence (softplus dt, clamps per reference, D skip, group-norm, z gate),
  sigmoid+bias router top-9 → norm → ×5.0, ReLU² experts, shexp add, latent
  down/up order; NoPE attention with GQA. Semantics mirrored from pinned
  llama.cpp (`mamba-base.cpp` build_mamba2_layer, CPU ssm_conv/ssm_scan/swiglu,
  ggml-quants dequants vendored in `cpu_ops.cpp` + `cpu_tables.hpp`).
- [x] Unit tests: `tests/models/nemotron_h_moe/reference_test.cpp` — in-memory
  synthetic 4-layer trunk (mamba/attn/moe) with snapshot goldens, determinism
  (bit-identical `Reset()` replay), order-sensitivity, context-capacity and
  error paths; real-GGUF greedy probe behind `GUFO_NEMOTRON_H_MOE_GGUF`
  (skip 77). llama.cpp parity gate GREEN vs `ff5a23376`: two IQ2_XXS dequant
  bugs fixed in `cpu_ops.cpp` (sub-block `qs` byte stride 4→8; per-`ib32`
  output advance so writes append instead of rewriting each block's first 32
  floats); oracle IQ2 output now bit-matches `dequantize_row_iq2_xxs` on raw
  expert bytes. See ledger 2026-10-19 parity rows.

## Phase 3 — HIP engine (`kernels/rocm/`)

- [ ] Mamba-2 kernels: decode recurrence + chunked prefill
  (dt A/d discretization; no deltanet reuse); conv-history state tail shared
  with cache design.
- [ ] MoE: reuse vendored mmq/mmid/moe_wave64 path (already IQ2_XXS+NVFP4
  capable) for grouped expert gemm; ReLU² fused act; router top-k kernel with
  sigmoid+bias and per-layer expert counts; latent up/down dense gemms (Q8_0).
- [ ] NoPE attention: attention prefill/decode minus rope (gemma4 closest).
- [ ] Session state: per-layer tagged caches (KV for 8 attn layers,
  conv+ssm state for 40 mamba layers), snapshot/rollback for speculative later.
- [ ] Engine `Model::Load/CreateSession` + executor prefill/step.

## Phase 4 — Runners + dispatch

- [ ] `cli_runner` (IsNemotronHMoe, Load, Generate, RunBenchmark),
  `serve_runner` (`LoadServeModel`, `CreateTextRunner`).
- [ ] Dispatch entries: `src/cli/prompt/prompt.cpp`, `src/cli/serve/inference_backend.{hpp,cpp}`,
  `src/cli/bench/bench.cpp`, `src/core/text_sampling_defaults.hpp`
  (temp 1.0 / top_p 0.95 from GGUF).
- [ ] CMake: `src/models/nemotron_h_moe/CMakeLists.txt` + add_subdirectory
  (HIP-gated) + `gufo_llm_cli` link + `gufo` serve_runner list.

## Phase 5 — Tests + exclusive-GPU verification

- [ ] `tests/models/nemotron_h_moe/` + CMake; targets `nemotron_h_moe.<case>`,
  labels `nemotron_h_moe;cpu` / `...;hip;gpu;gfx1151`; external-model cases
  behind `GUFO_NEMOTRON_H_MOE_GGUF` (skip 77).
- [ ] `python3 tools/ci/check-format.py` clean before every commit.
- [ ] gpu-test preset: `ctest --preset gpu-full -R '^nemotron_h_moe'`.
- [ ] Functional smoke through `gufo serve` + `tests/functional` wiring
  (impact.json profile `bolt-q2`, vision:false initially).
- [ ] All GPU benchmarks via `tools/bench/gpu_exclusive.sh` or
  `bench_bg.sh`+`bench_wait.sh` pair in ONE tool call (AGENTS protocol);
  iterate ≤16K context first.

## Phase 6 — Deferred (documented scope cuts)

- [ ] Vision: `nemotron_omni` CLIP projector + C-RADIO tower in mmproj path
  (gemma4 model-private vision precedent). Text-only first.
- [ ] MTP/LiLiCorr speculative decoding (drafter GGUF not local; MTP-on-hybrid
  has no Gufo precedent until state rollback lands).
- [ ] cache::Adapter continuation (qwen36 precedent: serve without adapter
  first).

## Verification ledger

| date | revision | check | result |
| --- | --- | --- | --- |
| 2026-10-18 | pre-impl | GGUF parse, arch facts vs llama.cpp master | consistent (no IQ2 size drift; Q8_0 B/elem exact) |
| 2026-10-18 | feat branch | NVFP4 mmq hardware probe (exclusive GPU): dense-vec N=1/3/8 + dense gemm tile boundaries vs exact f32 — probe now at `src/models/nemotron_h_moe/tools/nvfp4_probe.hip` (`nmb_` ABI) | 7/7 PASS, rel/scale ≤ 3.8e-3 (Q8_1-act floor); ssm_in-shape 5.27 ms, ssm_out-shape 3.01 ms |
| 2026-10-18 | feat branch | Core NVFP4=40 plumbing; `cmake --preset cpu-test` build + `ctest -R 'gguf\|quant\|dequant'` | 297/297 built, 3/3 pass |
| 2026-10-18 | feat branch | `Config::FromGguf` on the real 24GB GGUF; derived shapes vs tensor table (`.venv` gguf 0.19 sweep of all 843 tensors) | PASS: layer mix 8/40/40, heads 128 headdim 64, in_proj 18560, conv 10240; expert counts {240,304,368,432,496} × ff {1536,2048,2304,2560} |
| 2026-10-19 | feat branch | `ModelWeights::Bind` on the real 24GB GGUF (ad-hoc harness, per-layer shape/format/offset check); latent orientation fixed to stored [ne0=1024/4096] | PASS: 88 layers bound, payload 20.50 GiB, all shapes/formats match tensor sweep; `check-format` PASS |
| 2026-10-19 | feat branch | Latent orientation re-check: `ffn_latent_{up,down}` are standard [output rows × reduction cols], no transpose; weights.cpp comment corrected; oracle/real-GGUF runs confirm | PASS (supersedes the stored-output-first claim above) |
| 2026-10-19 | feat branch | CPU oracle `nemotron_h_moe.reference`: synthetic trunk — finite/non-degenerate logits, bit-identical `Reset()` replay, order sensitivity, capacity + error paths, locked goldens (tok-argmax 3 / sum 0.793073; seq 2 / 5.106880); `ctest -R '^nemotron_h_moe\.'` = tokenizer Pass + reference Skip; `check-format` PASS (756 files) | PASS |
| 2026-10-19 | feat branch | Real 24GB GGUF oracle probe (exclusive RAM via bench_bg/bench_wait): 3-token greedy steps + replay — argmax 103288 → 48756 → 70587, logit sums -436497/-355020/-283120, all finite, replay bit-stable; evidence `evidence/reference-cpu-oracle-2026-10-19.log` | SUPERSEDED by the two `DequantizeIq2Xxs` fixes below — those sums/argmax came from the broken IQ2 path |
| 2026-10-19 | feat branch | IQ2_XXS dequant fixes in `src/models/nemotron_h_moe/cpu_ops.cpp` vs ggml ground truth (byte-level probe on real expert rows decoded by `dequantize_row_iq2_xxs`): sub-block code stride `qs+4*ib32` (uint8) → `qs+8*ib32` (ggml field is `uint16_t qs[32]`); output pointer advanced `+32` per `ib32` (was rewriting each 256-block's first 32 floats, leaving 87.5% zeros). Post-fix oracle matches ggml element-for-element (up_r0 norm 0.427721 nz 2048/2048; shexp norm 0.548900). check-format PASS | PASS; nemotron ctest suite 2/2 green (with `LD_LIBRARY_PATH=/opt/rocm/lib`) |
| 2026-10-19 | feat branch | llama.cpp parity (exclusive RAM, probe @ pin `ff5a23376` incl. Nemotron Bolt support): per-layer bisection `parity_layers.py` — all 88 layer inputs match cos ≥ 0.99 for 3 steps; logits `parity_compare.py` (gate now top-100 rel ≤ 5%, cos ≥ 0.99, argmax): step argmax 1044/1010/1010 MATCH both sides, top-100 rel ≤ 0.0295, cos ≥ 0.999988, top-10 overlap 9–10; tail max_rel(all) ≤ 0.074 is summation-order noise on rank>100 logits. Evidence `evidence/llama-parity-{run-parity5-full,layers-parity5,compare-parity5}.log` | PASS — oracle valid as Phase-3 quality reference |

## Blockers / risks

- IQ2_XXS×fp16 activation grouped GEMM perf for 2–9B active params is the
  decode bottleneck hypothesis (240–496 experts/layer, top-k 9); validate with
  `tools/bench/moe_gemv_bench.hip` patterns early.
- Heterogeneous per-layer expert counts (240…496) — mmid launch params must be
  per-layer, not global `expert_count=496`.
- [resolved 2026-10-18] NVFP4 hardware path: probed end-to-end (generic Q8_1
  MMQ/MMVQ path via `mul_mat_q_case<GGML_TYPE_NVFP4>`, no FP4 tensor cores on
  gfx1151 — see ledger). Proven with a temporary additive instantiation in
  deepseek's mmq, since **reverted** per Golden rule; the same instantiation
  will be vendored into `src/models/nemotron_h_moe/kernels/rocm/mmq/`.
  Tensor-scale (`ssm_in.scale`/`ssm_out.scale`, per-tensor f32) is applied as
  a nemotron-side output epilogue multiply, never inside vendored mmq common
  code paths shared with other quant types.
- gpt2/pixtral pre-tokenizer fidelity vs HF (BPE regressions are silent).

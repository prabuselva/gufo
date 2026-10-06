# Gemma-4-26B-A4B (`gemma4`) — Development Plan & Log

Goal: implement GGUF arch `gemma4` (trunk), `gemma4-assistant` (MTP draft) and
the `gemma4v` vision projector in Gufo on Strix Halo `gfx1151`, isolated in
`src/models/gemma4` with model-private ROCm WMMA kernels, a CPU scalar oracle
and GPU parity tests. Each phase below is self-contained and auto-continuable:
it names its deliverables, its focused check, and its exit criterion. Update
the Status log at the bottom as phases land.

## Build / test environment

```sh
export PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export GUFO_GEMMA4_GGUF=/home/praburaja/projects/llm/models/gguf/Gemma4-26B-A4B-IT/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf
export GUFO_GEMMA4_MTP_GGUF=/home/praburaja/projects/llm/models/gguf/Gemma4-26B-A4B-IT/mtp-gemma-4-26B-A4B-it-Q8_0.gguf
export GUFO_GEMMA4_MMPROJ_GGUF=/home/praburaja/projects/llm/models/gguf/Gemma4-26B-A4B-IT/mmproj-BF16.gguf

cmake --preset gpu-test -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build --preset gpu-test --target <target> --parallel 4
ctest --preset gpu-full -R '^gemma4\.' --output-on-failure
```

GPU benchmarks run exclusively through `tools/bench/gpu_exclusive.sh` (see
AGENTS.md); iterate levers at ≤16K context first. The benchmark reference
column is the local llama.cpp fork (`/home/praburaja/projects/llm/llama.cpp/
llama.cpp`, ROCm build) which carries the ground-truth graphs
`src/models/gemma4.cpp`, `gemma4-assistant.cpp` and
`tools/mtmd/models/gemma4v.cpp`.

## Phases

### M1 — Config (`src/models/gemma4/config.{hpp,cpp}`)

Parse and validate `gemma4.*` / `gemma4-assistant.*` metadata: layer count,
hidden, vocab (from `tokenizer.ggml.tokens`), context, RMS eps, logit
softcap, per-layer `attention.head_count_kv` and
`attention.sliding_window_pattern`, dual rope bases/dimensions
(`rope.freq_base[_swa]`, `rope.dimension_count[_swa]`), key/value lengths
(512 full / 256 SWA), sliding window 1024, `shared_kv_layers`, expert
geometry (128 / top-8 / 704) and shared FFN 2112,
`embedding_length_per_layer_input == 0`, draft `embedding_length_out`.
Derived per-layer accessors: `IsSwa`, `HeadDim`, `RopeTheta`,
`AttentionQDim/KvDim`, `HasVProjection` (SWA only), `HasKv`
(`il < num_layers - shared_kv_layers`). `DraftMatches(trunk)` locks the
KV-share contract (draft SWA → target layer n−2, draft full → n−1, equal
head dims/kv counts/thetas/vocab). Lock the validated kernel geometry and
reject other profiles, as qwen36 does.

Check: `gemma4_config_test` (synthetic in-memory GGUF, CPU-only).
Exit: valid metadata parses; malformed/wrong-geometry metadata rejected.

### M2 — Weights (`weights.{hpp,cpp}`)

Tensor inventory and raw-block binding (no dequant): per-layer refs for
`attn_norm/q/k/v(0-28)/q_norm/k_norm/output`, the norms
(`post_attention_norm`, `ffn_norm`, `post_ffw_norm`, `pre_ffw_norm_2`,
`post_ffw_norm_1`, `post_ffw_norm_2`), `layer_output_scale`, shared FFN, router
(`ffn_gate_inp` F32 + `ffn_gate_inp.scale` F32 [2816]), expert stacks
(`ffn_gate_up_exps`, `ffn_down_exps` + `.scale`), global `output_norm`,
tied `token_embd` (lm_head), global `rope_freqs[256]`; draft tensors
(`nextn.pre_projection`, `nextn.post_projection`, draft `token_embd`,
draft norms/FFN). Accept F32/Q8_0/BF16 per tensor (layer 29 is BF16);
validate shapes against the config.

Check: `gemma4_weights_test` (headers only, skips 77 without
`GUFO_GEMMA4_GGUF`/`GUFO_GEMMA4_MTP_GGUF`).
Exit: every tensor name/shape/type resolves on both artifacts.

### M3 — CPU oracle (`reference/`)

Plain-C++ forward: embedding scale √2816, RMSNorm, per-head Q/K norms,
proportional rope (full layers, `rope_freqs` factors), V = rms_norm(wk·x) on
full layers / rms_norm(wv·x) on SWA, windowed + full attention, softmax
router top-8 (no bias), shared GELU-parallel FFN + expert FFN, the seven-norm
residual flow (`post_ffw_norm` on the combined FFN output before the residual
add), `layer_output_scale`, final norm → lm_head → softcap
(×1/30 → tanh → ×30). Also the draft step (pre/post projections, shared-KV
read, no softcap on draft logits).

Check: `gemma4_reference_test` (synthetic config unit tests + a few real
tokens via `GUFO_GEMMA4_GGUF`, slow, skips 77).
Exit: oracle produces coherent greedy text on the real artifact.

### M4 — Tokenizer (`tokenizer.{hpp,cpp}`)

SPM-style BPE: merge-rank table from 514 906 `tokenizer.ggml.merges`, raw
UTF-8 (no GPT-2 byte encoding), normalizer space → `▁` + clean spaces,
pre-split on newlines only, byte fallback through the 256 type-6 byte
tokens, special-token extraction (`<|turn>`, `<turn|>`, `<|image|>`,
`<|tool>`, `<tool|>`, `<|think|>`, …), add_bos. Decode with `▁` → space.

Check: `gemma4_tokenizer_test` — golden token vectors generated from the
llama.cpp fork (`llama-cli --tokens`) pinned in the test.
Exit: round-trip and golden parity on prose, code, CJK and newline runs.

### M5 — ROCm kernels (`kernels/rocm/`, docs in `docs/kernels/gemma4/`)

- `gemv.{hpp,hip.cpp}` — decode mat-vec, Q8_0/BF16/F32.
- `gemm.{hpp,hip.cpp}` — prefill: Q8_0 → `qfn_mmq_q8_0_dense`
  (`gufo_qwen38_flash_next_mmq`), BF16 → hipBLAS GemmEx, F32 → SGEMM.
- `routed_f16.{hpp,hip.cpp}` — port of the qwen36 WMMA MoE
  (dequant-after-LDS, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) to
  gemma4 geometry: `ffn_gate_up_exps [2816, 1408, 128]`
  (gate rows [0, 704), up rows [704, 1408)), `ffn_down_exps` with
  per-expert `.scale`, softmax weights applied in the down epilogue; accepts
  Q8_0 **and BF16** experts (layer 29). The source's fused gate+up `kPair`
  variant and `swiglu_gate` epilogue were stripped — gemma4 uses gelu-tanh
  via `GegluF16` between the two GEMMs.
Check: `gemma4.rocm_kernels` (synthetic, vs CPU formulas). Exit: each kernel
matches its CPU reference within the [QUALITY.md](QUALITY.md) tolerances.
Attention kernels ship with the M6 executor.

### M6 — Device model + executor + parity

`attention.{hpp,hip.cpp}` — fused per-head Q/K RMSNorm + NEOX rope
(proportional on full layers), V built from `attn_v` (SWA) or the K
projection (full) with weightless RMSNorm, F16 position-major KV cache,
flash-decode split + combine over the sliding window, and a causal windowed
tiled prefill for both head_dim 256 and 512. A WMMA fast path is deferred to
M11 tuning.

`device_model.{hpp,cpp}` (weight upload, native encodings),
`executor.{hpp,cpp}` (layer loop, per-class KV caches sized to the context,
positions, logits download via `hipMemcpy`).

Check: `gemma4_rocm_forward_test` (GPU logits vs oracle on real weights,
skips 77). Exit: parity within tolerance; greedy tokens match the oracle.

### M7 — Engine + CLI + chat template

`engine.{hpp,cpp}` (`Model`/`Session` like qwen36), hand-compiled gemma4
chat template (`chat_template.{hpp,cpp}`: `<|turn>role\n…<turn|>`, tool
calls, thinking) with a golden parity test against the GGUF Jinja template
rendered by the llama.cpp fork; wire `prompt.cpp`, `bench.cpp`,
`serve/inference_backend.cpp` on `general.architecture == "gemma4"`.

Check: `gemma4_chat_template_test`, `gemma4_rocm_inference_test`; manual
`gufo prompt`/`gufo bench`. Exit: coherent chat incl. tool-call formatting;
first retained `gufo bench` numbers in BENCHMARKS.md.

### M8 — MTP draft (`gemma4-assistant`)

Draft engine on the separate GGUF: input concat(target tok_embd × √2816,
target `h_nextn`), 4 layers reading target KV read-only (SWA → layer 28,
full → layer 29), draft logits (no softcap), verify + adaptive draft bounds
(qwen36 session policy).

Check: `gemma4.mtp` (draft forward vs CPU oracle) and `gemma4.speculative`
(greedy token-identical vs no-MTP at 48 tokens).
Exit: acceptance ≥ llama.cpp fork on the benchmark corpus. Output is
distribution-preserving (committed tokens are the trunk's), not bit-identical
to non-spec greedy at long range — the batched verify that yields the +57%
tg128 win is not bit-identical to single-token decode (see QUALITY.md). MTP is
opt-in; the default greedy path is unchanged.

### M9 — Vision (`gemma4v`)

`vision/encoder.{hpp,cpp}`: conv patch-embed, x/y position get_rows, 2D RoPE
neox θ=100 (x dims [0, 36), y dims [36, 72)), per-head q/k norms, avg-pool
3×3 merge, std-norm, ClippableLinear (optional clamp scalars, ±FLT_MAX
default), BF16 weights; bicubic preprocessing, `inp_raw × 2 − 1`; `<|image|>`
wiring in prompt/serve.

Check: `gemma4_vision_test` (encoder parity vs oracle on a fixed image).
Exit: image prompts answer correctly end-to-end.

### M10 — Quantizations (Q8_0, Q4_K_M)

Q8_0 artifact loads natively (same tiers). Q4_K_M: routed experts Q4_K/Q5_K
through `routed_f16` (already supports both), Q6_K tensors upcast to Q8_0 at
load — port the qwen36 Q4_K_XL path.

Check: weights + forward tests on each artifact. Exit: BENCHMARKS.md gains
per-quantization rows.

### M11 — Performance tuning + benchmarks

gfx1151 tuning (BN bucket selection, BM=256 instantiation, lm_head GEMV,
attention efficiency, roofline analysis per
[OPTIMIZATIONS.md](OPTIMIZATIONS.md)); full sweep pp512→102400, tg128,
MTP n=1/2, serve; interleaved candidate↔llama.cpp-fork sessions.

Exit: BENCHMARKS.md complete; retained wins documented in
OPTIMIZATIONS.md/EXPERIMENTS.md.

## Status log

- 2026-10-05 M0: docs scaffold; architecture contract pinned from the GGUFs
  and the llama.cpp fork graphs (correction: SWA layers **do** have
  `attn_v`; only full layers derive V from the K projection).
- 2026-10-05 M1: `src/models/gemma4/config.{hpp,cpp}` (`gufo_gemma4`),
  `tests/models/gemma4/config_test.cpp` (`gemma4.config`, labels
  `gemma4;cpu`) pass; both real artifacts parse (trunk 30×2816, draft
  4×1024→2816) and `DraftMatches` validates the share contract.
- 2026-10-05 M2: `src/models/gemma4/weights.{hpp,cpp}` +
  `tests/models/gemma4/weights_test.cpp` (`gemma4.weights`, labels
  `gemma4;cpu;external-model`, skips 77) pass on both real artifacts; every
  trunk and draft tensor resolves with the pinned shape and a decodable
  format (correction: layers carry **seven** norms — `post_ffw_norm` applies
  to the combined FFN output before the residual add, then
  `layer_output_scale`).
- 2026-10-05 M3: `src/models/gemma4/{cpu_ops,reference}.{hpp,cpp}`
  (`gufo_gemma4_reference`, EXCLUDE_FROM_ALL, -O2) +
  `tests/models/gemma4/reference_test.cpp` (`gemma4.reference`, labels
  `gemma4;cpu;external-model;slow`, skips 77) pass on both real artifacts:
  3 trunk steps, draft step and replay are finite, non-degenerate and stable
  (10.9 s wall). Semantics pinned from the fork graphs: NEOX rope with the
  proportional `rope_freqs` table (64 active pairs, the rest pass through),
  attention scale 1.0, weightless V norm, geglu experts with renormalized
  top-8 softmax weights (sum clamp 6.1e-5) and per-expert down scales, draft
  concat order [target emb ×√2816, trunk h].
- 2026-10-05 M4: `src/models/gemma4/tokenizer.{hpp,cpp}` (in
  `gufo_gemma4`) + `tests/models/gemma4/tokenizer_test.cpp`
  (`gemma4.tokenizer`, labels `gemma4;cpu;external-model`, skips 77) pass:
  synthetic merge-engine checks plus nine golden vectors (ids and decode)
  captured from the llama.cpp fork (`vocab_only` load) on the real artifact.
  SPM-style BPE replicated: `[^\n]+|[\n]+` word split, newline-only words
  looked up directly, longest-first special partition, priority-queue merges
  (min rank, min left) with lazy staleness via recorded side lengths,
  per-byte `<0xXX>` fallback, `▁` space escape, add_bos forced true.
- 2026-10-05 M5 (kernels tier): `src/models/gemma4/kernels/rocm/`
  (`gufo_gemma4_kernels`): `gemv`/`gemm`/`routed_f16` ported from the qwen36
  tier (geometry-agnostic; arch tag renamed), new `fused.{hpp,hip.cpp}` with
  `RmsNormRows`, `FusedAddRmsNorm`, `Add`, `RouterTopK` (softmax top-8,
  renorm, sum clamp 6.103515625e-5 — identical to the oracle), `ExpertCounts`,
  `MoeEpilogue` (gemma4 variant: `ids` + per-expert `expert_scale`, no shared
  expert), and `GegluF32`/`GegluF16` (gelu-tanh × up over the fused
  [rows][2·704] gate/up tensor; the routed path never uses the SwiGLU
  epilogues). `tests/models/gemma4/rocm_kernels_test.cpp`
  (`gemma4.rocm_kernels`, labels `gemma4;hip;gpu;gfx1151`) passes: GEMV
  Q8_0/BF16, EmbedRow, GemvGrouped + fused-layout GemvGroupedPair, Geglu
  F32/F16, RouterTopK vs a double reference, norms/Add, Q8_0 mmq GEMM batch 8,
  and the full routed MoE prefill pipeline (compact → gate/up m=1408 →
  geglu → down m=2816 → weighted combine) at artifact geometry vs a double
  reference (scale-relative worst 5.9e-4). Synthetic Q8_0 matrices use a
  realistic weight scale so the F16 geglu intermediates stay far below the
  half ceiling, as with RMS-normalized activations in the real model.
  Attention kernels land with the M6 executor.
- 2026-10-05 M5 (kernel docs): `docs/kernels/gemma4/` — README (kernel map,
  geometry, tier selection), `gemv.md`, `gemm.md`, `routed-f16-moe.md`,
  `fused-ops.md` with the measured parity errors; gemma4 row added to
  `docs/kernels/README.md`. All SiLU/SwiGLU surface removed from the ported
  kernels (`GemvGroupedSwiglu`, `RoutedGatedF16Gemm`, `kPair`, `swiglu_gate`);
  `RoutedF16Gemm` no longer takes a gate pointer. Parity rerun clean.
- 2026-10-05 M6 (attention kernels): `kernels/rocm/attention.{hpp,hip.cpp}` —
  `QknormRope` (per-head RMSNorm + NEOX rope, grid tokens×heads),
  `KvNormRopeWrite` (K norm+rope, weightless V norm, F16 cache write at
  `start+token`), `AttentionDecode` (flash-decode split over 8 waves/chunk,
  partials `[head][split][head_dim+2]`, combine in double; splits of ≥64 keys
  capped at 32) and `AttentionPrefill` (causal windowed tiles kQT=32×kKT=8,
  one thread per head dimension, masked score −1e30 so a fully masked leading
  tile cannot poison the online softmax). Both classes (256/8/θ1e4,
  512/2/θ1e6) pass vs double references over the F16-rounded cache:
  QknormRope ≤1.5e-6 absolute, KvNormRopeWrite exact, decode ≤1.3e-6
  (dense and windowed), prefill ≤8.5e-6 scale-relative.
- 2026-10-05 M6 (device model + executor + forward parity):
  `kernels/rocm/device_model.{hpp,cpp}` (owns the F16 position-major KV cache
  `[max_ctx][kvh·hd]` per layer, weight buffers, scratch) and
  `kernels/rocm/executor.{hpp,cpp}` (per-layer forward: QKV → QknormRope →
  KvNormRopeWrite → AttentionDecode/Prefill → O-proj → the five-norm residual
  flow → shared FFN + routed top-8 MoE → `layer_output_scale`; head =
  RmsNorm(cur, output_norm) → h_out → MatVec(output) → softcap).
  `tests/models/gemma4/rocm_forward_test.cpp` (`gemma4.rocm_forward`, labels
  `gemma4;hip;gpu;gfx1151;slow`) runs the GPU executor and the double oracle
  side by side over `GUFO_GEMMA4_GGUF` prompts and compares `h_out` and logits
  at positions 0–4. Two bugs found and fixed: the `act_` scratch was sized to
  the shared `ffn` (2112) but the MoE `GegluF32` writes `used·expert_ff` (5632)
  — resized to `max(ffn, used·expert_ff)`; and `KvNormRopeWriteKernel` was
  missing a `__syncthreads()` before the rope rotate. The oracle KV cache is
  F16-rounded (`RoundF16`) to match the GPU cache. Parity contract is
  chaos-robust (finite + argmax exact + worst_abs ≤ 2.0, see QUALITY.md):
  observed worst_abs 0.008/0.743/0.15/0.083/0.062, argmax matching at every
  position, `h_out` worst_abs 0.041. Verified through the exclusive-GPU
  harness (`bench_bg.sh` + `bench_wait.sh`): all 6 `gemma4.*` tests pass.
- M7: engine/CLI wiring + chat template — done. `engine.{hpp,cpp}`
  (`Model`/`Session`, `EncodeChat` renders the pinned template then encodes);
  `chat_template.{hpp,cpp}` is a deterministic C++ renderer validated
  byte-exact against the GGUF Jinja template (10/10 golden vectors,
  `gemma4.chat_template`, incl. tool-call and thinking formatting). Wired
  `bench.cpp` (`RunGemma4Benchmark`), `prompt.cpp`
  (`RunGemma4Prompt`/`RunGemma4Chat`) and `serve/inference_backend.cpp`
  (`Gemma4TextRunner` + `load(shared_ptr<models::gemma4::Model>,…)` +
  `general.architecture == "gemma4"` dispatch; single session, no MTP/vision
  yet). Serve smoke over HTTP (exclusive GPU): greedy chat returns the correct
  answer with `finish_reason=stop`. First `gufo bench` numbers (Q8_K_XL, no
  MTP): pp2048 505.49 t/s, pp8192 260.60 t/s, tg128 38.26 t/s.
- M8: MTP draft — done. `DeviceDraft`/`Executor::DraftStep` (draft forward,
  fixed position, trunk KV read-only) match the CPU oracle (`gemma4.mtp`);
  `Executor::Verify`/`RollbackVerify` + `Session::DecodeStep` speculative path
  give greedy token-identical output at 48 tokens (`gemma4.speculative`). CLI
  `--speculative mtp --mtp-model` wired into `bench.cpp` and `prompt.cpp`
  (`LoadGemma4Model` + `GenerateGemma4Response` now drive `DecodeStep`). tg128
  (Q8_K_XL, exclusive GPU, greedy): no MTP 38.22, MTP n=1 36.60, n=2 46.93,
  n=4 59.99 t/s (+57%). MTP is opt-in; committed tokens are the trunk's, so
  output stays on-target but is not bit-identical to non-spec greedy at long
  range (batched verify; see QUALITY.md). Serve path still loads no draft.
- M9: vision (`gemma4v`) — CPU-oracle foundation done. `vision/config.{hpp,cpp}`
  parses `clip.vision.*` + the `gemma4v` projector and locks the geometry
  (hidden 1152, heads 16, head_dim 72, layers 27, eps 1e-6, rope θ=100, merge 3,
  patch 16, projection 2816, align 48); `vision/weights.{hpp,cpp}` binds the real
  mmproj (patch_embd rank-4 `[16,16,3,1152]`, position_embd rank-3
  `[1152,10240,2]`, rank-1 norms, rank-2 projections, `mm.input_projection`
  `[1152,2816]`); `vision/reference.{hpp,cpp}` is the full CPU oracle
  (`ReferenceEncoder::Encode`). `gemma4.vision_reference` (external-model, skip
  77) runs the oracle on the real mmproj: all tensors bind, output finite
  `[4×2816]`, checksum −53.0989. Remaining: HIP kernels (2D RoPE, geglu_quick
  `x·sigmoid(1.702x)`, avg-pool 3×3, std-norm, non-causal attention kq_scale=1;
  reuse `Gemm`/`RmsNormRows`/`Add`/`ScaleInPlace`), device encoder, bicubic
  preprocessing, `<|image|>` engine wiring, GPU parity test.
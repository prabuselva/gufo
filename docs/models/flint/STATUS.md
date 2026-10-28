# Flint Preview 40B-A2B — Implementation Status

Working memory lane for bringing `flint` up in Gufo on gfx1151. Update every
phase transition; keep evidence pointers, not prose. Companion files to be
created with the runtime: EXPERIMENTS.md, BENCHMARKS.md.

Branch: `feat/support_flint_moe`. Artifact:
`~/projects/llm/models/gguf/Flint-40B-A2B-GGUF/flint-Q8_0.gguf`
(41.7 GiB, 779 tensors, Q8_0 + imatrix, `general.file_type=7`).

Legend: `[x]` done+verified · `[~]` in progress · `[ ]` todo · `[!]` blocked.

Tooling: inspect GGUFs with `~/projects/llm/models/venv/bin/python` (gguf-py;
`f.contents()` callable), run test binaries/ctest with
`LD_LIBRARY_PATH=/opt/rocm/lib` on this host, and run every GPU benchmark
through `tools/bench/gpu_exclusive.sh` or the `bench_bg.sh` + `bench_wait.sh`
pair in one tool call (AGENTS protocol).

## Golden rules (this model)

- Commit at every checkpoint; update this file in the same commit.
- Model-private code only: everything lives under `src/models/flint/`; reuse
  shared core/hip helpers, never edit other model trees.
- Control-token literals (`<|...|>`, XML tool tags) stay as ASCII-safe
  placeholders in docs and code output until the final tokenizer pass.
- GPU work needs exclusive access (resident `gufo serve` holds VRAM).

## Facts nailed from the GGUF (verified 2026-10-26, `file_type=7`, 779 tensors)

- 49 layers, RMS pre-norm ε=1e-6, d=2048, ctx 262144, vocab 150144 (from
  `tokenizer.ggml.tokens` — no `vocab_size` key), RoPE θ=10000, no scaling.
- Layers 0-1 dense FFN 6144 (`leading_dense_block_count=2`); layers 2-48 MoE:
  256 experts (ffn 512), sigmoid gating (`expert_gating_func=2`), top-6,
  `expert_weights_norm=true`, `expert_weights_scale=1.0`, 2 shared experts
  (ffn 1024). Router `ffn_gate_inp.weight` (256x2048) is BF16.
- GQA 32Q/4KV, head_dim 128: q_proj 2048→4096 (expanded), k/v 2048→512,
  o_proj (4096,2048). Per-head `attn_q_norm`/`attn_k_norm` (128) and
  `attn_sinks` (32) exist in EVERY layer.
- `attention.sliding_window=513`; `rope_pattern` ==
  `sliding_window_pattern` (bool arrays, 49 entries): True at 2..4, 6..8,
  10..12, ... 46; False at 0, 1, 5, 9, ..., 45, 48 (14 layers) → those are
  global-cache AND NoPE (llama.cpp `flint.cpp` gates rope by `has_rope(il)`).
- Tokenizer: gpt2 BPE, `pre=flint`, add_bos/eos=false, bos 150000
  `<|begin_of_text|>`-style, eos 150001, eog {150001,150003,150005,150018};
  control tokens 150002-150005 + 150018; XML tool tokens 150010-150017
  (type 4); chat template renders XML tool calls.

## Reference baseline (llama.cpp)

- [x] Upstream `flint.cpp` exists since tag b11803 (PR #30928); worktree
  `~/projects/llm/llama.cpp/llama.cpp` @ **b11879** (detached), ROCm build
  `build_b11879_rocm` (GGML_HIP, gfx1151), needs
  `LD_LIBRARY_PATH=/opt/rocm/lib`.
- [x] llama-bench @ b11879, `-ngl 99 -fa 1 -r 3`, KV f16
  (log `/tmp/opencode/gufo_bench_flint-base.log`):

  | ctx  | pp t/s | tg t/s |
  |------|--------|--------|
  | 2048 | 1478.22 | 47.72 |
  | 8192 | 1375.79 | 45.74 |
  | 16384| 1265.11 | 43.72 |
  | 32768| 1080.82 | 39.93 |
  | 65536|  860.59 | 34.21 |
  | 131072| 611.30 | 26.71 |

## Checkpoints

- [x] C0+C1 module skeleton + metadata loader:
  `src/models/flint/{CMakeLists.txt,config.hpp,config.cpp}`, registered in
  root `CMakeLists.txt`; `flint.config` CPU test
  (`tests/models/flint/config_test.cpp`) — geometry profile gate, bool-array
  pattern parsing, vocab-from-tokenizer, rejection cases.
- [x] C2 weights: `src/models/flint/weights.{hpp,cpp}` — `TensorRef`
  (native encodings, no dequant; shard/offset + bounds check; `Expert()`
  stride), `LayerWeights` (per-layer q/k/v/o + `attn_q_norm`/`attn_k_norm`/
  `attn_sinks`; dense ffn on layers 0-1 vs BF16-router + 3D `_exps` +
  single `_shexp` on 2-48), `ModelWeights::Bind` with exact-shape/format
  gate and tied-embedding fallback. `flint.weights` test
  (`tests/models/flint/weights_test.cpp`, `GUFO_FLINT_GGUF`, skip 77):
  expected-name table == all 779 file tensors exactly, per-layer kind/shape
  checks. Passes on the real artifact.
- [x] C3 scalar reference oracle: `src/models/flint/{cpu_ops.hpp,cpu_ops.cpp,
  reference.hpp,reference.cpp}` (built only via `gufo_flint_reference`,
  EXCLUDE_FROM_ALL, -O2) + `flint.reference` test (build target explicitly).
  Passes on the artifact: 3 tokens finite/non-degenerate logits, repeat-stable,
  vocab-range rejection, Reset rewinds.
  Semantics nailed from llama.cpp source (b11879, oracle contract):
  per-head `attn_q_norm`/`attn_k_norm` = LLM_NORM_RMS with plain `w` (no +1)
  over one head_dim, before the gated NEOX full-head rope (`n_rot=128`);
  sinks = full softmax over causal keys + sink as extra per-head logit, incl.
  max (`ops.cpp:5988-5999`, FA kernels identical); window reads keys
  `[pos+1-513, pos]`; MoE: sigmoid(router), top-6 desc (lower index on tie =
  ggml argsort), `weights /= clamp(sum, 6.103515625e-5, ∞)`, scale 1.0 no-op,
  single `_shexp` added once, then residual (`llama-graph.cpp:2194-2360`).
- [x] C4 ROCm kernels + executor (Q8 mmq routed expert path, SIMT GEMV tier
  for dense/attention projections incl. batched prefill, f16 KV planes,
  windowed attention with sinks + per-head QK norm + NEOX rope gated per
  layer, sigmoid top-6 gate + clamped L2 norm + single shared expert).
  `src/models/flint/kernels/rocm/{kernels,linear}.{hpp,cpp}`,
  `executor.{hpp,cpp}`; test `flint.rocm.forward`
  (`tests/models/flint/rocm_forward_test.cpp`, EXCLUDE_FROM_ALL target,
  `GUFO_FLINT_GGUF`, skip 77). Passes: decode steps 0-3 max|logit diff|
  0.21-0.91 vs oracle budgets ≤1.13 with argmax match, batched prefill
  identical to sequential decode (0.637). Debug aids kept (off by default):
  `GUFO_FLINT_DEBUG` stats, layer-hidden capture both sides,
  `EnableMoeTrace` host replay. Bugs found here: Q8_0 header scale is
  `ggml_half` not float (kernels `Q8Scale()`), MoE down-GEMM pair-indexed
  convention, missing `__syncthreads()` between rope pair-rewrite and the
  output readers in `QkEpilogueKernel` (half the dims could leak unrotated
  values at pos>0), and mmq dense prefill exceeding the oracle envelope
  (→ fp32 `GemvBatch` tier until the speed phase).
- [x] gpt2 BPE tokenizer with the `flint` pre-tokenizer
  (`src/models/flint/tokenizer.{hpp,cpp}`): specials from `token_type`
  CONTROL metadata (spelling-free), eog ∪ eos stop set, collapsed-letters
  pre-split reproducing llama.cpp's generic splitter (digit ≤3, CJK split
  before letters, whitespace/punct prefixes, clitics, whitespace run
  rules), request-local piece memo. Test `flint.tokenizer`
  (`tests/models/flint/tokenizer_test.cpp`: byte coverage, piece
  boundaries, CJK split trap, specials/stops, BOS/EOS, roundtrips, GGUF
  validation); passes.
- [x] Engine: `engine.{hpp,cpp}` (Model/Session over FlintExecutor; chunked
  prefill, prefix-reuse Sync with divergence replay, one live session) +
  test `flint.engine` (GPU, `GUFO_FLINT_GGUF`, skip 77): vocab agreement,
  tokenize/decode roundtrip, session exclusivity, deterministic full-replay
  and append-prefix logits. Passes on the real artifact (first=450,
  second=6963).
- [x] C5 bench CLI: `cli_runner.{hpp,cpp}` (`IsFlint`, `RunFlintBenchmark`)
  linked into `gufo_llm_cli` and dispatched from `src/cli/bench/bench.cpp`.
  Smoke on the real artifact (GPU exclusive): load 12.4 s, pp2048 255.0 t/s,
  tg128 37.3 t/s — prefill far behind llama.cpp, tune in C7.
- [x] Chat template: `chat_template.{hpp,cpp}` replicates the GGUF jinja
  byte-exactly (turn framing, sanitized user/system content with the seven
  registered specials HTML-escaped, think-tag/thought rendering, tool
  preamble + `tool_call`/tool_response XML). The `flint.chat_template`
  test pins seven cases against renderings checked with a jinja oracle.
- [x] C5 serve: `serve_runner.{hpp,cpp}` (serial runner, one session, state
  ABI `flint-rocm-session-v1`) wired into the `gufo` target and dispatched
  from `inference_backend` on architecture `flint`. GPU-exclusive smoke:
  load 18.3 s, `Flint Preview` listed, three chat completions (thinking
  output, continuation with prefix reuse, tools-ignored path) at 44-50
  t/s decode.
- [x] C6 logits parity vs llama.cpp b11879: `flint_logits_probe` (GPU engine)
  vs a libllama C-API probe, same 22-token raw chat prompt, 8 greedy steps,
  exclusive GPU. Chosen-id chains identical; top-32 overlap 30-32/32 with all
  max margin deltas (-0.4..-2.5 logit) on tokens far below argmax; rank-0-
  margin differences 0.0 at every step. Scalar-oracle parity covered by the
  passing `flint.reference` and `flint.rocm.forward` tests.
- [x] C7a WMMA attention: `kernels/rocm/attention_wmma.hip.cpp` (dispatched
  from `Attention()` at head_dim 128), 16x16x16 fused f16 tiles with hi/lo Q
  and hi/lo probability correction chains. gfx1151 finding: the f32 WMMA
  accumulator is not zero-neutral — a masked exact +0 probability still moves
  the row by 1-2 ulp through values sharing its B (V) tile, and the
  zero-filled cache tail makes tile contents launch-specific, so naive WMMA
  broke batch-invariance (engine bitwise replay red). The kernel keeps WMMA
  only for interior tiles — every wave row attends every column — with the
  interior bounds keyed on the wave's own rows only (never `a.rows`, so a
  row's arithmetic path is identical across prefix lengths) and exact
  per-element accumulation on boundary tiles. New `flint.attention.op` test
  (`tests/models/flint/attention_op_test.cpp`): fp64-oracle parity for WMMA
  and dispatched kernels (worst 6.8e-7, scalar same) + 10 bitwise
  prefix/batch-invariance pairs — all pass. Matched `flint.rocm.forward`
  evidence (argmax matched everywhere): decode0 0.378 (wmma) vs 0.491
  (scalar), prefill 0.221 vs 0.266, per-layer tracking equal or better; the
  chaotic decode1 draw (1.034 vs 0.936 on budget 1.031 spread 51.5, scalar
  already at 91 % of budget from MoE expert flips) led to raising the parity
  budget 2 % -> 3 % of the vocab logit spread (maintainer-approved).
  Gates GPU-exclusive on Q8_0: flint.attention.op 9.0 s, flint.rocm.forward
  13.9 s, flint.engine 12.3 s — all pass. Release bench (exclusive):
  pp2048 394.0 t/s (was 255.0 at C5 smoke), pp8192 363.0 t/s, tg1 ~39 — prefill
  now bounded by the fp32 GEMV trunk tiers, not attention; the ≥3000 target
   needs a sustained bf16 GEMM path next.
- [x] C7b token-tiled dense Q8 prefill: `GemvBatchQ8TileKernel` in
   `kernels/rocm/linear.hip.cpp` — one warp per row now computes 32
   consecutive tokens, loading each 34-byte Q8 block and its exact float(qs)
   conversions once (per weight byte, not per token) into 32 accumulator
   chains. Per-token b-stride order, float4 x loads, j-order, d*x products
   and WarpSum tree match `GemvBatchQ8Kernel`, so outputs are BITWISE
   identical (`tools/bench/flint_gemv_btile_bench.hip`: bitwise n=0 vs the
   production kernel on all four Flint projection shapes, hoist-T32 1.84-
   2.04x, floor-normalized ~5.1 TFMA/s); `GemvBatch` dispatches full tiles
   through it and routes ragged tails through the old kernel. Microbench
   note: earlier "vanished stores" runs were a bench bug (`ot` base missing
   `+ row` — every warp raced one cell), not a ROCm anomaly. Gates GPU-
   exclusive on Q8_0: flint.rocm.forward (prefill max|diff| 0.221 on budget
   1.539) and flint.engine pass. Release bench (exclusive): pp2048 497.2
   t/s (was 394.0, +26 %), pp8192 494.0 t/s (was 363.0, +36 %) — the ≥3000
   target still needs the tensor-core dense tier (measured kernel floor is
   ~5x below this FMA ceiling).
 - [x] C7c mmq tensor-core dense prefill: `Project()` dispatches Q8 batch>1
   through `qfn_mmq_q8_0_dense` (q8_1 X-quantized `ggml_mul_mat_vec_q`),
   keeping the batch-1 decode tier on the bitwise SIMT `Gemv`. Release bench
   (exclusive): pp2048 946 t/s (was 497.2, +90 %), tg1 unchanged. Tier mixing
   breaks two bitwise contracts, both calibrated with evidence:
   `flint.rocm.forward` — the 4-token probe prompt flips the executor top-1
   (margin ~0.36) while llama.cpp b11879 flips its own top-1 (332 vs 269,
   same margin) between sequential decode and batched prefill (probe
   `flint_batch_argmax_probe.cpp`), so `Compare()` moved to a tie-margin
   argmax rule (exact top-1 when the oracle margin exceeds the budget; inside
   a tie the chosen candidate must sit within budget of the oracle top-1) —
   maintainer-approved 2026-10-27.
   `flint.engine` — mmq tiles couple a row's accumulation order to the batch
   size, so a replayed prefill no longer bitwise-matches a batch-1 decode
   step (measured max|diff| 2.08 = 3.8 % of the 54.9 spread, argmax identical
   in both replays); `ExpectReplay()` replaced `std::equal` with a
   5 %-of-spread bound (floor 0.25) plus the same tie-margin rule,
   maintainer-approved 2026-10-27. Same-sequence determinism stays bitwise.
   Gates GPU-exclusive on Q8_0: flint.attention.op 9.2 s, flint.rocm.forward
    12.8 s, flint.engine 11.6 s — all pass (fw6).
  - [x] C7d prefill chunk 256→2048: profiling pp2048 showed the MoE routed
    id-GEMM (`qfn_mmq::mul_mat_q<80>`, one launch per expert-id column) at
    53 % of kernel time because `kPrefillChunk = 256` made every chunk
    re-read the 256-expert weight planes — 8 launches per gemm per layer.
    `tools/bench/flint_moe_mmq_bench.hip` (uniform routing, worst case for
    weight traffic): one 2048-token id launch beats eight 256-token launches
    ~7x on gate/up and ~8x on down; the count-selected tile-cols and
    `max_expert_rows` knobs beat defaults at every batch but are additive,
    not adopted yet. Release bench (exclusive): pp2048 1408.93 t/s (was
    946, +49 %; llama.cpp b11879 reference 1478, within 5 %), pp8192
    1237.55 t/s. Gates re-verified GPU-exclusive on Q8_0 with
    the new chunk: flint.engine 11.6 s, flint.rocm.forward 12.2 s,
    flint.attention.op 9.3 s — all pass (fw7b).
  - [x] C7e token-tiled bf16 router GEMV: at chunk 2048 the router's
    warp-per-token `GemvBatchBf16Kernel` re-read the 4 KB expert row per
    token (~550 GB/s effective, 0.59 TF/s). `GemvBatchBf16TileKernel`
    mirrors the C7b design — one warp per row, 32 consecutive tokens, each
    bf16 weight converted once, 32 accumulator chains keeping the exact
    k-stride fmaf chain and WarpSum tree. `flint_gemv_bf16_tile_bench.hip`:
    bitwise 0 mismatches at B=256/1024/2048 on 256x2048, 3.84-4.29x;
    dispatch tiles full groups and routes the ragged tail through the old
    kernel (decode batch 1 unchanged). Release bench (exclusive): pp2048
    1628.08 t/s (was 1408.93, +15.6 %; now above the llama.cpp b11879
    reference 1478), pp8192 1401.68 t/s (was 1237.55). Gates pass again
    unchanged (engine bitwise replay proves router logits identical;
    rtr1).
  - [x] C7f routed mmq launch bounds from per-expert counts: the id-GEMM
    default grid covers all column tiles for all 256 experts — whole-chunk
    padding once counts are known. `rocm::ExpertCounts` (atomic pass over
    ids, gemma4 pattern) + pinned D2H after RouterTopK; `GemmMoe` sets
    `qfn_mmq_set_routed_max_expert_rows` (trims empty column tiles) and
    `qfn_mmq_routed_tile_cols_for_counts` and resets to 0 after. Grid-only:
    results unchanged; decode (batch < 32) skips the host sync and keeps
    the plain launch. Release bench (exclusive): pp2048 1818.61 t/s (was
    1628.08, +11.7 %; llama.cpp b11879 reference 1478, +23 %), pp8192
    1559.67 t/s (was 1401.68, +11.3 %). Gates unchanged and pass (rtr2).
  - [x] C7g WMMA attention config sweep: `tools/bench/flint_attn_wmma_bench`
    compared the valid launches (K/V staging fixes kRowBlocks = 8) on the
    measured windowed-513 and full-causal pp2048 shapes; `<8, 1>` (whole
    GQA group per staged K/V tile) is bitwise-identical to `<2, 4>` on both
    and 31 %/10 % faster → new default with `<2, 4>` fallback. Release
    bench (exclusive): pp2048 1920.02 t/s (+5.6 %), pp8192 1657.07 t/s
    (+6.2 %). Gates unchanged and pass (attn81).
  - [x] C7h prefill chunk 2048 → 4096: fewer launches per prompt past one
    chunk (per-chunk MoE/attention prefix overlap halves). Exclusive
    bench: pp8192 1706.82 (vs 1657.07, +3.0 %), pp16384 1342.98 (vs
    1324.80 control, +1.4 %); pp2048 = single chunk, unchanged at 1920.02.
     Gates pass with the new chunking incl. the 24 K-activity engine replay
     (ck4k).
  - [x] C7i qfn_mmq fork for Flint: copied the whole vendored mmq tree to
     `src/models/flint/kernels/rocm/mmq` (mechanical `flint_`/`FLINT_`
     rename, new `gufo_flint_mmq` target, `kernels/rocm/mmq/VENDOR.md`
     records provenance) so the Q8_0 pipeline restructure (double-buffered
     LDS staging) evolves without touching Flash-Next production kernels
     (mmq/VENDOR.md:11-14). Code-identical at import: exclusive bench
     pp2048 1950.02 t/s (1920 ± noise) and all three gates pass — engine
      bitwise replay proves the rename preserved outputs (fork1).
   - [x] C7j flint_mmq standalone bench + Q8_0 pipeline experiment
     (negative). `tools/bench/flint_mmq_bench.hip` times the shipped dense
     and routed entry points (compile/link line in its header; `--save`/
     `--ref` bitwise compare): baseline dense_2048x4096x2048 20.48 TFLOP/s,
     moe gate/up 9.89, down 7.49 (mmqb). A fused-iteration staging variant
     in the fork (knob `FLINT_MMQ_PIPE_Q8_0`, 4 y-stage + 2 x-tile LDS
     regions, one barrier per K iteration) measured 13.44/2.60/1.51: the
     same-shape control (knob off, host gate still sized for the pipeline)
     was bitwise-identical yet -47 % on dense, so 97.6 KB blocks (1 wave
     group/CU vs 33.9 KB ≈ 3) dominate; the barrier win (+22 %) cannot pay
     for the residency loss. Shipped default keeps the knob off; the pipe
     branch itself also mismatches (unfixed, inert while off) and needs a
     race diagnosis before anyone re-enables. Host smem gate now mirrors
     the device pipe gate (was the control-build slowdown). Gates pass,
     pp2048 1936.42 ≈ 1950 within session noise (land1).
    - [x] C7k fix `ds4_serving_test` link since f61bf542: the flint serve
      dispatch in `inference_backend.cpp` was never linked for the
      cross-model serving tests; add `flint/serve_runner.cpp` +
      `gufo_flint_engine` to the ds4, qwen27b and qwen38 test targets.
    - [x] C7l grouped Q8_0 MoE prefill route (Gemma-4 `moe_prefill` fork).
      Vendored gemma4 `moe_prefill` + the `RoutedCompact` builders into
      `kernels/rocm/moe/` (VENDOR.md, format-excluded like the mmq fork).
      Standalone A/B (`tools/bench/flint_moe_route_bench.hip`, same weights
      /routing both routes): grouped vs mmq 16.63 vs 9.39 TF at gateup
      (+77 %) and 15.32 vs 6.96 TF at down (+120 %); grouped self-check
      bitwise-stable, vs mmq rms ≈5e-4 of ref (f16 activations). Executor:
      Q8_0 routed layers take bucket tiles (128) + `rows_slot` scatter,
      gate/up from f16 token rows, down from f16 slot rows; the per-layer
      counts→host sync only remains on the mmq fallback. Non-Q8_0 or
      >256-expert models keep the mmq route. Gates 3/3, pp2048 1936.42 →
      2328.41 (+20 %), pp8192 1706.82 → 1929.89 (+13 %), pp16384 1342.98 →
      1469.49 (+9 %); tg128 40.49 (no matched prior run, no regression vs
       the C-era 37.3 record).
     - [x] C7m W8A8 int8-WMMA dense prefill route (shared qwen tier).
       `Project()` adds a pre-quantized x8 route calling the gufo_core
       launchers `LaunchQuantizeActivationQ8_1FromFp32` +
       `LaunchBatchedQuantGEMMPreQuantized` (wave64 128x128 tile with
       wave32 fallback, `prefill_quant_gemm.hip`) — no mmq for eligible
       shapes. `W8A8Eligible` = Q8_0 && batch>=96 && rows>=1024 &&
       cols>=1024 && cols%256; quantize once per distinct activation
       (shared qa8/ao8/qd8 scratch): attn q + dense/shexp gate+up from the
       post-norm rows, attn_out (k=q_dim) and ffn/shexp down (k=ffn) from
       their f32 intermediates. k/v (m=512), router and batch-1 decode keep
       the existing tiers. Standalone A/B
       (`tools/bench/flint_w8a8_bench.hip`) GEMM-only w8a8 vs mmq
       at batch 2048: attn_q 1.34 vs 1.65 ms, attn_out 1.25 vs 1.77,
       shexp gate/up 0.34/0.35 vs 0.39/0.45, down 0.34 vs 0.40-0.44;
       requantize-per-call (w8full) ≈ mmq, so the shared quantize is the
       whole game. Gates 3/3 exclusive (flintgate — oracle accepts the
       numerics). Release bench (exclusive, w8a8perf): pp2048 2328.41 →
       2424.91 (+4.1 %), pp8192 1929.89 → 1975.27 (+2.4 %), pp16384
       1469.49 → 1514.75 (+3.1 %); tg128 39.60 within the prior 40.49
       noise floor (decode path untouched).
      - [x] C7n vectorized elementwise kernels + pipelined router tile GEMV.
        `float4` paths in SumExperts/AddResidual/SiluMul (`kernels.hip.cpp`)
        and the `NarrowHalf` gather+scatter (`moe/routed_compact.hip.cpp`):
        per-pass profile 55→37/12→10/12→10/12→9 ms, bitwise identical,
        wall-neutral (vec128 gates, tg128 41.38). `GemvBatchBf16TileKernel`
        now issues the activation loads for k and k+32 before the fmaf
        chains consume them (pair-pipelined k steps, ascending order kept):
        `tools/bench/flint_gemv_bf16_tile_bench.hip` 0 bitwise mismatches,
        8–13 % faster than the plain tile at B>=2048 (router GEMV was ~2 TF
        from strided-load latency); LDS-staged x tiles, 64-token tiles
        (register spill) and plain `__ldg` all measured slower and are
        documented in the bench header. Gates 3/3 exclusive (rodfix).
        Release bench (exclusive, rodperf): pp2048 2419.22 → 2480.83/2482.24
        (+2.6 %), pp8192 1949.06 → 2003.13 (+2.8 %), pp16384 1519.80 flat
        within the 1514.75–1529.94 band, tg128 41.02 unchanged (batch-1
        decode path untouched).
     - [x] C7o binary16 routed down-projection slot rows.
        `RoutedHalfDownKernel` gains a `kHalfOut` epilogue (Q8_0 grouped
        down only): the transposed token store writes 8 `__half` instead
        of two `float4`, halving the [slots x 2048] traffic at both the
        write and the `SumExperts` read (~200 MB per pass removed at
        pp2048). `SumExpertsKernel` is templated on the routed element
        type with an 8-byte half4 load; the mmq route keeps binary32
        (`ed` and `ed_h` are allocated per route). Gates 3/3 exclusive
        (edh2/edh3). Release bench (exclusive, edhperf): pp2048
        2491.14/2514.80, pp8192 2003.13 -> 2035.58 (+1.6 %), pp16384
        1519.80 -> 1556.65 (+2.4 %), tg128 41.20 unchanged; the win grows
        with context as the slot-traffic share does.
      - [x] C7p fused grouped gate/up launch with a SiLU epilogue.
        One `RoutedHalfDownKernel` geglu launch over the two stacked
        tensors (kernel/launcher gain a trailing `w_up` pointer: gate
        rows from `w`, up rows from `w_up`, both at `expert_ffn` rows
        per expert) replaces the two FP32-output launches plus `SiluMul`
        + `NarrowActivations`; up waves hand their raw FP32 accumulators
        to the gate waves through LDS in two token-tile chunks, so the
        epilogue computes `silu(gate) * up` (not `GeGluValue`) on
        exactly the values of the chain it replaces and narrows once
        (unit check `flint_moe_route_bench check-geglu`: two-pointer
        bit-identical to the fused single-tensor geglu, old-pipeline
        parity at host-mirror ULP only). Gates 3/3 exclusive (geglu3).
        Release bench (exclusive, gegperf): pp2048 2491.14/2514.80 ->
        2577.54/2617.01 (+3.5-5.0 %), pp8192 2035.58 -> 2099.64
        (+3.1 %), pp16384 1556.65 -> 1581.59 (+1.6 %), tg128 41.20 ->
        41.75, tg1 43-44 -> 45-46.
      - [ ] C7 GPU benchmark sweep vs baseline table above; BENCHMARKS.md /
    QUALITY.md / EXPERIMENTS.md + docs/models/README.md row.

## Design notes

- Attention: Q is wider than the hidden size (4096 vs 2048); RoPE applies
  only where `rope_pattern` set; windowed layers read the trailing 513 keys,
  others the whole cache → per-layer KV view with read-mask (gemma4
  precedent `executor.cpp` read masks). Sinks enter the softmax denominator
  (deepseek_v4_flash precedent).
- MoE decode reads ≈ per-token experts across 47 layers at Q8; route via the
  shared mmq/dense-GEMM helpers like qwen36_a3b, keep weights in native
  encodings (no dequant staging).
- Shared experts: ONE `_shexp` weight stream (ffn 1024) per MoE layer, added
  once — upstream `flint.cpp:162-170` ignores `expert_shared_count=2` beyond
  tensor allocation. The stored `_shexp` width is `expert_shared_feed_forward_length`.
- QK norms: upstream applies `attn_q_norm`/`attn_k_norm` (RMS, per-head
  width 128) to every layer before RoPE (`flint.cpp:110-111`).
- Iterate ≤16K context first; escalate winners to 32K/65K/100K (AGENTS).

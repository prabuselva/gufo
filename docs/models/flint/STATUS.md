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
- [ ] C5 CLI (`bench`/`prompt`/`serve`) dispatch + runners; engine
  (`engine.{hpp,cpp}`) first.
- [ ] C6 logits parity vs scalar oracle and llama.cpp b11879.
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

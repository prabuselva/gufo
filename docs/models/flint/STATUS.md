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
- [ ] C2 weights: tensor table (`ffn_gate_inp` BF16 router, 3D `_exps`
  experts, 2× `_shexp`, `attn_q_norm`/`attn_k_norm`, `attn_sinks`) +
  `flint.weights` test against the real artifact (`GUFO_FLINT_GGUF`, skip 77).
- [ ] C3 scalar reference oracle + numerical test.
- [ ] C4 ROCm kernels + executor (Q8 mmq routed path, dense f16 GEMM,
  windowed attention with sinks + QK norm, per-layer rope bypass,
  sigmoid top-6 gate + L2 norm + 2 shared experts).
- [ ] C5 CLI (`bench`/`prompt`/`serve`) dispatch + runners.
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

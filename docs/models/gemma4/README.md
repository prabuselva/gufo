# Gemma-4-26B-A4B (`gemma4`)

Gufo implementation of Google Gemma-4-26B-A4B (GGUF arch `gemma4`) on AMD
Strix Halo `gfx1151`: a 30-layer MoE trunk (128 experts, top-8, plus one
always-on shared FFN) with interleaved sliding-window (25 layers) and full
attention (5 layers), a separate 4-layer MTP draft (`gemma4-assistant`) that
shares the target KV cache, and a SigLIP-style vision projector
(`gemma4v` mmproj). The module is isolated in `src/models/gemma4` with its own
kernels; it shares no code with the Qwen engines beyond the generic
`gufo_core` GGUF reader and the `gufo_qwen38_flash_next_mmq` Q8_0 tensor-core
GEMM library.

## Artifacts

| File | Role |
| --- | --- |
| `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (27.6 GB) | trunk: F32 + Q8_0 + BF16 (layer 29 is BF16) |
| `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (461 MB) | MTP draft, arch `gemma4-assistant`, no K/V projections |
| `mmproj-BF16.gguf` (1.19 GB) | vision encoder + projector, `clip.vision.projector_type=gemma4v` |

Located under `/home/praburaja/projects/llm/models/gguf/Gemma4-26B-A4B-IT/`.
Future targets: Q8_0 and Q4_K_M quantizations of the trunk (M10).

## Architecture contract

- hidden 2816, 30 layers, vocab 262144, ctx 262144, RMS eps 1e-6,
  embedding scale √2816, final logit softcap 30 (scale → tanh → rescale).
- SWA layers (25, window 1024): 16 q-heads × 256, 8 kv-heads, rope θ=1e4,
  rotary 256, **has `attn_v`**.
- Full layers (il 5, 11, 17, 23, 29): 16 q-heads × 512, 2 kv-heads,
  rope θ=1e6, rotary 512, **no `attn_v`** (V = rms_norm(wk·x), unroped),
  proportional rope via the global `rope_freqs[256]` factors.
- Per-head RMS norms on Q and K (`attn_q_norm`/`attn_k_norm`, head_dim sized);
  V is normed with a weightless RMS norm. Attention scale 1.0.
- Layer flow: `attn_norm` → Q/K/V(+norms+rope) → attn → `wo` →
  `post_attention_norm` → +residual = `attn_out`. MoE: shared dense FFN
  (2112, GELU-parallel) from `ffn_norm` → `post_ffw_norm_1`; expert FFN from
  `pre_ffw_norm_2` → `post_ffw_norm_2`; router =
  softmax(`rms_norm(attn_out)`·(1/√2816)·`ffn_gate_inp.scale`·`ffn_gate_inp`)
  top-8 of 128, no bias; `cur = mlp + moe` → `post_ffw_norm` → +`attn_out`
  → ×`layer_output_scale`.
- Experts: `ffn_gate_up_exps [2816, 1408, 128]` (gate = rows [0, 704),
  up = [704, 1408)), `ffn_down_exps [704, 2816, 128]` + `ffn_down_exps.scale`.
- Tokenizer: SPM-style BPE (`tokenizer.ggml.model="gemma4"`): 514 906 merges,
  raw UTF-8 (no GPT-2 byte encoding), space → `▁`, pre-split on newlines only,
  256 byte tokens for fallback, specials `<|turn>`/`<turn|>` (bos 2, eos 106).
- MTP draft: 4 layers, hidden 1024, dense FFN 8192, own `token_embd`
  [1024, 262144] as lm_head (no softcap); input = concat(target tok_embd row ×
  √2816, target post-final-norm hidden) → `nextn.pre_projection [5632, 1024]`;
  output `nextn.post_projection [1024, 2816]`; reads target KV read-only
  (draft SWA → target layer 28, draft full → target layer 29).
- Vision: 27 blocks, embd 1152, 16 heads × 72, ffn 4304, patch 16, image 224,
  proj 2816; conv patch-embed, x/y position tables, 2D RoPE neox θ=100
  (x on dims [0, 36), y on [36, 72)), avg-pool 3×3 merge, std-norm,
  `mm.input_projection` BF16 ClippableLinear (clamp scalars optional; absent
  in this artifact). Bicubic resize, 70–1120 tokens/image.

## Documents

- [DEVELOPMENT.md](DEVELOPMENT.md) — phase plan M1–M11 with exit criteria and
  the running status log.
- [QUALITY.md](QUALITY.md) — CPU oracle, GPU parity tolerances, golden vectors.
- [OPTIMIZATIONS.md](OPTIMIZATIONS.md) — kernel tiers and retained wins.
- [EXPERIMENTS.md](EXPERIMENTS.md) — measured A/B experiments.
- [BENCHMARKS.md](BENCHMARKS.md) — retained performance numbers; reference
  column is the local llama.cpp fork build.

## Kernel documentation

[docs/kernels/gemma4/](../../kernels/gemma4/) (lands with M5).
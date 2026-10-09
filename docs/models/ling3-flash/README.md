# Ling 3.0 Flash (`bailingmoe3`)

Gufo implementation of Ling 3.0 Flash (GGUF arch `bailingmoe3`) on AMD
Strix Halo `gfx1151`: a 42-layer trunk of Kimi Delta Attention (KDA) linear
recurrent layers interleaved with absorbed Multi-head Latent Attention (MLA)
layers, a sigmoid-gated MoE FFN with group-limited top-k routing plus one
always-on shared expert, and a trailing self-speculative MTP block. The
module is isolated in `src/models/ling3_flash`; it shares no code with the
Qwen engines beyond the generic `gufo_core` GGUF reader, sampler, and the
HF-BPE tokenizer loader.

## Artifacts

| File | Role |
| --- | --- |
| `Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf` (~37 GB) | trunk shard 1: metadata + 938 of 1876 tensors |
| `Ling-3.0-flash-Q4_K_M-00002-of-00002.gguf` (~36 GB) | trunk shard 2 |

Split GGUFs are opened through the first shard; the reader resolves the
sequence. The GPU tests locate the file through `GUFO_LING3_FLASH_GGUF`.

## Architecture contract

- trunk 42 layers + 1 MTP block (`block_count` 43), hidden 2560,
  vocab 157184, ctx 262144, RMS eps 1e-6.
- MLA on every 6th trunk layer (il 5, 11, ..., 41) and the MTP block:
  32 heads, per-head query `nope|rope` halves (128+64), latent KV rank 512
  absorbed with a shared rope key (key width 576), value head 128,
  rope θ=6e6. All other layers are KDA.
- KDA (linear) layers: separate q/k/v projections, head_dim 128, short conv
  kernel 4, sigmoid decay gate lower-bounded at −5 (`safe_gate`), recurrent
  state instead of KV.
- MoE on trunk layers 2+: 512 experts, top-8 restricted to the best 4 of
  8 groups, sigmoid gate with renormalized weights ×2.5, always-on shared
  expert; expert FFN 768, dense prefix FFN 6144 on layers 0–1, per-expert
  SwiGLU clamps from `swiglu_clamp_exp`/`swiglu_clamp_shexp`.
- MTP block: present in the artifact (Q4_0 rows, shared vocabulary); Gufo
  loads it but the session API does not draft with it yet.
- Tokenizer: HF fast BPE loaded through the shared `QwenTokenizer`; the
  vocabulary specials delimit turns.

## Chat framing (Bailing V3)

`src/models/ling3_flash/chat_template.hpp` owns the framing: role turns open
with `<role>HUMAN</role>`/`<role>ASSISTANT</role>` (and `<role>SYSTEM</role>`
content), every turn closes with the eos token `kTurnEnd` (id 156895, with
bos/pad siblings 156891/156892 in the vocabulary), and the system header
carries the `detailed thinking on|off` directive. Reasoning output arrives
between the `kThinkStart` and `kThinkEnd` markers; the interactive chat and
the serve reasoning split both key on that boundary. Tool definitions, tool
calls and tool results are rejected by the renderer (and by `gufo serve`).

## Usage

```sh
gufo prompt --model .../Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf "..."
gufo prompt --model ... --chat            # interactive, re-renders history
gufo serve  --model .../Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf --port 8083
```

- CLI prompt/chat run at a fixed 4096-token context.
- `gufo serve` supports `--sessions 1..16` with the `--ctx` window divided
  across sessions; sessions share the executor and own their KV/recurrent
  state.
- Unsupported and rejected at load: image input, tool calling, speculative
  decoding (`--speculative`), the continuation disk cache.

## Tests

Configure per [docs/TESTING.md](../../TESTING.md), then (inside `nix develop`
for the GPU targets):

```sh
ctest --preset gpu-full -R ling3 --output-on-failure   # ROCm + model tests
ctest --preset cpu -R ling3 --output-on-failure        # config/weights
```

`tests/models/ling3_flash/` keeps the scalar oracle (`reference_test`), the
kernel/forward comparisons, and the model-gated inference test driven by
`GUFO_LING3_FLASH_GGUF`. A missing-model skip is not a quality pass.

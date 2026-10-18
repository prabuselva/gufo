# Ling 3.0 Tiny (`bailingmoe3`)

Gufo support effort for Ling 3.0 Tiny — the small sibling of
[Ling 3.0 Flash](../ling3-flash/README.md), same GGUF arch `bailingmoe3` and
identical hybrid design (KDA linear-recurrent layers interleaved with
absorbed MLA layers, sigmoid-gated group-limited MoE with one shared
expert). Tiny carries **no self-speculative MTP block**. Support is tracked
in the model-private [PLAN.md](PLAN.md) · [STATUS.md](STATUS.md) ·
[BENCHMARKS.md](BENCHMARKS.md) · [EXPERIMENTS.md](EXPERIMENTS.md); the
runtime code lives in `src/models/ling3_flash` under an explicit geometry
profile.

## Artifact

| File | Role |
| --- | --- |
| `/home/praburaja/projects/llm/models/gguf/Ling3.0-Tiny/Ling-3.0-tiny-Q8_0.gguf` | single-file weights (8.4 GB, GGUF v3, 526 tensors, 53 KV pairs, `general.size_label` `128x1.0B` = 128 experts) |

## Checkpoint facts (measured 2026-10-18 from the GGUF)

| Config field | Tiny actual | Flash runtime pin | Same? |
| --- | --- | --- | --- |
| `block_count` / `num_layers` | 24 / 24 (no MTP) | 43 / 42 | smaller, MTP absent |
| `nextn_predict_layers` | key absent → 0 | 1 | absent |
| `embedding_length` | 1536 | 2560 | smaller |
| `attention.head_count` | 16 | 32 | smaller |
| `vocab_size` | 157184 | 157184 | same |
| `key_length_mla` / `value_length_mla` | 192 / 128 | 192 / 128 | same |
| `kv_lora_rank` / `q_lora_rank` | 512 / 256 | 512 / 256 | same |
| `rope.dimension_count` / `rope.freq_base` | 64 / 6e6 | 64 | same / verify theta |
| `ssm.conv_kernel` / `kda.head_dim` | 4 / 128 | 4 / 128 | same |
| `kda.gate_lower_bound` / `kda.safe_gate` | −5.0 / true | −5.0 | same |
| `expert_count` / `expert_used_count` | 128 / 8 | 512 / 8 | smaller |
| `expert_group_count` / `_used_count` | 8 / 4 | 8 / 4 | same |
| `expert_feed_forward_length` / shared | 512 / 512 | 768 / 768 | smaller |
| `feed_forward_length` (dense) / `leading_dense_block_count` | 4608 / 1 | 6144 / 2 | smaller |
| `expert_weights_scale` / `expert_weights_norm` | 2.5 / true | 1.0 / false | parsed generically |
| eos / bos / pad token ids | 156895 / 156891 / 156892 | identical | shared framing |

Tensor-level structure (all 24 blocks probed from tensor tables): **MLA at
blocks 3, 7, 11, 15, 19, 23 — every 4th layer, not every 6th as in Flash**
(the guard loop `config.cpp:232` hardcodes `(il+1)%6==0` and needs a
per-profile stride); KDA on the other 18 blocks with 2048-extent
`attn_q/k/v` + `ssm_*` heads (16 × 128); dense FFN only in block 0
(`leading_dense_block_count`=1); blocks 1..23 carry 128-expert
`ffn_{gate,up,down}_exps` with `exp_probs_b` correction bias and one shared
expert; MLA blocks carry `attn_q_a/q_b/kv_a_mqa/v_b/k_b/attn_gate` with the
same 192/128/512/64 absorbed shapes as Flash, 16 heads. There are no
`nextn.*` tensors.

## Support status (2026-10-18)

Supported end to end: `gufo prompt`, `gufo chat` and `gufo serve` (native
Ling 3.0 tool dialect, shared with Flash) run Tiny through the same
`src/models/ling3_flash` engine via an explicit geometry profile selected
from artifact metadata — stride-4 MLA placement, stride-2 leading dense
blocks, low-rank split query projection with its own norm, no MTP path and
no SwiGLU clamp arrays. Gate evidence (greedy forward/hidden parity vs
`reference.cpp`, tool-call replay, exclusive pp/tg grid) is recorded in
[STATUS.md](STATUS.md) and [BENCHMARKS.md](BENCHMARKS.md) with commands in
[EXPERIMENTS.md](EXPERIMENTS.md).

```sh
gufo prompt --model /home/praburaja/projects/llm/models/gguf/Ling3.0-Tiny/Ling-3.0-tiny-Q8_0.gguf
```

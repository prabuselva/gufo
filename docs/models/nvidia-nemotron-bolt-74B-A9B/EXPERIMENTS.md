# NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB — Kernel Experiments

Retained/rejected log for `src/models/nemotron_h_moe` on gfx1151. Every row is
a matched same-session comparison through `tools/bench/gpu_exclusive.sh`
(exclusive VRAM) or the `bench_bg.sh`+`bench_wait.sh` pair launched in one
tool call; iteration happens at ≤16K context per AGENTS policy before any
32K/65K/100K confirmation. Production speeds live in
[BENCHMARKS.md](BENCHMARKS.md); integration progress in
[STATUS.md](STATUS.md).

## Weight format decisions

| question | decision | evidence |
| --- | --- | --- |
| IQ2_XXS (id 16) routed/shared experts | keep encoded; grouped GEMV/GEMM via vendored mmq (`gufo_qwen38_flash_next_mmq`, `mmq_type_traits` incl. IQ2_XXS) | byte-exact 2.0625 bpw from GGUF offsets; DS4 `iq2_tables.hip.hpp` table precedent |
| NVFP4 (id 40) `ssm_in`/`ssm_out` | keep encoded; dequant path must apply per-tensor `.scale` f32 | `block_nvfp4` = 4×UE4M3 + 32 B fp4 per 64 elems (mmq `ggml-common.h:148`) |
| Q8_0 (id 8) attn/latent/embd/output | existing Q8_0 GEMV/GEMM tiers | 1.0625 B/elem verified |
| Router + all norms + mamba A/D/conv/dt | stay F32 | GGUF dump |

## Planned measurements (fill as levers land)

| lever | scope | result | decision |
| --- | --- | --- | --- |
| Row-batched F32-activation prefill GEMMs (Q8/IQ2 block-per-row + tile LDS; NVFP4 k-split reuse; IQ2 experts sorted by `nbl_expert_index`) | end-to-end pp2048/pp16384 exclusive, matched | pp2048 30.81 → 63.19 (2.05×), pp16384 53.89 vs 28.41 baseline (1.90×), tg64 14.37 unchanged; stage profile: NVFP4 rows 55.3 %, IQ2 rows 28.9 %, router F32 5.9 % | adopt (default ≥64 tokens) |
| NVFP4 rows: LDS-tiled warp-per-row (4 x rows staged per k slice) | microbench | bank-conflict/swizzle redesign pending; device wave-64 (`-mwavefrontsize64`) breaks wave-32 lane math — derive from `warpSize` | open |
| Router GEMM (F32) | 5.9 % of rows-stage time (3.8 s / 63.7 s) | open |
| Mamba-2 decode: warp-per-head vs lane-per-head state layout | microbench | — | — |
| Chunked Mamba-2 prefill chunk size (256/512/1024/2048) | end-to-end pp8192 | — | — |
| MoE decode: per-layer n_expert (240…496) mmid launch params | microbench | — | — |
| ReLU² fused into expert epilogue vs separate pass | microbench | — | — |
| Latent down/up as fused prologue/epilogue vs dense gemm | end-to-end | — | — |
| NoPE attention: skip rope stage cost | microbench | — | — |

## Open questions

- NVFP4 tensor-scale handling in the vendored mmq traits path (apply scale on
  fp32 accumulate output vs dequant-time fold).
- Whether IQ2_XXS mmvq dequant dominates decode at top-k=9 × 40 MoE layers;
  price an expert-resident (n_expert_layer ≤ available L2/shared) scheduling
  variant before long-context work.
- Heterogeneous `expert_feed_forward_length` per layer (1536/2048 in
  `n_ff_exp` array) vs global `expert_count=496`: confirm router-derived
  per-layer count is the sole truth (llama.cpp `build_ffn_layer` uses
  `ffn_up_exps->ne[2]`).

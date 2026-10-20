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
| `NBL_ROW_NT` 4 → 8 (row-batch tile rows per block) | end-to-end pp2048 exclusive | pp2048 63.19 → 90.06 (1.42×), decode untouched (narrow path) | adopt |
| Vendored MMQ prefill tiers in the wide branch (`nmb_mmq_nvfp4_dense` ssm_in/out; `nmb_mmq_q8_0_dense` q/k/v/attn_out/latent; `nmb_mmq_iq2_xxs_dense` shared experts; `nmb_mmq_iq2_xxs_moe_raw` `up_exps` with per-token expert ids; `down_experts` keeps `nbl_gemv_rows_moe` on per-slot activations) | end-to-end pp2048/pp16384/tg64 exclusive + `device_step` wide gate (2 prompts, 132/197 tok, margin-aware rule) + `rocm_mmq` routed tier extended to 512 experts | pp2048 90.06 → 293.09 (3.25×), pp16384 53.89 → 165.30 (3.07×), tg64 14.44 unchanged; wide gate rel 1.4–1.5e-2 (q8_1-activation noise accumulation on oracle near-ties, margin 0.03–0.08; per-GEMM sites ≤4e-3), `nemotron_h_moe.*` contracts 7/7 | adopt (default ≥64 tokens) |
| NVFP4 rows: LDS-tiled warp-per-row (4 x rows staged per k slice) | microbench | superseded by MMQ prefill tiers for the wide path; rows kernels remain the decode/small-batch tier | closed (superseded) |
| Attention-Kernel LDS transpose (`skT[d][j+1]` padded) + wider token tile | `tools/bench/nem_attn_bench.hip` exclusive sweep (tr/kt grid, pp16384 shapes) + `rocm_attention` contract + `device_step` gate | kernel 26.0/5.7/2.4 -> 84.6/20.0/8.8 t/s (2048/8192/16384) with tr24 kt16 skT (skT alone +53%, tr24 +105% more; tr32/tr20/tr28, kt24/kt32, kt64 LDS-spill all worse); end-to-end pp16384 165.30 -> 240.69 (+45.6%), pp2048 293.09 -> 304.63, tg64 14.44 -> 14.52 | adopt (default) |
| Router GEMM (F32): LDS-tiled SGEMM 64x64 tile, k-step 32, 256 threads, 8 tokens x 2 experts per thread (float4 staging, `+1` row pad) | `tools/bench/nem_router_bench.hip` exclusive sweep (naive/k-split/tile grid) + `rocm_ops` contract + `device_step` gate | kernel 0.15 -> 3.0-4.4 TF/s at M=2048-16384 (~20x; prefetch, ks64, TN32, TGT128-tile variants worse; thread tile 8T x 2R ratio beat 4T x 4R); end-to-end pp2048 304.63 -> 425.83 (+39.8%), pp16384 240.69 -> 318.66 (+32.4%), tg64 14.52 -> 14.71 | adopt (default, k%32 fallback) |
| Routed `down_experts` on the MQ id tier (`nmb_mmq_iq2_xxs_moe_raw`, slots=t*top_k x 1 expert, X=per-slot act_ rows) instead of `nbl_gemv_rows_moe` GEMV | end-to-end pp2048/pp16384/tg64 exclusive + `device_step` gate | pp2048 425.83 → 508.61 (+19.4%), pp16384 318.66 → 357.58 (+12.2%, ref 488.80), tg64 unchanged; dead row-batched/expert-index path removed | adopt (wide branch) |
 | Attention score phase 2D register tile (2 rows x 2 kv cols per thread, 4 accumulators, one LDS word feeds two FFMAs; per-cell causal masks keep d-ascending chains bit-identical) | `tools/bench/nem_attn_bench.hip` exclusive RP/JP grid with median round-robin + duplicate controls; `rocm_attention`/`rocm_ops`/`rocm_mmq` + `device_step` gates; matched interleaved exclusive A/B | attention-only +24/+19/+18% median at 2048/8192/16384 shapes (T22 tr24/kt16; KJ column tiling and unroll variants lose ~10%; T41/T12 smaller, E22 tr32 1 block/CU); end-to-end pp2048 +1.2% (485.1 -> 490.9/492.0, both reps), pp16384 parity — gain shrinks under 256-token chunking where scores are a minor kernel share | adopt (default) |
| Q8_0 KV cache (llama-recipe blocks: f16 `amax/-127` + 32 int8, 34 B vs 128 B fp32) | full port (append quantize, staging dequantize, byte-cache contract, `device_model` sizing); `rocm_attention` (rel ≤2e-7, append ≤0.92× quantum bound) + `rocm_ops`/`rocm_mmq`; matched interleaved exclusive A/B 2 rounds + `device_step` gate | pp16384 357.36/352.76 -> 256.46/256.05 (-27.8), pp2048 471 avg -> 433 avg (-8.0), tg64 flat, gate pass — naive scalar staging (per-element byte+scale loads, i8/f16 cvt, mul) costs ~3 extra insts/elem; fp32 attention already runs ~180 GB/s (~71 % of LPDDR peak) so cutting traffic 3.7x while adding ALU loses; net regression | reject (reverted to F32 cache); revisit only as warp-owned vectorized unpack (4x u64 aligned loads, planes layout, dequant folded into LDS staging) priced in `nem_attn_bench.hip` first |
| Prefill chunk 256 → 2048 rows (`kPrefillChunk`; gemma4 3.2c donor — wide chunks amortize per-chunk weight re-streams of the row-batched/MMQ GEMMs and cut per-chunk Mamba state round-trips; scratch derives from the constant; scans keep per-output K-chain order) | end-to-end pp2048/pp8192/pp16384/tg64 exclusive + `device_step` gate + three kernel contracts | matched interleaved exclusive A/B 2 rounds (order swapped): pp2048 476.07/476.33 → 582.17/578.53 (+21.9 %), pp8192 423.64/420.17 → 508.87/509.55 (+20.7 %), pp16384 353.81/351.49 → 425.52/421.91 (+20.2 %, −13.3 % vs ref 488.80), tg64/pp1 flat, `device_step` PASS 266 s, no scratch OOM | adopt (default) |
| Mamba-2 decode: warp-per-head vs lane-per-head state layout | microbench | — | — |
| Chunked Mamba-2 prefill chunk size (256/512/1024/2048) | end-to-end pp8192 | device-wide prefill chunk lever landed above and subsumes it (mamba chunk = `kPrefillChunk` = 2048) | closed (superseded) |
| MoE decode: per-layer n_expert (240…496) mmid launch params | microbench | — | — |
| ReLU² fused into expert epilogue vs separate pass | microbench | — | — |
| Latent down/up as fused prologue/epilogue vs dense gemm | end-to-end | — | — |
| NoPE attention: skip rope stage cost | microbench | — | — |

## Open questions

- AttnKernel pp16384 moves ~2.9 TB of F32 KV per timed pass at ~180 GB/s
  (~71 % of the ~256 GB/s LPDDR ceiling). Price a vectorized-unpack Q8_0
  staging variant (plane layout, 4x u64 per-warp loads) in the microbench
  before another cache-format attempt.
- NVFP4 tensor-scale handling in the vendored mmq traits path (apply scale on
  fp32 accumulate output vs dequant-time fold).
- Whether IQ2_XXS mmvq dequant dominates decode at top-k=9 × 40 MoE layers;
  price an expert-resident (n_expert_layer ≤ available L2/shared) scheduling
  variant before long-context work.
- Heterogeneous `expert_feed_forward_length` per layer (1536/2048 in
  `n_ff_exp` array) vs global `expert_count=496`: confirm router-derived
  per-layer count is the sole truth (llama.cpp `build_ffn_layer` uses
  `ffn_up_exps->ne[2]`).

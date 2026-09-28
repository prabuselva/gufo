# Qwen3.6-35B-A3B — ROCm Kernel Map

All kernels for `qwen36_a3b` live in `src/models/qwen36_a3b/kernels/rocm/`
and are model-private. Geometry (Qwen3.5-MoE variant, Q8_K_XL artifact):
hidden 2048, 40 layers (GDN linear attention unless `(layer + 1) % 4 == 0`,
then GQA), attention 16 q-heads / 2 kv-heads / head_dim 256 / rotary 64,
SSM 16 k-heads / 32 v-heads / head_dim 128 / conv kernel 4, 256 routed
experts top-8 with expert_ff 512, shared expert ff 512, vocab 248320.
Weights stay quantized on device (Q8_0 / F32 / BF16 rows; routed experts
also Q4_K / Q5_K / Q5_1 through the F16 WMMA tier).

## Documents

| Document | Source | Contents |
| --- | --- | --- |
| [fused-ops.md](fused-ops.md) | `kernels.hip.cpp` (1319 lines) | norms, RoPE, elementwise, router, MoE epilogue, GDN decode + prefill, attention decode + prefill |
| [gemv.md](gemv.md) | `gemv.hip.cpp` (271 lines) | decode linear projections: Q8_0/F32/BF16 GEMV, grouped expert GEMV, embedding row dequant |
| [gemm.md](gemm.md) | `gemm.hip.cpp` (151 lines) | prefill linear projections: mmq Q8_0 dense, hipBLAS F32/BF16, MoE fallback |
| [routed-f16-moe.md](routed-f16-moe.md) | `routed_f16.hip.cpp` (903 lines) | production prefill MoE: bucket compaction + F16 WMMA routed expert GEMM |

## Tier selection

The executor (`executor.cpp`) picks a tier by batch size:

- **Decode (1 token)** — `Gemv` / `GemvGrouped` for every projection
  (memory-bound: one weight byte per MAC is the floor), fused operators
  from `kernels.hip.cpp` with the `head_dim`-specialized fast paths
  (`GdnDeltaDecodeKernel` at ssm head_dim 128, flash-decoding split
  attention at head_dim 256).
- **Prefill (chunked batch)** — `Gemm` / `GemmMoe` / `RoutedF16Gemm` for
  projections, the `*Prefill` fused operators, `GdnDeltaLoopKernel` for
  the linear-attention recurrence and `AttentionPrefillTiled` for GQA.

Dispatch rules inside the launchers:

| Launcher | Rule |
| --- | --- |
| `GdnDelta` | `head_dim == 128` → `GdnDeltaDecodeKernel`; else `GdnDeltaKernel` |
| `AttentionDecode` | `head_dim == 256 && n_kv > 0` → split + combine (`splits = ceil(n_kv/64)`, cap 32); else 3-pass `AttentionDecodeKernel` |
| `AttentionPrefill` | `head_dim <= 256` → `AttentionPrefillTiled`; else `AttentionPrefillNaive` |
| `GdnDeltaLoop` | `kRows = 8`, template `kCols = head_dim / warpSize` (switch 1–8) |
| `Gemm` / `GemmMoe` | Q8_0 → `qfn_mmq_*` (shared mmq target from `qwen38_flash_next`); F32 → hipBLAS SGEMM; BF16 → narrow + hipBLAS GemmEx; MoE non-Q8_0 → `MoeVecFallback` |
| `RoutedF16Gemm` | Q4_K/Q5_K only at `tile_rows <= 48`; `tile_rows ∈ {16, 48, 64}`; returns false on unsupported shape (executor falls back to `GemmMoe`) |

## Decode stage profile (33.3 tps, 32 steps)

From `profile.hpp` `StageProfiler` (sync-per-mark inflates the total by
~250 ms; see `docs/models/qwen3.6-35B-A3B/OPTIMIZATIONS.md`). Per-step
totals mark where each kernel document's optimization notes apply:

| Stage | ms/step | Kernel(s) |
| --- | --- | --- |
| `output` (lm_head) | 4.41 | `GemvQ8_0` (gap vs 2.34 ms standalone bench — open) |
| `lin_gemm_in` (ssm qkv/gate/alpha/beta) | 5.00 | `GemvQ8_0` |
| `moe_shared` | 2.00 | `Gemv` ×4 + `Swiglu` |
| `lin_gemm_out` (ssm out) | 1.69 | `GemvQ8_0` |
| `moe down/up/gate` | 0.058–0.062 each | `GemvGrouped*` |
| `moe_router` | 1.59 | `Gemv` + `RouterTopK` |
| `attn_gemm_qkv` | 1.44 | `Gemv` ×3 |
| `lin_delta` | 1.25 | `GdnDeltaDecodeKernel` |
| `attn_core` | 0.090 | `AttentionDecodeSplitKernel` + combine |
| `moe_epilogue` | 0.50 | `MoeEpilogueKernel` |
| `lin_conv` / `lin_outnorm` / `lin_normqk` | 0.38 / 0.34 / 0.31 | `GdnConvKernel`, `GdnOutNormKernel`, `GdnNormQkKernel` |
| `attn_rope_norm` | 0.24 | `RmsNormKernel` + `RopeKernel` |

Roofline: ~3.07 GB of weights read per token → ~78 tps at 240 GB/s.
Open items ranked in `OPTIMIZATIONS.md`: the `output` GEMV gap, router +
shared-expert launch consolidation, MTP speculative decoding.

## Oracle tests

| Test | Covers |
| --- | --- |
| `qwen36_a3b_rocm_gemv_test` | all GEMV/GEMM encodings vs scalar |
| `qwen36_a3b_rocm_gdn_test` | conv/norm/delta/loop vs reference recurrence (attn err 9.89e-5 / 1.83e-4, state 1.9e-5) |
| `qwen36_a3b_rocm_attention_test` | decode n_kv 1–2400, prefill tiles |

Run the smallest one covering the change (see AGENTS.md); model-level
quality is checked by the user-run forward test.
# Gemma-4-26B-A4B — ROCm Kernel Map

All kernels for `gemma4` live in `src/models/gemma4/kernels/rocm/` and are
model-private. Geometry (validated in `config.cpp`, Q8_K_XL artifact):
hidden 2816, 30 layers, 16 q-heads on both attention classes; sliding-window
attention on five of six layers (`layer % 6 != 5`, 8 kv-heads, head_dim 256,
own V projection, window 1024, rope theta 1e4) and full attention on every
sixth layer (2 kv-heads, head_dim 512, V derived from the K projection, rope
theta 1e6), both NEOX rope over the full head. Each layer has a shared dense
geglu FFN (ff 2112) and a 128-expert top-8 softmax MoE (expert_ff 704) with
per-expert down scales and a 6.103515625e-5 weight clamp. lm_head softcap 30.
The draft (`gemma4-assistant`) is a 4-layer dense MTP model (hidden 1024,
ffn 8192, kv heads {8,8,8,2}) reading trunk layers 28/29 KV read-only.
Weights stay quantized on device (Q8_0 / F32 / BF16 rows; routed experts also
Q4_K / Q5_K through the F16 WMMA tier).

Gemma-4's activation is **gelu-tanh**, never SiLU: the geglu split
(`GegluF32`/`GegluF16`) is the only activation path, and the ported routed
F16 GEMM had the source's SwiGLU epilogues stripped.

## Documents

| Document | Source | Contents |
| --- | --- | --- |
| [fused-ops.md](fused-ops.md) | `fused.hip.cpp` (338 lines) | norms, residual add, router, expert histogram, MoE combine, geglu |
| [gemv.md](gemv.md) | `gemv.hip.cpp` (675 lines) | decode linear projections: Q8_0/F32/BF16 GEMV, grouped expert GEMV, multi-projection fusion, embedding row dequant |
| [gemm.md](gemm.md) | `gemm.hip.cpp` (155 lines) | prefill linear projections: mmq Q8_0 dense, hipBLAS F32/BF16, MoE fallback |
| [routed-f16-moe.md](routed-f16-moe.md) | `routed_f16.hip.cpp` (800 lines) | production prefill MoE: bucket compaction + F16 WMMA routed expert GEMM |

Attention kernels (WMMA flash prefill/decode over the two head_dim classes,
windowed SWA ring + full KV, fused Q/K RMSNorm + NEOX rope) arrive with the
M6 executor and will extend this map.

## Tier selection

The executor picks a tier by batch size:

- **Decode (1 token)** — `Gemv` / `GemvMulti` for the dense projections,
  `GemvGrouped` / `GemvGroupedPair` for the routed experts (memory-bound:
  one weight byte per MAC is the floor), `fused.hip.cpp` operators around
  them.
- **Prefill (chunked batch)** — `Gemm` for the dense projections,
  `RoutedCompact` + `RoutedF16Gemm` + `GegluF16` + `MoeEpilogue` for the
  routed experts; `GemmMoe` only as the non-Q8_0 fallback.

Dispatch rules inside the launchers:

| Launcher | Rule |
| --- | --- |
| `Gemv` / `GemvRows` | Q8_0 → 4-wave row kernel; F32/BF16 → `GemvF32`/`GemvBf16`; K-quants are grouped-only |
| `GemvGrouped` | Q8_0 / Q4_K / Q5_K specialized; F32/BF16 → `GemvGroupedDense<T>` |
| `GemvGroupedPair` | Q8_0 only; returns false otherwise (caller runs two `GemvGrouped`) |
| `GemvMulti` | n in [1,4]; Q8_0 subset folded into one `GemvQ8_0Multi4`, the rest as plain `Gemv` |
| `Gemm` / `GemmMoe` | Q8_0 → `qfn_mmq_q8_0_dense` / `qfn_mmq_q8_0_moe_raw` (shared mmq target from `qwen38_flash_next`); F32 → hipBLAS SGEMM; BF16 → narrow + hipBLAS GemmEx; MoE non-Q8_0 → `MoeVecFallback` |
| `RoutedF16Gemm` | Q4_K/Q5_1/Q8_0/Q5_K/BF16; `tile_rows ∈ {16, 48, 64}`; exactly one of `out`/`out_half`; returns false on unsupported shape |

## Oracle tests

| Test | Covers |
| --- | --- |
| `gemma4.rocm_kernels` (`gemma4_rocm_kernels_test`) | all GEMV/GEMM encodings, grouped/pair/multi fusion, fused ops, router exact-match, and the full routed MoE pipeline vs the CPU oracle |

Measured parity (synthetic Q8_0 weights, `WorstRelative` /
`WorstRelativeToScale`): Gemv Q8_0 1.26e-3, GemvGroupedPair 9.8e-4/1.39e-3,
Geglu F32 2.07e-4, Gemm Q8_0 batch-8 3.58e-3 scale-relative, routed MoE
pipeline 5.93e-4 scale-relative, norms ≤ 2.4e-7, RouterTopK exact. Run the
smallest check covering the change (see AGENTS.md); model-level quality is
checked by the forward test.
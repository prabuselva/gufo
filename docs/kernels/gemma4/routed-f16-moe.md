# Gemma-4 Routed F16 MoE — `routed_f16.hip.cpp`

Production prefill path for the 128-expert top-8 MoE. Ported from the
Qwen 27B route (`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`);
the source's fused gate+up (`kPair`) variant and `swiglu_gate` epilogue were
**stripped** — Gemma-4 uses gelu-tanh, applied by `GegluF16` between the two
GEMMs. The dp4a MMQ grouped path is ~55x slower per MAC than the dense MMQ
on gfx1151; the matrix-core F16 route with the weights dequantized after the
LDS read is what makes routed prefill competitive.

## Layout contract

Routed assignments (`ids[t*k+s]`, -1 for none) are compacted by expert into
16-row padded buckets:

- `RoutedCompactRows(slots, n_experts) = slots + n_experts * 15` rows for
  the mapping arrays.
- `RoutedCompact(ids, counts, pad_bounds, cursors, rows_token, rows_slot,
  n_tokens, k, n_experts, stream)` — one block per expert bucket: exclusive
  scan of the padded counts into `pad_bounds[n_experts + 1]`, then a scatter
  of each slot into its bucket row. `rows_token` / `rows_slot` map each
  compacted row back to its token and slot; the launcher memsets both to -1
  first, so padding rows read as invalid.
- `tiles[y]` packs the expert in the low 16 bits and the token macro-tile
  index in the high 16, so no block is launched for an empty tile.

## Kernels

| Kernel | Launch | What it computes |
| --- | --- | --- |
| `RoutedF16GEMMKernel<kType, BM, BN, BK>` | grid (m/BM, n_tiles), 256 thr, `BM=128`, `BK=2`, `BN ∈ {16,48,64}` = `tile_rows` | WMMA GEMM of one expert against its bucket rows; weights dequantized to F16 after the LDS read; F32 accumulate over the whole K extent |
| `RoutedPadBoundsKernel` / `RoutedScatterKernel` | one block / grid | the two compaction stages |
| `NarrowKernel<T>` | 256 thr grid | `NarrowActivations`: F32 → F16 (or BF16) activation rows |

LDS plan per stage: code plane (BM rows x kChunks 16-byte chunks, permuted
for bank-group spread), Q5 high-bit plane, (scale, bias) plane, and the
activation plane `[kb][quarter][token][16 B]` so a fragment read is 256
contiguous bytes. A Q4_K code becomes a half through a byte permute into
the mantissa of 1024.0, a packed subtract, then one packed FMA; Q8_0/Q5_1/
BF16 decode straight into `__half2`. The epilogue transposes each 16x16
accumulator tile through LDS and scatters to `out[rows_out[c]][row]` as F32
or narrowed F16 (`out_half`); wide tiles (`BN >= 48`) use the full-row
128-byte-line variant. Short buckets skip the WMMA work of empty token
tiles.

## Launcher

`RoutedF16Gemm(w, type, x_half, tiles, n_tiles, tile_rows, pad_bounds,
rows_in, rows_out, out, out_half, m, k, stream)` — exactly one of `out` /
`out_half` non-null; `k` must be a multiple of 256 (Q4_K/Q5_K), 32 (BF16)
or 64 (Q8_0/Q5_1); returns false on unsupported shape/type (executor falls
back to `GemmMoe`). Supported types: Q4_K, Q5_1, Q8_0, Q5_K, BF16.

## Gemma-4 pipeline

For `ffn_gate_up_exps.weight` [128][1408][2816] (gate rows 0–703, up rows
704–1407) and `ffn_down_exps.weight` [128][2816][704] + `scale` [128]:

1. `NarrowActivations` the layer input to F16.
2. gate_up `RoutedF16Gemm` with `rows_in = rows_token`,
   `rows_out = rows_slot`, `out_half` — m = 1408.
3. `GegluF16` over the slot-indexed [rows][1408] buffer → [rows][704].
4. down `RoutedF16Gemm` with `rows_in = rows_slot`, `rows_out = rows_slot`,
   F32 `out` — m = 2816, k = 704.
5. `MoeEpilogue` combines slots with `weights * expert_scale(ids)`.

Everything between the compaction and the epilogue is slot-indexed; the
token only reappears in the epilogue.

## Numerical contract

Measured end-to-end (`gemma4.rocm_kernels`, synthetic Q8_0 experts, vs the
CPU oracle router + geglu + down + weighted combine): worst error 5.93e-4
scale-relative (tolerance 2e-2). The F16 narrowing of activations and the
F16 weight decode are the only lossy steps beyond the Q8_0 weights
themselves; the K accumulation is pure F32 WMMA.
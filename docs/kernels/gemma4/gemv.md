# Gemma-4 GEMV Tier — `gemv.hip.cpp`

Decode (and small-batch row) linear projections. Weights stay in their
native encoding; every kernel decodes on the fly. The tier is memory-bound:
the contract is one streamed weight byte per MAC at the ~240 GB/s streaming
ceiling, and the per-row dot order is pinned so decode numerics are stable
across launches.

## Formats

`GemvType { kQ8_0, kF32, kBF16, kQ4_K, kQ5_K }`. The Q8_K_XL artifact uses
Q8_0 for every dense projection and expert stack of trunk layers 0–28 and
BF16 for layer 29; Q4_K/Q5_K appear only for routed expert stacks of a
Q4_K_XL artifact. `row_bytes` must match `TensorRef::RowBytes` for the type
and `cols`.

## Kernels

| Kernel | Launch | What it computes |
| --- | --- | --- |
| `GemvQ8_0` | grid `rows/4`, 128 thr (4 waves) | one row per wave; four-block unrolled stream keeps several weight loads in flight; `WarpReduceSum` |
| `GemvRowsQ8_0` | grid `rows/4`, 128 thr | same dot for `tokens` activation rows (`x_stride`/`out_stride`), one wave per (row, token) pair |
| `GemvF32` / `GemvBf16` | grid `rows/4`, 128 thr | the same wave-per-row shape over unquantized rows |
| `GemvRowsDense<T>` | grid `rows/4`, 128 thr | F32/BF16 multi-token rows |
| `GemvGroupedQ8_0` | grid `used*rows/4`, 128 thr | routed decode: slot `s` uses expert `ids[s]`; `expert_stride` is the byte distance between stacked experts |
| `GemvGroupedQ4K` / `GemvGroupedQ5K` | same shape | K-quant decode of the routed rows (superblock scales staged per 32 blocks) |
| `GemvGroupedDense<T>` | same shape | F32/BF16 routed rows |
| `GemvGroupedQ8_0Pair` | grid `2*used*rows/4`, 128 thr | gate and up dots for the same (slot, row) in one wave; bit-identical to two `GemvGroupedQ8_0` launches |
| `GemvQ8_0Multi4` | grid `total_rows/4`, 128 thr | up to four Q8_0 matrices sharing one activation row; wave picks the owning matrix from cumulative row offsets |
| `EmbedRowQ8_0/F32/Bf16` | one block | `out[i] = W[row][i]` dequantized row (token embedding lookup) |

## Launchers and dispatch

- `Gemv(base, type, rows, cols, row_bytes, x, out, stream)` — Q8_0/F32/BF16.
- `GemvRows(...)` — multi-token variant of the same.
- `GemvGrouped(base, type, expert_stride, ids, used, rows, cols, x, x_stride,
  out, stream)` — one launch covers all selected experts; `ids` is never
  read back to the host.
- `GemvGroupedPair(wa, wb, ...)` — Q8_0 only, returns false otherwise. For
  the fused `ffn_gate_up_exps.weight` [128][1408][2816] the caller passes
  `wa = base`, `wb = base + 704 * row_bytes` (gate rows 0–703, up rows
  704–1407) and `expert_stride = 1408 * row_bytes`.
- `GemvMulti(projs, n, x, stream)` — folds the Q8_0 subset (attention q/k/v)
  into one `Multi4` launch; non-Q8_0 or single projections fall back to
  plain `Gemv`, bit-identical either way.

## Numerical contract

Per-row dots accumulate in F32 in block order; the oracle is the scalar
`MatVec` in `reference.cpp`. Measured (`gemma4.rocm_kernels`, synthetic
Q8_0): Gemv Q8_0 1.26e-3 (tol 2e-3, near-zero refs against a 1e-3 floor),
BF16 3.11e-5, EmbedRow Q8_0 exact, GemvGrouped Q8_0 4.14e-5,
GemvGroupedPair 9.81e-4 (gate) / 1.39e-3 (up).

## Notes

- The source tier's `GemvGroupedSwiglu` was not ported: Gemma-4 uses
  gelu-tanh, so a SiLU epilogue would be a wrong-activation trap. The geglu
  split lives in `fused.hip.cpp`.
- `WarpReduceSum` is `__shfl_down` over 32 lanes; the activation row is read
  from L1/L2 once per wave, which is why a row is never split across waves.
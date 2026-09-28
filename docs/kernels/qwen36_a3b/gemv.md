# Qwen3.6-35B-A3B — Decode GEMV Tier (`gemv.hip.cpp`)

Source: `src/models/qwen36_a3b/kernels/rocm/gemv.hip.cpp` (386 lines).
Public API: `gemv.hpp` (`GemvType` = `kQ8_0 | kF32 | kBF16`). This tier
runs every linear projection at decode (batch 1) and the shared-expert /
router projections; it is the model's dominant memory consumer. Weights
are never dequantized at load — the kernels decode in place.

`Q8_0Block` (:16) is the GGUF `block_q8_0`: `{__half d; int8 qs[32]}`,
**exactly 34 bytes** (no padding; a `static_assert`-equivalent contract
shared with the scalar oracle). The half scale always lands on an even
byte offset, so `d` loads are 2-byte aligned at worst.

All dot-product kernels share the shape: one **wave** reduces one output
row with `#pragma unroll 4` over the row, `WarpReduceSum`
(`shfl_down`, :21), lane 0 writes. The activation row is re-read from
L1/L2 per wave; splitting one row across waves would multiply that
traffic above the weight row (the design note at :29).

## Kernels

### `GemvQ8_0` :34 — launcher `Gemv` :223 (kQ8_0)

`out[r] = Σ_k W_q8[r][k] · x[k]`. Block 128 handles **four consecutive
rows**: wave `w` owns row `blockIdx.x·4 + w`; grid `(rows+3)/4`. Per
Q8_0 block `b`: `acc += d · qs[lane] · x[b·32 + lane]` — one int8 per
lane per block, the scale broadcast. The 4× unroll keeps several weight
loads in flight.

- Measured (standalone bench `tools/bench/lmhead_bench.hip`, lm_head
  248320×2048 Q8_0): **2.336 ms = 231 GB/s ≈ 96 % of the measured
  240 GB/s peak.** Rejected variants: misaligned-uint32 vec4 loads
  214 GB/s, vec2 227 GB/s, vec4 + LDS-staged x 120 GB/s (occupancy
  loss). The kernel is at roofline; do not micro-optimize it further.
- The in-model 4.41 ms vs 2.34 ms "gap" was a profiler-scope artifact:
  the `output` segment ran from `Mark("output")` to the next step's
  `Mark("embed")`, absorbing the engine's synchronous 1 MB logits D2H
  and host sampling. Fixed by `Mark("sample")` in the executor; the
  GEMV itself measures 2.34 ms in-model, at roofline.

### `GemvGroupedQ8_0` :59 — launcher `GemvGrouped` :326 (kQ8_0)

Routed-expert GEMV: one launch computes all `used` selected experts of
one projection. A 128-thread block folds four consecutive
**(slot, row) pairs**: `pair = blockIdx.x·4 + wave`, `s = pair / rows`,
`r = pair % rows`. The expert id is read **from device memory**
(`ids[s]`), so the routed MoE never round-trips to the host. Row base:
`w + ids[s]·expert_stride_blocks + r·nblocks`; activation row
`x + s·x_stride`. Grid `ceil(used·rows / 4)`.

- Decode MoE: gate/up/down each 0.058–0.062 ms across all 1280 calls —
  uniform, launch-overhead-free.
- `ids[s]` is a dependent load before the streaming loop; with used=8
  and rows=512 the grid is 1024 blocks, enough waves to hide it.

### `GemvQ8_0Pair` :94 — launcher `GemvPair` :354 (kQ8_0 only)

Fuses the two decode gate/up projections of one matrix pair (same
`cols`, two weight stacks, two output rows) into a single launch:
block 128 handles four consecutive rows of the **virtual concatenated**
row space `rows_a + rows_b`; wave `w` picks `wa`/`out_a` for rows below
`rows_a`, else `wb`/`out_b` with `r - rows_a`. The per-row dot is
identical to `GemvQ8_0` (same unroll, lane mapping and butterfly), so
results are **bit-exact** to two separate launches — verified by
`qwen36_a3b_rocm_gemv_test`. Halves launch count for the shared-expert
gate/up pair.

### `GemvGroupedQ8_0Pair` :131 — launcher `GemvGroupedPair` :368 (kQ8_0 only)

Routed-expert pair: one launch computes gate **and** up for all `used`
selected experts. Extends the (slot, row) pairing of
`GemvGroupedQ8_0` to `2·used·rows` virtual pairs — `pair ≥ used·rows`
selects the `wb`/`out_b` half. Expert id `ids[s]` read from device
memory as before; grid `ceil(2·used·rows / 4)`. Bit-exact to the two
separate `GemvGrouped` calls (same oracle test). Replaces the decode
routed gate + up launches (80 launches/step saved across 40 layers).

### `GemvGroupedDense<T>` :176 — launcher `GemvGrouped` (kF32/kBF16)

Same (slot, row) pairing for the dense expert stacks; element-strided
loop `i = lane; i < cols; i += 32` with `static_cast<float>(row[i])`.

### `GemvF32` :205 / `GemvBf16` :225 — launcher `Gemv`

Plain one-wave-per-row GEMV (grid = rows, block 32) for the dense
encodings. Same unroll-4 + butterfly contract; BF16 converts per
element.

### `EmbedRowQ8_0` :247 / `EmbedRowF32` :255 / `EmbedRowBf16` :264 — launcher `EmbedRow` :275

Token-embedding lookup: dequantize **one row** of the embedding matrix.
Q8_0: one warp per quant block (`grid = cols/32`, block 32),
`out[b·32+lane] = d · qs[lane]`. F32/BF16: grid-stride copy/convert,
grid cap 65535. Called once per token (and per MTP draft token).

## Launchers

| Launcher | Line | Grid / block |
| --- | --- | --- |
| `EmbedRow` | :275 | Q8_0: `cols/32` × 32; dense: `min(ceil(cols/256), 65535)` × 256 |
| `Gemv` | :306 | Q8_0: `ceil(rows/4)` × 128; dense: `rows` × 32 |
| `GemvGrouped` | :326 | `ceil(used·rows/4)` × 128, all types; `expert_stride` is in **bytes**, divided by the element/block size per type |
| `GemvPair` | :354 | `ceil((rows_a+rows_b)/4)` × 128; returns `false` for non-Q8_0 (caller falls back to two `Gemv` calls) |
| `GemvGroupedPair` | :368 | `ceil(2·used·rows/4)` × 128; `expert_stride` in bytes; returns `false` for non-Q8_0 |

## Numerics

- Q8_0 dot: `float(d) · float(qs) · x` accumulated in float per lane,
  then float butterfly sum. The oracle (`qwen36_a3b_rocm_gemv_test`)
  reproduces this order; changing the unroll or lane mapping changes
  low bits — re-run the test, don't assume associativity.
- No fast-math intrinsics anywhere in this file.

## Optimization notes

- The decode roofline (~3.07 GB/token → ~78 tps) is set by this tier:
  lm_head (1 GB) + per-layer projections. `output` is now confirmed at
  roofline in-model (2.34 ms/step); the remaining decode headroom is
  launch-latency bound, not bandwidth bound.
- `GemvGrouped` reads `ids` per pair; a future bucket-compaction
  (reusing `RoutedCompact` from the prefill tier) would let decode share
  the WMMA path for long batches — see the MTP/speculative work in
  `docs/models/qwen3.6-35B-A3B/OPTIMIZATIONS.md`.
- Done: shared-expert gate/up and routed gate/up each fused into one
  pair launch (`GemvPair` / `GemvGroupedPair`), 80 launches/step saved.
  Remaining candidates: router (`Gemv` + `RouterTopK`) fusion was
  rejected — it would reorder the router dot product and break
  token-parity; `moe_shared` (0.56 ms roofline) and `moe_router`
  (0.09 ms) are launch-latency bound, so hipGraph capture of the decode
  step is the better lever.
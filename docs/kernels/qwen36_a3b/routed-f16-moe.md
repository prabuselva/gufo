# Qwen3.6-35B-A3B — Routed F16 WMMA MoE (`routed_f16.hip.cpp`)

Source: `src/models/qwen36_a3b/kernels/rocm/routed_f16.hip.cpp` (903
lines). Public API: `routed_f16.hpp` (`WeightType` = GGUF ids: kF32 0,
kF16 1, kQ5_1 7, kQ8_0 8, kQ4_K 12, kQ5_K 13, kBF16 30). Ported verbatim
from the Qwen3.8-Flash-Next route
(`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`).

**Why this exists**: the dp4a MMQ grouped path is ~55× slower per MAC
than the dense MMQ on gfx1151. The matrix-core F16 route with the
weights dequantized **after the LDS read** is the production prefill MoE
path (executor.cpp:571–593; `GemmMoe` only when this returns false).

## Quantization blocks

| Struct | Line | Size | Notes |
| --- | --- | --- | --- |
| `Q8_0Block` | :37 | 34 B | `{half d; int8 qs[32]}` |
| `Q4KBlock` | :45 | 144 B | `{half d, dmin; u8 scales[12]; u8 qs[128]}` — 256-elem superblock |
| `Q5_1Block` | :53 | 24 B | `{half d, m; u32 qh; u8 qs[16]}` — 32 elems |
| Q5_K | :98 | 176 B | Q4_K header + 32 high-bit bytes + Q4_K nibbles |

`RoutedF16RowBytes<kType>` (:101) gives the encoded row stride.

## The dequant trick (:77–:94, :123–:145)

A 4/5-bit code becomes an F16 through a **byte permute into the mantissa
of 1024.0**: `0x6400 | q` is exactly `1024 + q` for `q < 32` (the half's
ulp is 1 there). Then a packed subtract of 1024 (1152 for signed Q8_0
bytes carried as `q + 128` via an `XOR 0x80808080` sign flip) and one
packed FMA apply `w = q·scale + bias`, with `(scale, bias)` staged per
(row, K block) as a half2: `(d·sc, −dmin·mn)` for Q4_K/Q5_K, `(d, 0)`
for Q8_0, `(d, m)` for Q5_1.

- `CodesToHalves` :125 — `__builtin_amdgcn_perm` against `kHalfMagic
  = 0x64646464` (:94), `__hadd2` magic, `__hfma2`.
- `GatherBit` :110 / `SpreadHighBits` :117 — bit-matrix tricks to move
  the Q5_K/Q5_1 fifth bits into per-byte bit 4.
- `Bf16x2ToF16x2` :141 — BF16 → F16 by shifting into the F32 top half
  and converting (exact in the expert-weight exponent range).

The codes stay **packed** in LDS (4 bits Q4_K, 5 bits Q5_1/Q5_K), which
keeps a two-K-block stage at 11–12 KB and five blocks resident per WGP.

## `RoutedF16GEMMKernel<WeightType, BM, BN, BK=2, kPair>` :147

The single production kernel (template-instantiated per encoding and
tile width). `__launch_bounds__(256)`, `BM = 128` (or 256), `BN ∈ {16,
48, 64}` token rows, `BK = 2` K blocks per stage (one 32-byte Q4_K
nibble group).

**Grid / tile map** (:86–:93, :199–:210): grid `(m / kRows, n_tiles)`.
`tiles[y]` packs the **expert in the low 16 bits** and the token macro
tile index in the high 16, so no block is launched for an empty tile.
Block (x, y) computes rows `x·BM..` of the expert against its compact
rows `[pad_bounds[e] + j·BN, +BN)` and scatters to
`out[rows_out[c]][row]`. Row blocks of one tile are consecutive in
dispatch order so they share the tile's gathered activations through L2.

**Work partition**: 8 waves; each wave owns a 16-row weight tile
(`kWaveRowTiles = BM/128`); `sub_lane = lane & 15`, `half_id = lane >>
4` match the WMMA fragment layout of
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (:33, `v16h`/`v8f` vector
types :30–:31).

**LDS plan** (:175–:197): code plane `BM × kChunks × 16 B` with a
row-dependent XOR swizzle (:281) so a fragment read (one row per lane)
covers all bank groups; high-bit plane (Q5 forms); scale plane (half2
per row per K block); activation plane `[kb][16-elem quarter][token]
[16 B]` with `kActStride = BN + 1` padding so a fragment read is 256
contiguous bytes and the gather writes land on distinct banks. The
epilogue reuses the whole allocation (min 8 KB).

**Pipeline** (:570–:579): `fetch_stage` (global → registers, one stage
ahead; per-type decode of the 34-B Q8_0 misalignment, Q4_K superblock
header caching every 8 blocks, Q5_K high-bit gathering) →
`commit_stage` (registers → LDS, deriving the (scale, bias) half2 only
at commit so nothing waits on the loads) → `__syncthreads` → prefetch
next → `compute_stage` (LDS → nibble unpack → `CodesToHalves` → WMMA
into `acc[kWaveRowTiles][kTokTiles]`) → `__syncthreads`. For `kPair` or
wide Q5/Q8 tiles, one token tile's fragments are kept live at a time
(compiler barrier :542) and `live_tok_tiles` skips WMMA work past a
short bucket's end (:546).

**Epilogues** (three variants):
1. `kPair` (:581–:634): four waves computed gate rows, four the matching
   up rows in one launch; the pair is folded through LDS and the SwiGLU
   `up·gate·sigmoid(gate)` is applied before narrowing to F16
   (`out_half` = the down projection's input). `asm volatile` guards
   (:614/:616) preserve the separate-projection F32 evaluation order so
   fast-math cannot flip an F16 rounding tie.
2. `BN ≥ 48` with `out_half` (:639–:685): one wave writes a complete
   128-byte line; optional `swiglu_gate` multiply (decode-path gate
   fusion); padded stride `BM+2` makes the accumulator scatter
   conflict-free.
3. Generic (:686–:722): transpose each 16×16 tile through wave-local
   LDS, scatter 16 rows per token to `rows_out` (F32 `out`, or F16 with
   optional SwiGLU). `dst < 0` (padding rows) is skipped.

**Numerics**: WMMA accumulates the whole K extent in F32 with no
per-block correction (that is the point of dequantizing after LDS).
`SigmoidF`/`SiluF` (:21–:26) use `__expf` — this path is **not** pinned
bit-exact to the scalar oracle; quality is guarded by the model forward
test.

## Compaction kernels

### `RoutedPadBoundsKernel` :727

Single block (1024 threads): `padded[e] = round_up(counts[e], 16)`,
zero `cursors[e]`, then thread 0 runs the exclusive scan into
`pad_bounds[0..n_experts]`. 16-row padding matches the WMMA token tile.

### `RoutedScatterKernel` :750

Per routed slot: `c = pad_bounds[e] + atomicAdd(&cursors[e], 1)`;
`rows_token[c] = slot / k`, `rows_slot[c] = slot`. Order inside a
bucket is arbitrary — every output row depends only on its own inputs.

### `NarrowKernel<T>` :769 — launcher `NarrowActivations` :784

F32 → F16 (or BF16) narrowing of the activation batch, block 256.

## Launchers

| Launcher | Line | Contract |
| --- | --- | --- |
| `RoutedCompactRows` | :797 | `slots + n_experts·15` rows for the padded layout |
| `RoutedCompact` | :801 | memset `rows_token/rows_slot` to −1 (0xFF), then pad-bounds + scatter |
| `LaunchRoutedF16<BN>` | :817 | per-encoding instantiation; Q4_K/Q5_K only at `BN ≤ 48`; `BM=128, BK=2` |
| `RoutedF16Gemm` | :871 | validates `k % block_elems` (256 K-quants / 64 Q5_1,Q8_0 / 32 BF16), `n_tiles > 0`, exactly one of `out`/`out_half`; dispatches `tile_rows ∈ {16, 48, 64}`; returns false → executor falls back to `GemmMoe` |

Executor flow per prefill MoE (:561–:608): `RoutedCompact` →
`NarrowActivations` (F16) → `RoutedF16Gemm` gate (`out_half` + SwiGLU
pairing) → up → down (F32 `out` indexed by slot) → `MoeEpilogue`.

## Optimization notes

- This kernel is compute-bound at prefill sizes; the measured prefill
  MoE cost dropped below the dense projections once it landed (see
  `docs/models/qwen3.6-35B-A3B/OPTIMIZATIONS.md`).
- Tuning knobs, in rough order of expected payoff: `BN` selection per
  bucket size (today the executor picks one `tile_rows` per call),
  `BM = 256` instantiation (allowed by the static_assert, never
  launched), and extending `kPair` (gate+up fused) to all encodings.
- Any change here must be checked through a full prefill forward
  (user-run) — there is no standalone oracle for the F16 rounding
  contract; the `asm volatile` order guards are load-bearing.
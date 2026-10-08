# Qwen3.6-35B-A3B — Fused Operators (`kernels.hip.cpp`)

Source: `src/models/qwen36_a3b/kernels/rocm/kernels.hip.cpp` (1319 lines).
Public API: `kernels.hpp`. Everything except `ExpertCountsKernel` is in an
anonymous namespace. Activations are row-major F32. The scalar oracle in
`reference.cpp` pins every formula; the elementwise helpers below use
exact libdevice math (`expf`, `log1pf`) so kernel and CPU agree to
rounding — do not replace them with fast intrinsics on pinned paths.

## Device helpers (not kernels)

| Helper | Line | Contract |
| --- | --- | --- |
| `DSigmoid(x)` | :11 | `1/(1+expf(-x))`, exact `expf` |
| `DSilu(x)` | :14 | `x * DSigmoid(x)` |
| `DSoftplus(x)` | :17 | `x > 20 ? x : log1pf(expf(x))` |
| `WarpReduceSum(double)` | :21 | `shfl_down` 16→1, valid in lane 0 |
| `WarpReduceSumF(float)` | :28 | butterfly `shfl_xor`, valid in **all** lanes |
| `WarpReduceSumD(double)` | :34 | butterfly `shfl_xor` in double, all lanes |
| `BlockReduceSum(double, shared[32])` | :43 | warp→shared→warp0, broadcast via `shared[0]` |
| `BlockReduceMaxIndex(v, idx, sv, si)` | :65 | max with **lowest-index tie-break**, broadcasts value+index |

`WarpReduceSumF/D` are the all-lane form used where every lane needs the
result without a broadcast (delta-rule dots). `BlockReduceSum` divides
nothing — callers apply `/dim` where the norm definition needs it.

## Norms and elementwise

### `RmsNormKernel` :103 — launcher `RmsNormRows` :1030

`out[r][d] = x[r][d] * rsqrt(mean_d x² + eps) * gamma[d]` (`gamma` may be
null; plain `w * x`, **not** `1 + w`). Grid = rows, block =
`min(dim, 256)`. Sum-of-squares accumulates in **double**; the scale is
computed in float. Two passes over the row (second hits L1/L2). Used for
input/post-attention norms and the pre-q/k norms.

- Memory: 2 reads + 1 write of `rows × dim` floats. Launch-bound at
  decode sizes (rows = 1).
- Optimization idea: fuse with the consumer GEMV prologue or with RoPE
  (`attn_rope_norm` is 3 launches today).

### `RopeKernel` :123 — launcher `Rope` :1037

NEOX partial rotary on `x[rows][heads][head_dim]`: pair `(i, i + half)`
turns by `pos[row] * theta^(-2i/rotary_dim)`; only the leading
`rotary_dim` channels rotate. Grid = `rows * heads`, block =
`rotary_dim / 2` (32 threads for rotary 64 — a sub-wave block). `freq`
is recomputed with `powf` per thread per call.

- Optimization idea: cache the frequency table in constant/LDS; fuse
  q-norm + rope (same rows) into one kernel.

### `SplitQGateKernel` :147 — launcher `SplitQGate` :1046

Deinterleaves the fused attention projection: `qg` is
`[tokens][heads][2·head_dim]` with per-head `[q | gate]`; writes `q` and
`gate` (each `[tokens][heads·head_dim]`) so q alone can be normed and
rotated. Grid-stride, block 256, grid capped at 65535. Pure bandwidth:
1 read + 2 writes per element.

### `SwigluKernel` :166 — launcher `Swiglu` :1057

`gate[i] = silu(gate[i]) * up[i]`, in place. Grid-stride, block 256,
grid cap 65535. In decode the executor runs one launch over
`used × expert_ff` (all 8 selected experts folded).

### `SigmoidMulKernel` :175 — launcher `SigmoidMul` :1068

`x[i] *= sigmoid(g[i])`. Grid-stride, block 256, cap 65535.

### `AddKernel` :184 — launcher `Add` :1079

`a[i] += b[i]`, the residual update. Grid-stride, block 256, cap 65535.
Decode: 2 launches per layer (`add` stage 0.78 ms/step total).

- Optimization idea: fuse the residual add into the producer's epilogue
  (attn_out / ffn GEMV writes) to halve the traffic.

## Router and MoE epilogue

### `RouterTopKKernel` :193 — launcher `RouterTopK` :1089

Softmax top-k per token. Grid = tokens, block = `max(n_experts, 32)`
(256 for this model), dynamic shared = `n_experts` floats (`prob`).

1. Max over logits (`BlockReduceMaxIndex` with dummy index).
2. `prob[e] = expf(v - max)`; **double** sum; normalize into shared.
3. `k` rounds of `BlockReduceMaxIndex` over `prob` (lowest index wins
   ties); thread 0 writes `ids[token*k+slot]`, `weights[...]`; the
   winner masks itself to `-inf`; `__syncthreads()` per round.
4. Thread 0 renormalizes by `max(selected_sum, 2⁻¹⁴ = 6.103515625e-5)`.

- Cost: `k=8` full-block reductions with barriers — serial, ~8×(log
  reduction + sync) per token. `moe_router` stage is 1.59 ms/step
  (GEMV + this).
- Optimization idea: per-warp bitonic top-8 on 256 probs (8 per lane)
  removes all barriers; or fuse the softmax+top-k into the router GEMV
  epilogue.

### `ExpertCountsKernel` :1098 — launcher `ExpertCounts` :1107

`counts[e] += 1` per routed slot (`ids[i] >= 0`), preceded by
`hipMemsetAsync`. Grid-stride, block 256. Bounds the routed MMQ/WMMA
grid to the real largest bucket. Non-static kernel (only launcher is in
the header).

### `MoeEpilogueKernel` :250 — launcher `MoeEpilogue` :1117

`out[t][i] = Σ_s weights[t][s] · expert_out[t·k+s][i] +
sigmoid(gate[t·gate_stride]) · shared[t][i]`. Grid-stride over
`tokens × dim`, block 256. Accumulates in **double** across the k=8
slots plus the shared expert. Reads `k+1` rows per output element
(2048-float stride between slots — strided but each row is contiguous
across threads).

- Decode: 0.50 ms/step. Traffic-bound: (k+2) rows read, 1 written.
- Optimization idea: fold the shared-expert `SigmoidMul` and this
  epilogue; or have the down-projection GEMV scale by weights directly.

## GDN decode (single token)

Gated DeltaNet, ssm geometry k_heads 16 / v_heads 32 / head_dim 128.
`state[h]` is `[head_dim][head_dim]` (row j = value dim, column i = key
dim), updated in place. Recurrence per value head h (k-head = `h %
k_heads`):

```
decay = expf(a[h] * softplus(alpha[h] + dt[h]))   b = sigmoid(beta[h])
u_j   = Σ_i (decay·s[j][i]) · k[i]                delta_j = (v_j - u_j) · b
s[j][i] += delta_j · k[i]                         attn_j = (Σ_i s[j][i]·q[i]) / √head_dim
```

### `GdnConvKernel` :272 — launcher `GdnConv` :1128

Causal depthwise conv (kernel 4) + SiLU over one token, advancing the
rolling `history` `[kernel-1][channels]` (oldest first) **in place**:
`acc = w[kernel-1]·qkv[ch] + Σ_{t<kernel-1} w[t]·history[t][ch]`,
`convolved = silu(acc)`, then shift history and append `qkv`.
Grid-stride over channels, block 256, grid cap 65535.

- Tiny launch (channels = 8192): 0.38 ms/step across 30 layers is
  launch-latency dominated, not bandwidth.
- Optimization idea: fuse conv + `GdnNormQk` (same producer chain).

### `GdnNormQkKernel` :293 — launcher `GdnNormQk` :1138

L2-normalizes q and k halves per key head: `scale = 1/√(Σx² + eps)`
(**no `/dim`** — this is L2, not RMS). Grid `(k_heads, 2)`, block
`head_dim`; `blockIdx.y` selects q (`convolved[0, k_heads)`) or k
(`[k_heads, 2·k_heads)`). Value half stays in place.

### `GdnDeltaKernel` :316 — fallback (head_dim ≠ 128)

One block per value head, block = `head_dim`, dynamic shared 2·head_dim
(`sq`, `sk` staged q/k rows). Thread `j` owns state row j: pass 1 decays
the row and dots with `sk` in **double** → `u`; `delta = (v_j - u)·b`;
pass 2 updates the row and dots with `sq` in double → `attn_j`. Two
serial 128-iteration loops per thread; kept only as a generic fallback.

### `GdnDeltaDecodeKernel` :361 — dispatched at head_dim 128 (:1152)

The production decode delta rule. **One warp per (head, state row)**:
grid `(head_dim/4, v_heads)` = `(32, 32)`, block 128 (4 warps), warp `w`
takes row `blockIdx.x·4 + w` of head `blockIdx.y`. Lane `l` owns the
`float4` at column `l` of that row; q/k rows are `float4[lane]`.

```
s = state_row4 · decay                       (float multiplies)
du = WarpReduceSumD(Σ s·kv)        → delta = (v_j - du)·b
o  = s + delta·kv                (rounded to float before the q dot)
attn_j = float(WarpReduceSumD(Σ o·qv)) / √head_dim
state_row4 = o
```

- State is read **once** and written once, fully coalesced 16 B/lane
  (512 B/wavefront-aligned rows). Two double butterfly reductions per
  row; the q dot needs no second pass because `o` is already in
  registers.
- Numerics: `o = s + delta·kv` is rounded to float before the q dot so
  the oracle error is bit-identical to the reference path (attn 9.89e-5
  / 1.83e-4, state 1.9e-5 on the two oracle cases).
- Perf: `lin_delta` 0.041 ms × 30 layers = 1.25 ms/step. State traffic
  is 32 heads × 64 KB = 2 MB read + 2 MB written per layer — at 240 GB/s
  the floor is ~0.033 ms; ~75 % of roofline.
- Optimization ideas: skip the write when `delta == 0` (rare); merge
  alpha/beta GEMVs' outputs; nothing large left here.

### `GdnOutNormKernel` :408 — launcher `GdnOutNorm` :1165

Per-head RMS norm of `attn` (`ss/head_dim`, double sum) then the SiLU
output gate `row[i] *= scale · norm_w[i] · silu(z[...])`, in place. Grid
= v_heads, block = head_dim.

## Attention decode

GQA with sigmoid output gate: `q[heads][head_dim]`,
`k_cache/v_cache[n_kv][kv_heads][head_dim]` (already rotated),
`out[h][i] = (Σ_j softmax(q·k_j·scale)·v_j[i]) · sigmoid(gate[h·hd+i])`,
`scale = 1/√head_dim`, `kvh = h / (heads/kv_heads)`.

### `AttentionDecodeKernel` :428 — fallback (head_dim ≠ 256)

One block per head, block = head_dim, three passes over the whole KV
range with `scratch[h][n_kv]` holding scores: (1) scores + max (double
dot per token, serial over head_dim), (2) exp + double sum, (3) lane `i`
accumulates the value mixture in double over all j. Serial over n_kv →
O(n_kv) per block; only correct-size fallback, not perf-relevant.

### `AttentionDecodeSplitKernel` :496 — flash-decoding pass 1 (:1280)

Dispatched for head_dim 256: `splits = ceil(n_kv/64)` capped at 32,
`chunk = ceil(n_kv/splits)` (recomputed so `splits·chunk ≥ n_kv`). Grid
`(heads, splits)`, block 256 = 8 waves (`kWaves`).

- Lane `l` holds `qv[m] = q[l + 32m]`, m < 8 → every K/V load is a
  coalesced 128 B wavefront.
- Wave `w` processes tokens `j = j0 + w, +8, …`: float dot +
  `WarpReduceSumF`·scale → online update of per-wave
  `(wmax, wsum, acc[8])` (`acc[m] = acc[m]·r + e·v_j[l+32m]`).
- Fold: lane 0 publishes `s_max/s_sum`; block max `bmax`; scaled
  accumulators scatter to `s_acc[wave][l+32m]`; per-thread sum over the
  8 waves writes `part[head][split][0..255]`, `part[..][256] = bmax`,
  `part[..][257] = Σ_w s_sum[w]·exp(s_max[w] − bmax)`.
- `part` layout: `[head][split][head_dim + 2]`, sized `heads·32·258`.
- All online-softmax math is **float** (no double) — validated against
  the oracle by `qwen36_a3b_rocm_attention_test` over n_kv 1–2400.
- Perf: `attn_core` 0.090 ms/step (was 1.28 ms serial). At n_kv ≈ 1k
  this reads ~2·n_kv·2·128·4 B ≈ 2 MB/layer for 10 layers — near
  launch-latency floor; splits keep ≥1 block/wave per token chunk.

### `AttentionDecodeCombineKernel` :580 — flash-decoding pass 2

Grid = heads, block = head_dim. Per element `i`: `m = max_s
part[s][256]`; `num = Σ_s part[s][i]·exp(part[s][256]−m)`; `den = Σ_s
part[s][257]·exp(...)`; `out = num/den · sigmoid(gate)`. ≤32 iterations
per thread, trivially small.

- Optimization idea: fold the combine into pass 1 via a second
  `__syncthreads`-free atomic/last-block pattern to drop one launch per
  layer (10 launches/step).

## Attention prefill

### `AttentionPrefillNaive` :609 — fallback (head_dim > 256)

One workgroup per (token, head), `kTile = 128` online softmax with one
score tile in shared; double accumulation (matches the decode fallback).
Reference implementation only.

### `AttentionPrefillTiled` :697 — dispatched at head_dim ≤ 256 (:1305)

One workgroup handles `kQT = 32` queries of one head and streams
`kKT = 8`-token KV tiles: each K/V element is read once per 32 queries
instead of once per query. Grid `(heads, ceil(tokens/32))`, block 256.

- Shared: `sK/sV[8][257]` (`kStride = 256+1` padding kills bank
  conflicts on the head_dim-strided reads), `sScore[32][8]`,
  `rmax/rsum/corr[32]`.
- Score phase: one thread per (query, key) pair, causal mask as
  `-INFINITY`, **float** dot over head_dim (differs from the naive and
  decode kernels' double dots — pinned by the prefill oracle test).
- Online softmax per query runs in threads `tid < kQT`; the value mix
  runs in threads `tid < head_dim`, each holding `acc[kQT]` in
  registers.
- Output: `acc[qi]/rsum[qi] · sigmoid(gate)`.
- Optimization ideas: `kKT` is only 8 (one KV token per 32 threads of
  dot work); a WMMA score tile (like the routed F16 kernel) or kKT 32
  with float4 LDS reads would raise arithmetic intensity. Prefill
  attention is not yet a top stage.

## GDN prefill (chunked)

Same recurrence as decode, sequential over the chunk but batched over
heads/rows. Buffers are `[tokens][...]`; `channels =
2·k_heads·head_dim + v_heads·head_dim` (8192).

### `GdnConvPrefillKernel` :814 — launcher `GdnConvPrefill` :1173

Causal depthwise conv + SiLU over the chunk; taps before the chunk read
`history` (`[kernel-1][channels]`, oldest first). Accumulation order
matches the decode kernel (newest tap first) so a chunk reproduces
per-token decode bit-for-bit. Grid-stride over `tokens·channels`, block
256, grid cap 65535.

### `GdnHistoryUpdateKernel` :847 — launcher `GdnHistoryUpdate` :1186

Advances the conv state past the chunk: `out[j][ch] = qkv[tokens −
(kernel−1) + j][ch]` (or the old history when that index is negative).
Writes a **disjoint** buffer (caller copies back) so there is no
aliasing. Grid-stride over `(kernel−1)·channels`.

### `GdnNormQkPrefillKernel` :871 — launcher `GdnNormQkPrefill` :1197

`GdnNormQkKernel` batched over tokens: grid `(tokens·k_heads, 2)`,
block `head_dim`, same L2 norm (`1/√(Σx² + eps)`, no `/dim`).

### `GdnDeltaLoopKernel<kRows, kCols>` :905 — launcher `GdnDeltaLoop` :1208

The prefill recurrence: sequential over tokens, parallel over
(head, state row). **One warp owns `kRows = 8` state rows** of one value
head; lane `l` keeps `kCols = head_dim / warpSize` columns per row in
registers (`s[8][kCols]`, addressed `lane + c·blockDim.x`), so the state
lives in registers across the whole token loop — global traffic is one
read + one write of the state per chunk. Grid `(head_dim/kRows,
v_heads)`, block = warp (queried once via `hipDeviceGetAttribute`,
static). `kCols` instantiated 1–8 (4 for head_dim 128 at wave32).

Per token t:

1. Prefetch token `t+1`'s `kn`/`qn` **before** the reductions so the
   loads overlap the shuffles.
2. `decay = expf(a[h]·softplus(alpha[t·v_heads+h] + dt[h]))`,
   `b = DSigmoid(beta[t·v_heads+h])` (alpha/beta are per-token here).
3. Per row r: `s[r] *= decay`; `du = Σ s·kc`, `dq = Σ s·qc` →
   `pu[r]`, `po[r]` via `WarpReduceSumF` (float, all-lane).
4. `kq = WarpReduceSumF(Σ kc·qc)` — **once per token**, reused by all 8
   rows: `attn = (po[r] + delta·kq)·q_scale` folds the second dot
   through `dot(s_new, q) = dot(s_decayed, q) + delta·dot(k, q)`.
5. `delta = (v_t − pu[r])·b`; `s[r] += delta·kc`; lane `r` writes
   `attn[t][h·hd + j0 + r]`.

- No shared memory, no barriers on the token recurrence — the serial
  dependency is carried entirely in registers + butterfly shuffles.
- Numerics are float (vs the decode kernel's double dots); the oracle
  test pins the prefill path separately.
- Occupancy: 16×32 blocks of 32 threads = 512 waves of work per launch;
  rows-per-warp (`kRows`) is the tuning knob (8 today).

### `GdnOutNormPrefillKernel` :1004 — launcher `GdnOutNormPrefill` :1266

`GdnOutNormKernel` batched: grid `tokens·v_heads`, block `head_dim`,
double sum, RMS (`/head_dim`) + SiLU gate, in place.

## Cross-cutting optimization notes

- Decode fused ops are launch-latency bound: 30 GDN layers × ~5 tiny
  kernels + 10 attention layers × ~4. The stage table in
  `docs/kernels/qwen36_a3b/README.md` shows several stages at 0.3 ms
  where the bandwidth floor is <0.1 ms. CUDA-graph-style capture or
  kernel fusion (conv+normqk, norm+rope, router+topk) is the lever.
- Grid-stride launchers cap at 65535 blocks; elementwise kernels never
  reach it at this model's sizes.
- Every `*Prefill`/decode pair must keep reproducing the same
  recurrence; when touching one, run `qwen36_a3b_rocm_gdn_test`.
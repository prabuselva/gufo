# Gemma-4 GEMM Tier — `gemm.hip.cpp`

Prefill (batched) linear projections: `out[t][r] = sum_k W[r][k] * x[t][k]`.
The same three encodings the GEMV tier decodes drive this tier; weights
always stay in their native encoding.

## Launchers

- `Gemm(base, type, rows, cols, row_bytes, x, out, batch, stream)`
- `GemmMoe(base, type, rows, cols, row_bytes, x, ids, out, n_tokens,
  n_experts, n_expert_used, stream)` — routed projection over the stacked
  [n_experts x rows x cols] tensor; `out[(t*k+s)]` indexed by (token, slot).

## Dispatch

| Type | Path |
| --- | --- |
| Q8_0 | `qfn_mmq_q8_0_dense` / `qfn_mmq_q8_0_moe_raw` — the shared mmq tensor-core target from `qwen38_flash_next` (the only cross-model dependency of this file; see `docs/kernels/qwen36_a3b/gemm.md` for the mmq internals) |
| F32 | hipBLAS SGEMM (row-major via `OP_T`/`OP_N`), process-wide handle rebound to the executor stream per call |
| BF16 | `NarrowBf16` narrows the activations into a lazily grown scratch, then hipBLAS GemmEx BF16-in / F32-out |
| Q4_K / Q5_K | never here: `ValidateTypes` restricts dense tensors to Q8_0/F32/BF16 and routed K-quant experts go through `RoutedF16Gemm` |
| MoE F32/BF16 | `MoeVecFallback<T>` — one block per (token, slot) output vector, expert row base selected from `ids` on device |

## Contract

- `x` is [batch x cols] F32, `out` is [batch x rows] F32 (dense) or
  [batch x n_expert_used x rows] (MoE), row-major. Launches are
  asynchronous on `stream` (null = default stream).
- The BF16 scratch is a single growable buffer: only the BF16 path touches
  it and the executor drives one stream.

## Numerical contract

Measured (`gemma4.rocm_kernels`): Gemm Q8_0 batch-8 worst error 3.58e-3
against the scalar oracle, scale-relative tolerance 2e-2. The mmq path
accumulates int8 products in I32 with per-32-block rescale, so its error is
inherently larger than the GEMV float path; the routed pipeline test pins
the end-to-end effect (see routed-f16-moe.md).
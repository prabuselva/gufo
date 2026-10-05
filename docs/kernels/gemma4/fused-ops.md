# Gemma-4 Fused Elementwise Ops — `fused.hip.cpp`

Small per-row / per-token ops around the GEMM tiers: norms, residual adds,
the MoE router, and the geglu activation. Every helper mirrors the scalar
oracle in `reference.cpp` (same float formulas) so kernel and CPU reference
agree to rounding.

## Kernels

| Kernel | Launch | What it computes |
| --- | --- | --- |
| `RmsNormKernel` | grid `rows`, `min(dim, 256)` thr | `out = x * rsqrt(mean(x^2) + eps) * gamma`; sum-of-squares accumulated in double, `gamma == nullptr` means identity |
| `FusedAddRmsNormKernel` | grid 1 | `x += addend` (written back in place), then the RMSNorm above — bit-identical to `AddKernel` + `RmsNormKernel` for one row |
| `AddKernel` | 256-thr grid-stride | `a += b` over `count` floats |
| `RouterTopKKernel` | grid `tokens`, `max(n_experts, 32)` thr | softmax (max-subtracted, double sum) over the router logits, then k argmax rounds with the winner poisoned to -inf; weights renormalized by `max(sum, 6.103515625e-5)`; ties resolve to the lowest expert index |
| `ExpertCountsKernel` | 256-thr grid-stride | `atomicAdd` histogram of the router ids (launcher memsets first) — feeds `RoutedCompact` |
| `MoeEpilogueKernel` | 256-thr grid-stride | `out[t][i] = sum_s weights[t*k+s] * expert_scale[ids[t*k+s]] * expert_out[(t*k+s)][i]`, double accumulator; `expert_scale` is the per-expert `ffn_down_exps.scale` [128] |
| `GegluKernel<T>` | 256-thr grid-stride | `act[r][i] = gelu_tanh(gu[r][i]) * gu[r][ff + i]` over rows of `2*ff`; T = float or `__half` (F16 path computes in float and narrows) |

`gelu_tanh` is `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))` — the only
activation in the model. There is no SiLU/SwiGLU path anywhere in the gemma4
kernels.

## Contract

- All launches are stream-ordered; `eps` comes from the config (1e-6).
- `RouterTopK` writes `ids` (-1 never occurs here; padding -1s are produced
  by `RoutedCompact`'s own memset) and `weights` in `[token * k + slot]`
  order, matching the oracle's top-8 ordering.
- `FusedAddRmsNorm` mutates `x`: after the call `x` is the new residual.

## Numerical contract

Measured (`gemma4.rocm_kernels`, vs the CPU oracle): RmsNormRows 2.19e-7,
FusedAddRmsNorm 2.37e-7, Add exact, RouterTopK exact ids and weights,
GegluF32 2.07e-4, GegluF16 4.09e-3 (F16 narrowing of the product).
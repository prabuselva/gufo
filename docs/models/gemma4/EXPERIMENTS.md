# Gemma-4-26B-A4B — Experiments

One entry per A/B or decision. Record the hypothesis, the matched
measurement (same build except the lever, exclusive GPU via
`tools/bench/gpu_exclusive.sh`, ≤16K context first), the quality check, and
the verdict. Promote wins to [OPTIMIZATIONS.md](OPTIMIZATIONS.md); keep
losses here so they are not retried.

## Format

```
### YYYY-MM-DD — <lever>
Hypothesis:
Change:
Measurement: (pp/tg cells, candidate vs baseline, GB/s or t/s)
Quality: (parity test / greedy hash)
Verdict: retained | rejected | inconclusive
```

## Log

### 2026-10-05 — Reuse `gufo_qwen38_flash_next_mmq` for Q8_0 dense GEMM
Hypothesis: the qwen38_flash_next Q8_0 WMMA GEMM transfers to gemma4
geometry (K=2816, N∈{4096, 8192, 2048, 1024, 2112, 262144}) without
modification, so only the routed MoE needs a gemma4 port.
Change: link `gufo_qwen38_flash_next_mmq` from the gemma4 device model.
Measurement: pending M5.
Quality: pending M5/M6.
Verdict: pending

### 2026-10-05 — Port qwen36 `routed_f16` WMMA MoE
Hypothesis: the qwen36 dequant-after-LDS WMMA MoE extends to gemma4
geometry (K=2816, N=1408 fused gate+up / 2816 down, 128 experts, top-8,
Q8_0 **and BF16** expert types, per-expert `.scale` on down) with the
softmax router weights applied in the down epilogue.
Change: `src/models/gemma4/kernels/rocm/routed_f16.{hpp,hip.cpp}`.
Measurement: pending M5.
Quality: `gemma4_rocm_moe_test` vs CPU formula.
Verdict: pending
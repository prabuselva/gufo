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

### 2026-10-06 — MTP draft forward kernel (M8a)
Hypothesis: the 4-layer MTP draft sidecar (`mtp-gemma-4-26B-A4B-it-Q8_0.gguf`,
hidden 1024→2816 out, dense FFN 8192, fixed position, trunk KV read-only)
reproduces an independent CPU oracle on the GPU.
Change: `DeviceDraft` + `Executor::AttachDraft`/`DraftStep`
(`kernels/rocm/{device_model,executor}.{hpp,cpp}`).
Measurement: `gemma4.mtp` — step1 rel-to-scale 0.116 (argmax 500==500),
step2 0.162 (argmax 800==800), h_next 0.069/0.108.
Quality: independent CPU `ReferenceModel` oracle; argmax matches both steps.
Verdict: retained

### 2026-10-06 — MTP speculative decode (M8b)
Hypothesis: chaining up to k drafts through the MTP block and verifying all
k+1 rows in one batched trunk pass raises tg throughput while the trunk verify
keeps the lossless accept prefix.
Change: `Executor::Verify`/`RollbackVerify` + `Session::DecodeStep` speculative
path (`engine.{hpp,cpp}`); CLI `--speculative mtp --mtp-model` in bench/prompt.
Measurement: tg128 (Q8_K_XL, depth 0, exclusive GPU, greedy) —
no MTP 38.22; MTP n=1 36.60 (−4.2%); n=2 46.93 (+22.8%); n=4 59.99 (+57.0%).
Acceptance at n=4: drafted 110 / accepted 92. `kMaxVerifyRows=5` caps k at 4.
Quality: `gemma4.speculative` (chat prompt, 48 tokens) token-identical.
Longer real-text chat (256 tokens) diverges from non-spec greedy at one
small-margin token (~char 399, "taken to perform" vs "spent performing"):
the batched trunk verify that produces the speedup is not bit-identical to
single-token decode, so it can flip a near-tie. Inherent to batched
verification (llama.cpp MTP behaves the same); MTP stays opt-in so the default
greedy path is unchanged.
Verdict: retained (opt-in, n=4 default cap)
### 2026-10-06 — Vision tower precision (M9)
Hypothesis: the `gemma4v` tower can run its projections in BF16 (native mmproj
type) like the trunk, halving vision weight bytes with no quality cost.
Change: `vision/encoder.{hpp,cpp}` device driver + `vision/kernels.{hpp,hip.cpp}`
(im2col, position add, 2D RoPE, geglu_quick, avg-pool 3×3, std-norm, flash-style
non-causal attention), reusing `Gemm`/`RmsNormRows`/`Add`.
Measurement: `gemma4.vision_encoder` vs the CPU oracle on a fixed 96×96 gradient.
BF16 `GemmEx` path: oracle checksum −53.0989 vs device −64.2804, max abs diff
5.45 (58% of ref max 9.35), mean 0.407 — BF16 activation narrowing compounds
across 27 blocks well past any acceptable bound.
F32 path (BF16 weights upcast on the host, SGEMM): checksums identical
(−53.0989), max abs diff 2.9e-05, mean 4.5e-06 — pure F32 rounding.
Quality: independent CPU `ReferenceEncoder` oracle; test tolerance tightened to
1e-3 relative (was a 5% BF16-era bound).
Verdict: F32 retained as the default. The tower is ~1.2 GB and a rounding pass
next to the 26B trunk, so full precision costs nothing measurable and removes a
real quality gap; BF16 rejected.

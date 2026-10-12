# Gemma-4-26B-A4B — Benchmarks

All numbers on Strix Halo gfx1151, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy sampling, `--n-gen 128`.
Reference column: the local llama.cpp fork
(`$LLAMA_CPP/llama.cpp`, ROCm build, same
GGUFs). Preserve compiler/dependency versions when comparing.

Artifacts: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (trunk),
`mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (draft), `mmproj-BF16.gguf` (vision)
under `$MODELS_DIR/`.

## Reproduce

```sh
# Gufo
build/release/gufo bench --model "$GUFO_GEMMA4_GGUF" \
  --n-prompt 512,1024,2048,4096,8192,16384,32768,65536,102400 --n-gen 128 \
  --speculative off --repetitions 3
# llama.cpp fork reference
llama-bench -m "$GUFO_GEMMA4_GGUF" -p 512,1024,2048,4096,8192,16384,32768,65536,102400 -n 128
```

## Prefill (t/s)

Matched exclusive-GPU sweep, `--repetitions 3` (mean of 3 timed runs after a
warmup; run-to-run spread ≤ ~18 t/s at pp512 and ≤ ~7 t/s elsewhere), greedy,
Q8_K_XL. Reference: llama.cpp fork HIP build (`/tmp/llama-hip`, gfx1151,
`-ngl 99`, flash attention on by default), build 843d57505 (unchanged). Gufo
column re-measured 2026-10-07 on the committed binary `efd90672`: the
model-private dense-F16 WMMA route (Q8_0 dense projections run through a
binary16 WMMA GEMM at wide batch; see [EXPERIMENTS.md](EXPERIMENTS.md)) lifts
prefill ~10–13% at short context and ~2–7% at long context over the prior int8
`mul_mat_q` build. The binary16 shared-FFN activation path added in `efd90672`
(RMSNorm/GeGLU outputs stored as binary16 so the WMMA GEMMs consume half inputs
directly) is bit-identical (output hash `2669d2fd…`) and within ~1% of its
parent on a controlled A/B. These reps3 numbers run ~3–4% under the old
`--repetitions 1` table at short context, but that is a methodology correction,
not a regression: re-measuring the *same* `cdbf18a3` build the old table was
taken on drops the same amount at reps3 (reps1 single-samples were optimistic),
while the current build is ~3–4% *faster* than `cdbf18a3` at reps3 (gelu-tanh
fusion + shared-FFN cumulative), the unaffected decode row matching across
sessions (41.75 vs 41.59 t/s). The two effects nearly cancel, so the honest
reps3 figure lands near the old optimistic reps1 figure.

Gufo column re-measured 2026-10-08 on `967f2370` plus the attention spill fix:
`AttentionPrefillWmmaKernel` now derives `kKSplit = kWaves / kSTiles` softmax
partials (generalizing the hardcoded 2-way K split) and the head_dim-512 launch
uses `kQueryRows = 16` (q_frag + o_acc = 64 + 64 VGPRs), so `<16, 16, 8, 512>`
fits the 256-VGPR `launch_bounds(256, 1)` budget with zero scratch on clang 23,
where the old `<32, …, 512>` shape spilled. A matched same-toolchain A/B (HEAD
vs fix, same exclusive session) gives +0.7–1.4% at 2K–16K, +0.9% at 65K and
+3.1% at 100K (−0.7% at 32K, inside session spread); the 2× K/V re-read of the
4-way split does not cost at depth. The head_dim-256 (sliding-window) launch is
unchanged (`kQueryRows = 32`, `kKSplit = 2`, bit-identical); the 512 path's
4-partial accumulation stays inside the 3e-3 kernel-test tolerance (worst error
2.0e-3 vs the double reference).

| pp | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| 512 | 1946.26 | 1219.75 | +59.6% |
| 1024 | 1976.77 | 1235.00 | +60.1% |
| 2048 | 1965.19 | 1181.73 | +66.3% |
| 4096 | 1931.24 | 1149.99 | +67.9% |
| 8192 | 1782.49 | 1019.03 | +74.9% |
| 16384 | 1564.36 | 881.18 | +77.5% |
| 32768 | 1232.47 | — | — |
| 65536 | 869.80 | — | — |
| 102400 | 659.38 | — | — |

The M11 WMMA flash-attention prefill kernel (`AttentionPrefillWmmaKernel`)
replaced the scalar per-query head_dim dot-product kernel
(`AttentionPrefillTiledKernel`, kept as the `AttentionPrefillScalar` oracle).
Gufo now leads llama.cpp at every measured depth and the previously quadratic
gap is gone: throughput falls only gently with context (1965 → 659 t/s from
2K → 100K) because 25 of 30 layers are sliding-window (window 1024) and only
the 5 full-attention layers (head_dim 512) grow quadratically. Versus the
pre-M11 scalar path the same build is 2.05× faster at 512 rising to 9.71× at
16384 (951 → 1946 and 161 → 1564).

At 4K this is at the practical ceiling for a 4B-active MoE on gfx1151: the
fully-optimized sibling `qwen3.6-35B-A3B` (3B active) reaches 2373 t/s
(reference llama 2425); scaled by active params (×3/4) that predicts
~1780 t/s for gemma4, and the dense-F16 route lifts gemma4 just past it (1931)
because gemma4 is the denser of the two (4B active of 26B, ~15%, vs 3B of 35B,
~9%), so its wide prefill spends a larger share in the dense projections the
route accelerates. A pp4096 profile shows the remaining time is two WMMA GEMMs
— routed experts (`RoutedF16GEMM`, ~51% MFU) and the dense projections now on
the model-private binary16 WMMA route (previously int8 `mul_mat_q`) — with
attention down to ~13% of prefill ([EXPERIMENTS.md](EXPERIMENTS.md)).

## Decode (t/s, tg128)

| Config | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| Q8_K_XL, no MTP | 41.82 | 41.83 | ~0% |
| Q8_K_XL, MTP n=1 | 36.60 | — | — |
| Q8_K_XL, MTP n=2 | 46.93 | — | +12.2% |
| Q8_K_XL, MTP n=4 | 59.99 | — | +43.4% |

The no-MTP row is the current vec4-GEMV + float4-RMSNorm build (pp512 tg128
reps3; re-measured 41.82 on 2026-10-08 with the attention spill fix — the
decode path is unchanged, −0.5% is session spread). The vec4 Q8_0 GEMV lifted
it from 38.08 → 40.15 (+5.4%) and the float4
RMSNorm lifted it further to 42.02 (+4.6%); both are bit-identical end-to-end
(see [EXPERIMENTS.md](EXPERIMENTS.md)). The MTP rows predate both and were
measured on the scalar build; they run the same projections and norms in the
verify pass and would gain similarly, but are not re-measured here.

MTP is opt-in (`--speculative mtp --mtp-model <draft>`); `n` is
`--draft-tokens`, capped at 4 by `kMaxVerifyRows=5`. n=4 is the best measured
(acceptance 92/110). Speculative decode is distribution-preserving but not
bit-identical to non-speculative greedy beyond ~90 tokens (batched verify is
the speedup source; see [EXPERIMENTS.md](EXPERIMENTS.md)).

## Quantizations (Q4_K_M vs Q8_K_XL)

`gufo bench --repetitions 1`, exclusive GPU, greedy. Q4_K_M upcasts the dense
Q4_K/Q5_K/Q6_K tensors to Q8_0 at load and keeps the routed experts native
(Q4_K gate/up + Q8_0 down); see [EXPERIMENTS.md](EXPERIMENTS.md).

| Test | Q4_K_M (17.09 GiB) | Q8_K_XL (25.74 GiB) | gain |
| --- | --- | --- | --- |
| pp512 | 1004.83 | 966.06 | +4.0% |
| pp8192 | 261.66 | 251.19 | +4.2% |
| pp16384 | 163.14 | 160.66 | +1.5% |
| tg128 | 41.87 | 38.30 | +9.3% |

Q4_K_M is 33.6% smaller and faster on every axis: the dense upcast costs no
throughput (Q8_0 is the native dense tier) and the smaller resident footprint
helps decode bandwidth-bound tg most.

This table predates the vec4 Q8_0 GEMV (see the Decode section); both columns
were measured on the scalar build and both would gain from vec4, so the
relative Q4-vs-Q8 comparison is unaffected. It also predates the M11 WMMA
flash-attention prefill kernel, so its prefill rows are far below current-build
throughput — see "Q4_K_M on the current build" below for the superseding
figures.

## Q4_K_M on the current build (prefill + MTP)

`gufo bench --repetitions 1`, exclusive GPU, greedy, committed binary
`73f9dc60` (the bench kernels are unchanged since the WMMA-prefill + dense-F16
build; that commit only touched the serve path). Q4_K_M trunk + Q8_0 MTP draft.
Prefill is unaffected by MTP; the two prefill columns agree within run-to-run
noise.

| Test | no MTP | MTP n=4 | gain |
| --- | --- | --- | --- |
| pp2048 | 2143.30 | 2117.80 | ~0% |
| pp8192 | 1948.33 | 1956.52 | ~0% |
| pp16384 | 1641.22 | 1647.51 | ~0% |
| tg128 | 46.23 | 81.05 | +75.3% |

These are the current-build Q4_K_M figures and supersede the pre-M11 scalar
prefill rows in the Quantizations table above (pp8192 261 → 1948). The MTP
tg128 gain (+75%, same-build) is larger than Q8_K_XL's same-build +57%
(38.22 → 59.99, [DEVELOPMENT.md](DEVELOPMENT.md)): Q4_K_M decode is more
bandwidth-bound, so accepted drafts move throughput more.

## Serve (t/s, streaming)

| Scenario | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| short chat, MTP | — | — | — |
| 16K prompt + gen, MTP | — | — | — |

## Vision

| Scenario | Gufo | llama.cpp (mtmd) | gain |
| --- | --- | --- | --- |
| single 448px image pp | — | — | — |

## Notes

- (fill per phase; record build versions per table refresh)
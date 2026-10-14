# Gemma-4-26B-A4B — Benchmarks

All numbers on Strix Halo gfx1151, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy sampling, `--n-gen 128`.
Reference column: llama.cpp b11557 (`5c8854fd8`) ROCm build (HIP/Clang 23,
same GGUFs). Preserve compiler/dependency versions when comparing.
Main tables re-swept 2026-10-14: Gufo `d94fc396` and the reference measured
in matched exclusive-GPU windows, both `--repetitions 3`.

Artifacts: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (trunk),
`mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (draft), `mmproj-BF16.gguf` (vision)
under `$MODELS_DIR/`.

## Reproduce

```sh
# Gufo
build/release/gufo bench --model "$GUFO_GEMMA4_GGUF" \
  --n-prompt 512,1024,2048,4096,8192,16384,32768,65536,102400 --n-gen 128 \
  --speculative off --repetitions 3
# llama.cpp reference
llama-bench -m "$GUFO_GEMMA4_GGUF" -p 512,1024,2048,4096,8192,16384,32768,65536,102400 -n 128 -r 3
# MTP decode rows (Gufo only; llama.cpp has no gemma4 MTP path)
build/release/gufo bench --model "$GUFO_GEMMA4_GGUF" --n-prompt 512 --n-gen 128 \
  --speculative mtp --mtp-model "$GUFO_GEMMA4_MTP_GGUF" --draft-tokens 2 --repetitions 3
```

## Prefill (t/s)

Matched exclusive-GPU sweep, `--repetitions 3` (mean of timed runs; Gufo
run-to-run spread ≤ ~15 t/s except pp8192 ±46, reference noisier at small
depths: ±98 at pp512, ±65 at pp1024), greedy, Q8_K_XL. Both columns
re-measured 2026-10-14; the reference binary changed (b11557 `5c8854fd8`
replaces the previous `843d57505` build and is itself faster, e.g. pp2048
1181.73 → 1416.68), so reference rows were re-swept rather than reused, and
llama.cpp now runs 32K–100K too (the old `—` entries are gone).

Gufo column re-measured 2026-10-14 on `d94fc396`. Since the previous card
binary (`967f2370` + attention spill fix) the whole prefill program landed:
the fork prefill kernel stack `def99473` (fused MoeFinish, F16 dense window,
96-row MoE GEMM), upload-time qkv/gate/up weight fusion for wide prefill GEMM
`3046def5`, warp-wide QK/KV norm-rope kernels `d357cbdf` (+2.9% pp2048),
one-warp-per-token rope angles `c897e712` (+8.5% pp2048), the two-blocks-per-WGP
WMMA prefill attention `9e4d43dd`, 2048-row chunked prefill `be9a5d16` (+22%
pp2048/pp8192, fork parity), batched embedding lookup `42d78f7a`, merged/norm
fusions `4fd73934`/`858a9446`/`77e33192`, 128-row Q8_0 MoE tiles `9ef6ecc4`
and the templatized post-rope kernel `86d1191d`. Prefill is +49%…+91% above
the 2026-10-08 table from pp1024 up (+19% at launch-bound pp512). Note the shape: pp512 (2321) now sits *below*
pp1024 (2942) and the peak is at pp2048 (3213) — chunked prefill batches rows
into ~2048-row chunks, so sub-chunk prompts are launch-bound and reach their
best throughput at full chunks; pre-M11 tables were monotonically falling.

| pp | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| 512 | 2321.39 | 1414.00 | +64.2% |
| 1024 | 2941.99 | 1427.78 | +106.1% |
| 2048 | 3212.87 | 1416.68 | +126.8% |
| 4096 | 3152.43 | 1325.96 | +137.7% |
| 8192 | 2952.61 | 1214.88 | +143.0% |
| 16384 | 2636.22 | 1053.70 | +150.2% |
| 32768 | 2214.35 | 814.09 | +172.0% |
| 65536 | 1655.96 | 571.43 | +189.8% |
| 102400 | 1258.91 | 430.36 | +192.5% |

The M11 WMMA flash-attention prefill kernel (`AttentionPrefillWmmaKernel`)
replaced the scalar per-query head_dim dot-product kernel
(`AttentionPrefillTiledKernel`, kept as the `AttentionPrefillScalar` oracle).
Gufo now leads llama.cpp at every measured depth and the gap *widens* with
context (+64% at 512 → +192% at 100K): llama.cpp's full-attention layers grow
much faster with depth, while Gufo falls only gently (3213 → 1259 t/s from
2K → 100K) because 25 of 30 layers are sliding-window (window 1024) and only
the 5 full-attention layers (head_dim 512) grow quadratically. Versus the
pre-M11 scalar path (951 t/s at 512, 161 at 16384) the current build is 2.4×
at 512 and 16.4× at 16384.

This is far past the old 4K "practical ceiling" estimate for a 4B-active MoE
on gfx1151: scaling the then-optimized sibling `qwen3.6-35B-A3B` (2373 t/s at
pp4096, 3B active) by active params (×3/4) had predicted ~1780 t/s, and the
Gemma4 prefill program has since doubled past it (3152) — that estimate
ignored that gemma4 is the denser sibling (4B active of 26B, ~15%, so wide
prefill spends a larger share in the dense projections the WMMA route and the
upload-time weight fusion accelerate) and predates the chunked-prefill and
norm-fusion work entirely. Per-kernel profiles of the current build are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Decode (t/s, tg128)

| Config | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| Q8_K_XL, no MTP | 42.06 | 41.95 | +0.3% |
| Q8_K_XL, MTP n=2 | 57.76 | — | +37.3% |
| Q8_K_XL, MTP n=4 | 70.47 | — | +67.5% |

All rows re-measured 2026-10-14 on `d94fc396` (pp512 tg128 reps3, matched
windows; the MTP gain column is versus this build's no-MTP row). The no-MTP
row carries the full decode lever stack on top of the vec4 Q8_0 GEMV +
float4 RMSNorm (38.08 → 40.15 → 42.02 historically, both bit-identical; see
[EXPERIMENTS.md](EXPERIMENTS.md)): F16 attention-output twin in the epilogue
feeding half-in o_proj `22424308`, merged norm+RoPE launches `77e33192`, and
the fused dense-FFN up+gate decode GEMV `b4ca07bc`. Parity with llama.cpp
(42.06 vs 41.95, measured in one window) holds from the previous card (41.82
vs 41.83): both engines sustain the same effective per-token weight bandwidth
over the ~4B active experts and neither has moved much past it. The MTP rows
replaced the scalar-era entries
(46.93 / 59.99, and the below-no-MTP 36.60 at n=1, all pre-lever and now in
git history); the verify pass runs these same fused kernels, and n=1 was not
re-measured.

MTP is opt-in (`--speculative mtp --mtp-model <draft>`); `n` is
`--draft-tokens`, capped at 4 by `kMaxVerifyRows=5`. n=4 is the best measured
(92/110 draft acceptance tallied when MTP was introduced; not re-tallied in
the 2026-10-14 sweep). Speculative decode is distribution-preserving but not
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
throughput — see the Q4_K_M snapshot below for the superseding (but itself
pre-2026-10-program) figures.

## Q4_K_M snapshot (`73f9dc60`, prefill + MTP)

`gufo bench --repetitions 1`, exclusive GPU, greedy, committed binary
`73f9dc60` — a 2026-10-05 snapshot: it predates this card's 2026-10-14
sweep binary entirely (WMMA-prefill stack, chunked prefill, norm fusions and
all decode levers are missing), so these Q4 rows do not track the Q8 tables
above; 2026-10-14 sweep scope was Q8_K_XL. Q4_K_M trunk + Q8_0 MTP draft.
Prefill is unaffected by MTP; the two prefill columns agree within run-to-run
noise.

| Test | no MTP | MTP n=4 | gain |
| --- | --- | --- | --- |
| pp2048 | 2143.30 | 2117.80 | ~0% |
| pp8192 | 1948.33 | 1956.52 | ~0% |
| pp16384 | 1641.22 | 1647.51 | ~0% |
| tg128 | 46.23 | 81.05 | +75.3% |

These supersede the pre-M11 scalar prefill rows in the Quantizations table
above (pp8192 261 → 1948) but are themselves historical. The MTP tg128 gain
(+75%, same-build Q4 pair) is larger than Q8_K_XL's same-build +57%
(38.22 → 59.99, [DEVELOPMENT.md](DEVELOPMENT.md)): Q4_K_M decode is more
bandwidth-bound, so accepted drafts move throughput more. The current-build
Q8_K_XL MTP gain is now +37%/+68% (n=2/n=4 over no-MTP); this Q4 +75%
predates the whole 2026-10 program and needs re-measurement before its deltas
are quoted next to current figures.

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

- 2026-10-14 table refresh: Gufo `d94fc396` (release preset, plain toolchain;
  `platform_profile` N/A on this host), llama.cpp reference b11557
  (`5c8854fd8`, HIP/Clang 23, local build), exclusive-GPU windows, reps3,
  temperatures 35–61 °C. Prefill/Decode tables replaced the 2026-10-07/08
  entries (binaries `efd90672`/`967f2370`+spill-fix vs reference `843d57505`;
  git history retains them). The 2026-10 program between them: fork prefill
  kernel stack, chunked prefill, WMMA attention port, norm/RoPE fusion,
  upload-time weight fusion, and the decode lever stack (STATUS.md levers
  1–12, lever 13 rejected).
- Serve and Vision tables are still unmeasured.
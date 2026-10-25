# NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. ROCm 7.15 (HIP 7.15.26333,
AMD clang 23.0.0), `cmake --preset release` (RelWithDebInfo). All cells are
measured through `tools/bench/gpu_exclusive.sh` (exclusive VRAM; no resident
`gufo serve`) with the standard `gufo bench` protocol (greedy argmax, no chat
template, 3 repetitions, per-request phases; margins 5 % / 3 ms per
AGENTS.md). No cell is filled until the implementation lands; see
[STATUS.md](STATUS.md) for progress.

Exclusive windows drift ±5 % in absolute t/s from session to session
(2026-10-20: same binary 485–509 t/s at pp2048), so lever deltas are taken
from matched same-window interleaved A/B runs, never from table history
across sessions.

Artifact: `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (~23 GiB, IQ2_XXS
experts + NVFP4 Mamba + Q8_0 attention, 262144-token context, 74B total /
~9B active). Reference runtime for parity checks: latest llama.cpp master
(NVIDIA-verified recipe: `--cache-type-k/v q8_0` equivalent is **not** applied
by default in Gufo; text recipe temp 1.0 / top-p 0.95 is a serve default, not
`gufo bench` greedy).

## Prefill (t/s, prompt-only)

| depth | gufo | reference | date/rev |
| --- | --- | --- | --- |
| pp512 | — | — | — |
| pp2048 | 623.80 ± 0.00 | — | 2026-10-20 (prefill chunk 2048 → 8192; single-chunk depth, matched A/B flat −1.1..+2.2 % over 5 pairs) |
| pp8192 | 560.28 ± 0.00 | — | 2026-10-20 (prefill chunk 2048 → 8192 halves MoE weight re-streams; matched interleaved A/B +2.5/+2.4 %) |
| pp16384 | 464.20 ± 0.00 | 488.80 | 2026-10-20 (prefill chunk 2048 → 8192; matched interleaved A/B +3.3/+0.6 %; −5.0 % vs ref, absolute windows drift ±5 %) |
| pp32768 | — | — | — |
| pp65536 | — | — | — |
| pp102400 | — | — | — |

## UD-Q4_K_XL prefill vs llama.cpp (46.77 GiB artifact)

Same model weights (`UD-Q4_K_XL`, 46.77 GiB, shard 1 loaded with the rest of
the split), measured 2026-10-24 in consecutive exclusive windows: `gufo` at
`a8bb20dd` (`gufo bench --n-prompt D --n-gen 1`, one run/depth) vs llama.cpp
`ff5a23376` `llama-bench -fa on -ngl 99 -r 2` (HIP build
`build_b11721_rocm`, Vulkan build `build_nemotron_vulkan`). The reference
runtimes use all GPU layers and flat attention, matching Gufo settings.

| depth | gufo | llama ROCm | llama Vulkan | gufo vs ROCm |
| --- | --- | --- | --- | --- |
| pp2048 | 611.52 ± 0.00 | 521.15 ± 7.37 | 538.95 ± 1.15 | +17.3 % |
| pp4096 | 610.85 ± 0.00 | 507.95 ± 0.50 | 518.68 ± 1.92 | +20.3 % |
| pp8192 | 560.56 ± 0.00 | 503.51 ± 0.21 | 510.78 ± 0.50 | +11.3 % |
| pp16384 | 471.08 ± 0.00 | 492.43 ± 0.52 | 494.46 ± 3.39 | −4.3 % |
| pp32768 | 313.39 ± 0.00 | 466.49 ± 3.18 | 449.88 ± 0.27 | −32.8 % |
| pp65536 | 177.42 ± 0.00 | 436.76 ± 0.67 | 412.55 ± 0.39 | −59.4 % |

Gufo wins ≤8K depth and loses from 16K on: throughput degrades super-linearly
with depth (llama.cpp stays >430 t/s at 64K) and one pp65536 pass costs ~795 s
wall vs ~150 s for the reference. This depth collapse — the open prefill lever
at this revision — was fixed the same day by the WMMA attention kernel in the
next section. Reference decode (tg) cells for this artifact are not measured
yet.

## UD-Q4_K_XL prefill vs llama.cpp — WMMA attention (same artifact)

Profile of the table above put `AttnKernel<24,16>` at 52.3 % of the pp32768
GPU trace. Replaced by a WMMA fp16x3 hi/lo-split prefill kernel (see
[EXPERIMENTS.md](EXPERIMENTS.md) for the split-accumulator story and gates);
decode and partial tiles keep the F32 kernel. `gufo` built on `a8bb20dd` with
the WMMA kernel, 2026-10-24 exclusive windows, same bench command; llama
columns carried from the section above (unchanged references).

| depth | gufo | llama ROCm | llama Vulkan | gufo vs ROCm |
| --- | --- | --- | --- | --- |
| pp2048 | 616.76 ± 0.00 | 521.15 ± 7.37 | 538.95 ± 1.15 | +18.3 % |
| pp4096 | 636.04 ± 0.00 | 507.95 ± 0.50 | 518.68 ± 1.92 | +25.2 % |
| pp8192 | 637.86 ± 0.00 | 503.51 ± 0.21 | 510.78 ± 0.50 | +26.7 % |
| pp16384 | 613.48 ± 0.00 | 492.43 ± 0.52 | 494.46 ± 3.39 | +24.6 % |
| pp32768 | 551.44 ± 0.00 | 466.49 ± 3.18 | 449.88 ± 0.27 | +18.2 % |
| pp65536 | 449.28 ± 0.00 | 436.76 ± 0.67 | 412.55 ± 0.39 | +2.9 % |
| pp102400 | 379.66 ± 0.00 | — | — | — |

The super-linear depth collapse is gone: 16K→64K now costs 613→449 t/s
(−26.8 %) instead of 471→177 (−62.4 %), and Gufo leads llama ROCm at every
measured depth. The remaining ≥65K falloff is KV read traffic
(F32 cache streamed per tile), the Q8_0-cache lever, not the score phase.
Quality gates on the WMMA build: extended `rocm_attention` (≤2e-5 rel),
`rocm_ops`, and shipped-artifact `reference` + `device_step` all PASS.

## UD-Q4_K_XL decode vs llama.cpp (46.77 GiB artifact, batch 1)

Same artifact and bench command as the prefill sections, windows 2026-10-24/25
evening. "gufo pre" is `a8bb20dd` + WMMA; "gufo ILP" adds the two-chain Q6_K
decode gemv; "gufo ILP-all" extends the dual chain to every K-quant expert
tier ([EXPERIMENTS.md](EXPERIMENTS.md)); llama ROCm is
`build_b11721_rocm` `llama-bench -fa on -ngl 99 -r 2` with the prompt depth set
explicitly (earlier single cells were depth-ambiguous).

| context | gufo pre | gufo ILP | gufo ILP-all | matched A/B | llama ROCm | ILP-all vs llama |
| --- | --- | --- | --- | --- | --- | --- |
| tg128 @0 | 19.05/19.07/18.98 | 19.78/19.92/19.90 | 20.19/20.13/20.04 | +1.9 %, 3/3 pairs vs ILP | 22.23 ± 0.06 | −9.5 % |
| tg128 @16K | 18.96/18.88 | 19.61/19.58 | 19.86/19.87/19.90 | +1.8 %, 3/3 pairs vs ILP | 22.27 ± 0.10 | −10.7 % |
| pp16384 control | 615.4/610.0 | 610.2/612.9 | — | untouched | 490.26 ± 0.39 | +24.7 % |

`device_step` (UD artifact, exclusive) PASS on the ILP-all build; decode remains
10–12 % behind llama ROCm, the remaining profiled cost being Q8_0 gemv traffic
(33 % busy) and the Q5_K/funnel remainder around the ~240 GB/s ceiling.

## Decode (t/s, batch 1)

| context | gufo | reference | date/rev |
| --- | --- | --- | --- |
| tg128 @0 | 26.34 ± 0.11 | — | 2026-10-21 (parallel router top-k; n-prompt 1; matched interleaved A/B +19.7 % vs ring gemm 22.00; tg512 24.95) |
| tg128 @16K | 24.95 ± 0.06 | — | 2026-10-21 (n-prompt 16384; matched interleaved A/B +17.1 % vs ring gemm 21.28) |
| tg128 @64K | — | — | — |

## Memory

| metric | value | date/rev |
| --- | --- | --- |
| resident + weights @ load | — | — |
| KV + mamba state @ 262144 ctx | — | — |
| max ctx fits 24 GB recipe (≤98304 default?) | — | — |

## TODO cells

- Mamba-state bytes/token accounting (40 layers × d_inner×state per head).
- MoE decode roofline vs `tools/bench/build.sh` microbenches (IQ2_XXS GB/s).
- 28GB-recipe parity vs README guidance at ctx 65536.

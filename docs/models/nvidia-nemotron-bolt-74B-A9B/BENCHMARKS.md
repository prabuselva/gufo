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

## Decode (t/s, batch 1)

| context | gufo | reference | date/rev |
| --- | --- | --- | --- |
| tg128 @0 | 17.34 ± 0.09 | — | 2026-10-21 (packed-row decode gemv; n-prompt 1; matched interleaved A/B +19.6 % vs row-gemv 14.44) |
| tg128 @16K | 17.20 ± 0.03 | — | 2026-10-21 (n-prompt 16384; matched interleaved A/B +20.2 % vs row-gemv 14.29) |
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

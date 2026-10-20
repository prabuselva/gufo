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
| pp2048 | 486.51 ± 0.00 | — | 2026-10-20 (attn score 2x2 tile; matched A/B +1.2 %) |
| pp8192 | — | — | — |
| pp16384 | 354.24 ± 0.00 | 488.80 | 2026-10-20 (attn score 2x2 tile; matched A/B parity) |
| pp32768 | — | — | — |
| pp65536 | — | — | — |
| pp102400 | — | — | — |

## Decode (t/s, batch 1)

| context | gufo | reference | date/rev |
| --- | --- | --- | --- |
| tg128 @0 | — | — | — |
| tg128 @16K | — | — | — |
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

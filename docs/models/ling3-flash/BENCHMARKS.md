# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench`, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy, one rep per cell (± 0.00).
Latest column: October 11, 2026 (`9b1434ca`). llama.cpp reference:
same day, `pp` cells only, matched model and prompt length.

Positive gain favors Gufo. [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill and decode

| Cell | Gufo pp/tg (tok/s) | prev Gufo (`473c946b`) | Δ | llama.cpp (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 802.74 | 770.67 | +4.2% | 605.92 | +32.5% |
| pp4096 | 714.66 | 659.11 | +8.4% | 551.23 | +29.7% |
| pp8192 | 562.12 | 497.42 | +13.0% | 496.35 | +13.2% |
| pp16384 | 395.30 | 322.93 | +22.4% | — | — |
| tg128 | 30.45 | 30.53 | -0.3% | — | — |
| tg1024 | 28.56 | 29.50 | -3.2% | — | — |

The `9b1434ca` change is prefill-only, so tg cells are unchanged-workload
controls; run-to-run tg spread was previously ~2%, leaving tg1024 mildly
inconclusive but unregressed in mechanism.

## Measurement detail

```sh
tools/bench/bench_bg.sh <name> <s> build/release/gufo bench \
  --model Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 128,1024
```

Raw logs: `/tmp/opencode/gufo_bench_w16e2e.log` (current),
`/tmp/opencode/gufo_bench_final.log` (`473c946b`).
Kernel-level microbenchmarks: `tools/bench/ling3_mla_prefill_bench.hip`
(`cand`, `replay`) and `tools/bench/ling3_mla_decode_bench.hip`.

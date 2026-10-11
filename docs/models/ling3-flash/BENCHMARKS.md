# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench`, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy. Latest column: October 11, 2026,
working-tree routed-GEMM live-token-tile change; pp cells are medians of two
matched ABBA reps against the `9b1434ca` binary in the same exclusive
sessions, tg1024 medians of three. llama.cpp reference: October 11,
`pp` cells only, matched model and prompt length.

Positive gain favors Gufo. [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill and decode

| Cell | Gufo pp/tg (tok/s) | prev Gufo (`9b1434ca`) | Δ | llama.cpp (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 832.56 | 802.74 | +3.7% | 605.92 | +37.4% |
| pp4096 | 735.71 | 714.66 | +2.9% | 551.23 | +33.5% |
| pp8192 | 578.04 | 562.12 | +2.8% | 496.35 | +16.5% |
| pp16384 | 401.51 | 395.30 | +1.6% | — | — |
| tg128 | 30.45 | 30.45 | — | — | — |
| tg1024 | 28.95 | 28.56 | +1.4% | — | — |

The routed-GEMM change leaves decode on the unchanged GEMV path: matched
tg1024 runs overlapped (base 28.53/28.95/28.75, candidate 29.18/28.75/28.95),
so the tg delta is noise, not a gain; tg128 is carried forward from
`9b1434ca` un-re-measured.

## Measurement detail

```sh
tools/bench/bench_bg.sh <name> <s> build/release/gufo bench \
  --model Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 128,1024
```

Raw logs: `/tmp/opencode/gufo_bench_e2eab.log`,
`/tmp/opencode/gufo_bench_e2etg.log` (current),
`/tmp/opencode/gufo_bench_w16e2e.log` (`9b1434ca`),
`/tmp/opencode/gufo_bench_final.log` (`473c946b`).
Kernel-level microbenchmarks: `tools/bench/ling3_mla_prefill_bench.hip`
(`cand`, `replay`), `tools/bench/ling3_mla_decode_bench.hip`, and
`tools/bench/ling3_routed_bench.hip` (routed MoE GEMM/Fnv gate, `--probe`
for byte-level one-hot/dense oracle checks).

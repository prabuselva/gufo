# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench`, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy. Latest column: October 11, 2026,
bit-exact parallel router top-k; all cells are means of the two matched
ABBA reps against the `b59fdf3a` binary within the same full-recipe
exclusive runs (`gufo_bench_rrfull`). llama.cpp reference: October 11,
`pp` cells only, matched model and prompt length.

Positive gain favors Gufo. [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill and decode

| Cell | Gufo pp/tg (tok/s) | prev Gufo (`b59fdf3a`) | Δ | llama.cpp (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 836.79 | 809.03 | +3.4% | 605.92 | +38.1% |
| pp4096 | 738.07 | 715.15 | +3.2% | 551.23 | +33.9% |
| pp8192 | 575.05 | 563.95 | +2.0% | 496.35 | +15.9% |
| pp16384 | 393.45 | 388.38 | +1.3% | — | — |
| tg128 | 32.58 | 30.10 | +8.2% | — | — |
| tg1024 | 31.47 | 29.24 | +7.6% | — | — |

Both decode cells separate cleanly (new min above prev max in the matched
ABBA order). pp16384 is noisy in both binaries (spread ≈4%, orders overlap);
its direction matches the other pp cells but the magnitude is within noise.

## Measurement detail

```sh
tools/bench/bench_bg.sh <name> <s> build/release/gufo bench \
  --model Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 128,1024
```

Raw logs: `/tmp/opencode/gufo_bench_rrfull.log` (current, matched ABBA full
recipe), `/tmp/opencode/gufo_bench_rrabba2.log` (decode-only pair),
`/tmp/opencode/gufo_bench_e2eab.log`,
`/tmp/opencode/gufo_bench_e2etg.log`,
`/tmp/opencode/gufo_bench_w16e2e.log` (`9b1434ca`),
`/tmp/opencode/gufo_bench_final.log` (`473c946b`).
Kernel-level microbenchmarks: `tools/bench/ling3_mla_prefill_bench.hip`
(`cand`, `replay`), `tools/bench/ling3_mla_decode_bench.hip`, and
`tools/bench/ling3_routed_bench.hip` (routed MoE GEMM/Fnv gate, `--probe`
for byte-level one-hot/dense oracle checks).

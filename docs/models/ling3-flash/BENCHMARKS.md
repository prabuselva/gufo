# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench`, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy. Latest column: October 11, 2026,
vec4 Q8_0 dense GEMV on top of the bit-exact parallel router; all cells are
means of the two matched BAAB reps against the `967de4ba` binary within the
same full-recipe exclusive runs (`gufo_bench_v4full`). llama.cpp reference:
October 11, `pp` cells only, matched model and prompt length.

Positive gain favors Gufo. [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill and decode

| Cell | Gufo pp/tg (tok/s) | prev Gufo (`967de4ba`) | Δ | llama.cpp (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 834.67 | 837.12 | −0.3% | 605.92 | +37.8% |
| pp4096 | 736.66 | 733.60 | +0.4% | 551.23 | +33.6% |
| pp8192 | 575.25 | 573.90 | +0.2% | 496.35 | +15.9% |
| pp16384 | 393.27 | 384.19 | +2.4% | — | — |
| tg128 | 33.29 | 32.36 | +2.9% | — | — |
| tg1024 | 32.23 | 31.24 | +3.2% | — | — |

Both decode cells separate cleanly (new min 33.07/32.01 above prev max
32.45/31.33 in the matched BAAB order), and an earlier dedicated tg1024
BAAB (each `−r 3`, `gufo_bench_v4tg1k`) showed the same separation
(31.89 ± 0.27 and 31.93 ± 0.21 vs 31.50 ± 0.10 and 31.63 ± 0.24).
The pp cells move within noise except pp16384, which is noisy in both
binaries (spread ≈4%, orders overlap).

Decode is no longer dominated by the Q8_0 GEMV tier: the vec4 kernel is
~30% faster on every dense shape at kernel level, yet realizes ~3% decode,
so remaining wall time sits in MoE GEMV, other quant tiers and launch gaps.

## Measurement detail

```sh
tools/bench/bench_bg.sh <name> <s> build/release/gufo bench \
  --model Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 128,1024
```

Raw logs: `/tmp/opencode/gufo_bench_v4full.log` (current, matched BAAB full
recipe), `/tmp/opencode/gufo_bench_v4tg1k.log` (dedicated tg1024 ×3 pair),
`/tmp/opencode/gufo_bench_v4micro.log` (Q8_0 scalar vs vec4 kernel micro),
`/tmp/opencode/gufo_bench_rrfull.log` (router change, matched ABBA full
recipe), `/tmp/opencode/gufo_bench_rrabba2.log` (decode-only pair),
`/tmp/opencode/gufo_bench_e2eab.log`,
`/tmp/opencode/gufo_bench_e2etg.log`,
`/tmp/opencode/gufo_bench_w16e2e.log` (`9b1434ca`),
`/tmp/opencode/gufo_bench_final.log` (`473c946b`).
Kernel-level microbenchmarks: `tools/bench/ling3_mla_prefill_bench.hip`
(`cand`, `replay`), `tools/bench/ling3_mla_decode_bench.hip`, and
`tools/bench/ling3_routed_bench.hip` (routed MoE GEMM/Fnv gate, `--probe`
for byte-level one-hot/dense oracle checks).

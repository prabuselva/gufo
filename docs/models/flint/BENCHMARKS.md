# Flint Preview 40B-A2B — Benchmarks

Strix Halo gfx1151, exclusive GPU, Q8_0 GGUF. Dates and provenance:
[QUALITY.md](QUALITY.md). Raw rows:
[`artifacts/flint-c7-native-sweep.log`](artifacts/flint-c7-native-sweep.log).

## Single user, autoregressive

Native CLI scope on both sides (no HTTP): Gufo `gufo bench`, llama.cpp
`llama-bench` @ b11879 (`-ngl 99 -fa 1 -r 3`, KV f16). Each row matches
llama-bench's depth semantics: prefill a 512-token block at frontier
`depth = ctx − 640`, generate 128 tokens at that frontier. Gufo is one
warmed sample, mean `± sd` degenerate by design.

| ctx      | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
|----------|----------------:|---------------------:|-----:|----------------:|---------------------:|-----:|
| 2048     | 2043.02 | 1478.22 | +38.2 % | 41.81 | 47.72 | −12.4 % |
| 8192     | 2056.97 | 1375.79 | +49.5 % | 42.40 | 45.74 | −7.3 % |
| 16384    | 2013.30 | 1265.11 | +59.1 % | 41.74 | 43.72 | −4.5 % |
| 32768    | 2004.04 | 1080.82 | +85.4 % | 42.80 | 39.93 | +7.2 % |
| 65536    | 1960.51 |  860.59 | +127.8 % | 42.52 | 34.21 | +24.3 % |
| 131072   | 1966.43 |  611.30 | +221.7 % | 41.26 | 26.71 | +54.5 % |

Gufo's incremental pp512 and decode stay flat to the deepest frontier
(4 KV heads → the global layers read only 3.8 GB of KV per 512-token block
at 131 K); llama-bench degrades on both from ctx 32 K. The remaining
reference wins are the short-depth decode rows (ctx ≤ 16384): the C7
campaign optimized prefill only — decode stays a follow-up lever.

## Full-prompt chunked prefill (Gufo scope)

Whole prompt of `ctx` tokens prefilled in 4096-token chunks, tg128 after
the last chunk, one warmed sample. No llama.cpp counterpart: llama-bench
rows are incremental 512-token prefills, not whole-prompt rate.

| prompt tokens | pp (tok/s) | tg128 (tok/s) |
|--------------:|-----------:|-------------:|
| 2048    | 2677.90 | 42.05 |
| 8192    | 2159.85 | 41.81 |
| 16384   | 1628.35 | 42.69 |
| 32768   | 1077.38 | 42.66 |
| 65536   |  642.31 | 42.22 |
| 131072  |  332.55 | 42.37 |

The whole-prompt rate falls with prompt length because the global
(sliding-window-off) layers attend the whole prefix; the per-chunk MoE and
GEMM tiers are flat (chunk 8192 trial C7w changed nothing — see
[EXPERIMENTS.md](EXPERIMENTS.md)).

## Not measured / not supported

- HTTP serve sweep, loading time and memory tables: `TODO` — the C5 Flint
  serve runner is single-session and has not been through the shared
  `model-bench.py` harness.
- Concurrency and speculative tables: `N/A` — Flint serve is one serial
  session and no draft model exists for Flint.
- Q8 is the only quantized artifact in scope (`file_type=7` imatrix Q8_0).

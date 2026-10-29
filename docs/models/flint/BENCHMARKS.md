# Flint Preview 40B-A2B — Benchmarks

Strix Halo gfx1151, exclusive GPU, Q8_0 GGUF. Dates and provenance:
[QUALITY.md](QUALITY.md). Raw rows: full depth sweep
[`artifacts/flint-full-sweep.log`](artifacts/flint-full-sweep.log), earlier
native sweeps [`artifacts/flint-c7-native-sweep.log`](artifacts/flint-c7-native-sweep.log).

## Single user, autoregressive

Native CLI scope on both sides (no HTTP): Gufo `gufo bench`, llama.cpp
`llama-bench` @ b11879 (`-ngl 99 -fa 1 -r 3`, KV f16). Each row matches
llama-bench's depth semantics: prefill a 512-token block at frontier
`depth = ctx − 640`, generate 128 tokens at that frontier. Both engines were
measured in one exclusive session (full sweep log in `artifacts/`). Gufo is
one warmed sample, mean `± sd` degenerate by design.

| ctx      | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
|----------|----------------:|---------------------:|-----:|----------------:|---------------------:|-----:|
| 2048     | 2065.42 | 1425.93 | +44.8 % | 42.69 | 48.05 | −11.2 % |
| 4096     | 2065.62 | 1381.61 | +49.5 % | 41.65 | 47.44 | −12.2 % |
| 8192     | 2064.02 | 1269.81 | +62.5 % | 41.94 | 46.26 |  −9.3 % |
| 16384    | 2023.39 | 1093.01 | +85.1 % | 42.65 | 44.19 |  −3.5 % |
| 32768    | 2019.60 |  881.95 | +129.0 % | 42.89 | 40.49 |  +5.9 % |
| 65536    | 2011.00 |  619.08 | +224.8 % | 42.25 | 34.53 | +22.4 % |
| 131072   | 1962.88 |  385.39 | +409.3 % | 41.19 | 27.05 | +52.3 % |

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
| 2048    | 2778.63 | 42.01 |
| 8192    | 2246.51 | 41.19 |
| 16384   | 1628.35 | 42.69 |
| 32768   | 1077.38 | 42.66 |
| 65536   |  642.31 | 42.22 |
| 131072  |  332.55 | 42.37 |

The 2048 and 8192 rows were re-measured after the C8a multi-row router
tile (+3.7 % / +4.0 % over 2677.90 / 2159.85 in the same session); the
remaining rows are still at the C7v-revision measurements. The
whole-prompt rate falls with prompt length because the global
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

# Ling 3.0 Tiny benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. `Ling-3.0-tiny-Q8_0`
(7.83 GiB GGUF, single shard, F16 MoE with stride-4 MLA). Gufo `bench`,
exclusive GPU (`tools/bench/bench_bg.sh` + `bench_wait.sh` watchdog),
greedy, `build/release/gufo` on the Tiny-support branch (same tree as the
131/131 CPU + 13/13 `gpu-full -R ling3` close-out,
see [STATUS.md](STATUS.md)); measured 2026-10-18 16:59–17:05, log
`/tmp/opencode/gufo_bench_bgrid10.log`. Grid mirrors the retained
[Ling 3.0 Flash](../ling3-flash/BENCHMARKS.md) cells. [Status](STATUS.md) ·
[Experiment history](EXPERIMENTS.md)

No llama.cpp reference columns are measured for Tiny: the model shares the
`bailingmoe3` arch with Flash but a llama.cpp comparison run was not part
of this enablement. The Flash document is the perf reference for the
shared kernel stack.

## Prefill

Same prompt length, fresh context, timed prefill phase only.
`--n-prompt 2048,4096,8192,16384 --n-gen 1 -r 2`.

| Cell | Gufo (tok/s) | Spread |
| ---: | ---: | ---: |
| pp2048 | 5361.05 | ± 29.64 |
| pp4096 | 4454.50 | ± 0.70 |
| pp8192 | 3266.39 | ± 12.46 |
| pp16384 | 2020.17 | ± 2.15 |

Tiny keeps the same 4× attention/KV reduction as Flash (MLA rank 512/64)
on ~1/16 of the routed weight volume, so prefill runs 3.0–4.1× Flash's
retained column at equal depth and decays faster in relative terms
(5361→2020, −62 % across the 8× range, vs −50 % on Flash): the smaller
model hides kernel-launch and gather overheads less well at short depth
and its prefill becomes KV-bandwidth-shaped sooner. These cells were
measured on the binary with the prefill `v_b` gather-row fix
(EXPERIMENTS "Prefill MLA v_b contracted over token rows"); the earlier
smoke values (pp2048 5117.28, pre-fix, single rep) were superseded.

## Decode

Cold context (no cached prompt), timed generation only:
`--n-prompt 0 --n-gen 128,1024 -r 3`.

| Cell | Gufo (tok/s) | Spread |
| ---: | ---: | ---: |
| tg128 | 107.13 | ± 0.06 |
| tg1024 | 99.60 | ± 0.07 |

Decode runs at ~2.8× Flash's cold column (38.3/37.2) with a similar
depth shape (−7.0 % tg128→tg1024 vs −2.8 % on Flash); Tiny has no MTP
draft head (unlike Flash's speculative runs), so these are plain-decode
rates. Tiny is not a perf target — no decode kernel work was done for it
beyond the shared stack.

## Reproduce

```sh
source ./rocm_env.sh
tools/bench/bench_bg.sh <name> 300 build/release/gufo bench \
  --model Ling-3.0-tiny-Q8_0.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 1 -r 2   # prefill
tools/bench/bench_bg.sh <name> 300 build/release/gufo bench \
  --model Ling-3.0-tiny-Q8_0.gguf \
  --n-prompt 0 --n-gen 128,1024 -r 3               # cold decode
```

Raw logs: `/tmp/opencode/gufo_bench_bgrid10.log` (both tables above, one
exclusive window that also carries the Flash regression grid),
`/tmp/opencode/gufo_bench_tinysmoke.log` (superseded pre-fix smoke).

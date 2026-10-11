# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench` and llama.cpp
`llama-bench`, exclusive GPU (`tools/bench/gpu_exclusive.sh` /
`tools/bench/bench_bg.sh` + watchdog), greedy. Latest columns: October 11–12,
2026, Gufo HEAD `b25c42ed` (with direct MLA cache-row writes) as
`build/release/gufo`, against llama.cpp **b11557** (build `5c8854fd8`,
`-ngl 999`, flash-attn auto), freshly built clean on this host for both
backends: ROCm (`GGML_HIP=ON`, gfx1151 fatbin, HIP 7.15.26333, `GGML_NATIVE=OFF`)
and Vulkan (`GGML_VULKAN=ON`, Release, RADV `STRIX_HALO`). The model loads on
llama.cpp as arch `bailingmoe3 124B.A5.1B`. Positive gain favors Gufo; all
cells are single same-session matched runs (Gufo `-r 2`/`-r 3`, llama-bench
`-r 2`/`-r 3`). [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill

Same prompt length, fresh context, timed prefill phase only on both sides.

| Cell | Gufo (tok/s) | llama.cpp ROCm b11557 | Gain | llama.cpp Vulkan b11557 | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 840.25 | 659.34 | **+27.4%** | 618.64 | **+35.8%** |
| pp4096 | 732.80 | 633.69 | +15.6% | 590.35 | +24.1% |
| pp8192 | 576.11 | 583.71 | −1.3% | 559.02 | +3.1% |
| pp16384 | 402.21 | 507.39 | −20.7% | 504.40 | −20.3% |

Gufo wins short-context prefill decisively but the engines cross between 4K
and 8K: Gufo decays 840→402 (−52%) with depth while llama.cpp holds
659→507 (−23%) on both backends, so at 16K Gufo trails by about 20%. This is
the same prefill-attention decay tracked in `EXPERIMENTS.md`; the reference
now quantifies how much of it is hardware-ceiling and how much is kernel
shape (the ROCm reference is not special — Vulkan matches it here).
llama.cpp ROCm pp2048 repeats agreed within 0.5 tok/s across two exclusive
sessions (659.30 / 659.34; the run-1 standard deviation reflects one
clock-ramp rep).

## Decode

Cold context (no cached prompt), timed generation only: `gufo bench
--n-prompt 0 --n-gen 128,1024 -r 3` vs the direct `-pg 0,128`/`-pg 0,1024`
llama-bench tests. llama.cpp b11557 only yields split phase rates for
prefill-only and cold-decode tests; `pp,tg` pairs report one blended rate,
so pairs are kept as evidence notes rather than table cells.

| Cell | Gufo (tok/s) | llama.cpp ROCm b11557 | Gain | llama.cpp Vulkan b11557 | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| tg128 | 33.57 | 33.78 | −0.6% | 41.42 | −19.0% |
| tg1024 | 32.41 | 34.59 | −6.3% | 41.59 | −22.1% |

Gufo cold decode is on par with the ROCm backend at tg128, 6% behind at
tg1024, and about 20% behind llama.cpp **Vulkan**, whose decode (41.4–41.6)
is the strongest decode path measured on this host for the model. llama
decode does not slow as it generates (its tg1024 ≥ tg128 on both backends);
Gufo loses 3.5% from tg128→tg1024. Preparing a 2048-token frontier before
timed generation (`gufo bench --n-prompt 0 --n-gen 128,1024 -d 2048`) costs
Gufo a further ~4–6% (31.40 / 31.35), and the retained `pp2048+tg` blends
(Gufo `--n-prompt 2048` runs: 33.62 / 32.51 in-pipeline; llama ROCm 93.30,
Vulkan 108.25 per combined test) indicate llama.cpp's decode is nearly flat
with cache depth — consistent with its slower depth decay in prefill. The
decode-frontier depth cost is a concrete Gufo-side follow-up (see
`EXPERIMENTS.md` open items).

## MTP speculative decode

`--speculative mtp`, tg1024 after pp2048, matched BAAB exclusive pair
(October 11, `gufo_bench_ropebaab`): direct MLA cache-row writes (rope-tail
and kpe rows written by the kernels instead of per-token `hipMemcpyAsync`
appends) move 36.22 ± 0.18 / 36.22 ± 0.28 to 36.91 ± 0.22 / 37.33 ± 0.05
(**+2.5%**), each candidate run above both controls. The same change is
+1.4% on plain tg1024 (31.80/31.53 → 32.07/32.15, `gufo_bench_plain`). The
plain pipeline scope (after pp2048) differs from the cold decode table
above; llama.cpp b11557 has no accepted path for this model's MTP head, and
no speculative reference cells are claimed. Prefill paths are untouched
(prefill already wrote cache rows directly).

## Measurement detail

```sh
# Gufo (exclusive, warmed):
tools/bench/bench_bg.sh <name> <s> build/release/gufo bench \
  --model Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  --n-prompt 2048,4096,8192,16384 --n-gen 128,1024          # prefill + in-pipeline tg
build/release/gufo bench --model ... --n-prompt 0 --n-gen 128,1024 -r 3   # cold decode
# llama.cpp b11557 (same exclusive session), per backend:
llama-bench -m Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf \
  -pg 2048,0 -pg 4096,0 -pg 8192,0 -pg 16384,0 -pg 0,128 -pg 0,1024 \
  -r 2 -ngl 999 -o jsonl
```

Raw logs: `/tmp/opencode/gufo_bench_llgrid.log` (first full three-engine
grid, markdown), `gufo_bench_llgrid2.log` (split-phase JSONL grids, source
of the cells above), `gufo_bench_llcold.log` (cold tg128/tg1024),
`gufo_bench_gucid.log` (Gufo cold + `-d 2048` decode),
`gufo_bench_lltg.log` (pp2048+tg blends, depth-evidence),
`gufo_bench_ropebaab.log` and `gufo_bench_plain.log` (MTP/plain tg1024 BAAB
pairs for the MLA copy removal), `/tmp/opencode/gufo_bench_v4full.log`
(router/vec4-era matched BAAB full recipe),
`gufo_bench_v4tg1k.log`, `gufo_bench_rrfull.log`.
Build logs: `/tmp/opencode/llama_b11557_{rocm,vulkan}_{cfg,build}.log`.
Kernel-level microbenchmarks: `tools/bench/ling3_mla_prefill_bench.hip`,
`tools/bench/ling3_mla_decode_bench.hip`,
`tools/bench/ling3_routed_bench.hip`.

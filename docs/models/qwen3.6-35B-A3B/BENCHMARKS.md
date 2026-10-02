# Qwen3.6-35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo `b5b897be` (branch
`feat/support_qwen36_35b_a3b`), ROCm 7.15 (HIP 7.15.26333, AMD clang 23.0.0),
`cmake --preset release` (RelWithDebInfo, assertions off). Measured October 2,
2026 through `tools/bench/gpu_exclusive.sh` with no other GPU workload
resident. Artifact: `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (36.41 GiB).

`gufo bench` is greedy argmax with no chat template (thinking off), 3
repetitions, matching the reference methodology exactly (same flags, same
16-token decode prefix at depth 0). The `±` is the stddev over the timed
repetitions.

## Qwen3.6-35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | gufo (t/s) | reference (t/s) | ratio |
| ---: | ---: | ---: | ---: |
| 512 | 1518.26 ± 3.28 | 1475.15 | 103 % |
| 1024 | 1887.60 ± 1.64 | 2101.67 | 90 % |
| 2048 | 2054.91 ± 2.24 | 2311.31 | 89 % |
| 4096 | 1982.20 ± 5.33 | 2378.75 | 83 % |
| 8192 | 1777.60 ± 1.37 | 2299.53 | 77 % |
| 16384 | 1516.51 ± 2.47 | 2168.04 | 70 % |

| Workload | gufo (t/s) | reference (t/s) | ratio |
| --- | ---: | ---: | ---: |
| tg128 (greedy, no MTP) | 44.85 ± 0.06 | 51.59 | 87 % |

## MTP decode

MTP uses the draft head embedded in the same GGUF (`--speculative mtp
--mtp-model <same file> --draft-tokens N --min-draft-tokens N`), which pins the
per-round draft count to `N`. Acceptance is over the benchmark's synthetic
prompt.

| Draft tokens | tg128-mtp (t/s) | acceptance | tg128 no-MTP (t/s) | reference tg128-mtp (t/s) | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58.98 ± 0.05 | 88 % (56/64) | 44.85 | 60.23 | 98 % |
| 2 | 58.64 ± 0.03 | 70 % (63/90) | 44.85 | 65.01 | 90 % |

## Serving (production path)

`gufo serve` with `--speculative mtp --min-draft-tokens 2 --draft-tokens 2`
(the launch script default) reaches **65.93 t/s** at 84 % acceptance (147/175),
measured over HTTP with a synchronized client (see
[EXPERIMENTS.md](EXPERIMENTS.md) "MTP is the throughput lever"). This meets and
exceeds the ~60 t/s target and the reference's 65.01 t/s.

## Reading the numbers

The production serving path (MTP, adaptive draft) meets the throughput target.
On a strict like-for-like `gufo bench` basis, however, three real gaps to the
reference remain; they are kernel-efficiency differences, not methodology:

- **Long-context prefill (70 % at 16384).** The 10 full-attention layers are
  O(n²) and the WMMA kernel re-reads each KV head's prefix once per head-pair
  block (`kWmmaHeads=2` splits the 8-head GQA group across 4 blocks) — a 4×
  redundant global KV read that makes long-context attention memory-bound. The
  reference's prefill stays flat (~2.2 K t/s) where ours collapses to 1517.
  Root cause and the planned fix (one block per KV group) are in
  [OPTIMIZATIONS.md](OPTIMIZATIONS.md) "Prefill attention WMMA + row-split GDN"
  and [EXPERIMENTS.md](EXPERIMENTS.md) "Prefill attention".
- **Greedy decode (87 %).** At 44.85 t/s the step moves ~3.07 GB of active
  weights/token, i.e. ~138 GB/s effective against the ~240 GB/s streaming peak
  (roofline ~78 t/s, [OPTIMIZATIONS.md](OPTIMIZATIONS.md)). The Q8_0 `lm_head`
  GEMV is at the DRAM ceiling (231 GB/s), but the grouped MoE and `lin_gemm_in`
  GEMVs run well below it, so batch-1 greedy is efficiency-bound, not
  bandwidth-bound. The reference's 51.59 t/s (~159 GB/s) closes part of that
  gap. MTP is the production lever precisely because it amortizes these reads
  over k+1 verified tokens.
- **MTP n=2 (90 %).** Our second draft token is accepted far less often (70 %
  vs 88 % at n=1), so pinning n=2 does not improve on n=1, whereas the
  reference's n=2 (65.01) clearly beats its n=1 (60.23). The reference itself
  flags the n=1/n=2 spread on this quant as run-dependent; the adaptive serving
  path (84 % acceptance, 65.93 t/s) is the retained configuration.

## Reproduce

With no other GPU workload resident (wrap each command in
`tools/bench/gpu_exclusive.sh` for exclusive VRAM):

```sh
cmake --preset release && cmake --build --preset release --parallel 4
MODEL=.../Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
./build/release/gufo bench --model "$MODEL" \
  --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
for N in 1 2; do
  ./build/release/gufo bench --model "$MODEL" \
    --speculative mtp --mtp-model "$MODEL" \
    --draft-tokens $N --min-draft-tokens $N \
    --n-prompt 512 --n-gen 128 --repetitions 3
done
```

Greedy, thinking off. Kernel findings in [EXPERIMENTS.md](EXPERIMENTS.md) and
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).
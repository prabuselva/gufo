# Qwen3.6-35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. ROCm 7.15 (HIP 7.15.26333,
AMD clang 23.0.0), `cmake --preset release` (RelWithDebInfo, assertions off).
**Both columns are measured on this machine in the same exclusive-GPU session**
on October 3, 2026 through `tools/bench/gpu_exclusive.sh` (no other GPU workload
resident): **gufo** = this fork `569c96d9` + the MTP-draft-skip fix
(`feat/support_qwen36_35b_a3b`, `build/release/gufo`); **reference** =
`gufo_slimsami` `388ff05` (`feat/qwen35moe-35b-a3b`, `build_rocm10/gufo`).
Artifact: `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (36.41 GiB).

`gufo bench` is greedy argmax with no chat template (thinking off), 3
repetitions, identical flags for both binaries. The `±` is the stddev over the
timed repetitions. Measuring the reference on this machine rather than quoting
its own numbers matters: the reference runs faster here than on its origin
machine, so the same-machine ratios are lower than its self-reported ones.

## Qwen3.6-35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | gufo (t/s) | reference (t/s) | ratio |
| ---: | ---: | ---: | ---: |
| 512 | 1533.78 ± 9.71 | 1666.62 ± 5.12 | 92 % |
| 1024 | 1962.93 ± 25.56 | 2219.82 ± 1.89 | 88 % |
| 2048 | 2171.57 ± 8.08 | 2424.57 ± 1.13 | 90 % |
| 4096 | 2153.26 ± 1.59 | 2467.13 ± 2.72 | 87 % |
| 8192 | 2039.56 ± 1.94 | 2359.84 ± 10.53 | 86 % |
| 16384 | 1883.42 ± 24.09 | 2234.33 ± 0.98 | 84 % |

| Workload | gufo (t/s) | reference (t/s) | ratio |
| --- | ---: | ---: | ---: |
| tg128 (greedy, no MTP) | 44.43 ± 0.04 | 50.87 ± 0.05 | 87 % |

## MTP decode

MTP uses the draft head embedded in the same GGUF (`--speculative mtp
--mtp-model <same file> --draft-tokens N --min-draft-tokens N`), which pins the
per-round draft count to `N`. Acceptance is over the benchmark's synthetic
prompt.

| Draft tokens | gufo tg128-mtp (t/s) | gufo acceptance | reference tg128-mtp (t/s) | reference acceptance | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58.85 ± 0.16 | 88 % (56/64) | 59.37 ± 0.10 | 81 % | 99 % |
| 2 | 58.62 ± 0.09 | 70 % (63/90) | 63.41 ± 0.15 | 73 % | 92 % |

## Serving (production path)

`gufo serve` with `--speculative mtp --min-draft-tokens 2 --draft-tokens 2`
(the launch script default) reaches **65.93 t/s** at 84 % acceptance (147/175),
measured over HTTP with a synchronized client (see
[EXPERIMENTS.md](EXPERIMENTS.md) "MTP is the throughput lever"). This meets and
exceeds the ~60 t/s target and the reference's measured bench MTP n=2 (63.41).

## Reading the numbers

The production serving path (MTP, adaptive draft) meets the throughput target.
On a strict like-for-like `gufo bench` basis measured on this machine, the
reference is faster on every point; the gaps are kernel-efficiency differences,
not methodology:

- **Long-context prefill (92 % at 512 → 84 % at 16384).** The earlier monotonic
  collapse (91 % → 68 %) is fixed. The dominant cause was not the attention
  kernel itself: every plain (non-speculative) prefill was still running the MTP
  draft layer through the scalar oracle attention (`AttentionPrefillTiled`,
  19.3 % of pp16384 GPU time) because the draft had no f16 KV mirror and so fell
  off the WMMA path. The fix skips the draft entirely unless MTP is enabled
  (matching the reference, which gates the draft on the speculative session
  mode) and gives the draft an f16 mirror so that when MTP *is* enabled it uses
  the same WMMA kernel as the trunk. That lifted pp16384 from 1515 to 1883 t/s
  and flattened the curve (peak 2172, only a mild decline to 1883 at 16 K). The
  residual 84 % at 16384 is the remaining kernel-efficiency gap: the reference's
  **head-major packed KV fast path** (`attention_wmma.hip`, gated to
  `batch_size >= 1024`) repacks each KV head into a contiguous `[context]
  [head_dim]` slab so consecutive key rows sit 512 B apart (fully coalesced)
  instead of our position-major 1 KB stride (the other KV head interleaved), and
  it stores V pre-transposed straight into the WMMA fragment layout, dropping a
  per-tile LDS transpose + barrier. That port is the next lever; see
  [OPTIMIZATIONS.md](OPTIMIZATIONS.md) "Prefill attention WMMA + row-split GDN".
- **Greedy decode (87 %).** At 44.43 t/s the step moves ~3.07 GB of active
  weights/token, i.e. ~136 GB/s effective against the ~240 GB/s streaming peak
  (roofline ~78 t/s, [OPTIMIZATIONS.md](OPTIMIZATIONS.md)). The Q8_0 `lm_head`
  GEMV is at the DRAM ceiling (231 GB/s), but the grouped MoE and `lin_gemm_in`
  GEMVs run well below it, so batch-1 greedy is efficiency-bound, not
  bandwidth-bound. The reference's 50.87 t/s (~155 GB/s) closes part of that
  gap. MTP is the production lever precisely because it amortizes these reads
  over k+1 verified tokens.
- **MTP n=1 is parity (99 %); n=2 lags (92 %).** At n=1 we match the reference
  (58.85 vs 59.37) while accepting *more* (88 % vs 81 %). At n=2 our acceptance
  collapses to 70 % (vs the reference's 73 %), so pinning n=2 does not improve
  on n=1 (58.62 ≈ 58.85), whereas the reference's n=2 (63.41) clearly beats its
  n=1 (59.37). The adaptive serving path (84 % acceptance, 65.93 t/s) is the
  retained configuration. The draft now runs the WMMA kernel through its f16 KV
  mirror; acceptance is bit-identical to the pre-fix scalar-oracle draft (56/64
  and 63/90) with an identical output hash, confirming the WMMA draft reproduces
  the oracle exactly.

## Reproduce

With no other GPU workload resident (wrap each command in
`tools/bench/gpu_exclusive.sh` for exclusive VRAM). Run the same sequence for
both binaries — `B=./build/release/gufo` (gufo) and
`B=../gufo_slimsami/build_rocm10/gufo` (reference):

```sh
MODEL=.../Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
"$B" bench --model "$MODEL" \
  --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
for N in 1 2; do
  "$B" bench --model "$MODEL" \
    --speculative mtp --mtp-model "$MODEL" \
    --draft-tokens $N --min-draft-tokens $N \
    --n-prompt 512 --n-gen 128 --repetitions 3
done
```

Greedy, thinking off. Kernel findings in [EXPERIMENTS.md](EXPERIMENTS.md) and
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).
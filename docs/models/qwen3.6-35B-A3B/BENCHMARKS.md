# Qwen3.6-35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. ROCm 7.15 (HIP 7.15.26333,
AMD clang 23.0.0), `cmake --preset release` (RelWithDebInfo, assertions off).
**Both columns are measured on this machine in the same exclusive-GPU session**
on October 3, 2026 through `tools/bench/gpu_exclusive.sh` (no other GPU workload
resident), interleaved candidate→reference→reference→candidate to cancel
order/thermal drift: **gufo** = this fork `082a8432` + the fused GDN prefill
conv+RMSNorm kernel (`feat/support_qwen36_35b_a3b`, `build/release/gufo`);
**reference** = `gufo_slimsami` `388ff05` (`feat/qwen35moe-35b-a3b`,
`build_rocm10/gufo`). Artifact: `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (36.41 GiB,
262144-token context).

`gufo bench` is greedy argmax with no chat template (thinking off), 3
repetitions, identical flags for both binaries. The `±` is the stddev over the
timed repetitions. Measuring the reference on this machine rather than quoting
its own numbers matters: the reference runs faster here than on its origin
machine, so the same-machine ratios are lower than its self-reported ones.

## Qwen3.6-35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | gufo (t/s) | reference (t/s) | ratio |
| ---: | ---: | ---: | ---: |
| 512 | 1585.5 ± 7.8 | 1648.1 ± 10.0 | 96 % |
| 1024 | 2009.6 ± 4.4 | 2201.6 ± 6.1 | 91 % |
| 2048 | 2203.1 ± 3.8 | 2403.3 ± 4.1 | 92 % |
| 4096 | 2194.7 ± 3.8 | 2425.1 ± 2.2 | 90 % |
| 8192 | 2077.8 ± 2.4 | 2341.1 ± 4.2 | 89 % |
| 16384 | 1952.7 ± 1.7 | 2217.8 ± 1.7 | 88 % |
| 32768 | 1703.3 ± 11.8 | 1941.7 ± 15.1 | 88 % |
| 65536 | 1358.4 ± 6.7 | 1592.3 ± 1.1 | 85 % |
| 102400 | 1114.7 ± 7.9 | 1331.4 ± 2.3 | 84 % |

| Workload | gufo (t/s) | reference (t/s) | ratio |
| --- | ---: | ---: | ---: |
| tg128 (greedy, no MTP) | 44.87 ± 0.04 | 50.80 ± 0.04 | 88 % |

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

- **Long-context prefill (96 % at 512 → 88 % at 16–32 K → 84 % at 100 K).** The
  earlier monotonic collapse (91 % → 68 %) is fixed. The dominant cause was not
  the attention kernel itself: every plain (non-speculative) prefill was still
  running the MTP draft layer through the scalar oracle attention
  (`AttentionPrefillTiled`,
  19.3 % of pp16384 GPU time) because the draft had no f16 KV mirror and so fell
  off the WMMA path. The fix skips the draft entirely unless MTP is enabled
  (matching the reference, which gates the draft on the speculative session
  mode) and gives the draft an f16 mirror so that when MTP *is* enabled it uses
  the same WMMA kernel as the trunk. That lifted pp16384 from 1515 to 1883 t/s
  and flattened the curve; the fused GDN prefill conv+RMSNorm front-end
  ([EXPERIMENTS.md](EXPERIMENTS.md)) adds ~1 % more, to a peak 2203 (pp2048)
  and 1953 at 16 K (the table's gufo column). Hoisting the GDN decay/beta/q·k
  out of the serial recurrence (prep kernels, [EXPERIMENTS.md](EXPERIMENTS.md))
  adds a further ~2 % bit-exact, and prefetching the recurrence's critical-path
  scalars one token ahead adds ~0.5 % more, so the current tree is ~2-3 % above
  the table at every depth (prep measured pp2048 +3.4 %, pp16384 +2.8 %,
  pp102400 +1.6 %; prefetch +0.4-0.8 % on top). The
  residual 88 % at 16384 is the remaining kernel-efficiency gap in the attention
  core. The reference's
  **head-major packed KV fast path** (`attention_wmma.hip`, gated to
  `batch_size >= 1024`) repacks each KV head into a contiguous `[context]
  [head_dim]` slab so consecutive key rows sit 512 B apart (fully coalesced)
  instead of our position-major 1 KB stride (the other KV head interleaved), and
  it stores V pre-transposed straight into the WMMA fragment layout, dropping a
  per-tile LDS transpose + barrier. We ported that path bit-exactly (repacked ==
  current, 0 mismatches over 4.2 M outputs) and measured it interleaved: it is a
  **regression** for us — pp32768 −4 %, pp65536 −6 %, pp102400 −9 % — because our
  production kernel already reads V coalesced and transposes it cheaply through
  LDS (packed-V), so the per-chunk full-prefix repack (an O(context) copy every
  chunk, growing with context) costs more than the transpose it removes. The
  repack only pays when the *unpacked* path is the slow one; ours is the packed-V
  path, so the repack was reverted and packed-V stays the default. A fresh phase
  ablation of the production packed tile at 100 K confirms the attention core is
  now **WMMA-compute-bound** (S + PV matmuls ≈ 55 % of the kernel, global KV load
  only ≈ 5 %), so neither the repack nor a full-KV-group re-read cut can close
  the gap; the residual widening past 16 K (88 % at 32 K → 85 % at 64 K → 84 % at
  100 K) is matmul efficiency, not data movement. See [EXPERIMENTS.md](EXPERIMENTS.md)
  and [OPTIMIZATIONS.md](OPTIMIZATIONS.md) "Prefill attention WMMA + row-split GDN".
- **Greedy decode (88 %).** At 44.87 t/s the step moves ~3.07 GB of active
  weights/token, i.e. ~136 GB/s effective against the ~240 GB/s streaming peak
  (roofline ~78 t/s, [OPTIMIZATIONS.md](OPTIMIZATIONS.md)). The Q8_0 `lm_head`
  GEMV is at the DRAM ceiling (231 GB/s), but the grouped MoE and `lin_gemm_in`
  GEMVs run well below it, so batch-1 greedy is efficiency-bound, not
  bandwidth-bound. The reference's 50.80 t/s (~155 GB/s) closes part of that
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
# Long-context rows (32 K/64 K/100 K); n-gen 1 skips decode, ~8 min per binary:
"$B" bench --model "$MODEL" \
  --n-prompt 32768,65536,102400 --n-gen 1 --repetitions 3
for N in 1 2; do
  "$B" bench --model "$MODEL" \
    --speculative mtp --mtp-model "$MODEL" \
    --draft-tokens $N --min-draft-tokens $N \
    --n-prompt 512 --n-gen 128 --repetitions 3
done
```

Greedy, thinking off. Kernel findings in [EXPERIMENTS.md](EXPERIMENTS.md) and
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).
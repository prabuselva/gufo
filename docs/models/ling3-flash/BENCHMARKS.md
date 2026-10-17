# Ling 3.0 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory.
`Ling-3.0-flash-Q4_K_M` (two-shard GGUF). Gufo `bench` and llama.cpp
`llama-bench`, exclusive GPU (`tools/bench/gpu_exclusive.sh` /
`tools/bench/bench_bg.sh` + watchdog), greedy. Reference columns: October
11–12, 2026, llama.cpp **b11557** (build `5c8854fd8`,
`-ngl 999`, flash-attn auto), freshly built clean on this host for both
backends: ROCm (`GGML_HIP=ON`, gfx1151 fatbin, HIP 7.15.26333, `GGML_NATIVE=OFF`)
and Vulkan (`GGML_VULKAN=ON`, Release, RADV `STRIX_HALO`). Prefill and
decode Gufo columns refreshed October 17, 2026 on HEAD `2127c37d`
(5.3c plus the accepted 5.4b→5.9 decode levers and the 6.0/6.1 Q4_K/Q5_K
unpack folds; see `EXPERIMENTS.md`) as `build/release/gufo`,
log `gufo_bench_p6grid.log`. The model loads on
llama.cpp as arch `bailingmoe3 124B.A5.1B`. Positive gain favors Gufo; all
cells are single same-session matched runs (Gufo `-r 2`/`-r 3`, llama-bench
`-r 2`/`-r 3`), except the refreshed Gufo columns (noted below). [Quality evidence](QUALITY.md) ·
[Experiment history](EXPERIMENTS.md)

## Prefill

Same prompt length, fresh context, timed prefill phase only on both sides.

| Cell | Gufo (tok/s) | llama.cpp ROCm b11557 | Gain | llama.cpp Vulkan b11557 | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| pp2048 | 1323.69 | 659.34 | **+100.8%** | 618.64 | **+114.0%** |
| pp4096 | 1167.65 | 633.69 | **+84.3%** | 590.35 | **+97.8%** |
| pp8192 | 920.27 | 583.71 | **+57.7%** | 559.02 | **+64.6%** |
| pp16384 | 663.98 | 507.39 | **+30.9%** | 504.40 | **+31.6%** |

The October 17 refresh is 0.8–2.0 % below the October 16 values in all
four prefill cells (pp4096 within its ±23.5 spread, the other three
tight); no prefill path changed in 5.4b→6.1, so this is treated as
session variance, not attributed to a lever.

Gufo now wins at every measured depth, but still decays faster with
depth than the references: 1334→670 (−50%) across the 8× range against
659→507 (−23%) on both llama.cpp backends; after the `5.3c` WMMA hi/lo
MLA prefill attention the 16K lead is **+32%**. After the October 15
`__expf` online-softmax lever the column
read 868→446 with a −12% 16K gap; the binary16 router retype and WMMA
route (`5.3a`, `EXPERIMENTS.md`) lifted every depth to 1074→496 with
the crossing landing exactly at 16K, and the head-bucket WMMA route for
the absorbed MLA `k_b`/`v_b` projections (`5.3b`, same doc — heads as
virtual experts on the routed F16 tier) lifted it again to 1249→530
(+16/+13/+9/+7% over the 5.3a cells at 2048/4096/8192/16384; matched
BAAB against the 5.3a build +14.5/+9.9% at pp2048/pp8192). The decay
that remained was the non-WMMA fp32 MLA prefill kernel:
61.5 % of device time at 16K at ≈ 3.2 TF/s against a 55.25 TFLOPS WMMA
ceiling, while the reference engines cross the same shape in WMMA tiers
(`EXPERIMENTS.md`; the ROCm reference is not special — Vulkan matches
it here). `5.3c` puts that kernel — scores and P·V — on the tensor
cores (matched BAAB against the `5.3b` build: +6.2/+18.7/+25.4 % at
pp2048/pp8192/pp16384, `gufo_bench_53cpair.log`,
`gufo_bench_53c16k.log`); the residual −50 % vs −23 % decay is not yet
attributed to a single kernel. The October 16 Gufo column is the standard-grid run
`gufo_bench_53cgrid.log` on the `5.3c` build; paired evidence for
earlier refreshes is `gufo_bench_53bpair2.log` (`5.3b` vs the rebuilt
`5.3a` control), the earlier `5.3a` refresh `gufo_bench_bench53ab.log`
(control `5.4b`,
`gufo_bench_ba53a.log`), and the `__expf`-window refresh used
`gufo_bench_expf_ab.log`, `gufo_bench_pp4096ab.log` (control drift
≤0.7%, gains +3.0/+5.0/+8.4/+10.9%).
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
| tg128 | 38.30 | 33.78 | **+13.4%** | 41.42 | −7.5% |
| tg1024 | 36.94 | 34.59 | **+6.8%** | 41.59 | −11.2% |

Gufo cold decode now leads the ROCm backend at tg128 (+13.4 %) and
tg1024 (+6.8 %), 7–11 % behind llama.cpp **Vulkan**, whose decode
(41.4–41.6) is still the strongest decode path measured on this host for
the model (the column cells are the October 17 refresh
`gufo_bench_p6grid.log`, `-r 3`, against the retained reference runs;
cumulative 5.2a→5.4b lever
gains per depth are in the paragraph below). llama
decode does not slow as it generates (its tg1024 ≥ tg128 on both backends);
Gufo loses 3.6% from tg128→tg1024. Preparing a 2048-token frontier before
timed generation (`gufo bench --n-prompt 0 --n-gen 128,1024 -d 2048`) costs
Gufo a further ~4–6% (31.40 / 31.35), and the retained `pp2048+tg` blends
(Gufo `--n-prompt 2048` runs: 33.62 / 32.51 in-pipeline; llama ROCm 93.30,
Vulkan 108.25 per combined test) indicate llama.cpp's decode is nearly flat
with cache depth — consistent with its slower depth decay in prefill. The
decode-frontier depth cost is a concrete Gufo-side follow-up (see
`EXPERIMENTS.md` open items). After 5.2a (row-broadcast decode scoring,
October 15, matched BAAB pair in one exclusive window,
`gufo_bench_ba52a.log`), frontier decode moves tg128 @ d2048
30.33/30.39 → 31.08/31.17 (**+2.5 %**) and tg128 @ d8192 23.47/23.59 →
25.69/25.75 (**+9.4 %**), with cold tg128 neutral (+0.7 %); depth
prefill shares the path (`pp8 @ d8192` 67.56/68.42 → 72.82/72.42,
+7.0 %). The residual @d2048 frontier gap is −7.0 % vs cold, still open.
5.4a (four rows per block in the F32 GEMV tier, October 15,
bit-identical, matched BAAB pair `gufo_bench_ba54.log`) lifts decode at
every depth: tg128 cold 33.71/33.83 → 33.80/34.10 (+0.5 %), @d2048
32.02/31.99 → 32.60/32.65 (**+1.9 %**), @d8192 27.50/27.75 →
27.91/27.91 (**+1.0 %**), with `pp8` within noise. 5.4b (two rows per
wave in the grouped Q4_K/Q6_K tiers, also bit-identical — grouped worst
errors unchanged to the digit) extends it: matched BAAB pair, twice
against the same pre-5.4 control (`gufo_bench_ba54b.log`), tg128 cold
33.39/33.64 → 34.88/35.26 (**+4.7 %** cumulative), @d2048 32.36/31.74 →
33.51/33.15 (**+3.9 %**), @d8192 27.98/27.31 → 28.92/28.53
(**+3.9 %**), `pp8` still within noise. 5.3a (binary16 router retype,
matched BAAB pair, twice, `gufo_bench_ba53a.log`) adds cold tg128
35.35/35.57 → 35.91/36.04 (+1.6 %/+1.3 %) from the halved decode router
read on top. 5.3b (head-bucket WMMA route for the absorbed MLA
prefill projections, October 16) changes no decode path; its dedicated
decode BAABs are neutral (cold d0 `gufo_bench_53btg.log`,
`tg128 @ d8192` `gufo_bench_53btgd2.log`) and the cells above are the
`5.3b` grid refresh. 5.4c (October 16) extends the 5.4b fold to the
MTP-draft grouped Q4_0/Q8_0 tiers — bit-identical, +8.6 % on the
cold-L2 down-shape microbench — but it covers only the single draft
layer, so the matched `--speculative mtp` BAAB tg1024 pair is neutral
(40.51/40.56 → 40.50/40.49, `gufo_bench_mo54tgm.log`) and no cell
above moves. 5.7 (October 17) runs the decode shared-expert chain and
routed `up` on non-blocking side streams, overlapped with the router →
gate → down chain, kernels and reduction orders unchanged; matched
BAAB ×2 (`gufo_bench_baabov`) at short context: tg1024 after pp2048
34.79/34.14 → 36.19/36.17 (**+5.0 %**), tg128 after pp256
35.79/35.77 → 37.25/37.17 (**+4.0 %**); the cold grid above was not
re-measured. 5.8 (October 17) fans the six decode KDA projection GEMVs
and the MLA `attn_kv_a`/`attn_gate` chains over the same two side
streams, joins unchanged downstream, kernels and reduction orders
untouched; matched BAAB ×2 (`gufo_bench_fanoutbaab`) at short context:
tg1024 after pp2048 36.16/36.00 → 36.61/36.45 (**+1.2 %/+1.3 %**),
tg128 after pp256 37.30/36.85 → 37.50/37.30 (+0.5 %/+1.2 %); spec and
pp2048 controls neutral, cold grid not re-measured. 5.9 (October 17)
keeps each `KdaDelta` decode thread's state row in registers between
the two dependent passes (`KdaDeltaVec4Kernel`, d = 128, bit-identical);
matched BAAB ×2 (`gufo_bench_kdaab`): tg1024 after pp2048 36.46/36.48 →
36.97/37.03 (**+1.4 %/+1.5 %**), tg128 after pp256 37.58/37.67 →
37.97/37.82 (+1.0 %/+0.4 %), speculative tg128 +2.5 %/+1.2 % and tg1024
+1.4 %/+1.7 %; pp2048 tg1 controls within noise, cold grid not
re-measured. 6.0 (October 17) folds the
Q4_K grouped expert-GEMV sub-block unpack into three word loads and exact
integer products (`Q4ScaleMinFromWords`, bit-identical, kernel probe
−8.9 %); matched BAAB ×2 (`gufo_bench_moeab`): tg1024 after pp2048
+0.2 %/+0.5 % plain, +1.0 %/+1.2 % speculative (7/8 pairs ahead, one
−1.4 % spec outlier); cold grid not re-measured.
6.1 (October 17) folds the same unpack into the dense and batched
Q4_K/Q5_K GEMV kernels (probe −11.0 %/−4.7 % bit-identical); matched
BAAB ×2 (`gufo_bench_moeab2`): tg1024-plain +0.6 %/+0.7 %, speculative
+0.5 %/+0.7 % at both tg128 and tg1024; cold grid not re-measured.

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

5.6a (October 16) made the Q8_0 Vec4 multi-row GEMV bit-exact with the
per-row kernel (`GemvRowsQ8_0Vec4`); the verify rows route through it in
the draft loop, so the matched `--speculative mtp` BAAB pair after pp8
rose 39.67 → 40.34 (+1.7 %, `tg1024`) and 37.04 → 37.50 (+1.2 %,
`tg1024 @ d2048`), both pairs consistent, while every plain pp/tg cell
stayed inside ±0.4 % (`gufo_bench_q8baab`). The after-pp2048 table above
is a different prefill prefix and was not re-measured; it stands.

5.6b (October 16) batched the spec prefill (`Executor::MtpPrefill`) and
made the bench pp rows honour `--speculative mtp` — every `--speculative`
pp figure recorded before this change was measured with MTP disabled and
was plain prefill. Re-recorded on the serve path's spec Sync, matched
BAAB ×2 plus an 8K pair against a sequential-sync control carrying the
same bench fix (`gufo_bench_p3h`): pp2048 32.35 → 604.8 t/s (**18.7×**)
and pp8192 29.53 → 445.3 (**15.1×**). The draft rows now come from WMMA
batch hidden states instead of per-token GEMV ones, which trims accept
rate: the same pairs move spec tg1024 40.28 → 39.25 (−2.6 %) and 39.75 →
39.20 (−1.4 %), and tg256 after pp8192 39.79 → 38.83 (−2.4 %), every
pair consistent; outputs stay bit-exact under greedy verification. Plain
cells unchanged. The tg rows quoted above stand as the sequential-sync
pipeline's history.

5.6c (October 17) runs the draft layer itself as a batched WMMA tier in
`MtpPrefill` and adds Q4_0 decoding to the F16 GEMM kernels: the MTP
checkpoint stores every matrix as Q4_0, which the tier could not decode,
so each chunk tail had kept a sequential per-token GEMV draft (~0.9 ms/token).
Matched BAAB ×2, exclusive (`gufo_bench_baab56c`): spec pp2048
605.6–607.5 → 1270.9–1280.1 t/s (**2.1×**), pp4096 546.5 → 1140.5
(**2.1×**) — spec prefill is now within ~4 % of plain; spec tg1024 after
pp2048 39.37 → 39.55 (+0.5 %, no accept-rate cost) and the plain pp2048
control is unchanged (1329.6/1329.3). Outputs stay bit-exact under greedy
verification.

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

Raw logs: `/tmp/opencode/gufo_bench_p6grid.log` (October 17 close-out
refresh of the Gufo prefill/cold-decode columns, HEAD `2127c37d`),
`/tmp/opencode/gufo_bench_llgrid.log` (first full three-engine
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

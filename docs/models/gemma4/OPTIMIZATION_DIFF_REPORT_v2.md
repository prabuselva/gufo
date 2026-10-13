# Gemma-4-26B-A4B — Diff report v2: fidelity verdict and measured attribution

Extends [OPTIMIZATION_DIFF.md](OPTIMIZATION_DIFF.md) (v1). v1 answered *where* the
time goes. This report answers the question v1 left open: whether the franzmoca
implementation (`gufo-org/gufo` PR #426, branch `feat/gemma`) is actually
computing Gemma 4, whether *our* CPU oracle is wrong, and what the remaining
speed gap is really made of — with a fresh matched measurement instead of
published model-card numbers.

## 1. Verdict

1. **The fork is not cheating.** Its prefill runs binary16 activations and
   binary16 WMMA attention with **F32 accumulation**, computes all top-8 routed
   experts with exact weights, and applies exact causal/window masks. Every
   precision compromise is documented with a KL number measured against its own
   oracle instead of asserted, and the compromise it makes is the smaller one
   (oracle ablation: binary16 projections → KL `3.7e-5`; Q8_1 projections →
   `0.74`).
2. **Our oracle is not wrong either.** Both `reference.cpp` implementations
   reproduce llama.cpp's `gemma4` graph in every detail checked (§4). The
   difference is capability: our oracle is single-token, hard-codes the
   binary16 KV rounding as truth, cannot ablate activation precision, and our
   GPU parity gates are absolute-logit error + argmax, so they cannot see the
   failure modes the fork quantifies (bulk/chunked prefill drift, activation
   rounding, near-tie routing flips).
 3. **The speed gap is architectural, not an accuracy trade.** Measured today on
    this machine, exclusive, same artifact: **1.67× at pp2048, widening to 1.92×
    at pp2048 with a 16 K cache** (§3). Our decode gap is 1.14× autoregressive
    at Q8 and **1.79× on the same model's Q4_K_XL artifact, where the fork is
    11.5× faster to load** (§3.1). With matched draft policies, MTP verification
    step cost is at parity (§3, §7).
4. **llama.cpp is the least faithful of the three**, using Q8_1 activations in
   its ROCm path (KL ~0.42 vs the same FP32 reference) — being slower than the
   fork does not make us more correct.
5. **This A/B found two real quality defects on our side**, both in paths the
   fork has hardened: our MTP verification is not width-invariant (the accepted
   trajectory differs from autoregressive decoding, §6.1) and our long-context
   prefill collapses to a `-\n\n-\n\n-` loop at 16 K depth where the fork keeps
   generating (§6.2).

## 2. Method and provenance

| Side | Revision | Binary | Notes |
|---|---|---|---|
| gufo | `0d1053bf` (2026-10-12) | `build/release/gufo`, built 2026-10-11 23:18 | no source newer than the binary |
| franzmoca | `22341ae7` (2026-10-11) | `gufo_franzmoca/build/release/gufo`, **rebuilt for this run** 2026-10-12 08:55 | the retained Oct-7 binary predates four gemma4 perf commits, so it was rebuilt with the same system toolchain (`/usr/bin/c++`, RelWithDebInfo, `/opt/rocm`) |
| shared artifact | `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (25.74 GiB), `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` | identical for both engines | |
| shared artifact (S3.1) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (13.27 GiB, "smart Q4_0, QAT-lossless") | identical for both engines | |

Protocol — identical flags for both binaries, greyscale of the retained method
(`BENCHMARKS.md`): exclusive GPU through `tools/bench/gpu_exclusive.sh`
(detached `tools/bench/bench_bg.sh` + foreground `tools/bench/bench_wait.sh`
watchdog), greedy, `--repetitions 3`, and the two binaries **interleaved inside
one exclusive window** (`s1_gufo, s1_fork, s2_gufo, …`) so thermal and memory
drift hit both sides equally.

```sh
bash /tmp/opencode/gemma4_ab.sh    # 6 runs, each individually `timeout`-bounded
bash /tmp/opencode/gemma4_qat.sh   # follow-up window: K-quant matrix + MTP fixed-policy
```

Raw per-run logs (ephemeral, on this host): `/tmp/opencode/gemma4_ab/s{1,2,3}_{gufo,fork}.log`
and `/tmp/opencode/gemma4_qat/s4_*.log` / `s5_*_mtpfixed.log`, harness markers
`/tmp/opencode/gufo_bench_gemma4{ab,qat}.log`. Model load 10–16 s per
invocation, so every cell is a cold-load measurement; nothing was reused across
invocations.

Consistency checks: the fork reproduces its own published model card within 3 %
(pp2048 3326.05 measured vs 3428 published; v1 measured 3231 on the older
binary) and sits inside its claimed decode ranges (AR 45.7–51.1, MTP
68.9–78.8). Its d16384 numbers reproduce v1's independent re-measurement
(2349.04 now vs 2292.2 in v1; tg128 43.94 vs 43.86). The measurement harness is
therefore sound.

## 3. Measured result (2026-10-12, exclusive, greedy, r3, Q8_K_XL)

| workload | gufo `0d1053bf` | franzmoca `22341ae7` | fork / gufo |
|---|---|---|---|
| pp2048 @ d0 | 1992.40 ± 4.85 | 3326.05 ± 25.08 | **1.67×** |
| pp8192 @ d0 | 1799.52 ± 23.53 | 3133.36 ± 2.59 | **1.74×** |
| pp16384 @ d0 | 1559.33 ± 0.41 | 2771.69 ± 2.47 | **1.78×** |
| pp2048 @ d16384 | 1222.89 ± 3.80 | 2349.04 ± 53.12 | **1.92×** |
| tg128 @ d0 | 42.29 ± 0.02 | 48.21 ± 0.05 | 1.14× |
| tg128 @ d16384 | 36.94 ± 0.05 | 43.94 ± 0.03 | 1.19× |
| tg128 @ d0, MTP n=4 | 70.39 ± 0.05 | 75.63 ± 0.51 | 1.07× |

Notes:

* The gap **widens monotonically with context depth** (1.67 → 1.78 → 1.92).
  That is the signature of per-token/per-key traffic and attention occupancy,
  not of a single GEMM.
* Our own current binary is ~3–5 % faster than the retained `BENCHMARKS.md`
  table (pp2048 1934.99 → 1992.40, pp8192 1733.34 → 1799.52, pp16384 1488.44 →
  1559.33, MTP n=4 59.99 → 70.39) from the device-side tile build and the
  binary16 shared-FFN work. **The gap does not close on its own**: at the same
  pp2048 point v1 measured 1.67× and it is still 1.67×.
* MTP: our accept rate is *better* (92 accepted / 106 drafted = 0.87) than the
  fork's calibrated default (70/113 = 0.62) and we are still slower. Re-running
  the fork with `--draft-policy fixed --draft-tokens 4` (139 drafted, 70
  accepted) gives **70.80 ± 0.02 vs our 70.20 ± 0.18** — verification-step cost
  is at parity; the 1.07× was the calibrated draft policy, not kernels. Our
  high accept rate is not a win either: §6.1 shows our accepted trajectory
  differs from our own autoregressive one, i.e. the extra accepts include
  false accepts.

### 3.1 K-quant artifact (`qat-UD-Q4_K_XL`, 2026-10-12 second exclusive window)

Same protocol, same interleaving; pp/tg at depth 0:

| workload | gufo | franzmoca | fork / gufo | Q8 comparison |
|---|---|---|---|---|
| pp2048 | 2129.01 ± 7.44 | 3611.83 ± 14.55 | **1.70×** | 1.67× |
| pp8192 | 1927.18 ± 25.04 | 3352.60 ± 12.02 | **1.74×** | 1.74× |
| pp16384 | 1651.06 ± 0.72 | 2979.01 ± 1.55 | **1.80×** | 1.78× |
| tg128 | 43.49 ± 0.01 | 77.98 ± 0.42 | **1.79×** | 1.14× |
| model load | 69.6 s | 6.0 s | **11.5×** | 10.0 s vs 9.9 s |

The artifact halves the bytes and the fork scales with it (tg 48.2 → 78.0,
pp16384 2772 → 2979); we do not (tg 42.3 → 43.5, pp16384 1559 → 1651) and our
decode stops being bandwidth-bound. That is the §7 `hipblasSgemm` / narrowing
fallback catching K-quantized weights row-by-row, and it also costs us an 11.5×
load-time tax (per-tensor dequant at upload). Greedy 128-token outputs are
coherent on both engines but **not identical** — both parse the pangram prompt
correctly, then wording diverges inside the window ("a reference to a
high-performance computing architecture" vs "a technical reference to a
high-performance processor"). Cross-engine greedy divergence of this kind is
expected from projection precision differences and is exactly what the missing
oracle-KL harness (§9) would quantify; no claim is made about its size here.

## 4. Oracle fidelity audit

Three arbiters were used, because an implementation can agree with its own
oracle while both share a modelling error:

* **A** — our oracle: `src/models/gemma4/reference.cpp` + `cpu_ops.cpp`.
* **B** — the fork's oracle: `gufo_franzmoca/src/models/gemma4/reference.cpp`.
* **C** — an independent source of truth: `llama.cpp/src/models/gemma4.cpp`
  plus `src/llama-graph.cpp` (`build_ffn:1693`, `build_moe_ffn:1941`,
  `build_lora_mm_id:1542`).

### 4.1 Semantics checklist (A, B and C all agree)

| Graph detail | llama.cpp (C) | gufo (A) | franzmoca (B) |
|---|---|---|---|
| embedding × √n_embd, token rows only | `gemma4.cpp` | ✓ | ✓ |
| per-head RMSNorm on Q and K, V weightless | `gemma4.cpp` | ✓ | ✓ |
| V = `rms_norm(wk·x)` when `wv` absent, **never roped** | `gemma4.cpp` | ✓ | ✓ |
| K roped, `f_attention_scale = 1.0` (no 1/√d) | `gemma4.cpp` | ✓ | ✓ |
| router = `rms_norm → ×1/√n_embd → ×ffn_gate_inp_s → mm` | `build_moe_ffn` | ✓ | ✓ |
| softmax gating + `norm_w=true` → renormalised top-8, clamp `6.103515625e-5` | `build_moe_ffn` | ✓ | ✓ |
| shared FFN `gelu(gate) * up` (`LLM_FFN_GELU`), parallel gate/up | `build_ffn` | ✓ | ✓ |
| `ffn_down_exps_s` is a **per-expert scalar** gathered by expert id | `build_lora_mm_id:1549-1555` (`reshape [1,E,1]` → `get_rows(ids)`) | ✓ `down_scale[e]` | ✓ `expert_scale[e]` |
| post-FFN norms on the shared and routed branches *before* the add | `build_moe_ffn` | ✓ | ✓ |
| `out_scale` on the layer output, final norm → `lm_head` | `gemma4.cpp` | ✓ | ✓ |
| logit softcap `×1/30 → tanh → ×30` | `gemma4.cpp` | ✓ | ✓ |

Both oracles also round cached K/V through binary16, which is what the GPU cache
does. **No modelling error was found in either oracle.** The one candidate that
looked like a shared bug — a per-tensor rather than per-expert down-projection
scale — is settled by `build_lora_mm_id` itself.

### 4.2 Where the oracles differ (capability, not correctness)

| Property | gufo (A) | franzmoca (B) |
|---|---|---|
| Driving mode | single-token `Step` + `DraftStep` | multi-row `Forward` (real chunked prefill), image-row window `ends` |
| KV precision | binary16 rounding **unconditional** (`reference.cpp:24` `RoundF16`, applied at `:170-171`) | a mode: `Storage::kFloat32 / kHalfKv / kQ8Activations` (`reference.hpp:23-28`, applied at `reference.cpp:98,330`, exposed as `--half-kv` in `tests/models/gemma4/reference_probe.cpp:51-52`) |
| Activation ablation | none | rounds *any* projection input to Q8_1 inside the oracle → predicts a GPU path's KL before writing a kernel |
| Inner precision | float accum, `powf` rope | double rope + double attention accumulation (a stricter truth than llama.cpp's own float) |

The important consequence is the second row. Our oracle bakes the GPU's binary16
KV into "truth", so it *cannot* report how much that rounding costs; the fork
runs the same reference at FP32 KV and at binary16 KV and reports the
difference. Ours therefore tests "GPU matches the runtime's assumed rounding",
not "GPU matches the model".

### 4.3 GPU-side quality gates

| | gufo | franzmoca |
|---|---|---|
| metric | absolute logit error + argmax equality | KL divergence vs FP32 oracle + top-1 count, per position |
| decode | Q8_0 logits ≤ `2e-3` (measured `1.12e-3`); WMMA MoE ≤ `5e-3` | mean KL `7.5e-7` / `4.8e-7` / `8.8e-7` (Q4_K/Q6_K/Q8_K), max ≤ `9e-6`, top-1 22/22 |
| prefill | `AttentionPrefill` WMMA ≤ `3e-3` (measured 2.01e-3, attributed to an "F16-Q encoding floor") | binary16 activations + binary16 attention: mean KL `3.7e-5` |
| bulk / chunked prefill | no equivalent test | 1542-token prompt vs exact rows: KL `0.027`, top-1 1489/1542, gated at `0.06` |
| full-model forward | worst-abs ≤ `2.0` (Q8_0), ≤ `6.0` (Q4_K_M, observed 4.60) | per-position KL table per quant |

Two things follow. First, max-abs-logit error is nearly blind to what this model
does wrong: with `f_attention_scale = 1.0` the softmax is extremely peaked, so
routing near-ties flip experts and move a handful of logits hard while leaving
generation sane — the fork measured this directly (changing only the RMSNorm
block-sum width moved mean KL from `1.3e-6` to `2.7e-5` and produced single
positions at `3e-4`). Second, our "F16-Q floor" of ~2e-3 is suspicious: our
prefill kernel and the fork's use the *same* instruction
(`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `attention.hip.cpp:399` vs
`attention_wmma.hip.cpp:40`), both accumulate in F32, and the fork's whole
pipeline is binary16 yet its KL is 3.7e-5. Our 2e-3 is more likely near-tie
routing noise plus F32-vs-F16 divergence elsewhere than an irreducible floor —
and it is currently unfalsifiable with the metric we use.

## 5. Is the fork's speed bought with accuracy? Lever by lever

| Fork lever | Exactness evidence | Measured cost (fork's own oracle) |
|---|---|---|
| binary16 activations end-to-end (prefill `half_prefill_`) | values rounded once at the producer | KL **3.7e-5** (ablation: every projection binary16) |
| Q8_1 activation quantisation as fallback for non-WMMA formats | `executor.cpp:496` only for formats no WMMA GEMM decodes | KL **0.74** — kept off the hot path; this is what llama.cpp does and why its KL is ~0.42 |
| BF16 weights uploaded as binary16 (`weights.cpp:48-51`) | bf16 ⊂ fp16 for weight magnitudes → **lossless** | none |
| No `kF32` projections accepted (`weights.cpp:49-51`) | forces the tensor through a real GEMM path instead of FP32 SGEMM | none (it removes a rounding *less*, since the F32 path also feeds F32 rows) |
| WMMA prefill attention (`attention_wmma.hip.cpp:1-22`) | `v_wmma_f32_16x16x16_f16`, F32 accumulators, online softmax, causal + window + key-limit masks applied per element | binary16 attention included in the 3.7e-5 above (prefill row) |
| "Derived keys" for the 5 global layers | V and roped K come from one projection (`V = rms_norm(raw)`, `K = rope(norm(raw))`), so caching one and deriving both is algebraic | rounding point moves; covered by the prefill KL row |
| Routed experts | `moe.hip.cpp` / `moe_prefill.hip.cpp` contain **no threshold, no expert dropping, no cap** — the only "skip" is not processing 16-row tile parts past a bucket (`moe_prefill.hip.cpp:8`), which is padding, not approximation | all 8 experts computed |
| Tuned GEMM plans (`executor.cpp:427-437`) | plan is a tiling choice; same F32 accumulation | none expected |
| Calibrated draft policy, chains to 15 | drafting only *guesses*; the target model verifies | not applicable if verification is exact — see §6.1 for ours |

**Verdict: no lever is paid for with unreported accuracy.** The two places the
fork trades precision (binary16 activations, binary16 KV) are each measured
against an oracle that can express both sides, both are two to five orders of
magnitude inside our own tolerance, and the aggressive option (Q8_1 activations)
is the one it avoided.

## 6. What this A/B found on our side

### 6.1 MTP verification is not width-invariant

`output_sha256` of the 128-token generations (`--n-gen 128`, greedy, pp2048):

| engine | autoregressive | MTP n=4 | same text? |
|---|---|---|---|
| franzmoca | `de facto facto-__ (de facto facto-_ _). …` | identical text | **yes** — drafted 113 (calibrated) and 139 (`--draft-policy fixed`), accepted 70, identical text under both policies |
| gufo | `2669d2fd…` `… freez ing-freezing-…` | `2d874d6f…` `… -user-fact-of-fact-of-…` | **no** — drafted 106, accepted 92 |

A verified draft can only change throughput, never the accepted token sequence:
each accepted token must be the target model's own argmax. Our MTP path changes
the sequence, so our verification logits are not the same numbers as our
decode logits — exactly the divergence that costs less accuracy than it looks
but costs correctness. The fork's lever for this is explicit: *"Drafts are
verified with width-invariant WMMA projections … stay bit-identical to
single-token decode"* (PR #426 update), implemented by routing verification
widths through the same WMMA kernel (`executor.cpp:510-514`: `w.cols >= 2048 &&
(has_draft() || rows >= 6)`). Our `QUALITY.md` currently frames MTP drift as
expected ("not bit-identical beyond ~90 tokens") instead of as a defect.

### 6.2 Long-context generation collapses earlier than the fork

Same prompt, same depth 16384, greedy, 128 tokens:

* franzmoca: keeps cycling the prompt sentence coherently.
* gufo: `The quick brown fox jumps over the-` then `-` `-` `-` … to the end.

The bench prompt is synthetic and repetitive by design, so degeneration itself
is not a bug (the fork documents the same regime, reporting bulk KL 1.71–2.10 on
a 1353-token repetitive prompt rather than gating it). What differs is *how
early* we fall off, at the same depth, which is the behavioural counterpart of
the 1.92× prefill gap at depth: our long-context path both costs more and
drifts more. We have no test for it — `--validate-prefill` exists in the CLI
(`src/cli/bench/bench.cpp:162`) but gemma4 dispatches to
`models::gemma4::RunGemma4Benchmark` at `src/cli/bench/bench.cpp:1277` before
the generic `ValidatePrefill` (`:1402`), so **the flag is a silent no-op for
gemma4** (same in the fork: `gufo_franzmoca/src/cli/bench/bench.cpp:1461-1462`
before `:1583` — it prints nothing in either engine's `s1` log).

### 6.3 Our tolerance table cannot support the claims we make in it

Absolute-error bands (`2e-3`, `5e-3`, forward `≤6.0`) plus argmax equality do
not distinguish "attention F16 rounding" from "a near-tied expert flipped at
position 812", which is why §4.3's floor claim exists. Any prefill/activation
change ported from §8 must therefore be gated on KL and top-1 mass, not on
worst-abs.

## 7. Where the prefill gap comes from

v1's pp2048 profile of our pipeline, mapped to the fork's implementation:

| our kernel group | share | fork's equivalent | why it is cheaper there |
|---|---|---|---|
| `RoutedF16GEMM` | 34.2 % | `moe_prefill.hip.cpp` tiled expert GEMMs | half inputs/outputs already matched; theirs bucket 16-row parts and stage tiles per expert |
| `DenseF16GEMM` | 18.8 % | `executor.cpp:485-490` `DenseF16Gemm` + `HalfPlan` | **shape-tuned plans**: `kStagedRowGroups8` for K ≥ 4096, `kRowGroups8` for K = 2112; quoted 2013 → 1759 µs per 2048 rows on the attention output, 605 → 582 µs on the MLP down. Ours takes no plan at all (`kernels/rocm/dense_f16_gemm.hpp:26`) |
| `hipblasSgemm` (F32 dense) | 6.6 % | *does not exist* — `kF32` is not an accepted projection type | FP32 GEMM is 4× the activation traffic and 2× the weight traffic of WMMA for no accuracy benefit |
| `Narrow*` | ~2 % | producer writes the half row | one extra full hidden pass per producer we cannot remove until producers are end-to-end binary16 (`executor.cpp:679` `NarrowActivations`) |
| `AttentionPrefillWmma` | ~10 % | `attention_wmma.hip.cpp:84-85` | `__attribute__((amdgpu_waves_per_eu(8)))` where we ask for **one block per WGP** (`attention.hip.cpp:418` `__launch_bounds__(kWaves * 32, 1)`), plus next-tile register prefetch and derived keys halving global-layer KV traffic |
| `QknormRope` / `KvNormRope` / `RmsNorm` | ~10 % | fused into the producing GEMM epilogue | separate elementwise kernels on FP32 rows; note `AddKernel`/`ScaleKernel`/`MulKernel`/`SoftcapKernel` are still distinct launches (`fused.hip.cpp:273,401,410,420`) |
| `MoeEpilogue` | 5.7 % | folded into the down-projection epilogue | extra full-width read/write per layer |

The depth-widening (1.67 → 1.92) is explained by the rows that scale with
context — attention occupancy and KV traffic — plus the FP32 row traffic our
producers still emit. On decode, the Q8 autoregressive 1.14× is GEMV/split-K
work; MTP step cost is already at parity under a matched fixed draft policy
(70.20 vs 70.80, §3), so that cell is a drafting-policy gap (port 8), not a
kernel gap. The K-quant artifact makes the same fallback visible in isolation:
halving the bytes widens our decode gap to 1.79× and our load time to 11.5×
(§3.1) — K-quant rows reach dequant → FP32 scratch → `hipblasSgemm` instead of
a fused dequant-to-binary16 path.

## 8. Ranked port plan

Ordered by expected gain per unit of risk. Every gate is KL + top-1 against the
oracle plus a matched `pp2048`/`pp8192` timing A/B in one exclusive window;
worst-abs alone does not qualify a change.

1. **Shape-tuned dense GEMM plans.** Add a plan parameter to
   `kernels/rocm/dense_f16_gemm.hpp:26` mirroring
   `gufo_franzmoca/src/models/gemma4/kernels/rocm/executor.cpp:427-437`
   (staged row-groups-8 for K ≥ 4096, row-groups-8 for K = 2112).
   *Gain:* ~12 % of the 18.8 % dense group. *Accuracy:* none expected (tiling
   only). *Gate:* dense-GEMM operator test + matched pp2048.
2. **Kill the FP32/BF16 dense GEMM route.** Drop `hipblasSgemm` and the BF16
   narrowing scratch (`kernels/rocm/gemm.hip.cpp:140-176`), run BF16 as binary16
   at upload like `weights.cpp:48-51`, and forbid `kF32` projections (ours
   allows it: `weights.cpp:126-128,192-194,232-234`).
   *Gain:* removes ~6.6 % kernel time plus its traffic, and is a prerequisite for
    the K-quant comparison. *Accuracy:* BF16→binary16 is lossless; loading-time
    rejection is strictly safer than a silent upcast. *Gate:* weight-binding test
    + pp2048 KL. *Measured stake:* this is the route the Q4_K_XL artifact falls
    back to — 1.79× decode gap, 1.70–1.80× prefill gap and 11.5× load time (§3.1).
3. **End-to-end binary16 activations.** Remove `NarrowActivations`
   (`executor.cpp:679`): every producer (`RmsNormRowsHalf`, `GeGLU`,
   QK-norm-rope, MoE epilogue) already has a half output — extend it to the
   residual stream so no FP32 row is ever written or re-read.
   *Gain:* the remaining `Narrow*` ~2 % plus the dominant traffic term.
   *Accuracy:* expected within the fork's measured 3.7e-5 — **verify, do not
   assume**. *Gate:* new KL harness (§9) + full-model forward + our existing
   output hash.
4. **Attention occupancy and traffic.** Raise to 2 blocks/WGP via
   `amdgpu_waves_per_eu` (ref. `attention_wmma.hip.cpp:84-85`), add next-tile
   register prefetch, skip the rescale when the row max is unchanged, emit
   binary16 output, and store derived keys for the 5 global layers.
   *Gain:* the largest single contributor to the depth-widening gap.
   *Accuracy:* rounding point moves for global-layer K; gate on per-position KL
   at 16 K, not max-abs.
5. **Fuse the small elementwise layers.** `QknormRope`, `KvNormRope`,
   `MoeEpilogue`, plus `Add`/`Scale`/`Mul`/`Softcap`
   (`fused.hip.cpp:273-428`) into GEMM epilogues and into the residual write.
   *Gain:* ~15 % of kernel time in launch + traffic, more at short context.
   *Accuracy:* summation width changes silently flip near-tied experts (fork
   measured `1.3e-6` → `2.7e-5` from RMSNorm block width alone) — keep the
   reduction order pinned and gate on KL.
6. **Router path in one GEMM.** Fuse `rms_norm × 1/√n_embd × ffn_gate_inp_s`
   into the router GEMM prologue instead of `MulKernel` + a separate small GEMM
   + `RouterTopKKernel` (`fused.hip.cpp:282`).
   *Gain:* 30 launches/layer of tiny work. *Accuracy:* keep the router-input
   scale in F32; layer-0 checks in the fork already pin router logits to 9.5e-7
   relative and top-8 membership exact — adopt those checks as the gate.
7. **Decode: width-invariant verification + split-K K-quant GEMV.** Route
   verification widths (1–16 rows) through the same binary16 WMMA GEMV the
    single-token path uses (ref. `executor.cpp:510-514`), add split-K for the
    K-quant GEMVs, and fix §6.1.
    *Gain:* the 1.14× AR (Q8) and 1.79× AR (K-quant, §3.1) decode gaps; MTP step
    cost already matches at fixed policy (§7).
    *Accuracy:* this one
   *improves* accuracy — the gate is bit-identical `output_sha256` between
   `--speculative mtp` and `--speculative off` on the bench prompt.
8. **Draft policy.** Then, and only then, look at the fork's calibrated gemma4
   drafting and 15-token chains. This is where the whole MTP throughput gap
   lives: the fork's own calibrated policy beats its fixed one, 75.63 vs 70.80
   (+6.8 %), at identical accepted-token correctness (§3, §6.1). Correctness
   first (§6.1), this is a throughput lever on top.

## 9. Quality harness required before landing §8

1. **Port the KL harness**: `reference_probe.cpp` +
   `gufo_franzmoca/tools/gemma4/compare_logits.py`, producing per-position KL
   and top-1 counts against the oracle for decode, single-token-at-depth, and
   full-prefill logits.
2. **Add oracle `Storage` modes to our reference** (`kFloat32 / kHalfKv /
   kQ8Activations`, `reference.hpp:23-28`) so we can predict a lever's cost
   before writing the kernel and quantify our binary16-KV assumption (§4.2).
3. **Implement bulk-vs-exact prefill for gemma4**, i.e. make
   `--validate-prefill` work in `models/gemma4/cli_runner.cpp` instead of
   no-op'ing at `src/cli/bench/bench.cpp:1277`, and report KL, not only rmse and
   argmax rank.
4. **Add an MTP width-invariance test**: `output_sha256` equality between
   speculative and autoregressive at several depths; keep it in
   `tests/functional/` with `cache`/`long-context` selected.
5. **Record the degenerate long-prompt trajectory** at 16 K/32 K as a tracked
   (reported, not gated) artefact so §6.2 stops being invisible.

## 10. Do not port

* **Q8A16 / Q8_1 activation quantisation** on any prefill path — measured 0.74
  mean KL against 3.7e-5 for binary16, and it is the reason llama.cpp's ROCm
  teacher-forced logits sit at KL ~0.42.
* Any lever landed on the strength of a max-abs-logit gate (§6.3).
* Fine-tuned-model parity claims from the fork's docs: its own QAT
  `Q4_K_M` prefill is KL `0.013`, failing its `1e-3` limit, and `Q6_K` is
  `0.0026` — that needs its own measurement here before we claim support.

## 11. Caveats and next measurements

* Revisions are pinned in §2; the fork binary was rebuilt locally (its repo
  carries an uncommitted `CMakeLists.txt` workaround that pre-includes
  `<format>` before HIP headers, identical to ours).
* The MTP draft-policy confound is resolved (§3): fork fixed-policy 70.80 ± 0.02
  vs ours 70.20 ± 0.18. Only the *calibrated* fork side of the earlier cell
  relied on a fork-only CLI option; the fixed-policy re-run used `--draft-policy
  fixed --draft-tokens 4` and matched our drafting.
* No same-run oracle KL exists for either engine on these cells; §5's
  accuracy numbers are the fork's published measurements against the same
  artifact, not re-measured here. §9 item 1 makes them re-measurable.
* The retained `BENCHMARKS.md` table is ~3–5 % pessimistic for the current
  binary (§3) and should be re-measured when the next gemma4 change lands.
* The QAT `UD-Q4_K_XL` cells are now measured (§3.1): decode 1.79×, prefill
  1.70–1.80×, load 69.6 s vs 6.0 s. No KL comparison was run on the QAT
  artifact; port 2's §8 gate must include it, and §10's fine-tune parity claim
  (fork `Q4_K_M` KL 0.013 > 1e-3) still awaits our own harness.

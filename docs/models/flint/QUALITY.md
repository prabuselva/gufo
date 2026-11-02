# Flint Preview 40B-A2B — Quality

Independent agreement, engine contracts and their replay commands. Timings
live in [BENCHMARKS.md](BENCHMARKS.md).

## Agreement with llama.cpp (independent implementation)

- Logits parity, `flint_logits_probe` (GPU engine) vs a libllama C-API probe
  @ **b11879**, same 22-token raw chat prompt, 8 greedy steps, exclusive
  GPU: chosen-id chains identical; rank-0 margin difference 0.0 at every
  step; top-32 overlap 30–32/32 with all max margin deltas on tokens far
  below argmax (−0.4..−2.5 logit). This is agreement with another
  inference engine on one greedy trajectory, not a capability score.

## Engine contracts (bitwise)

- `flint.reference` — scalar CPU oracle forward vs the reference vectors.
- `flint.rocm.forward` — GPU forward bitwise-stable across device replays.
- `flint.engine` — chunked prefill, KV replay incl. the 24 K-activity
  continuation (`ck4k`); kernel changes must keep these bitwise-identical.
- `flint.attention.op` — WMMA attention operator equivalence.
- `flint.config` / `flint.weights` / `flint.tokenizer` / `flint.chat_template`
  — GGUF metadata, tensor binding, gpt2-BPE + `flint` pre-tokenizer and the
  jinja tool-call template.

Reproduce (exclusive GPU, model path via `GUFO_FLINT_GGUF`):

```sh
cmake --preset gpu-test
cmake --build --preset gpu-test --parallel 4
env GUFO_FLINT_GGUF=<flint-Q8_0.gguf> ctest --test-dir build/gpu-test \
  -R 'flint\.' --output-on-failure
```

## Measurement provenance

- Speed rows 2026-10-29, rev `062c616d`, binary `build/release/gufo`
  (SHA-256 prefix `841c712c403e6ad7`), exclusive GPU through the
  `gpu_exclusive` harness; llama.cpp worktree @ b11879 baseline retained
  in STATUS (ROCm build, `-ngl 99 -fa 1 -r 3`, KV f16). The matched
  2048–131072 comparison table (both engines, one session) is
  `artifacts/flint-full-sweep.log`, rev `57d22867` on the same binary; it
  supersedes the earlier 6-point llama baseline numbers.
- Artifact `flint-Q8_0.gguf` (41.7 GiB, 779 tensors, imatrix Q8_0);
  weights are not part of the runtime package.
- Power profile node is absent on this host
  (`/sys/firmware/acpi/platform_profile` unreadable); all rows were taken
  in exclusive runs with the resident server stopped.
- Fused-kernel quality gates (C7v): probe `flint_norm_probe` reports
  mismatches=0 across 9 batches; the Q8_1 MoE tiers are checked
  bitwise-stable by the route benches' `--save`/`--ref` compares.
- C9 KV W8A8 dispatch (rev `0371a1fd` + dispatch, binary prefix
  `d02818172e3b29df`): outputs are byte-identical to `0371a1fd` below
  batch 1024 (850-token `flint_logits_probe` control), so the agreement
  point above (batch 22) and every bitwise contract are untouched by
  construction; at 2116 prompt tokens the greedy trajectory diverges at
  a 0.23-margin near-tie top-1 with magnitude no larger than a
  layer-0-only control — the int8 accumulation-order class shipped
  since C7m (statistics in `artifacts/flint-c9-kv-w8a8-adoption.log`).
- No hosted functional (HTTP) profile exists for Flint yet; serve parity
  beyond the single-session C5 runner is an open gap, not a pass.

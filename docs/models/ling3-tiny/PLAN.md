# Plan: enable Ling 3.0 Tiny on the Ling 3.0 Flash model stack

Model facts (measured): [README.md](README.md). Experiment log:
[EXPERIMENTS.md](EXPERIMENTS.md). Tracker: [STATUS.md](STATUS.md).
Runtime code stays in `src/models/ling3_0`. GPU rules:
[.agents/skills/optimize-kernel/SKILL.md](../../../.agents/skills/optimize-kernel/SKILL.md)
and AGENTS.md (exclusive runs, `bench_bg.sh` + `bench_wait.sh` in one tool
call). Do not invoke `nix`.

## Why it is close

Tiny is the same Bailing-MoE hybrid family: every dimension the specialized
KDA, absorbed-MLA and grouped-expert kernels key on is identical or a
smaller multiple of the same tile bases (README table, measured 2026-10-18).
No architectural blocker — the blockers are (1) the conservative exact-shape
pin `config.cpp:217-228`, (2) the Flash-specific MLA stride in the guard
loop `config.cpp:232` (Tiny: every 4th layer, not every 6th), (3) the
unconditional executor MTP path while Tiny ships no `nextn` block, and
(4) unverified small-shape kernel behavior.

## Strategy

1. **Profiles, not a pin.** Replace the exact 21-value check with an
   explicit allow-list of measured `{Flash, Tiny}` geometry profiles, each
   profile carrying its MLA stride and MTP presence. Divisibility alone
   never qualifies for the list; being listed means measured end-to-end.
2. **`nextn_layers == 0` is a first-class config.** Gate MTP allocation,
   weight validation and prefill/decode helpers on `nextn_layers > 0`
   (`kernels/rocm/executor.cpp:205-208, 1396-1413, 1610-1613`); speculative
   decoding stays rejected for this family regardless.
3. **Audit kernels by running them tiny, not by reading them.** The executor
   allocates from config-driven head counts (`executor.cpp:312, 341-417`)
   and a grep audit found no Flash constants in the kernels; the suspicious
   shapes are Q4_K rows with 2 super-blocks (expert_ff 512) vs 3 (768),
   16-head KDA/MLA grids, and 128-expert buckets. Resolve each by execution
   against `reference.cpp` and record in EXPERIMENTS.md — accepted and
   rejected alike.
4. **Quality bar identical to Flash.** Reference/oracle greedy parity,
   tool-call E2E replay in both transports, pp/tg smoke recorded (Tiny
   speed is not a gate).

## Phases (detail + checkboxes in STATUS.md)

1. Config profiles and loader (CPU): guard table, `nextn=0` accepted,
   per-profile MLA stride, config unit tests, weights allowlist check.
2. MTP-free executor: conditional MTP path + runner test.
3. Kernel shape audit and fixes: routed Q4_K 2-super-block rows, 16-head
   KDA/MLA, dense 4608 tiles, 128-expert buckets; Flash stays bit-exact
   (Fnv-sum evidence per QUALITY conventions).
4. Quality and E2E on hardware: greedy logit parity, GPU load/prompt/smoke,
   tool-call E2E non-stream + stream, doc and label updates
   (`ling3_tiny` ctest label, model README support text,
   [docs/TESTING.md](../../TESTING.md)).

## Definition of done

`gufo serve --model .../Ling-3.0-tiny-Q8_0.gguf` loads and serves chat +
tool calls on gfx1151; Tiny unit/E2E evidence in STATUS.md; Flash
regression (bit-identity + full `cpu-test` suite) unchanged; no version or
changelog edits.

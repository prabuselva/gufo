---
name: verify-model-change
description: Verify a functional (not performance) change to a Gufo model on Strix Halo gfx1151 - generation/streaming, stop tokens, vision attach or a new CLI/HTTP feature. Runs the smallest covering ctest golden set, interprets skips versus fixture-mismatch failures, and does one GPU-exclusive end-to-end generation check. Leave perf work to optimize-kernel and benchmark-model.
metadata:
  origin: gufo
---

# Verify a functional model change

Confirm a model behaves correctly after an edit: focused goldens first, then one
real end-to-end generation. Read the model's `docs/models/<model>/{README,QUALITY}.md`
and `docs/TESTING.md`, plus root [AGENTS.md](../../../AGENTS.md) for the GPU-
exclusive loop and build presets. Follow the user's machine, time and Git
instructions.

## Order

1. `python3 tools/ci/check-format.py` (add `--fix`, rerun) before any C++ commit.
2. Build the smallest covering target with the `gpu-test` preset. Model/kernel
   test targets are `EXCLUDE_FROM_ALL`: `ctest` never compiles, so always pass
   `--target` for the golden you intend to run after editing sources.
3. Run the focused goldens, then ONE GPU-exclusive end-to-end generation.

## Running goldens on this host

GPU-linked test binaries need the ROCm loader path even when labelled `cpu`:

```sh
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
  ctest --preset gpu-full -R '^<model>\.(tokenizer|chat_template|<...>)$' --output-on-failure
```

Goldens tagged `external-model` **skip** (they pass) without the model env var
(`GUFO_<MODEL>_GGUF`); a skip is not a quality pass. Pass the var to exercise
tokenizer/chat-template assertions against real metadata.

Trap - numeric oracle tests (`reference`, `*_reference`, `weights`) validate a
tiny synthetic fixture, not a production checkpoint. Pointing `GUFO_<MODEL>_GGUF`
at the real multi-GB model makes hand-computed logits disagree and prints spurious
`FAIL: step N logits are non-degenerate`. That failure is wrong-fixture, not a
regression: read the test to find the fixture it actually wants, and do not chase
the number. Confirm a suspected regression against an unmodified build instead.

## Stop tokens and streamed generation

Generation stopping and text surfacing are the same token; verify both.

- Some models carry a dedicated end-of-turn id in GGUF metadata
  (`tokenizer.ggml.eot_token_id`) that is distinct from `tokenizer.ggml.eos_token_id`.
  The end-of-turn marker, not generic EOS, is the generation stop. Check the
  tokenizer golden test for the intended `EosId()` before adding a parallel field;
  aligning the existing EOS usually beats branching on a second id.
- The decode path that surfaces text (`TokenText`) must suppress the stop token.
  The non-speculative loop breaks on the stop token before emitting, but the
  MTP/speculative streaming runner emits `TokenText(token)` for EVERY accepted
  token with no per-token stop filter, so a raw turn marker leaks there even
  though the plain path looks clean. Fix surfacing and stopping together, and
  confirm the streamed/speculative path, not only greedy single-step output.

## End-to-end generation check

A resident `gufo serve` holds VRAM, so never time or verify a model while it is
up. Use the detached-run + foreground-watchdog pattern from AGENTS.md, launching
both in the SAME tool call. Confirm the behaviour under test, not just a zero
exit: for image input verify the `--image`/`--mmproj` attach and grounding, and
that a reply ends at a turn boundary with no control token in the text. Read the
`.log`, not only the watchdog line. The watchdog respawns the resident server and
must report `/v1/models -> 200` before you continue.

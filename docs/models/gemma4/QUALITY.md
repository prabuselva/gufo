# Gemma-4-26B-A4B — Quality Contracts

The CPU scalar oracle (`src/models/gemma4/reference/`) is the numerical
ground truth. Every GPU path must reproduce it within the tolerances below;
end-to-end greedy token streams are the final gate.

## Oracle

- Plain C++ (double-accumulated where noted), built only by validation
  targets (`gufo_gemma4_reference`, `-O2`).
- Pins the exact operation order of the reference graph
  (`llama.cpp/src/models/gemma4.cpp`, `gemma4-assistant.cpp`):
  embedding scale √2816 on token rows only; per-head RMSNorm (eps 1e-6) on
  Q and K with weights, on V without weight; proportional rope on full
  layers (`inv_freq[i] × rope_freqs[i]`); V = rms_norm(wk·x) on full layers,
  rms_norm(wv·x) on SWA; attention scale 1.0; softmax router over
  `rms_norm(attn_out)·(1/√2816)·ffn_gate_inp.scale` (elementwise) then
  `ffn_gate_inp`; top-8 renormalized weights; GELU-parallel shared FFN;
  the five-norm residual flow; `layer_output_scale`; final norm → lm_head →
  softcap ×(1/30) → tanh → ×30. Draft logits are **not** softcapped.

## Parity tolerances

| Path | Contract |
| --- | --- |
| F32 fused ops (norms, rope, softmax) | ≤ 1e-5 abs vs oracle |
| Q8_0 GEMV/GEMM (mmq, dense + routed) | accumulation order pinned; logits ≤ 2e-3 abs |
| WMMA MoE (routed_f16, F16 narrowing) | not bit-exact; guarded by forward test, logits ≤ 5e-3 abs |
| Attention WMMA (F16 KV mirror) | ≤ 5e-3 abs on attention output |
| Full-model forward (`rocm_forward`, pos 0–4) | logits finite + argmax exact + worst_abs ≤ 2.0; `h_out` worst_abs ≤ 2.0 |
| End-to-end greedy (serve, M7+) | identical token stream vs oracle on the pinned prompts |
| MTP speculative decode (M8b, opt-in) | lossless accept from the trunk verify pass; `gemma4.speculative` token-identical at 48 tokens. Not bit-identical to non-spec greedy at long range: the batched k+1-row verify that yields the speedup differs from single-token decode by the same GEMM-vs-GEMV rounding as the rows above, so a near-tied argmax can flip. Opt-in only; the default greedy path is unchanged. |
| Vision tower F32 (`gemma4.vision_encoder`) | device vs CPU oracle on a fixed 96×96 image: checksums identical (−53.0989), max abs diff 2.9e-05, mean 4.5e-06 (F32 rounding). mmproj uploaded as F32 (BF16 upcast on host). |
| Vision preprocessing (`gemma4.vision_prompt`) | smart-resize targets vs hand-computed `calc_size_preserved_ratio`; solid-color images survive resize unchanged in CHW `[0,1]`; `<|image|>` marker expands to `<|image>` + N fillers + `<image|>` with correct offsets/counts. |
| Vision end-to-end OCR (M9) | greedy `gufo prompt --mmproj --image` on the mtmd moon-landing page returns the exact headline "A Powdery Surface Is Closely Explored" (Q8_K_XL trunk + BF16 mmproj). |

The full-model forward contract is absolute-error + argmax, not relative: the
graph is unscaled attention over 30 layers with a top-8 MoE, so a rare
near-tied router flip shifts even the top-1 logit by O(1) and relative error
is unbounded on near-zero logits. Observed worst_abs per position (0–4):
0.008 / 0.743 / 0.15 / 0.083 / 0.062, argmax matching at every position; a
logic bug against the ±30 softcap would show O(10) absolute error.

## Golden vectors

- Tokenizer: `tests/models/gemma4/tokenizer_golden.h` — token id vectors for
  a pinned corpus (prose, code, CJK, newline runs, tool-call markup),
  generated with the llama.cpp fork and regenerated only by an explicit
  decision recorded in EXPERIMENTS.md.
- Chat template: rendered strings for pinned message sets (plain, tools,
  thinking) compared against the GGUF `tokenizer.chat_template` rendered by
  the llama.cpp fork's chat parser.
- Forward: `GUFO_GEMMA4_GGUF` prompts pinned in
  `tests/models/gemma4/rocm_forward_test.cpp`; output hash recorded per
  phase so regressions are visible.
- Serve (M7): greedy HTTP chat through `Gemma4TextRunner` returns the correct
  answer with `finish_reason=stop` (exclusive-GPU smoke); the full end-to-end
  greedy token-stream parity vs the oracle on pinned prompts is the M7+ gate in
  the tolerance table above.

## Rules

- A missing-model skip (exit 77) is not a quality pass.
- Any kernel change that alters rounding must keep the end-to-end greedy
  token stream identical, or the change must be justified in EXPERIMENTS.md
  with acceptance/quality measurements.
- MTP is opt-in (`--speculative mtp`); the default greedy path must be
  unchanged. When enabled, every committed token comes from the trunk verify
  pass (never the draft), so output stays on the target's own distribution;
  the batched verify is not bit-identical to single-token decode, so greedy
  can diverge from non-speculative greedy at near-tied tokens at long range
  (measured ~90 tokens on one chat prompt). This matches llama.cpp MTP and is
  the accepted cost of batched verification.
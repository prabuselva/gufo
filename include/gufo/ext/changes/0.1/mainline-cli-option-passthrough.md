# Mainline CLI option passthrough

Carry the CLI-sourced tuning fields the mainline model drivers consume through
the published option records, so in-tree drivers keep their behaviour when the
shared CLI builds their requests:

- Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
  `TextBenchOptions.draft_tokens_specified` records whether `gufo bench`
  received an explicit `--draft-tokens` count; drivers fall back to their
  calibrated default only when it is `false`.
- Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
  `TextPromptOptions.attn_window` and `TextPromptOptions.attn_sink` carry the
  sparse-attention prefill settings parsed from `--attn-window`/`--attn-sink`.
- Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
  `TextSpeculativeConfig.max_draft_tokens_specified` mirrors the flag for the
  serving `spec decoding` route.

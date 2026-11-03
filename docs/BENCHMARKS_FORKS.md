# Fork-wide benchmarks (Strix Halo, gfx1151)

Measured on the merged fork of `mainline_fork` with `cmake --preset release`
(LLM inference only, ROCm/HIP), Strix Halo `gfx1151`. One `gufo bench` process
per cell, greedy sampling, nothing else on the GPU (resident `gufo serve`
stopped for the sweep window). Cell names follow upstream llama-bench:
`ppK` = single prefill of that many tokens (`pp512` ... `pp100K`), `tg` =
token generation from a 512-token prompt, `+ MTP` = the same decode test with
MTP speculative decoding enabled (speedup vs the base column in brackets).
All values are t/s. `Load` is the median cold model-load seconds over that
model's runs. Drive a sweep with `tools/bench/fork_sweep.sh <key>` (the
exclusive-GPU protocol) and parse with `tools/bench/fork_sweep_report.py`.

Sweep dates: 2026-11-02/03, revision `74888ea0`. All 8 models, every cell rc=0.

| Model | Load | pp512 | pp2K | pp4K | pp8K | pp16K | pp32K | pp65K | pp100K | tg128 | tg1024 | tg128 + MTP | tg1024 + MTP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flint Preview 40B-A2B (Q8_0) | 12.9 | 2169.8 | 2960.6 | 2860.2 | 2429.4 | 1823.4 | 1165.6 | 688.8 | 466.0 | 46.6 | 37.7 | - | - |
| Ling 3.0 Tiny (Q8_0) | 11.4 | 5486.7 | 5398.4 | 4513.8 | 3317.2 | 2155.8 | 1208.2 | 657.3 | 416.3 | 107.4 | 99.9 | - | - |
| Gemma4 26B-A4B IT (Q8_K_XL) | 10.6 | 2359.9 | 3225.0 | 3177.6 | 3041.3 | 2737.7 | 2218.6 | 1661.6 | 1303.4 | 42.5 | 41.4 | 57.6 (1.35x) | 64.2 (1.55x) |
| Qwen3.6 35B-A3B (Q8_K_XL) | 23.9 | 1734.3 | 2547.3 | 2524.1 | 2455.3 | 2287.6 | 1949.1 | 1523.6 | 1254.1 | 45.0 | 44.6 | 62.6 (1.39x) | 64.6 (1.45x) |
| Qwen3.8 Flash Next (UD-Q4_K_XL) | 34.3 | 1039.2 | 1745.5 | 1799.2 | 1790.1 | 1746.0 | 1654.4 | 1618.9 | 1582.9 | 28.4 | 28.4 | 29.6 (1.04x) | 43.6 (1.54x) |
| Ling 3.0 Flash (Q4_K_M) | 33.1 | 1233.4 | 1366.5 | 1233.3 | 1009.0 | 688.1 | 436.8 | 240.4 | 164.9 | 38.4 | 37.3 | 43.0 (1.12x) | 42.3 (1.13x) |
| Nemotron 3.5 Bolt 74B (24 GB) | 7.3 | 555.1 | 619.9 | 654.5 | 664.9 | 640.3 | 571.3 | 478.8 | 392.7 | 26.3 | 24.0 | - | - |
| Nemotron 3.5 Bolt 74B (UD-Q4_K_XL) | 22.8 | 527.4 | 624.1 | 642.7 | 646.4 | 607.5 | 556.3 | 459.6 | 384.6 | 20.2 | 18.9 | - | - |

Artifacts and load spread under `$HOME/projects/llm/models/gguf`:

| Model | GGUF artifact | Load range (s) | MTP draft |
| --- | --- | --- | --- |
| Flint Preview 40B-A2B | `Flint-40B-A2B-GGUF/flint-Q8_0.gguf` (39.40 GiB) | 12.3-21.5 | - |
| Ling 3.0 Tiny | `Ling3.0-Tiny/Ling-3.0-tiny-Q8_0.gguf` (7.83 GiB) | 2.5-14.3 | - |
| Gemma4 26B-A4B IT | `Gemma4-26B-A4B-IT/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (25.74 GiB) | 9.0-15.3 | `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`, `--draft-tokens 2` |
| Qwen3.6 35B-A3B | `Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | 22.8-35.5 | same file (`--mtp-model <model>`) |
| Qwen3.8 Flash Next | `Qwen3.8-Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL` (4 shards) | 26.4-55.8 | `mtp-Qwen3.8-Flash-Next-Q8_0_unsloth.gguf` |
| Ling 3.0 Flash | `Ling3.0-Flash/Ling-3.0-flash-Q4_K_M` (2 shards, 49.4 GiB) | 29.3-47.3 | in-model (builtin MTP heads) |
| Nemotron 3.5 Bolt 74B | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (22.7 GiB IQ2-style distill) | 6.7-18.3 | - |
| Nemotron 3.5 Bolt 74B | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD_4_K_XL-GGUF/UD-Q4_K_XL/` (3 shards, 43.08 GiB) | 20.5-34.2 | - |

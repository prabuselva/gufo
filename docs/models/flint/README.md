# Flint Preview 40B-A2B

Preview MoE text model (49 layers, 256+2 experts, top-6 sigmoid routing,
GQA 32Q/4KV with per-head QK norms and attention sinks, 14 global/NoPE
layers + 513-token sliding window, ctx 262144). Weights: Q8_0 imatrix GGUF
(`flint-Q8_0.gguf`, 41.7 GiB), acquired separately.

Run (Linux gfx1151 build, weights path required by the tests via
`GUFO_FLINT_GGUF`):

```sh
gufo bench --model flint-Q8_0.gguf -p 8192 -n 128
gufo serve llm --model flint-Q8_0.gguf -p 8080 -c 32768   # single session
```

- [Status and checkpoint log](STATUS.md)
- [Benchmarks vs llama.cpp b11879](BENCHMARKS.md)
- [Quality and contracts](QUALITY.md)
- [Retained and rejected experiments](EXPERIMENTS.md)

Serve is a single serial session; no concurrency, no speculative draft.

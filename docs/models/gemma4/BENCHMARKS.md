# Gemma-4-26B-A4B — Benchmarks

All numbers on Strix Halo gfx1151, exclusive GPU
(`tools/bench/gpu_exclusive.sh`), greedy sampling, `--n-gen 128`.
Reference column: the local llama.cpp fork
(`/home/praburaja/projects/llm/llama.cpp/llama.cpp`, ROCm build, same
GGUFs). Preserve compiler/dependency versions when comparing.

Artifacts: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` (trunk),
`mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (draft), `mmproj-BF16.gguf` (vision)
under `/home/praburaja/projects/llm/models/gguf/Gemma4-26B-A4B-IT/`.

## Reproduce

```sh
# Gufo
build/release/gufo bench --model "$GUFO_GEMMA4_GGUF" --n-prompt 16384 --n-gen 128
# llama.cpp fork reference
llama-bench -m "$GUFO_GEMMA4_GGUF" -p 512,1024,2048,4096,8192,16384,32768,65536,102400 -n 128
```

## Prefill (t/s)

| pp | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| 512 | — | — | — |
| 1024 | — | — | — |
| 2048 | 505.49 | — | — |
| 4096 | — | — | — |
| 8192 | 260.60 | — | — |
| 16384 | — | — | — |
| 32768 | — | — | — |
| 65536 | — | — | — |
| 102400 | — | — | — |

## Decode (t/s, tg128)

| Config | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| Q8_K_XL, no MTP | 38.22 | — | — |
| Q8_K_XL, MTP n=1 | 36.60 | — | −4.2% |
| Q8_K_XL, MTP n=2 | 46.93 | — | +22.8% |
| Q8_K_XL, MTP n=4 | 59.99 | — | +57.0% |

MTP is opt-in (`--speculative mtp --mtp-model <draft>`); `n` is
`--draft-tokens`, capped at 4 by `kMaxVerifyRows=5`. n=4 is the best measured
(acceptance 92/110). Speculative decode is distribution-preserving but not
bit-identical to non-speculative greedy beyond ~90 tokens (batched verify is
the speedup source; see [EXPERIMENTS.md](EXPERIMENTS.md)).

## Quantizations (Q4_K_M vs Q8_K_XL)

`gufo bench --repetitions 1`, exclusive GPU, greedy. Q4_K_M upcasts the dense
Q4_K/Q5_K/Q6_K tensors to Q8_0 at load and keeps the routed experts native
(Q4_K gate/up + Q8_0 down); see [EXPERIMENTS.md](EXPERIMENTS.md).

| Test | Q4_K_M (17.09 GiB) | Q8_K_XL (25.74 GiB) | gain |
| --- | --- | --- | --- |
| pp512 | 1004.83 | 966.06 | +4.0% |
| pp8192 | 261.66 | 251.19 | +4.2% |
| pp16384 | 163.14 | 160.66 | +1.5% |
| tg128 | 41.87 | 38.30 | +9.3% |

Q4_K_M is 33.6% smaller and faster on every axis: the dense upcast costs no
throughput (Q8_0 is the native dense tier) and the smaller resident footprint
helps decode bandwidth-bound tg most.

## Serve (t/s, streaming)

| Scenario | Gufo | llama.cpp | gain |
| --- | --- | --- | --- |
| short chat, MTP | — | — | — |
| 16K prompt + gen, MTP | — | — | — |

## Vision

| Scenario | Gufo | llama.cpp (mtmd) | gain |
| --- | --- | --- | --- |
| single 448px image pp | — | — | — |

## Notes

- (fill per phase; record build versions per table refresh)
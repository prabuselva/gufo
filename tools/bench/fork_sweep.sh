#!/usr/bin/env bash
# Fork-wide benchmark sweep for ONE model: model load + prefill depth grid
# (512..100000) + decode (tg128/tg1024) + optional MTP speculative decode.
# Every cell is a fresh `gufo bench` process, so model load runs are
# un-warmed context and one failing depth cannot kill the others.
#
# Run INSIDE an exclusive-GPU window. From a harness use:
#   tools/bench/bench_bg.sh <key> <inner_timeout_s> tools/bench/fork_sweep.sh <key>
# and then tools/bench/bench_wait.sh in the foreground (AGENTS.md protocol).
#
# Outputs under ${GUFO_FORK_SWEEP_DIR:-/tmp/opencode/sweep}/<key>/:
#   <cell>.out / <cell>.err per cell; run markers land on stdout (the
#   bench_bg log). Parse the table rows "| model | size | ROCm (HIP) |
#   ppNNN | t/s |" and "[Model Load]: N s" lines from the .out files.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
key="${1:?usage: fork_sweep.sh <model-key>}"
state="${GUFO_FORK_SWEEP_DIR:-/tmp/opencode/sweep}/$key"
mkdir -p "$state"
gufo=build/release/gufo
M="$HOME/projects/llm/models/gguf"

model=""
draft=""
spec=none
draft_tokens=""
case "$key" in
  qwen38-flash-next)
    model="$M/Qwen3.8-Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    draft="$M/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0_unsloth.gguf"
    spec=external
    ;;
  qwen36-35b-a3b)
    model="$M/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    spec=same-file
    ;;
  gemma4-26b-a4b)
    model="$M/Gemma4-26B-A4B-IT/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
    draft="$M/Gemma4-26B-A4B-IT/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    spec=external
    draft_tokens=2
    ;;
  ling30-flash)
    model="$M/Ling3.0-Flash/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf"
    spec=builtin
    ;;
  ling30-tiny)
    model="$M/Ling3.0-Tiny/Ling-3.0-tiny-Q8_0.gguf"
    ;;
  flint-40b-a2b)
    model="$M/Flint-40B-A2B-GGUF/flint-Q8_0.gguf"
    ;;
  nemotron-bolt-24gb)
    model="$M/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
    ;;
  nemotron-bolt-ud-q4k-xl)
    model="$M/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD_4_K_XL-GGUF/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00001-of-00003.gguf"
    ;;
  *)
    echo "fork_sweep: unknown model key '$key'" >&2
    exit 2
    ;;
esac
if [ "$spec" = same-file ]; then draft="$model"; fi

if [ ! -f "$model" ]; then
  echo "fork_sweep: model missing: $model" >&2
  exit 2
fi
if [ "$spec" = external ] && [ ! -f "$draft" ]; then
  echo "fork_sweep: draft missing: $draft (disabling spec rows)" >&2
  spec=none
fi

run() { # <label> <bench args...>
  local label="$1"
  shift
  echo "=== RUN $key/$label $(date -Is) :: gufo bench $*"
  if "$gufo" bench "$@" > "$state/$label.out" 2> "$state/$label.err"; then
    echo "--- DONE $key/$label rc=0"
  else
    echo "--- DONE $key/$label rc=$? (last stderr below)"
    tail -n 3 "$state/$label.err" | sed "s/^/    /"
  fi
}

# pp512 and tg128 share one process (same as llama-bench short cells).
run pp512-tg128 --model "$model" --n-prompt 512 --n-gen 128
for d in 2048 4096 8192 16384 32768 65536 100000; do
  run "pp$d" --model "$model" --n-prompt "$d" --n-gen 1
done
run pp512-tg1024 --model "$model" --n-prompt 512 --n-gen 1024
if [ "$spec" != none ]; then
  specargs=(--speculative mtp --mtp-model "$draft")
  [ -n "$draft_tokens" ] && specargs+=(--draft-tokens "$draft_tokens")
  run spec-tg128 --model "$model" "${specargs[@]}" --n-prompt 512 --n-gen 128
  run spec-tg1024 --model "$model" "${specargs[@]}" --n-prompt 512 --n-gen 1024
fi
echo "=== SWEEP $key COMPLETE $(date -Is)"

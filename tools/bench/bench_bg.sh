#!/usr/bin/env bash
# Launch ONE exclusive-GPU benchmark in the background and return immediately.
#
# gpu_exclusive.sh blocks for the whole benchmark (stop server -> run -> respawn
# server -> poll /v1/models), which can exceed a tool's wall-clock limit. So run
# it detached here and wait on it in the foreground with bench_wait.sh, which is
# the harness process that stays alive until the benchmark finishes. This run
# uses --no-respawn: it only stops the server and runs the benchmark. The
# FOREGROUND bench_wait.sh does the respawn, so a killed benchmark can never take
# an in-flight respawn down with it and leave the server down.
#
# Usage: bench_bg.sh <name> <inner_timeout_sec> <cmd...>
#   <inner_timeout_sec> bounds the benchmark itself (timeout -k 60).
#
# State (override dir with GUFO_BENCH_STATE_DIR, default /tmp/opencode):
#   $STATE/gufo_bench_<name>.{log,pid,status}
#
# Example:
#   tools/bench/bench_bg.sh pp16k 600 \
#     build/release/gufo bench --model "$MODEL" --n-prompt 16384 --n-gen 1
set -uo pipefail
name="$1"; shift
inner_timeout="$1"; shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state="${GUFO_BENCH_STATE_DIR:-/tmp/opencode}"
mkdir -p "$state"
base="${state}/gufo_bench_${name}"
log="${base}.log"
status="${base}.status"
rm -f "$status"
: > "$log"
echo "$$" > "${base}.pid"
{
  echo "START $(date -Is) name=$name inner_timeout=${inner_timeout}s pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
  echo "CMD: $*"
} | tee -a "$log"
cd "$root"
# shellcheck disable=SC1091
source ./rocm_env.sh >>"$log" 2>&1
timeout -k 60 "$inner_timeout" tools/bench/gpu_exclusive.sh --no-respawn -- "$@" >>"$log" 2>&1
rc=$?
echo "END $(date -Is) rc=$rc" >> "$log"
echo "$rc" > "$status"
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

# Fallback respawner. bench_wait.sh owns the respawn and normally restores the
# server immediately after the status appears. If that watchdog is killed
# (e.g. a harness call timeout) before respawning, the resident `gufo serve`
# stays down. This detached process outlives the harness tree, so after the
# status exists it waits a grace period for the watchdog, then respawns itself
# once the watchdog is proven gone and no other exclusive-GPU run is active.
# flock serializes both respawners. Override grace/window with
# GUFO_BENCH_FALLBACK_GRACE / GUFO_BENCH_FALLBACK_SEC.
fallback_respawn() {
  local url="http://127.0.0.1:${GUFO_SERVE_PORT:-8083}/v1/models"
  local grace="${GUFO_BENCH_FALLBACK_GRACE:-120}" window="${GUFO_BENCH_FALLBACK_SEC:-1800}"
  local elapsed=0 wp
  while [ "$elapsed" -lt "$window" ]; do
    sleep 10
    elapsed=$((elapsed + 10))
    curl -fsS --max-time 5 "$url" >/dev/null 2>&1 && return 0
    wp="$(cat "${base}.watchdog.pid" 2>/dev/null || true)"
    if [ -n "$wp" ] && kill -0 "$wp" 2>/dev/null; then continue; fi
    # Never contend with another gpu_exclusive run (next benchmark in flight)
    # or with a watchdog that has not started yet within the grace period.
    pgrep -f 'tools/bench/gpu_exclusive\.sh' >/dev/null 2>&1 && continue
    [ "$elapsed" -lt "$grace" ] && continue
    # gpu_respawn_server.sh exits 0 without launching when the server already
    # serves, so racing the watchdog across the flock cannot double-start.
    echo "FALLBACK $(date -Is) name=$name watchdog gone and server down; respawning" >>"$log"
    if command -v flock >/dev/null 2>&1; then
      flock "${state}/gufo_serve_respawn.lock" \
        tools/bench/gpu_respawn_server.sh >>"$log" 2>&1 \
        || echo "FALLBACK $(date -Is) name=$name respawn reported failure" >>"$log"
    else
      tools/bench/gpu_respawn_server.sh >>"$log" 2>&1 \
        || echo "FALLBACK $(date -Is) name=$name respawn reported failure" >>"$log"
    fi
    return 0
  done
  echo "FALLBACK $(date -Is) name=$name window expired; server not checked further" >>"$log"
}
fallback_respawn
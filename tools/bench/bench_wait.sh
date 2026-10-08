#!/usr/bin/env bash
# Foreground watchdog + respawner for a background exclusive-GPU benchmark
# (bench_bg.sh).
#
# This is the harness process: it stays alive in the FOREGROUND, polling the
# background run until the benchmark finishes (or a total wall-clock budget
# elapses, then it kills the whole process group). Because bench_bg.sh runs with
# --no-respawn, THIS process then respawns the resident server in the foreground
# (gpu_respawn_server.sh) and confirms /v1/models before returning -- so the
# server always comes back, even if the benchmark was killed. Run it with a tool
# timeout LARGER than total_sec plus the respawn time.
#
# Usage: bench_wait.sh <name> [total_sec] [ready_grace]
#   total_sec    wall-clock budget before the run is killed   [4200]
#   ready_grace  wait between SIGTERM and SIGKILL of the group [90]
#
# State dir must match bench_bg.sh (GUFO_BENCH_STATE_DIR, default /tmp/opencode).
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
name="$1"; shift
total="${1:-4200}"
grace="${2:-90}"
state="${GUFO_BENCH_STATE_DIR:-/tmp/opencode}"
base="${state}/gufo_bench_${name}"
log="${base}.log"
status="${base}.status"
pidfile="${base}.pid"
pid="$(cat "$pidfile" 2>/dev/null || true)"
start=$(date +%s)
echo "WATCHDOG start name=$name pid=${pid:-?} total=${total}s $(date -Is)"
while :; do
  if [ -f "$status" ]; then
    echo "WATCHDOG benchmark finished rc=$(cat "$status") $(date -Is)"
    break
  fi
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    [ -f "$status" ] && { echo "WATCHDOG finished rc=$(cat "$status")"; break; }
    echo "WATCHDOG pid ${pid:-?} gone without status $(date -Is)"
    break
  fi
  now=$(date +%s); el=$((now-start))
  if [ "$el" -ge "$total" ]; then
    echo "WATCHDOG TIMEOUT after ${el}s -> SIGTERM group -${pid}"
    kill -TERM "-${pid}" 2>/dev/null || true
    sleep "$grace"
    kill -KILL "-${pid}" 2>/dev/null || true
    echo "WATCHDOG terminated $(date -Is)"
    break
  fi
  last="$(tail -n 1 "$log" 2>/dev/null || true)"
  echo "  [${el}s] ${last}"
  sleep 15
done
# bench_bg.sh ran with --no-respawn, so THIS foreground process brings the
# resident server back. Skip if something is already serving (avoid a second
# server on the port).
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8083/v1/models 2>/dev/null || echo 000)"
if [ "$code" = 200 ]; then
  echo "WATCHDOG server already serving; skipping respawn"
else
  echo "WATCHDOG respawning resident server (foreground) $(date -Is)"
  "$root/tools/bench/gpu_respawn_server.sh" --settle 30 \
    || echo "WATCHDOG respawn reported failure (see /tmp/gufo_serve_respawn.log)"
  code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8083/v1/models 2>/dev/null || echo 000)"
fi
echo "WATCHDOG server /v1/models -> ${code}"
echo "WATCHDOG done $(date -Is)"
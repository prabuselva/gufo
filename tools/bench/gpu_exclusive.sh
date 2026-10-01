#!/usr/bin/env bash
# Run one benchmark with exclusive GPU/VRAM, then restore the resident server.
#
# The host keeps a `gufo serve` instance resident, holding tens of GB of VRAM
# and contending for the GPU. Kernel and model benchmarks need that VRAM and an
# uncontended device, so this wrapper stops the server, runs one command, and
# ALWAYS brings the server back before returning -- even if the benchmark fails
# or the wrapper is interrupted. The benchmark's exit status is preserved.
#
# Usage:
#   tools/bench/gpu_exclusive.sh [options] -- <command> [args...]
#
# Options:
#   --launch-script PATH   Server launch script used to respawn
#                          [./run_rocm_qwen38_flash_next_mmproj.sh]
#   --host HOST            Host polled for readiness                       [127.0.0.1]
#   --port PORT            Port polled for readiness                       [8083]
#   --expect-model SUBSTR  Require this substring in /v1/models before ready [any model]
#   --ready-timeout SEC    Max wait for the server to serve again          [900]
#   --settle SEC           Extra wait after readiness before returning     [180]
#   --no-kill              Skip stopping a running server (assume VRAM free)
#   -h, --help             Show this help
#
# Examples:
#   tools/bench/gpu_exclusive.sh -- /tmp/attn_causal_bench
#   tools/bench/gpu_exclusive.sh --expect-model Qwen3.8 -- \
#       env GUFO_QWEN36_A3B_BENCH=2048,4096,100000 \
#       build/gpu-test/tests/models/qwen36_a3b/qwen36_a3b_rocm_forward_test
#   tools/bench/gpu_exclusive.sh -- bash -c '/tmp/bench_a && /tmp/bench_b'
set -euo pipefail

cd "$(dirname "$0")/../.."

launch_script="./run_rocm_qwen38_flash_next_mmproj.sh"
host="127.0.0.1"
port="8083"
expect_model=""
ready_timeout=900
settle=60
do_kill=1

while [ $# -gt 0 ]; do
  case "$1" in
    --launch-script) launch_script="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --expect-model) expect_model="$2"; shift 2 ;;
    --ready-timeout) ready_timeout="$2"; shift 2 ;;
    --settle) settle="$2"; shift 2 ;;
    --no-kill) do_kill=0; shift ;;
    -h | --help)
      sed -n '2,32p' "$0"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "gpu_exclusive: unknown option '$1' (put the command after --)" >&2
      exit 2
      ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "gpu_exclusive: no benchmark command given after --" >&2
  exit 2
fi

bench_cmd=("$@")
url="http://${host}:${port}/v1/models"
respawn_log="/tmp/gufo_serve_respawn.log"
respawned=0

log() { printf '\n=== [gpu_exclusive] %s ===\n' "$*"; }

server_pids() { pgrep -f 'gufo serve' 2>/dev/null || true; }

stop_server() {
  local pids
  pids="$(server_pids)"
  if [ -z "$pids" ]; then
    log "no running 'gufo serve' to stop"
    return 0
  fi
  log "stopping gufo serve (pid $pids)"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    [ -z "$(server_pids)" ] && break
    sleep 1
  done
  pids="$(server_pids)"
  if [ -n "$pids" ]; then
    log "force killing gufo serve (pid $pids)"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi
  # Let the driver reclaim VRAM before the benchmark allocates.
  sleep 5
  log "server stopped; VRAM released"
}

respawn_server() {
  [ "$respawned" = 1 ] && return 0
  respawned=1
  if [ ! -x "$launch_script" ]; then
    log "ERROR: launch script not executable: $launch_script -- server NOT restored"
    return 1
  fi
  log "respawning server via $launch_script (log: $respawn_log)"
  nohup "$launch_script" >"$respawn_log" 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  log "server launching (pid $pid); polling $url (timeout ${ready_timeout}s)"
  local waited=0 body
  while [ "$waited" -lt "$ready_timeout" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR: server process exited early; tail of $respawn_log:"
      tail -n 20 "$respawn_log" 2>/dev/null || true
      return 1
    fi
    body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
    if [ -n "$body" ] && { [ -z "$expect_model" ] || printf '%s' "$body" | grep -q "$expect_model"; }; then
      log "server ready after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "ERROR: server not ready within ${ready_timeout}s; tail of $respawn_log:"
  tail -n 20 "$respawn_log" 2>/dev/null || true
  return 1
}

restore() {
  local rc=$?
  trap - EXIT INT TERM
  respawn_server || true
  if [ "$settle" -gt 0 ]; then
    log "settling ${settle}s before returning"
    sleep "$settle"
  fi
  log "done (benchmark exit ${rc})"
  exit "$rc"
}
trap restore EXIT INT TERM

if [ "$do_kill" = 1 ]; then stop_server; fi

log "running benchmark: ${bench_cmd[*]}"
"${bench_cmd[@]}"
# Reaching the end (or any failure) fires the EXIT trap -> restore().

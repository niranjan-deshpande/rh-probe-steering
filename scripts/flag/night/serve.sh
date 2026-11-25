#!/bin/bash
# Start and stop the vLLM server for Qwen3.8-27B on the pod.
#
# Usage:
#   serve.sh start plain            raw-completions server, prefix caching on (rollouts, unsteered generation)
#   serve.sh start steer TABLE      per-request steering: applies scripts/flag/night/patch_vllm_steer_multi.py to the
#                                   venv if it is not applied yet, loads TABLE once, prefix caching off
#   serve.sh stop                   kill the server's whole process group and wait until the GPU memory is free
#   serve.sh status                 prints SERVER_READY <mode>, SERVER_STARTING <mode> or SERVER_DOWN
#   serve.sh unpatch                put the pristine vLLM model file back (patch script --restore; server must be down)
# `start` blocks until GET /v1/models answers, then prints the exact line SERVER_READY and exits 0; the server keeps
# running in its own session. If the server dies or is not ready in READY_TIMEOUT_MIN it prints SERVER_FAILED <reason>
# and exits 1 with nothing left running. `stop` ends with the exact line SERVER_STOPPED.
# Server log: /workspace/logs/server_<mode>.log (appended). State: /workspace/running/server.pid and server.mode.
# Knobs (environment): PORT 8000, MAX_SEQS 80, GPU_UTIL 0.97, MAX_MODEL_LEN (plain 65536, steer 20000), LAYER 36,
#   READY_TIMEOUT_MIN 25, REQUIRE_VID 1 (steer: a request without a condition id is an error), EXTRA_ARGS (appended).
# A patched install with RLP_STEER_LAYER unset does no steering, so `start plain` works with or without the patch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
PORT="${PORT:-8000}"; LAYER="${LAYER:-36}"
VLLM_BIN="${VLLM_BIN:-$NIGHT_VENV/bin/vllm}"; PY="${NIGHT_PY:-$NIGHT_VENV/bin/python}"
PATCH="${STEER_PATCH:-$HERE/patch_vllm_steer_multi.py}"
mkdir -p "$NIGHT_LOGS" "$NIGHT_RUN"
PIDF="${NIGHT_RUN:?}/server.pid"; MODEF="${NIGHT_RUN:?}/server.mode"

pid() { cat "$PIDF" 2> /dev/null || true; }
# Alive if the group or the leader exists (the leader alone, for the instant before setsid has made the group).
up() { local p; p="$(pid)"; [ -n "$p" ] && { kill -0 -- "-$p" 2> /dev/null || kill -0 "$p" 2> /dev/null; }; }
ready() {
  if command -v curl > /dev/null; then curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null
  else "$PY" -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" 2> /dev/null; fi
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }

stop_server() {
  local p; p="$(pid)"
  if [ -n "$p" ]; then
    kill -TERM -- "-$p" 2> /dev/null || kill -TERM "$p" 2> /dev/null
    for _ in $(seq 1 30); do up || break; sleep 1; done
    kill -KILL -- "-$p" 2> /dev/null; kill -KILL "$p" 2> /dev/null
  fi
  # A GPU process that left the group but is still in the server's session (same session id) is the server's:
  # kill it by pid. Anything else on the GPU (a transformers stage) is never touched.
  if command -v nvidia-smi > /dev/null; then
    if [ -n "$p" ]; then
      for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null); do
        if [ "$(ps -o sid= -p "$q" 2> /dev/null | tr -d ' ')" = "$p" ]; then kill -KILL "$q" 2> /dev/null; fi
      done
    fi
    for _ in $(seq 1 60); do
      u="$(gpu_used)"; [ -z "$u" ] && break; [ "$u" -lt "${GPU_FREE_MIB:-2000}" ] && break; sleep 1
    done
    echo "gpu memory in use after stop: ${u:-?} MiB"
  fi
  [ -e "$PIDF" ] && unlink "$PIDF"; [ -e "$MODEF" ] && unlink "$MODEF"
  return 0
}
fail() { echo "SERVER_FAILED $*"; stop_server > /dev/null 2>&1; exit 1; }

model_file() {
  "$PY" -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('vllm').origin), 'models/qwen3_5/nvidia/model.py'))"
}
ensure_patched() {   # the patch script is idempotent: same version = no change, other version = restore and re-patch
  [ -f "$PATCH" ] || fail "steering patch not found at $PATCH"
  local mp; mp="$(model_file)" || fail "cannot locate vLLM in $NIGHT_VENV"
  [ -f "$mp" ] || fail "vLLM model file missing: $mp"
  if ! "$PY" "$PATCH"; then
    grep -q "rlp-steer" "$mp" || fail "steering patch does not apply to a clean vLLM (see PATCH_FAILED above)"
    echo "an older steering patch is in the way: reinstalling vllm==${VLLM_VERSION:-0.34.0} without dependencies, then patching"
    "$NIGHT_VENV/bin/uv" pip install -q --python "$PY" --no-deps --reinstall "vllm==${VLLM_VERSION:-0.34.0}" \
      || fail "could not reinstall vLLM"
    "$PY" "$PATCH" || fail "steering patch does not apply"
  fi
  grep -q "rlp-steer" "$mp" || fail "patch script ran but $mp is unchanged"
}

start() {
  local mode="${1:-}" table="${2:-}" len log
  case "$mode" in
    plain) len="${MAX_MODEL_LEN:-65536}" ;;
    steer) len="${MAX_MODEL_LEN:-20000}"
           [ -n "$table" ] || fail "usage: serve.sh start steer TABLE"
           [ -s "$table" ] || fail "steering table missing or empty: $table" ;;
    *) echo "usage: serve.sh start plain | start steer TABLE | stop | status | unpatch" >&2; exit 2 ;;
  esac
  if up; then
    if [ "$(cat "$MODEF" 2> /dev/null)" = "$mode $table" ] && ready; then echo "already running"; echo SERVER_READY; exit 0; fi
    echo "SERVER_FAILED a server is already running ($(cat "$MODEF" 2> /dev/null)); run serve.sh stop first"; exit 1
  fi
  ready && { echo "SERVER_FAILED port $PORT already answers but this script did not start it"; exit 1; }
  log="$NIGHT_LOGS/server_$mode.log"
  set -- serve "$NIGHT_MODEL" --dtype bfloat16 --language-model-only --max-model-len "$len" \
    --gpu-memory-utilization "${GPU_UTIL:-0.97}" --port "$PORT" --max-num-seqs "${MAX_SEQS:-80}"
  unset RLP_STEER_LAYER RLP_STEER_TABLE RLP_STEER_REQUIRE_VID RLP_STEER_FILE
  if [ "$mode" = steer ]; then
    ensure_patched
    for suf in err loaded stats; do [ -e "$table.$suf" ] && unlink "$table.$suf"; done
    export RLP_STEER_LAYER="$LAYER" RLP_STEER_TABLE="$table" RLP_STEER_REQUIRE_VID="${REQUIRE_VID:-1}"
    set -- "$@" --no-enable-prefix-caching
  fi
  # shellcheck disable=SC2086
  set -- "$@" ${EXTRA_ARGS:-}
  echo "SERVER_START $mode $(date -u +%FT%TZ): $VLLM_BIN $*" >> "$log"
  setsid "$VLLM_BIN" "$@" >> "$log" 2>&1 < /dev/null &
  echo $! > "$PIDF"; echo "$mode $table" > "$MODEF"
  local t0 lim; t0=$(date +%s); lim=$(( ${READY_TIMEOUT_MIN:-25} * 60 ))
  until ready; do
    up || fail "server exited during start-up: $(grep -v '^SERVER_START' "$log" | tail -n 5 | tr '\n' ' ' | cut -c1-600)"
    [ "$mode" = steer ] && [ -s "$table.err" ] && fail "steering patch reported: $(tail -n 2 "$table.err" | tr '\n' ' ')"
    [ $(( $(date +%s) - t0 )) -ge "$lim" ] && fail "not ready after ${READY_TIMEOUT_MIN:-25} min"
    sleep "${READY_POLL_S:-5}"
  done
  if [ "$mode" = steer ]; then
    [ -s "$table.err" ] && fail "steering patch reported: $(tail -n 2 "$table.err" | tr '\n' ' ')"
    [ -s "$table.loaded" ] || fail "server is up but did not write $table.loaded: the steering patch is not active"
    echo "table loaded: $(head -c 300 "$table.loaded")"
  fi
  echo "ready after $(( $(date +%s) - t0 )) s, mode $mode, log $log"
  echo SERVER_READY
}

case "${1:-}" in
  start) shift; start "$@" ;;
  stop) stop_server; echo SERVER_STOPPED ;;
  status)
    if up && ready; then echo "SERVER_READY $(cat "$MODEF" 2> /dev/null)"
    elif up; then echo "SERVER_STARTING $(cat "$MODEF" 2> /dev/null)"
    else echo SERVER_DOWN; fi ;;
  unpatch)
    up && { echo "stop the server first" >&2; exit 1; }
    mp="$(model_file)" || exit 1
    if grep -q "rlp-steer" "$mp"; then "$PY" "$PATCH" --restore > /dev/null; fi
    if grep -q "rlp-steer" "$mp"; then echo UNPATCH_FAILED; exit 1; else echo UNPATCHED; fi ;;
  *) echo "usage: serve.sh start plain | start steer TABLE | stop | status | unpatch" >&2; exit 2 ;;
esac

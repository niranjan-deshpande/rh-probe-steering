#!/bin/bash
# Laptop loop that copies a pod's outputs into the repo every few minutes. It only ever copies FROM the pod.
#
# Usage: laptop_sync.sh POD_SSH_HOST PORT [POD_NAME]          (POD_NAME defaults to HOST_PORT; use A or B)
#   nohup bash scripts/flag/night/laptop_sync.sh 1.2.3.4 12345 A > logs/flag/night/sync_A.log 2>&1 &
#
#   /workspace/rlp/cache/flag/   ->  cache/flag/
#   /workspace/logs/             ->  logs/flag/night/<pod>/logs/
#   /workspace/done/             ->  logs/flag/night/<pod>/done/
#   /workspace/failed/           ->  logs/flag/night/<pod>/failed/
#   /workspace/watchdog/         ->  logs/flag/night/<pod>/watchdog/     (heartbeat and watchdog log)
#
# Safety: no --delete, so a local file is never removed. A local file is replaced only when the pod's copy is newer,
# and the replaced version is kept under cache/_backup_night_sync/<pod>/ (same relative path). .env, .venv and
# __pycache__ are never copied. One line per pass: SYNC_OK <pod> <time ET> or SYNC_FAIL <pod> <time ET> <what failed>.
# The loop keeps going when the pod is unreachable (it may be restarting). It ends when the file
# logs/flag/night/<pod>/STOP_SYNC exists, or after one pass with ONCE=1.
# Knobs: INTERVAL seconds between passes (default 180), ONCE=1, EXTRA_EXCLUDE="pattern pattern" (rsync patterns).
set -uo pipefail
[ $# -ge 2 ] || { echo "usage: laptop_sync.sh POD_SSH_HOST PORT [POD_NAME]" >&2; exit 2; }
HOST="$1"; PORT="$2"
source "$(dirname "${BASH_SOURCE[0]}")/laptop_common.sh"
NAME="${3:-${HOST}_${PORT}}"
case "$NAME" in *[!A-Za-z0-9._-]*|""|.*) echo "bad POD_NAME '$NAME'" >&2; exit 2 ;; esac
OUT="$REPO/logs/flag/night/$NAME"
BAK="$REPO/cache/_backup_night_sync/$NAME"
mkdir -p "$OUT/logs" "$OUT/done" "$OUT/failed" "$OUT/watchdog" "$REPO/cache/flag" "$BAK"
EXTRA=()
for x in ${EXTRA_EXCLUDE:-}; do EXTRA+=(--exclude "$x"); done

pull() {   # pull REMOTE_DIR LOCAL_DIR BACKUP_SUBDIR
  "${RSYNC[@]}" -rlt --timeout=120 --update "${NEVER[@]}" --exclude "*.tmp" \
    ${EXTRA[@]+"${EXTRA[@]}"} "${REMOTE}$1/" "$2/"
}
while true; do
  failed=""
  pull /workspace/rlp/cache/flag "$REPO/cache/flag" cache_flag || failed="$failed cache/flag"
  pull /workspace/logs "$OUT/logs" logs || failed="$failed logs"
  pull /workspace/done "$OUT/done" done || failed="$failed done"
  pull /workspace/failed "$OUT/failed" failed || failed="$failed failed"
  pull /workspace/watchdog "$OUT/watchdog" watchdog || failed="$failed watchdog"
  if [ -z "$failed" ]; then
    echo "SYNC_OK $NAME $(et_now) done=[$(ls "$OUT/done" | tr '\n' ' ' | sed 's/ $//')] failed=[$(ls "$OUT/failed" | tr '\n' ' ' | sed 's/ $//')]"
    date -u +%FT%TZ > "$OUT/last_sync"
  else
    echo "SYNC_FAIL $NAME $(et_now)$failed"
  fi
  [ "${ONCE:-0}" = 1 ] && { [ -z "$failed" ]; exit $?; }
  [ -e "$OUT/STOP_SYNC" ] && { echo "SYNC_STOPPED $NAME $(et_now)"; exit 0; }
  sleep "${INTERVAL:-180}"
done

#!/bin/bash
# Run one stage under a time limit, in its own session, with done / failed markers.
#
# Usage: stage.sh NAME TIMEOUT_MIN CMD [ARG...]
#   log      /workspace/logs/NAME.log   (appended; the stage's stdout and stderr)
#   markers  /workspace/done/NAME on success, /workspace/failed/NAME otherwise (one line: reason, exit code, time)
#   skip     if /workspace/done/NAME exists nothing runs (remove the marker to force a re-run)
#   while running: /workspace/running/NAME holds the process-group id (the watchdog reads it)
# Exit code: 0 done or skipped, 124 timed out, otherwise the command's exit code. A failed stage does not stop the
# caller unless the caller checks the exit code, so independent stages can simply be listed one after another.
# The last line of the log, and of stdout, is exactly one of:  STAGE_DONE NAME | STAGE_FAILED NAME | STAGE_SKIPPED NAME
# The command runs from the repo root with the pod environment of env.sh (venv first on PATH, HF offline).
# To run a pipeline as one stage: stage.sh H1 75 bash -c 'set -euo pipefail; step1; step2'
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
[ $# -ge 3 ] || { echo "usage: stage.sh NAME TIMEOUT_MIN CMD [ARG...]" >&2; exit 2; }
NAME="$1"; TMIN="$2"; shift 2
case "$NAME" in *[!A-Za-z0-9_.-]*|""|.*) echo "stage.sh: bad stage name '$NAME'" >&2; exit 2 ;; esac
LIMIT=$(awk -v m="$TMIN" 'BEGIN { if (m !~ /^[0-9]+(\.[0-9]+)?$/ || m <= 0) exit 1; printf "%d", m * 60 + 0.5 }') \
  || { echo "stage.sh: bad TIMEOUT_MIN '$TMIN'" >&2; exit 2; }
mkdir -p "$NIGHT_LOGS" "$NIGHT_DONE" "$NIGHT_FAILED" "$NIGHT_RUN"
LOG="$NIGHT_LOGS/$NAME.log"
RUNF="${NIGHT_RUN:?}/${NAME:?}"; FAILF="${NIGHT_FAILED:?}/${NAME:?}"; DONEF="${NIGHT_DONE:?}/${NAME:?}"
if [ -e "$DONEF" ]; then echo "STAGE_SKIPPED $NAME"; exit 0; fi
if [ -s "$RUNF" ] && kill -0 -- "-$(cat "$RUNF")" 2> /dev/null; then
  echo "stage.sh: stage $NAME is already running (group $(cat "$RUNF"))" >&2; exit 3
fi
[ -e "$RUNF.timeout" ] && unlink "$RUNF.timeout"
[ -e "$FAILF" ] && unlink "$FAILF"
now() { date -u +%FT%TZ; }
cd "$NIGHT_REPO" 2> /dev/null || cd "$NIGHT_ROOT" || exit 2
echo "STAGE_START $NAME $(now) limit=${LIMIT}s cmd: $*" >> "$LOG"
setsid "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNF"
killgroup() {      # the stage is a session leader, so its pid is its process-group id
  kill -TERM -- "-$PID" 2> /dev/null || kill -TERM "$PID" 2> /dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- "-$PID" 2> /dev/null || return 0; sleep 1; done
  kill -KILL -- "-$PID" 2> /dev/null
}
finish() {         # finish RC REASON
  [ -e "$RUNF" ] && unlink "$RUNF"
  [ -e "$RUNF.timeout" ] && unlink "$RUNF.timeout"
  [ -n "${TIMER:-}" ] && kill "$TIMER" 2> /dev/null
  if [ "$1" = 0 ]; then
    now > "$DONEF"; echo "STAGE_DONE $NAME" >> "$LOG"; echo "STAGE_DONE $NAME"
  else
    echo "$2 rc=$1 $(now)" > "$FAILF"
    echo "stage $NAME failed: $2 rc=$1 $(now)" >> "$LOG"; echo "STAGE_FAILED $NAME" >> "$LOG"; echo "STAGE_FAILED $NAME"
  fi
  exit "$1"
}
trap 'killgroup; finish 143 "interrupted"' TERM INT HUP
T0=$(date +%s)
( while kill -0 "$PID" 2> /dev/null; do
    if [ $(( $(date +%s) - T0 )) -ge "$LIMIT" ]; then echo timeout > "$RUNF.timeout"; break; fi
    sleep 1
  done ) &
TIMER=$!
while kill -0 "$PID" 2> /dev/null; do
  if [ -e "$RUNF.timeout" ]; then
    killgroup; wait "$PID" 2> /dev/null
    finish 124 "timeout after ${TMIN} min"
  fi
  sleep 1
done
wait "$PID"; RC=$?
# Children the stage left behind in its group (a server it forgot to stop) must not hold the GPU.
kill -0 -- "-$PID" 2> /dev/null && killgroup
if [ "$RC" = 0 ]; then finish 0 ok; else finish "$RC" "command failed"; fi

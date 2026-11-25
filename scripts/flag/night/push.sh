#!/bin/bash
# Copy the code and the needed inputs from the laptop to /workspace/rlp on a pod. Never copies .env, .venv or caches
# that are not listed.
#
# Usage: push.sh POD_SSH_HOST PORT [INCLUDE_LIST]
#   Code: the folders rlp/ and scripts/ are always copied (2 MB).
#   Inputs: every path in INCLUDE_LIST (default scripts/flag/night/push_include.txt), one per line, relative to the
#   repo root; a folder is copied whole; blank lines and lines starting with # are skipped. A listed path that does
#   not exist is an error, so a typo cannot silently leave an input behind.
#   DRY_RUN=1 lists what would be copied and copies nothing.
# It also creates /workspace/{logs,done,failed,running,watchdog}. Nothing on the pod is deleted. Uses rsync -rlt (not
# -a: chown fails on the pod's network volume). The last line is exactly PUSH_OK on success.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: push.sh POD_SSH_HOST PORT [INCLUDE_LIST]" >&2; exit 2; }
HOST="$1"; PORT="$2"
source "$(dirname "${BASH_SOURCE[0]}")/laptop_common.sh"
LIST="${3:-$REPO/scripts/flag/night/push_include.txt}"
[ -f "$LIST" ] || { echo "include list not found: $LIST" >&2; exit 2; }
cd "$REPO"
TMP="$(mktemp -t night_push.XXXXXX)"
trap 'unlink "$TMP" 2> /dev/null || true' EXIT
bad=0
while IFS= read -r line || [ -n "$line" ]; do
  p="$(printf '%s' "$line" | sed -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's:/*$::')"
  [ -n "$p" ] || continue
  case "/$p/" in
    */../*|//*) echo "push.sh: path must be relative to the repo and stay inside it: $p" >&2; bad=1; continue ;;
    */.env/*|*/.env.*|*/.venv/*) echo "push.sh: refusing to copy $p" >&2; bad=1; continue ;;
  esac
  [ -e "$p" ] || { echo "push.sh: listed input does not exist: $p" >&2; bad=1; continue; }
  printf '%s\n' "$p" >> "$TMP"
done < "$LIST"
[ "$bad" = 0 ] || { echo "PUSH_FAILED fix $LIST" >&2; exit 2; }
DRY=(); [ "${DRY_RUN:-0}" = 1 ] && DRY=(-n -v)
echo "push to $HOST:$PORT at $(et_now): code (rlp, scripts) and $(wc -l < "$TMP" | tr -d ' ') listed inputs, $(tr '\n' '\0' < "$TMP" | xargs -0 du -ch 2> /dev/null | tail -n 1 | cut -f1) of inputs"
if [ "${DRY_RUN:-0}" != 1 ]; then
  pod "mkdir -p /workspace/rlp /workspace/logs /workspace/done /workspace/failed /workspace/running /workspace/watchdog"
fi
"${RSYNC[@]}" -rlt --timeout=120 ${DRY[@]+"${DRY[@]}"} "${NEVER[@]}" rlp scripts "${REMOTE}/workspace/rlp/"
if [ -s "$TMP" ]; then
  "${RSYNC[@]}" -rlt --timeout=120 ${DRY[@]+"${DRY[@]}"} "${NEVER[@]}" --files-from="$TMP" . "${REMOTE}/workspace/rlp/"
fi
echo PUSH_OK

#!/bin/bash
# Local test of stage.sh, serve.sh, push.sh and laptop_sync.sh with fakes (no pod, no GPU, no ssh, no network).
# Usage: bash scripts/flag/night/test_night_scripts.sh [WORK_DIR]      last line: NIGHT_SCRIPTS_TEST_OK
# The work directory (default: a new temporary one) is left in place for inspection.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="${1:-$(mktemp -d -t night_test.XXXXXX)}"
mkdir -p "$T/bin" "$T/pod" "$T/venv/bin" "$T/laptop"
echo "work dir: $T"
FAILS=0
ok() { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; FAILS=$((FAILS + 1)); }
check() { local what="$1"; shift; if "$@" > /dev/null 2>&1; then ok "$what"; else bad "$what"; fi; }
checknot() { local what="$1"; shift; if "$@" > /dev/null 2>&1; then bad "$what"; else ok "$what"; fi; }

# ---- shims for tools that macOS lacks ------------------------------------------------------------------------------
if ! command -v setsid > /dev/null; then
  printf '%s\n' '#!/bin/bash' 'exec perl -e "use POSIX; POSIX::setsid(); exec @ARGV or exit 127" -- "$@"' > "$T/bin/setsid"
  chmod +x "$T/bin/setsid"
fi
export PATH="$T/bin:$PATH"
export NIGHT_ROOT="$T/pod" NIGHT_REPO="$T/pod/rlp" NIGHT_VENV="$T/venv" HF_HOME="$T/hf"
mkdir -p "$NIGHT_REPO"

echo "== stage.sh"
out="$(bash "$HERE/stage.sh" S_ok 1 bash -c 'echo hello; pwd')"; rc=$?
check "success: exit 0" [ "$rc" = 0 ]
check "success: last stdout line" [ "$(echo "$out" | tail -n 1)" = "STAGE_DONE S_ok" ]
check "success: done marker, no failed marker" [ -e "$T/pod/done/S_ok" -a ! -e "$T/pod/failed/S_ok" ]
check "success: log has output and exact final line" [ "$(tail -n 1 "$T/pod/logs/S_ok.log")" = "STAGE_DONE S_ok" ]
check "success: ran in the repo dir" grep -qx "$(cd "$NIGHT_REPO" && pwd -P)\|$NIGHT_REPO" "$T/pod/logs/S_ok.log"
check "success: running marker removed" [ ! -e "$T/pod/running/S_ok" ]
out="$(bash "$HERE/stage.sh" S_ok 1 bash -c 'echo SHOULD_NOT_RUN')"; rc=$?
check "skip when done: STAGE_SKIPPED, exit 0" [ "$rc" = 0 -a "$out" = "STAGE_SKIPPED S_ok" ]
checknot "skip when done: command not run" grep -q SHOULD_NOT_RUN "$T/pod/logs/S_ok.log"

out="$(bash "$HERE/stage.sh" S_fail 1 bash -c 'echo before; exit 7')"; rc=$?
check "failure: exit code passed through" [ "$rc" = 7 ]
check "failure: failed marker with rc" grep -q "rc=7" "$T/pod/failed/S_fail"
check "failure: no done marker, final line" [ ! -e "$T/pod/done/S_fail" -a "$(tail -n 1 "$T/pod/logs/S_fail.log")" = "STAGE_FAILED S_fail" ]
out="$(bash "$HERE/stage.sh" S_fail 1 bash -c 'echo second try')"; rc=$?
check "re-run after failure: succeeds and clears the failed marker" [ "$rc" = 0 -a -e "$T/pod/done/S_fail" -a ! -e "$T/pod/failed/S_fail" ]

t0=$(date +%s)
out="$(bash "$HERE/stage.sh" S_slow 0.05 bash -c 'echo $$ > '"$T"'/slow.pid; sleep 300 & echo $! > '"$T"'/slow.child; wait')"; rc=$?
dt=$(( $(date +%s) - t0 ))
check "timeout: exit 124 (got $rc) within 20 s (took ${dt}s)" [ "$rc" = 124 -a "$dt" -lt 20 ]
check "timeout: failed marker says timeout" grep -q "timeout" "$T/pod/failed/S_slow"
sleep 1
checknot "timeout: stage process is gone" kill -0 "$(cat "$T/slow.pid")"
checknot "timeout: its child is gone too (whole group killed)" kill -0 "$(cat "$T/slow.child")"
check "timeout: final log line" [ "$(tail -n 1 "$T/pod/logs/S_slow.log")" = "STAGE_FAILED S_slow" ]
bash "$HERE/stage.sh" "bad/name" 1 true > /dev/null 2>&1; check "bad name refused (exit 2)" [ $? = 2 ]
bash "$HERE/stage.sh" S_x abc true > /dev/null 2>&1; check "bad timeout refused (exit 2)" [ $? = 2 ]
bash "$HERE/stage.sh" S_indep 1 true > /dev/null; check "a later stage still runs after earlier failures" [ -e "$T/pod/done/S_indep" ]

echo "== watchdog.py --dry-run against the stage markers"
python3 "$HERE/watchdog.py" --root "$T/pod" --deadline $(( $(date +%s) + 3600 )) --max-hours 9 --start-epoch "$(date +%s)" --dry-run --once > "$T/wd.out" 2>&1
check "dry run: tick with reason=ok and a heartbeat file" [ -s "$T/pod/watchdog/heartbeat.json" -a -n "$(grep 'reason=ok' "$T/wd.out")" ]
python3 "$HERE/watchdog.py" --root "$T/pod" --deadline $(( $(date +%s) - 5 )) --max-hours 9 --start-epoch "$(date +%s)" --dry-run --once > "$T/wd2.out" 2>&1
check "dry run: past the deadline it says what it would do" grep -q "DRY_RUN would stop pod .* (reason: deadline)" "$T/wd2.out"

echo "== serve.sh (fake vllm binary, fake vllm package, the real patch script is NOT used)"
PORT_T=$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
mkdir -p "$T/fakepkg/vllm/models/qwen3_5/nvidia" "$T/www/v1"
: > "$T/fakepkg/vllm/__init__.py"; echo "# pristine model file" > "$T/fakepkg/vllm/models/qwen3_5/nvidia/model.py"
echo '{"data": []}' > "$T/www/v1/models"
cat > "$T/venv/bin/vllm" <<EOF
#!/bin/bash
# fake server: records its arguments and environment, optionally fails, then serves /v1/models
echo "\$@" > "$T/vllm.args"; env | grep '^RLP_STEER' > "$T/vllm.env"
[ -e "$T/make_server_fail" ] && { echo "fake crash"; exit 9; }
[ -n "\${RLP_STEER_TABLE:-}" ] && grep -q rlp-steer "$T/fakepkg/vllm/models/qwen3_5/nvidia/model.py" && echo '{"K": 3}' > "\$RLP_STEER_TABLE.loaded"
sleep 2
cd "$T/www" && exec python3 -m http.server "$PORT_T" --bind 127.0.0.1
EOF
cat > "$T/fake_patch.py" <<'EOF'
import importlib.util, os, sys
mp = os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "models/qwen3_5/nvidia/model.py")
s = open(mp).read()
if "--restore" in sys.argv:
    open(mp, "w").write(open(mp + ".orig").read())
elif "rlp-steer" not in s:
    open(mp + ".orig", "w").write(s)
    open(mp, "w").write(s + "# rlp-steer\n")
    open(os.path.join(os.path.dirname(mp), "patch_count"), "a").write("x")
EOF
chmod +x "$T/venv/bin/vllm"
export PYTHONPATH="$T/fakepkg" NIGHT_PY="$(command -v python3)" STEER_PATCH="$T/fake_patch.py" PORT="$PORT_T" READY_POLL_S=1
S="$HERE/serve.sh"
check "status before start: SERVER_DOWN" [ "$(bash "$S" status)" = "SERVER_DOWN" ]
out="$(bash "$S" start plain 2>&1)"; rc=$?
check "start plain: exit 0, last line SERVER_READY" [ "$rc" = 0 -a "$(echo "$out" | tail -n 1)" = "SERVER_READY" ]
check "start plain: model, text-only flag, no steering env" [ -n "$(grep -- '--language-model-only' "$T/vllm.args")" -a -n "$(grep 'Qwen/Qwen3.8-27B' "$T/vllm.args")" -a ! -s "$T/vllm.env" ]
checknot "start plain: prefix caching left on" grep -q -- "--no-enable-prefix-caching" "$T/vllm.args"
check "status: SERVER_READY plain" [ "$(bash "$S" status | cut -d' ' -f1-2)" = "SERVER_READY plain" ]
out="$(bash "$S" start plain 2>&1)"; check "second start of the same mode is a no-op that prints SERVER_READY" [ "$(echo "$out" | tail -n 1)" = "SERVER_READY" ]
echo table > "$T/table.pt"
out="$(bash "$S" start steer "$T/table.pt" 2>&1)"; check "start steer while plain runs is refused" [ $? = 1 -a -n "$(echo "$out" | grep '^SERVER_FAILED')" ]
SPID="$(cat "$T/pod/running/server.pid")"
out="$(bash "$S" stop 2>&1)"; check "stop: last line SERVER_STOPPED" [ "$(echo "$out" | tail -n 1)" = "SERVER_STOPPED" ]
checknot "stop: server process group is gone" kill -0 -- "-$SPID"
checknot "stop: port no longer answers" curl -sf -m 2 "http://127.0.0.1:$PORT_T/v1/models"
check "stop when nothing runs is harmless" [ "$(bash "$S" stop 2>&1 | tail -n 1)" = "SERVER_STOPPED" ]

out="$(bash "$S" start steer "$T/missing.pt" 2>&1)"; check "steer without a table file fails" [ $? = 1 -a -n "$(echo "$out" | grep '^SERVER_FAILED')" ]
out="$(bash "$S" start steer "$T/table.pt" 2>&1)"; rc=$?
check "start steer: SERVER_READY" [ "$rc" = 0 -a "$(echo "$out" | tail -n 1)" = "SERVER_READY" ]
check "start steer: patch applied, env exported, prefix caching off" [ -n "$(grep rlp-steer "$T/fakepkg/vllm/models/qwen3_5/nvidia/model.py")" -a -n "$(grep "RLP_STEER_TABLE=$T/table.pt" "$T/vllm.env")" -a -n "$(grep 'RLP_STEER_LAYER=36' "$T/vllm.env")" -a -n "$(grep -- '--no-enable-prefix-caching' "$T/vllm.args")" ]
bash "$S" stop > /dev/null 2>&1
out="$(bash "$S" start plain 2>&1)"; check "plain on a patched install: no steering env" [ "$(echo "$out" | tail -n 1)" = "SERVER_READY" -a ! -s "$T/vllm.env" ]
bash "$S" stop > /dev/null 2>&1
check "unpatch restores the pristine file" [ "$(bash "$S" unpatch)" = "UNPATCHED" -a -z "$(grep rlp-steer "$T/fakepkg/vllm/models/qwen3_5/nvidia/model.py")" ]
touch "$T/make_server_fail"
out="$(bash "$S" start plain 2>&1)"; rc=$?
check "server that dies at start-up: exit 1 and SERVER_FAILED with the log tail" [ "$rc" = 1 -a -n "$(echo "$out" | grep '^SERVER_FAILED server exited during start-up.*fake crash')" ]
check "after a failed start nothing is left: SERVER_DOWN" [ "$(bash "$S" status)" = "SERVER_DOWN" ]
unlink "$T/make_server_fail"
unset PYTHONPATH PORT

echo "== push.sh and laptop_sync.sh (a local directory stands in for the pod)"
R="$T/laptop/repo"; P="$T/podfs"
mkdir -p "$R/rlp/flag/__pycache__" "$R/scripts/flag/night" "$R/cache/flag/onp" "$R/cache/big" "$R/data" "$R/.venv/lib" "$P"
echo code > "$R/rlp/flag/a.py"; echo junk > "$R/rlp/flag/__pycache__/a.pyc"; echo sh > "$R/scripts/flag/night/x.sh"
echo SECRET > "$R/.env"; echo SECRET > "$R/rlp/.env"; echo SECRET > "$R/cache/flag/onp/.env"; echo v > "$R/.venv/lib/x"
echo input > "$R/cache/flag/onp/means.npy"; echo other > "$R/cache/flag/onp/not_listed.npy"; echo big > "$R/cache/big/huge.bin"; echo d > "$R/data/records.jsonl"
printf '%s\n' '# comment' 'cache/flag/onp/means.npy   # trailing comment' '' 'data/' > "$T/list.txt"
export NIGHT_LAPTOP_REPO="$R" LAPTOP_TEST_REMOTE="$P"
out="$(DRY_RUN=1 bash "$HERE/push.sh" testhost 22 "$T/list.txt" 2>&1)"
check "push dry run copies nothing" [ ! -e "$P/workspace/rlp/rlp" -a "$(echo "$out" | tail -n 1)" = "PUSH_OK" ]
mkdir -p "$P/workspace/rlp"
out="$(bash "$HERE/push.sh" testhost 22 "$T/list.txt" 2>&1)"; rc=$?
check "push: PUSH_OK" [ "$rc" = 0 -a "$(echo "$out" | tail -n 1)" = "PUSH_OK" ]
check "push: code and listed inputs arrived" [ -f "$P/workspace/rlp/rlp/flag/a.py" -a -f "$P/workspace/rlp/scripts/flag/night/x.sh" -a -f "$P/workspace/rlp/cache/flag/onp/means.npy" -a -f "$P/workspace/rlp/data/records.jsonl" ]
check "push: unlisted caches, .venv and __pycache__ stayed home" [ ! -e "$P/workspace/rlp/cache/flag/onp/not_listed.npy" -a ! -e "$P/workspace/rlp/cache/big" -a ! -e "$P/workspace/rlp/.venv" -a ! -e "$P/workspace/rlp/rlp/flag/__pycache__" ]
check "push: no .env anywhere on the pod" [ -z "$(find "$P" -name '.env*')" ]
printf '%s\n' 'cache/flag/onp' > "$T/list2.txt"
bash "$HERE/push.sh" testhost 22 "$T/list2.txt" > /dev/null 2>&1
check "push: a listed folder that contains .env is copied without it" [ -f "$P/workspace/rlp/cache/flag/onp/not_listed.npy" -a -z "$(find "$P" -name '.env*')" ]
printf '%s\n' 'cache/flag/onp/typo.npy' > "$T/list3.txt"
bash "$HERE/push.sh" testhost 22 "$T/list3.txt" > /dev/null 2>&1; check "push: a missing listed input is an error" [ $? = 2 ]
for p in '.env' '../outside' '.venv'; do
  printf '%s\n' "$p" > "$T/list4.txt"; bash "$HERE/push.sh" testhost 22 "$T/list4.txt" > /dev/null 2>&1
  check "push: list entry '$p' is refused" [ $? = 2 ]
done
bash "$HERE/push.sh" 'bad host' 22 > /dev/null 2>&1; check "push: bad host refused" [ $? = 2 ]

mkdir -p "$P/workspace/rlp/cache/flag/mt/run1" "$P/workspace/logs" "$P/workspace/done" "$P/workspace/failed" "$P/workspace/watchdog"
echo result > "$P/workspace/rlp/cache/flag/mt/run1/out.jsonl"; echo SECRET > "$P/workspace/rlp/cache/flag/.env"
echo log > "$P/workspace/logs/H1.log"; echo t > "$P/workspace/done/H1"; echo f > "$P/workspace/failed/V1"; echo '{}' > "$P/workspace/watchdog/heartbeat.json"
echo pod_version > "$P/workspace/rlp/cache/flag/onp/means.npy"; touch -t 203001010000 "$P/workspace/rlp/cache/flag/onp/means.npy"   # newer on the pod
echo laptop_only > "$R/cache/flag/laptop_only.json"
echo laptop_newer > "$R/cache/flag/onp/not_listed.npy"; touch -t 203001010000 "$R/cache/flag/onp/not_listed.npy"                      # newer on the laptop
out="$(ONCE=1 bash "$HERE/laptop_sync.sh" testhost 22 A 2>&1)"; rc=$?
check "sync: one pass, SYNC_OK A (got: $(echo "$out" | tail -n 1 | cut -c1-60))" [ "$rc" = 0 -a -n "$(echo "$out" | grep '^SYNC_OK A [0-9]*:[0-9][0-9] [ap]m ET done=\[H1\] failed=\[V1\]$')" ]
check "sync: outputs, logs, markers and heartbeat arrived" [ -f "$R/cache/flag/mt/run1/out.jsonl" -a -f "$R/logs/flag/night/A/logs/H1.log" -a -f "$R/logs/flag/night/A/done/H1" -a -f "$R/logs/flag/night/A/failed/V1" -a -f "$R/logs/flag/night/A/watchdog/heartbeat.json" ]
check "sync: local-only file kept" [ "$(cat "$R/cache/flag/laptop_only.json")" = "laptop_only" ]
check "sync: the pod's .env was not copied (the laptop's own one is untouched)" [ ! -e "$R/cache/flag/.env" -a "$(cat "$R/cache/flag/onp/.env")" = "SECRET" ]
check "sync: file newer on the laptop was not replaced" [ "$(cat "$R/cache/flag/onp/not_listed.npy")" = "laptop_newer" ]
check "sync: file newer on the pod replaced the local one, old version backed up" [ "$(cat "$R/cache/flag/onp/means.npy")" = "pod_version" -a "$(cat "$R/cache/_backup_night_sync/A/cache_flag/onp/means.npy")" = "input" ]
mv "$P/workspace" "$P/workspace_gone"
out="$(ONCE=1 bash "$HERE/laptop_sync.sh" testhost 22 A 2>&1)"; rc=$?
check "sync: unreachable source gives SYNC_FAIL and a non-zero exit, local files intact" [ "$rc" != 0 -a -n "$(echo "$out" | grep '^SYNC_FAIL A')" -a -f "$R/cache/flag/mt/run1/out.jsonl" ]

echo
if [ "$FAILS" = 0 ]; then echo NIGHT_SCRIPTS_TEST_OK; else echo "NIGHT_SCRIPTS_TEST_FAILED $FAILS"; exit 1; fi

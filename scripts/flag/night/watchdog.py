"""Watchdog that runs ON the pod and stops it. Standard library only; run it with any python3.

    setsid -f python3 /workspace/rlp/scripts/flag/night/watchdog.py --deadline 1795627800 --max-hours 9 \
        < /dev/null > /dev/null 2>&1
    python3 watchdog.py --deadline 1795627800 --max-hours 9 --dry-run --once      # print the decision, stop nothing

The pod is stopped (GraphQL podStop with the key RunPod injects into the container) as soon as ONE of these holds:
  deadline   the wall clock passed --deadline (epoch seconds);
  max_hours  the pod has run for --max-hours (the spend limit for this pod: hours x hourly price). The start is
             --start-epoch if given, else the start time of process 1 (the container), else the watchdog's own start;
  idle       for --idle-min minutes (default 20) there was no stage process, GPU utilisation stayed under
             --util-threshold percent (default 5) and no file under /workspace/logs was modified;
  stop_file  /workspace/STOP exists.
"Stage process" means a live process group recorded by stage.sh in /workspace/running/, or any process whose command
line contains stage.sh or setup_pod.sh. A server with nothing using it does NOT count as activity. To hold the pod
open on purpose, touch a file under /workspace/logs (that restarts the idle clock). /workspace is a volume that
survives a stop, so remove /workspace/STOP before restarting a pod that was stopped that way.

Files (in /workspace/watchdog/, which is deliberately outside /workspace/logs so that the watchdog's own writes do not
look like activity): heartbeat.json (rewritten every tick), watchdog.log (one line per tick), stopped (written when a
stop was requested). The API key is read from the environment or /proc/1/environ, is sent only to api.runpod.io, and
is never printed or logged.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

GRAPHQL = "https://api.runpod.io/graphql"
STAGE_WORDS = ("stage.sh", "setup_pod.sh")


# ----------------------------------------------------------------------------------------------------- decision logic
def decide(now, start, last_active, deadline, max_hours, idle_min, stop_file):
    """Pure function. Returns (stop: bool, reason: str). Checks are ordered so that the reason is the hardest limit."""
    if stop_file:
        return True, "stop_file"
    if deadline is not None and now >= deadline:
        return True, "deadline"
    if max_hours is not None and now - start >= max_hours * 3600:
        return True, "max_hours"
    if idle_min is not None and now - last_active >= idle_min * 60:
        return True, "idle"
    return False, "ok"


def update_last_active(last_active, now, stage_alive, gpu_util, newest_log_mtime, util_threshold=5.0):
    """Pure function. The idle clock restarts when a stage is alive, the GPU is busy, or a log file changed.
    An unreadable GPU utilisation (None) is not evidence of activity."""
    if stage_alive or (gpu_util is not None and gpu_util >= util_threshold):
        last_active = max(last_active, now)
    if newest_log_mtime is not None:
        last_active = max(last_active, min(newest_log_mtime, now))    # a timestamp in the future cannot buy time
    return last_active


def scrub(text, secret):
    """Remove the key from anything that is about to be printed."""
    text = str(text)
    return text.replace(secret, "<key>") if secret else text


# -------------------------------------------------------------------------------------------------------------- probes
def slurp(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace")


def read_pid1_env(path="/proc/1/environ"):
    try:
        raw = slurp(path)
    except OSError:
        return {}
    return dict(x.split("=", 1) for x in raw.split("\0") if "=" in x)


def credentials(environ=None, pid1=None):
    environ = os.environ if environ is None else environ
    pid1 = read_pid1_env() if pid1 is None else pid1
    key = environ.get("RUNPOD_API_KEY") or pid1.get("RUNPOD_API_KEY") or ""
    pod = environ.get("RUNPOD_POD_ID") or pid1.get("RUNPOD_POD_ID") or ""
    return key, pod


def pid1_start_epoch():
    """Start time of process 1 = when the container started (a little after billing started)."""
    try:
        ticks = int(slurp("/proc/1/stat").rsplit(")", 1)[1].split()[19])     # field 22, counted after the command name
        btime = next(int(x.split()[1]) for x in slurp("/proc/stat").splitlines() if x.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return None


def gpu_util():
    """Highest utilisation over the GPUs, in percent, or None if nvidia-smi cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        vals = [float(x) for x in out.split() if x.strip()]
        return max(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def stage_alive(root, proc="/proc"):
    """True if a stage recorded in <root>/running is alive, or a stage.sh / setup_pod.sh process exists."""
    run = os.path.join(root, "running")
    try:
        names = os.listdir(run)
    except OSError:
        names = []
    for n in names:
        if n.startswith("server.") or n.endswith(".timeout"):     # the server alone is not a stage
            continue
        try:
            pgid = int(slurp(os.path.join(run, n)).split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if pgid > 1 and group_alive(pgid):
            return True
    me = os.getpid()
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit() and int(p) != me]
    except OSError:
        pids = []
    for p in pids:
        try:
            cmd = slurp(os.path.join(proc, p, "cmdline")).replace("\0", " ")
        except OSError:
            continue
        if "watchdog.py" in cmd:
            continue
        if any(w in cmd for w in STAGE_WORDS):
            return True
    return False


def newest_mtime(path):
    """Newest modification time of any file under path (None if there is none)."""
    best = None
    for d, _, files in os.walk(path):
        for f in files:
            try:
                m = os.stat(os.path.join(d, f)).st_mtime
            except OSError:
                continue
            best = m if best is None or m > best else best
    return best


# ------------------------------------------------------------------------------------------------------------- actions
def stop_pod(key, pod, opener=urllib.request.urlopen):
    """Ask RunPod to stop this pod. Returns (ok, message); the message never contains the key."""
    if not key or not pod:
        return False, "no RUNPOD_API_KEY / RUNPOD_POD_ID in the environment or /proc/1/environ"
    if not all(c.isalnum() or c in "-_" for c in pod):
        return False, "RUNPOD_POD_ID has unexpected characters"
    body = json.dumps({"query": 'mutation{podStop(input:{podId:"%s"}){id}}' % pod}).encode()
    msg = "not tried"
    for url, headers in ((GRAPHQL, {"Authorization": "Bearer " + key}), (GRAPHQL + "?api_key=" + key, {})):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "pod-watchdog", **headers})
        try:
            with opener(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")
            if '"errors"' in text:
                msg = "GraphQL error: " + scrub(text[:300], key)
                continue
            return True, "podStop accepted: " + scrub(text[:200], key)
        except urllib.error.HTTPError as e:
            msg = "HTTP %s" % e.code
        except Exception as e:  # noqa: BLE001
            msg = scrub("%s: %s" % (type(e).__name__, e), key)
    return False, msg


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deadline", type=float, required=True, help="wall-clock hard stop, epoch seconds")
    ap.add_argument("--max-hours", type=float, required=True, help="spend limit for this pod, in hours of running")
    ap.add_argument("--idle-min", type=float, default=20.0)
    ap.add_argument("--util-threshold", type=float, default=5.0)
    ap.add_argument("--start-epoch", type=float, default=None, help="when billing started (default: container start)")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between checks")
    ap.add_argument("--root", default=os.environ.get("NIGHT_ROOT", "/workspace"))
    ap.add_argument("--dry-run", action="store_true", help="print the decisions, never call the API")
    ap.add_argument("--once", action="store_true", help="one check, then exit")
    a = ap.parse_args(argv)

    wd = os.path.join(a.root, "watchdog")
    logs = os.path.join(a.root, "logs")
    os.makedirs(wd, exist_ok=True)
    key, pod = credentials()
    began = time.time()
    start = a.start_epoch or pid1_start_epoch() or began
    last_active = began                       # a fresh watchdog grants one full idle window
    logf = open(os.path.join(wd, "watchdog.log"), "a")

    def say(line):
        line = scrub("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), line), key)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    say("watchdog start: pod=%s key=%s deadline=%d (in %.2f h) max_hours=%.2f (ran %.2f h so far) idle_min=%g dry_run=%s"
        % (pod or "MISSING", "present" if key else "MISSING", a.deadline, (a.deadline - began) / 3600, a.max_hours,
           (began - start) / 3600, a.idle_min, a.dry_run))
    if not (key and pod) and not a.dry_run:
        say("WATCHDOG_CANNOT_STOP: credentials missing; the laptop guard is the only protection")

    while True:
        now = time.time()
        alive, util, mt = stage_alive(a.root), gpu_util(), newest_mtime(logs)
        last_active = update_last_active(last_active, now, alive, util, mt, a.util_threshold)
        stop, reason = decide(now, start, last_active, a.deadline, a.max_hours, a.idle_min,
                              os.path.exists(os.path.join(a.root, "STOP")))
        state = dict(time=now, pod=pod, stop=stop, reason=reason, stage_alive=alive, gpu_util=util,
                     idle_min=round((now - last_active) / 60, 2), hours_run=round((now - start) / 3600, 3),
                     hours_to_deadline=round((a.deadline - now) / 3600, 3), max_hours=a.max_hours, dry_run=a.dry_run)
        write_json(os.path.join(wd, "heartbeat.json"), state)
        say("tick stop=%s reason=%s stage=%s util=%s idle_min=%.1f hours_run=%.2f to_deadline_h=%.2f"
            % (stop, reason, alive, util, state["idle_min"], state["hours_run"], state["hours_to_deadline"]))
        if stop:
            if a.dry_run:
                say("DRY_RUN would stop pod %s now (reason: %s)" % (pod or "MISSING", reason))
            else:
                write_json(os.path.join(wd, "stopped"), state)
                try:
                    os.sync()
                except Exception:  # noqa: BLE001
                    pass
                ok, msg = stop_pod(key, pod)
                say("STOP_REQUESTED reason=%s ok=%s %s" % (reason, ok, msg))
                if a.once:
                    return 0 if ok else 1
                time.sleep(120)               # if we are still alive the stop did not take: the loop tries again
                continue
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

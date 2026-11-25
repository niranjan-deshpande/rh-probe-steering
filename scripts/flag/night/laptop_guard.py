"""Second-layer hard stop, run on the LAPTOP. It only computes and prints decisions; it calls no API and stops nothing.
The lead reads the STOP lines and terminates those pods with its own tool. Standard library only.

    python3 scripts/flag/night/laptop_guard.py --deadline 1795627800 --budget 38 \
        --pod A,<pod id>,<start>,2.19 --pod B,<pod id>,<start>,2.19,9

--pod NAME,ID,START,PRICE[,MAX_HOURS[,STOPPED_AT]]
    START and STOPPED_AT are epoch seconds or ISO-8601 with an offset (2026-11-25T01:10:00-05:00, or ...Z). PRICE is
    dollars per hour. MAX_HOURS is that pod's own limit (leave empty for none). STOPPED_AT marks a pod that is already
    stopped or terminated: it still counts toward the spend, and it is never told to stop again.
--deadline  epoch seconds or ISO-8601. Every running pod gets STOP once the clock reaches it.
--budget    combined dollars. Spend = --spent + sum over pods of PRICE x hours from START to now (or STOPPED_AT).
            Every running pod gets STOP when the spend would reach the budget before the next check, that is when
            spend + (combined hourly rate) x --poll-min / 60 >= budget.
--now       epoch or ISO, for tests and what-if questions (default: the current time).
--loop-min  repeat every N minutes until interrupted (default: one pass).

Output, one line per pod, then a summary:
    STOP <pod id> name=<name> reason=<deadline|budget|max_hours>
    OK <pod id> name=<name> ...
    SPEND ...
Exit code: 3 if any line is STOP, else 0 (in loop mode: when interrupted). Clock times are printed in US Eastern.
The estimate ignores what RunPod bills between pod creation and START and for disks of stopped pods; add those with
--spent if they matter.
"""
import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    ET = dt.timezone(dt.timedelta(hours=-5), "ET")


@dataclass
class Pod:
    name: str
    id: str
    start: float
    price: float
    max_hours: Optional[float] = None
    stopped_at: Optional[float] = None

    def hours(self, now):
        end = now if self.stopped_at is None else min(self.stopped_at, now)
        return max(0.0, end - self.start) / 3600

    def spend(self, now):
        return self.price * self.hours(now)

    def running(self, now):
        return self.start <= now and (self.stopped_at is None or self.stopped_at > now)


def parse_time(s):
    """Epoch seconds, or ISO-8601 with an explicit offset. A time without an offset is refused, not guessed."""
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("time %r has no UTC offset; write e.g. 2026-11-25T12:30:00-05:00 or epoch seconds" % s)
    return t.timestamp()


def parse_pod(s):
    f = [x.strip() for x in s.split(",")]
    if not 4 <= len(f) <= 6 or not f[0] or not f[1]:
        raise ValueError("--pod wants NAME,ID,START,PRICE[,MAX_HOURS[,STOPPED_AT]], got %r" % s)
    f += [""] * (6 - len(f))
    price = float(f[3])
    if price <= 0:
        raise ValueError("price must be positive in %r" % s)
    return Pod(f[0], f[1], parse_time(f[2]), price, float(f[4]) if f[4] else None, parse_time(f[5]) if f[5] else None)


def et(epoch):
    t = dt.datetime.fromtimestamp(epoch, ET)
    return "%d:%02d %s ET %s" % (t.hour % 12 or 12, t.minute, "am" if t.hour < 12 else "pm", t.strftime("%b %d"))


def decide(pods, now, deadline, budget, spent=0.0, poll_min=5.0):
    """Pure function. Returns (decisions, summary): decisions is a list of (pod, "STOP" | "OK", reason)."""
    total = spent + sum(p.spend(now) for p in pods)
    live = [p for p in pods if p.running(now)]
    rate = sum(p.price for p in live)
    over_deadline = now >= deadline
    over_budget = bool(live) and total + rate * poll_min / 60 >= budget
    out = []
    for p in pods:
        if not p.running(now):
            out.append((p, "OK", "not running"))
        elif over_deadline:
            out.append((p, "STOP", "deadline"))
        elif over_budget:
            out.append((p, "STOP", "budget"))
        elif p.max_hours is not None and p.hours(now) + poll_min / 60 >= p.max_hours:
            out.append((p, "STOP", "max_hours"))
        else:
            out.append((p, "OK", "running"))
    left = budget - total
    summary = dict(total=total, budget=budget, rate=rate, left=left, n_running=len(live),
                   budget_out_at=(now + left / rate * 3600) if rate > 0 and left > 0 else None,
                   deadline=deadline, now=now)
    return out, summary


def report(pods, now, deadline, budget, spent, poll_min, out=None):
    out = sys.stdout if out is None else out
    decisions, s = decide(pods, now, deadline, budget, spent, poll_min)
    for p, verdict, reason in decisions:
        if verdict == "STOP":
            print("STOP %s name=%s reason=%s" % (p.id, p.name, reason), file=out)
        else:
            print("OK %s name=%s %s hours=%.2f spend=$%.2f" % (p.id, p.name, reason.replace(" ", "_"), p.hours(now),
                                                              p.spend(now)), file=out)
    when = et(s["budget_out_at"]) if s["budget_out_at"] else "n/a"
    print("SPEND total=$%.2f of $%.2f, rate=$%.2f/h on %d running, budget lasts until %s, deadline %s, now %s"
          % (s["total"], budget, s["rate"], s["n_running"], when, et(deadline), et(now)), file=out)
    return any(v == "STOP" for _, v, _ in decisions)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Laptop-side hard-stop calculator (prints decisions, calls nothing).")
    ap.add_argument("--pod", action="append", required=True, metavar="NAME,ID,START,PRICE[,MAX_HOURS[,STOPPED_AT]]")
    ap.add_argument("--deadline", required=True)
    ap.add_argument("--budget", type=float, required=True, help="combined dollars")
    ap.add_argument("--spent", type=float, default=0.0, help="dollars already spent outside these pods")
    ap.add_argument("--poll-min", type=float, default=5.0, help="minutes until the next check (budget look-ahead)")
    ap.add_argument("--now", default=None)
    ap.add_argument("--loop-min", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        pods = [parse_pod(x) for x in a.pod]
        deadline = parse_time(a.deadline)
        fixed_now = parse_time(a.now) if a.now else None
    except ValueError as e:
        ap.error(str(e))
    if len({p.id for p in pods}) != len(pods):
        ap.error("duplicate pod id")
    while True:
        stop = report(pods, fixed_now if fixed_now is not None else time.time(), deadline, a.budget, a.spent, a.poll_min)
        sys.stdout.flush()
        if a.loop_min is None:
            return 3 if stop else 0
        time.sleep(a.loop_min * 60)


if __name__ == "__main__":
    sys.exit(main())

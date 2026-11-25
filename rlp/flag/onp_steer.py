"""Tabulate the on-policy steering runs (Stage 4 of rlp/flag/onp.py).

Each condition is a base_rate run named st_<vec> on the same prompts. A hack is a detector hit; the rate is hits over
all completions (invalid and truncated answers count as no hack, and are reported separately). Differences from the
unsteered run are paired by prompt and bootstrapped over prompts within environment.

    .venv/bin/python -m rlp.flag.onp_steer base probe20_neg probe20_pos ...
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("cache/flag/base_rate")
FIT = ["length_pad", "quiz_key", "leak", "hardcode"]          # environments the directions were fitted on (other items)
HELD = ["save_tamper", "fake_pass", "fake_cite"]              # environments no direction saw
ZERO = ["false_report", "ticket_close", "reward_tamper", "runtests"]   # base rate near zero


def load(vec):
    out = {}
    for f in sorted((ROOT / f"st_{vec}").glob("*.labels.jsonl")):
        beh = f.name.split("__")[0]
        out[beh] = {}
        for line in open(f):
            r = json.loads(line)
            out[beh][r["uid"]] = r
    return out


def pooled(d, envs, key="hit"):
    return np.array([float(bool(r.get(key))) for e in envs for _, r in sorted(d[e].items())])


def diff_ci(b, d, envs, key="hit", n=4000, seed=0):
    rng = np.random.default_rng(seed)
    per = []
    for e in envs:
        uids = sorted(set(b[e]) & set(d[e]))
        per.append(np.array([float(bool(d[e][u].get(key))) - float(bool(b[e][u].get(key))) for u in uids]))
    x = np.concatenate(per)
    boots = [np.concatenate([p[rng.integers(len(p), size=len(p))] for p in per]).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(vecs):
    runs = {v: load(v) for v in vecs}
    base = runs[vecs[0]]
    envs = [e for e in FIT + HELD + ZERO if e in base]
    out = {}
    print(f"{'condition':13s} " + " ".join(f"{e[:11]:>11s}" for e in envs))
    for v, d in runs.items():
        cells = []
        for e in envs:
            rs = list(d[e].values())
            cells.append(f"{sum(bool(r['hit']) for r in rs):3d}/{len(rs):2d} v{sum(bool(r['valid']) for r in rs):2d}")
        print(f"{v:13s} " + " ".join(f"{c:>11s}" for c in cells))
    print()
    for v, d in runs.items():
        row = {}
        for name, es in (("fit_envs", FIT), ("held_envs", HELD), ("zero_envs", ZERO)):
            es = [e for e in es if e in d]
            row[name] = dict(rate=float(pooled(d, es).mean()), n=int(len(pooled(d, es))),
                             valid=float(pooled(d, es, "valid").mean()), trunc=float(pooled(d, es, "trunc").mean()))
            if v != vecs[0]:
                row[name]["diff"], row[name]["lo"], row[name]["hi"] = diff_ci(base, d, es)
        for e in FIT + HELD:
            row[e] = dict(rate=float(pooled(d, [e]).mean()))
            if v != vecs[0]:
                row[e]["diff"], row[e]["lo"], row[e]["hi"] = diff_ci(base, d, [e])
        solve_envs = [e for e in ("hardcode", "leak", "fake_pass", "runtests") if e in d]
        row["solves"] = float(pooled(d, solve_envs, "solves").mean())
        row["quiz_correct"] = float(pooled(d, ["quiz_key"], "correct").mean())
        row["tokens"] = float(np.mean([r["tokens"] for e in envs for r in d[e].values()]))
        out[v] = row
        f = lambda k: (f"{100*row[k]['rate']:5.1f}%" + (f" ({100*row[k]['diff']:+5.1f}, {100*row[k]['lo']:+5.1f} to {100*row[k]['hi']:+5.1f})" if "diff" in row[k] else " " * 25))
        print(f"{v:13s} fit {f('fit_envs')}  held {f('held_envs')}  zero {100*row['zero_envs']['rate']:4.1f}%  "
              f"valid {100*row['fit_envs']['valid']:3.0f}/{100*row['held_envs']['valid']:3.0f}%  solves {100*row['solves']:3.0f}%  "
              f"quiz {100*row['quiz_correct']:3.0f}%  tok {row['tokens']:5.0f}")
    json.dump(out, open("cache/flag/onp/steer_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])

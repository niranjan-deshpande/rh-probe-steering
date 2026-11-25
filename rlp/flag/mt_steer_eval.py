"""Steering inside rollouts: hack rate of the steered runs against the unsteered run on the same episodes.

    .venv/bin/python -m rlp.flag.mt_steer_eval [--base m1] [--runs v3_sprobe_sub v3_gf_sub] [--families save_game release_gate]

Episodes are paired on (task, k): the runner derives every turn's seed from the task key, k and the turn number, so a
steered episode starts from the same prompt and seed as its unsteered partner. An episode counts as valid unless it
stopped with "truncated" or "error". Intervals are 95% bootstrap intervals over tasks (2,000 draws). Writes
cache/flag/onp/v3_rollsteer.json.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MT = ROOT / "cache/flag/mt"


def load(run, fam):
    p = MT / run / f"{fam}__hack.jsonl"
    return {(e["task"], e["k"]): e for e in map(json.loads, open(p))} if p.exists() else {}


def boot(rows, f, n=2000, seed=0):
    """rows: list of per-task lists; f maps a flat list of items to a number."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = [x for i in rng.integers(0, len(rows), len(rows)) for x in rows[i]]
        v = f(s)
        if v is not None:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def valid(e):
    return e["info"]["stop"] not in ("truncated", "error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="m1")
    ap.add_argument("--runs", nargs="*", default=["v3_sprobe_sub", "v3_gf_sub"])
    ap.add_argument("--families", nargs="*", default=["save_game", "release_gate"])
    a = ap.parse_args()
    rate = lambda s: (sum(x for x in s) / len(s)) if s else None
    res = {"base": a.base, "runs": {}}
    for run in a.runs:
        R = {"families": {}}
        pooled = {}
        for fam in a.families:
            B, S = load(a.base, fam), load(run, fam)
            keys = sorted(k for k in B if k in S and valid(B[k]) and valid(S[k]))
            tasks = sorted({k[0] for k in keys})
            per = [[(bool(B[k]["info"]["hit"]), bool(S[k]["info"]["hit"])) for k in keys if k[0] == t] for t in tasks]
            pooled.update({t: p for t, p in zip(tasks, per)})
            flat = [x for p in per for x in p]
            d = {"n_base": len(B), "n_steered": len(S), "n_invalid_steered": sum(not valid(e) for e in S.values()),
                 "n_pairs": len(flat), "n_tasks": len(tasks),
                 "base_rate": rate([b for b, _ in flat]), "steered_rate": rate([s for _, s in flat]),
                 "steered_rate_all": rate([bool(e["info"]["hit"]) for e in S.values()]),
                 "stops": {}, "kinds": {}, "solved": rate([bool(e["info"].get("solved")) for e in S.values()]),
                 "base_solved": rate([bool(e["info"].get("solved")) for e in B.values()]),
                 "gen_tokens_median": float(np.median([e["info"]["n_gen_tokens"] for e in S.values()])) if S else None,
                 "turns_median": float(np.median([e["info"]["n_turns"] for e in S.values()])) if S else None}
            for e in S.values():
                d["stops"][e["info"]["stop"]] = d["stops"].get(e["info"]["stop"], 0) + 1
                k = e["info"].get("kind") or "none"
                d["kinds"][k] = d["kinds"].get(k, 0) + 1
            if flat:
                d["diff"] = d["steered_rate"] - d["base_rate"]
                d["diff_ci"] = boot(per, lambda s: rate([y for _, y in s]) - rate([x for x, _ in s]))
            R["families"][fam] = d
        per = list(pooled.values())
        flat = [x for p in per for x in p]
        if flat:
            R["pooled"] = {"n_pairs": len(flat), "n_tasks": len(per), "base_rate": rate([b for b, _ in flat]),
                           "steered_rate": rate([s for _, s in flat]),
                           "diff": rate([s for _, s in flat]) - rate([b for b, _ in flat]),
                           "diff_ci": boot(per, lambda s: rate([y for _, y in s]) - rate([x for x, _ in s]))}
        res["runs"][run] = R
    if len(a.runs) == 2:   # head to head on episodes present and valid in both steered runs
        per = []
        for fam in a.families:
            X, Y = load(a.runs[0], fam), load(a.runs[1], fam)
            keys = sorted(k for k in X if k in Y and valid(X[k]) and valid(Y[k]))
            for t in sorted({k[0] for k in keys}):
                per.append([(bool(X[k]["info"]["hit"]), bool(Y[k]["info"]["hit"])) for k in keys if k[0] == t])
        flat = [x for p in per for x in p]
        if flat:
            res["head_to_head"] = {"runs": a.runs, "n_pairs": len(flat), "n_tasks": len(per),
                                   "diff": rate([x for x, _ in flat]) - rate([y for _, y in flat]),
                                   "diff_ci": boot(per, lambda s: rate([x for x, _ in s]) - rate([y for _, y in s]))}
    out = ROOT / "cache/flag/onp/v3_rollsteer.json"
    out.write_text(json.dumps(res, indent=1))
    pc = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}"
    for run, R in res["runs"].items():
        for fam, d in list(R["families"].items()) + ([("POOLED", R["pooled"])] if "pooled" in R else []):
            ci = d.get("diff_ci") or [None, None]
            print(f"{run:15s} {fam:13s} pairs {d['n_pairs']:3d} tasks {d['n_tasks']:2d}  base {pc(d['base_rate'])}  steered {pc(d['steered_rate'])}"
                  f"  diff {pc(d.get('diff'))} [{pc(ci[0])}, {pc(ci[1])}]" + (f"  stops {d['stops']} kinds {d['kinds']} solved {pc(d['solved'])} (base {pc(d['base_solved'])}) gen med {d['gen_tokens_median']} turns med {d['turns_median']}" if "stops" in d else ""))
    if "head_to_head" in res:
        h = res["head_to_head"]
        print(f"head to head {h['runs'][0]} minus {h['runs'][1]}: {pc(h['diff'])} [{pc(h['diff_ci'][0])}, {pc(h['diff_ci'][1])}] on {h['n_pairs']} pairs, {h['n_tasks']} tasks")
    print("wrote", out)


if __name__ == "__main__":
    main()

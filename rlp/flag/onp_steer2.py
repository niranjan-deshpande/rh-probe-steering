"""Signed steering contrasts against a random-direction null (steering block of 2026-11-24, runs s2_<dir><norm>_{pos,neg}).

For each direction the statistic is the hack rate with the vector added minus the rate with it subtracted, paired by
prompt, averaged with equal weight over environments. It needs no unsteered run, it cancels the part of a perturbation
that raises or lowers hacking whatever its sign, and under the null (a random direction u is as likely as -u) it is
symmetric about zero. A candidate is compared with the spread of the random directions' contrasts:

    z = s_cand / RMS(s_random), referred to a t distribution with R degrees of freedom (one-sided),

and by its rank among {+s_i, -s_i}. The interval printed beside each contrast is a bootstrap over prompts only; it is
not the test.

    .venv/bin/python -m rlp.flag.onp_steer2 --cands sprobe20 lda20 dom20 --rand r220 r320 ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("cache/flag/base_rate")
GROUPS = {"held": ["save_tamper", "fake_pass"], "fit": ["length_pad", "leak", "quiz_key"]}


def load(run):
    out = {}
    for f in sorted((ROOT / run).glob("*.labels.jsonl")):
        out[f.name.split("__")[0]] = {json.loads(l)["uid"]: json.loads(l) for l in open(f)}
    return out


def contrast(pos, neg, envs, key="hit", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    per = []
    for e in envs:
        uids = sorted(set(pos[e]) & set(neg[e]))
        per.append(np.array([float(bool(pos[e][u].get(key))) - float(bool(neg[e][u].get(key))) for u in uids]))
    s = float(np.mean([p.mean() for p in per]))
    bs = [np.mean([p[rng.integers(len(p), size=len(p))].mean() for p in per]) for _ in range(n)]
    return s, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rate(d, envs, key="hit"):
    return float(np.mean([np.mean([bool(r.get(key)) for r in d[e].values()]) for e in envs]))


def main(a):
    if a.groups:
        GROUPS.clear()
        GROUPS.update(json.loads(a.groups))
    runs = {d: (load(f"{a.prefix}{d}_pos"), load(f"{a.prefix}{d}_neg")) for d in a.cands + a.rand}
    out = {}
    for g, envs in GROUPS.items():
        rs = np.array([contrast(*runs[d], envs, n=1)[0] for d in a.rand])
        rms = float(np.sqrt(np.mean(rs ** 2)))
        print(f"\n{g} environments ({', '.join(envs)}): {len(rs)} random directions, contrasts in points "
              f"{np.round(100 * np.sort(rs), 1).tolist()}, RMS {100 * rms:.1f}")
        out[g] = {"random": {d: float(x) for d, x in zip(a.rand, rs)}, "rms": rms}
        null = np.concatenate([rs, -rs])
        for d in a.cands:
            s, lo, hi = contrast(*runs[d], envs)
            z = s / rms
            p = float(stats.t.sf(abs(z), len(rs)))
            rank = int((np.abs(null) >= abs(s)).sum())
            pos, neg = runs[d]
            out[g][d] = dict(contrast=s, lo=lo, hi=hi, z=z, p_one_sided=p, n_null_as_large=rank,
                             rate_pos=rate(pos, envs), rate_neg=rate(neg, envs), valid_pos=rate(pos, envs, "valid"),
                             valid_neg=rate(neg, envs, "valid"),
                             per_env={e: contrast(pos, neg, [e], n=1)[0] for e in envs})
            print(f"  {d:12s} added {100 * rate(pos, envs):5.1f}%  subtracted {100 * rate(neg, envs):5.1f}%  contrast {100 * s:+6.1f} "
                  f"({100 * lo:+.1f} to {100 * hi:+.1f})  z {z:+.2f}  p {p:.3f}  null values as large: {rank}/{len(null)}  "
                  f"valid {100 * rate(pos, envs, 'valid'):.0f}/{100 * rate(neg, envs, 'valid'):.0f}%  "
                  + " ".join(f"{e[:9]} {100 * x:+.0f}" for e, x in out[g][d]["per_env"].items()))
    if a.base:
        # Secondary: each condition against the unsteered run on the same prompts. The null is the 2R random conditions
        # (a random vector added or subtracted is one draw each). Solve rates show whether the model was degraded.
        base = load(f"{a.prefix}{a.base}")
        extra = lambda d, k, envs: float(np.mean([bool(r.get(k)) for e in envs if e in d for r in d[e].values()]))
        for g, envs in GROUPS.items():
            b0 = rate(base, envs)
            nd = np.array([rate(runs[d][i], envs) - b0 for d in a.rand for i in (0, 1)])
            print(f"\n{g} environments, change from the unsteered rate {100 * b0:.1f}% (valid {100 * rate(base, envs, 'valid'):.0f}%): "
                  f"{len(nd)} random conditions range {100 * nd.min():+.1f} to {100 * nd.max():+.1f}, mean {100 * nd.mean():+.1f}, SD {100 * nd.std(ddof=1):.1f}")
            out[g]["base"] = dict(rate=b0, random_diffs=nd.tolist())
            for d in a.cands + a.extra:
                if d not in runs:
                    runs[d] = (load(f"{a.prefix}{d}_pos"), load(f"{a.prefix}{d}_neg"))
                row = {}
                for name, run in zip(("added", "subtracted"), runs[d]):
                    x = rate(run, envs) - b0
                    row[name] = dict(diff=x, n_random_beyond=int((nd >= x).sum() if name == "added" else (nd <= x).sum()),
                                     valid=rate(run, envs, "valid"))
                out[g].setdefault(d, {})["vs_base"] = row
                print(f"  {d:12s} added {100 * row['added']['diff']:+6.1f} (random at least as high: {row['added']['n_random_beyond']}/{len(nd)}, valid {100 * row['added']['valid']:.0f}%)   "
                      f"subtracted {100 * row['subtracted']['diff']:+6.1f} (random at least as low: {row['subtracted']['n_random_beyond']}/{len(nd)}, valid {100 * row['subtracted']['valid']:.0f}%)")
        S = [e for e in ("fake_pass", "leak") if e in base]
        if not S:
            json.dump(out, open(a.out, "w"), indent=1)
            return
        print(f"\nhonest capability: tests solved in {', '.join(S)} / quiz_key correct. unsteered {100 * extra(base, 'solves', S):.0f}% / {100 * extra(base, 'correct', ['quiz_key']):.0f}%")
        rs = [(extra(runs[d][i], "solves", S), extra(runs[d][i], "correct", ["quiz_key"])) for d in a.rand for i in (0, 1)]
        print(f"  random conditions: solves {100 * min(x for x, _ in rs):.0f} to {100 * max(x for x, _ in rs):.0f}%, quiz {100 * min(y for _, y in rs):.0f} to {100 * max(y for _, y in rs):.0f}%")
        out["capability"] = {"base": [extra(base, "solves", S), extra(base, "correct", ["quiz_key"])], "random": rs}
        for d in a.cands + a.extra:
            c = [(extra(runs[d][i], "solves", S), extra(runs[d][i], "correct", ["quiz_key"])) for i in (0, 1)]
            out["capability"][d] = c
            print(f"  {d:12s} added: solves {100 * c[0][0]:.0f}%, quiz {100 * c[0][1]:.0f}%   subtracted: solves {100 * c[1][0]:.0f}%, quiz {100 * c[1][1]:.0f}%")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--rand", nargs="+", required=True)
    ap.add_argument("--base", help="name of the unsteered run on the same prompts")
    ap.add_argument("--extra", nargs="*", default=[], help="further directions for the change-from-base table only")
    ap.add_argument("--groups", help='JSON {group name: [environments]} replacing the default groups')
    ap.add_argument("--prefix", default="s2_")
    ap.add_argument("--out", default="cache/flag/onp/steer2_summary.json")
    main(ap.parse_args())

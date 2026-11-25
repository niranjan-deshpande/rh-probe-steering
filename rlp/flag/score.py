"""Passage-level detection scores for every direction (CPU).   python -m rlp.flag.score --rollouts ib_qwen38

Positives are passages the judge flagged in all votes, negatives are passages no vote flagged; ambiguous passages are
left out. Reported per direction, at the layer chosen in dirs.py, with max-pooling over the passage's tokens (the
primary, as Goodfire's token-level probe is used) and mean-pooling:

    auc          AUROC, hack vs clean passages
    ci           95% bootstrap interval, resampling tasks (a task's three splits stay together)
    auc_len      AUROC after the log token count of the passage is regressed out of the score
    enacted      AUROC with positives restricted to hacks that were attempted or enacted, not only considered
    diff vs gf   paired bootstrap of (auc - auc of Goodfire's pooled direction), same resamples

Random directions give the floor (their 95th percentile). `gf_oracle` is the best of Goodfire's single-contrast and
pooled probes over both harvests and all layers, chosen on these same test passages, so it is an upper bound on what
their selection procedure could reach.
"""
import argparse
import json

import numpy as np

from .. import config
from . import rollouts as RO, synth as S
from .dirs import auroc

OUT = config.CACHE / "flag"
CH = ("reasoning", "content", "tool_calls")
MAIN = ("close:gf_pool", "close:dom_swe", "close:dom_ours", "close:probe_ours", "gd_s0", "gd_s1", "gd_s2", "gd_mean")


def load(name):
    d = OUT / "read" / name
    meta = json.load(open(d / "dirs.json"))
    rs = RO.load(S.GF / name)
    P, y, enacted, task, ntok, chan = [], [], [], [], [], []
    for r in rs:
        f = d / f"{r['trace_id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for i, (mi, ci) in enumerate(zip(z["mi"], z["channel"])):
            p = r["passages"][(int(mi), CH[int(ci)])]
            if p["consensus"] == "ambiguous":
                continue
            P.append(z["proj"][i])
            y.append(p["consensus"] == "positive")
            enacted.append(bool(set(p["statuses"]) & {"enacted", "attempted"}))
            task.append(r["task"])
            ntok.append(int(z["ntok"][i]))
            chan.append(int(ci))
    return meta, np.stack(P), np.array(y), np.array(enacted), np.array(task), np.array(ntok), np.array(chan)


def resid(s, loglen):
    A = np.stack([np.ones_like(loglen), loglen], 1)
    return s - A @ np.linalg.lstsq(A, s, rcond=None)[0]


def boot(scores, y, task, n=1000, seed=0):
    """scores: {name: [N]}. -> {name: [n] AUROCs} with tasks resampled, the same resamples for every name."""
    rng = np.random.default_rng(seed)
    ut = np.unique(task)
    idx_of = {t: np.where(task == t)[0] for t in ut}
    out = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([idx_of[t] for t in rng.choice(ut, len(ut))])
        yy = y[idx]
        if yy.sum() < 2 or (~yy).sum() < 2:
            continue
        for k, s in scores.items():
            ss = s[idx]
            out[k].append(auroc(ss[yy], ss[~yy]))
    return {k: np.array(v) for k, v in out.items()}


def table(meta, P, y, enacted, task, ntok, pool=0, layer=None, names=MAIN, nboot=1000):
    L = layer or meta["layer"]
    li = meta["layers"].index(L)
    nm = meta["names"]
    ll = np.log(np.maximum(ntok, 1))
    rnd = [i for i, k in enumerate(nm) if k.startswith("rnd")]
    res = {"layer": L, "pool": ("max", "mean")[pool], "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
           "n_enacted": int((y & enacted).sum()), "length_auc": auroc(ll[y], ll[~y])}
    ra = np.array([auroc(P[y, li, i, pool], P[~y, li, i, pool]) for i in rnd])
    ral = np.array([auroc(resid(P[:, li, i, pool], ll)[y], resid(P[:, li, i, pool], ll)[~y]) for i in rnd])
    res["random"] = dict(p50=float(np.median(ra)), p95=float(np.percentile(ra, 95)), max=float(ra.max()),
                         len_p95=float(np.percentile(ral, 95)))
    # Goodfire oracle over their probes, harvests, layers and pooling stays within Goodfire's own data
    best = (-1, None)
    for i, k in enumerate(nm):
        if ":gf_" in k:
            for lj, Lj in enumerate(meta["layers"]):
                a = auroc(P[y, lj, i, pool], P[~y, lj, i, pool])
                if a > best[0]:
                    best = (a, (k, Lj, lj, i))
    scores = {k: P[:, li, nm.index(k), pool] for k in names if k in nm}
    if best[1]:
        scores["gf_oracle"] = P[:, best[1][2], best[1][3], pool]
        res["gf_oracle_is"] = [best[1][0], best[1][1]]
    B = boot(scores, y, task, nboot)
    ref = "close:gf_pool"
    res["dirs"] = {}
    for k, s in scores.items():
        e = y & enacted
        r = resid(s, ll)
        d = dict(auc=auroc(s[y], s[~y]), ci=[float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))],
                 auc_len=auroc(r[y], r[~y]), enacted=auroc(s[e], s[~y]) if e.sum() >= 5 else None)
        if ref in B and k != ref:
            df = B[k] - B[ref]
            d["diff_vs_gf"] = [float(df.mean()), float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))]
        res["dirs"][k] = d
    return res


def show(res):
    print(f"layer {res['layer']}, {res['pool']}-pool: {res['n_pos']} hack passages ({res['n_enacted']} enacted) vs "
          f"{res['n_neg']} clean; length alone {res['length_auc']:.3f}; random directions median "
          f"{res['random']['p50']:.3f}, p95 {res['random']['p95']:.3f} (length-controlled p95 {res['random']['len_p95']:.3f})")
    for k, d in res["dirs"].items():
        df = d.get("diff_vs_gf")
        print(f"  {k:18s} auc {d['auc']:.3f} ({d['ci'][0]:.3f}, {d['ci'][1]:.3f})  length-controlled {d['auc_len']:.3f}  "
              f"enacted {d['enacted'] if d['enacted'] is None else round(d['enacted'], 3)}"
              + (f"  vs Goodfire {df[0]:+.3f} ({df[1]:+.3f}, {df[2]:+.3f})" if df else ""))
    if "gf_oracle_is" in res:
        print("  gf_oracle =", res["gf_oracle_is"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="ib_qwen38")
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    meta, P, y, enacted, task, ntok, chan = load(a.rollouts)
    out = {}
    for pool in (0, 1):
        r = table(meta, P, y, enacted, task, ntok, pool, nboot=a.nboot)
        show(r)
        out[r["pool"]] = r
    out["by_layer"] = {L: {k: auroc(P[y, li, meta["names"].index(k), 0], P[~y, li, meta["names"].index(k), 0])
                           for k in MAIN[:4] if k in meta["names"]} for li, L in enumerate(meta["layers"])}
    print("max-pool AUROC by layer:", {L: {k: round(v, 3) for k, v in d.items()} for L, d in out["by_layer"].items()})
    json.dump(out, open(OUT / f"score_{a.rollouts}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Post hoc checks on the rollout detection of 2026-11-25 (written after seeing the main tables; laptop only).

1. Step-level head to head: step AUROC (the step that contains the hack against every step of honest episodes of the
   family) for the logistic probe fitted on all single-turn environments and for gf_pool, with the paired difference,
   bootstrap over tasks, and the equal-weight mean over families.
2. Position baseline: the same step AUROC when the score is only the step's position in its episode (index among the
   assistant messages, and that index divided by the episode's number of assistant messages).
3. The same step AUROC restricted to honest steps at the same or a later position than the earliest hack step.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.mt_posthoc --runs m1 ib_full_oneoff ...
"""
import argparse, json
from pathlib import Path
import numpy as np
from . import mt_select as M


def load_steps(run_dirs):
    eps, SM, rows, off = [], [], [], 0
    for rd in run_dirs:
        z, j = np.load(rd / "acts_36.npz"), json.load(open(rd / "acts_36.json"))
        eps += j["episodes"]; SM.append(z["step_mean"].astype(np.float32))
        n_in_ep = {}
        for s in j["steps"]:
            k = n_in_ep.get(s["ep"], 0); n_in_ep[s["ep"]] = k + 1
            rows.append(dict(ep=s["ep"] + off, pos=k, hack=bool(s["is_hack_step"])))
        for r in rows[len(rows) - len(j["steps"]):]:
            r["n"] = n_in_ep[r["ep"] - off]
        off += len(j["episodes"])
    return eps, np.concatenate(SM), rows


def boot(by, f, n_boot, rng):
    out = []
    for idx in M.boot_idx(len(by), n_boot, rng):
        v = f([by[i] for i in idx])
        if v == v: out.append(v)
    return M.ci(out) if out else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(M.ONP / "night_posthoc.json")); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    b = M.parser().parse_args([])
    st = M.load_st(b.st_feats, b.st_meta, None)
    gf_pool, _, _ = M.load_gf(b.gf_dir, b.gf_readout, b.gf_units)
    W = {"probe": M.fit_probe(st, b.C, not b.no_balance), "gf_pool": gf_pool}
    eps, SM, rows = load_steps([M.MTD / r for r in a.runs])
    res = {}
    for fam in sorted({e["family"] for e in eps}):
        keep = [i for i, r in enumerate(rows) if eps[r["ep"]]["family"] == fam and eps[r["ep"]]["y"] is not None
                and (eps[r["ep"]]["y"] == 0 or r["hack"])]
        if not keep: continue
        S = {k: SM[keep] @ np.asarray(w, np.float32) for k, w in W.items()}
        S["position"] = np.array([rows[i]["pos"] for i in keep], float)
        S["relative position"] = np.array([(rows[i]["pos"] + 1) / rows[i]["n"] for i in keep], float)
        pos = np.array([eps[rows[i]["ep"]]["y"] == 1 for i in keep]); task = np.array([eps[rows[i]["ep"]]["task"] for i in keep])
        p0 = S["position"][pos].min() if pos.any() else 0
        late = pos | (S["position"] >= p0)
        tasks = sorted(set(task)); by = [np.where(task == t)[0] for t in tasks]
        rng = np.random.default_rng(0)
        def au(k, mask=None):
            def f(groups):
                ix = np.concatenate(groups)
                if mask is not None: ix = ix[mask[ix]]
                return M.auroc(S[k][ix][pos[ix]], S[k][ix][~pos[ix]])
            return f
        out = {"n_hack_steps": int(pos.sum()), "n_honest_steps": int((~pos).sum()), "n_tasks": len(tasks), "earliest_hack_position": float(p0),
               "n_honest_steps_late": int((late & ~pos).sum())}
        for k in S:
            out[k] = {"auc": au(k)(by), "ci": boot(by, au(k), a.n_boot, rng), "auc_late": au(k, late)(by), "ci_late": boot(by, au(k, late), a.n_boot, rng)}
        d = lambda g: au("probe")(g) - au("gf_pool")(g)
        out["probe_minus_gf"] = {"diff": d(by), "ci": boot(by, d, a.n_boot, rng)}
        res[fam] = out
        print(f"{fam:13s} hack steps {out['n_hack_steps']:3d} honest steps {out['n_honest_steps']:4d} (late {out['n_honest_steps_late']})  " +
              "  ".join(f"{k} {out[k]['auc']:.3f} [{out[k]['ci'][0]:.3f}, {out[k]['ci'][1]:.3f}] late {out[k]['auc_late']:.3f}" for k in S) +
              f"  probe-gf {out['probe_minus_gf']['diff']:+.3f} [{out['probe_minus_gf']['ci'][0]:+.3f}, {out['probe_minus_gf']['ci'][1]:+.3f}]")
    fams = [f for f in res if res[f]["n_hack_steps"] >= 5]
    # equal weight per family, bootstrap tasks within family
    rng = np.random.default_rng(1); bs = []
    cache = {}
    for fam in fams:
        keep = [i for i, r in enumerate(rows) if eps[r["ep"]]["family"] == fam and eps[r["ep"]]["y"] is not None and (eps[r["ep"]]["y"] == 0 or r["hack"])]
        sp, sg = SM[keep] @ np.asarray(W["probe"], np.float32), SM[keep] @ np.asarray(W["gf_pool"], np.float32)
        pos = np.array([eps[rows[i]["ep"]]["y"] == 1 for i in keep]); task = np.array([eps[rows[i]["ep"]]["task"] for i in keep])
        cache[fam] = (sp, sg, pos, [np.where(task == t)[0] for t in sorted(set(task))])
    def dfam(fam, idx=None):
        sp, sg, pos, by = cache[fam]; ix = np.concatenate(by if idx is None else [by[i] for i in idx])
        return M.auroc(sp[ix][pos[ix]], sp[ix][~pos[ix]]) - M.auroc(sg[ix][pos[ix]], sg[ix][~pos[ix]])
    point = float(np.mean([dfam(f) for f in fams]))
    for _ in range(a.n_boot):
        v = [dfam(f, rng.integers(0, len(cache[f][3]), len(cache[f][3]))) for f in fams]
        if all(x == x for x in v): bs.append(np.mean(v))
    res["overall_probe_minus_gf"] = {"families": fams, "diff": point, "ci": M.ci(bs)}
    print(f"overall (families with at least 5 hack steps: {fams}) probe minus gf_pool, step AUROC: {point:+.3f} [{M.ci(bs)[0]:+.3f}, {M.ci(bs)[1]:+.3f}]")
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()

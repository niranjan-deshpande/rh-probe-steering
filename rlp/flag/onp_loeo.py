"""Does fitting the detector on more environments help on unseen environments and unseen hack categories?

Post hoc analysis on cached layer-36 completion means. It reuses the three held-out environments of
results/05_onpolicy_directions.md as fitting data in some folds, so after this they are no longer untouched.

  loeo       fit on every other environment, test on the one left out
  category   fit on every other hack category, test on all environments of the one left out
  curve      held-out matched AUROC against the number of fitting environments (random subsets of the others)

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.onp_loeo [--feats path.npy --meta path.json]
"""
import argparse
import itertools
import json

import numpy as np

from . import onp

CATEGORY = {"length_pad": "Game", "hardcode": "Game", "fake_cite": "Game", "quiz_key": "Peek", "leak": "Peek",
            "save_tamper": "Tamper", "fake_pass": "Misreport"}


def groups(meta, envs):
    """{group: (hack rows, honest rows)} over every prompt of these environments that has both classes."""
    g = {}
    for i, m in enumerate(meta):
        if m["beh"] in envs and m["y"] is not None:
            g.setdefault(m["group"], ([], []))[1 - m["y"]].append(i)
    return {k: v for k, v in g.items() if v[0] and v[1]}


def fit(meta, X, envs, Sinv=None, C=0.01, balance=False):
    """-> {'probe': w, 'lda': w, 'dom': w} fitted on all both-class prompts of envs (same recipe as onp.closed_form)."""
    from sklearn.linear_model import LogisticRegression
    rows, ys, D, De, re = [], [], [], [], []
    for k, (h, n) in groups(meta, envs).items():
        mu = X[h + n].mean(0)
        rows += [X[i] - mu for i in h + n]
        ys += [1] * len(h) + [0] * len(n)
        re += [k.split("|")[0]] * (len(h) + len(n))
        D.append(X[h].mean(0) - X[n].mean(0))
        De.append(k.split("|")[0])
    R = np.stack(rows)
    sd = R.std(0) + 1e-6
    sw = None
    if balance:                           # every environment gets the same total weight
        cnt = {e: re.count(e) for e in set(re)}
        sw = np.array([len(re) / (len(cnt) * cnt[e]) for e in re])
    out = {"probe": LogisticRegression(C=C, max_iter=2000, fit_intercept=False).fit(R / sd, ys, sample_weight=sw).coef_[0] / sd}
    D = np.stack(D)
    out["dom"] = np.mean([D[[e == b for e in De]].mean(0) for b in sorted(set(De))], 0)
    if Sinv is not None:
        out["lda"] = Sinv @ out["dom"]
    return out


def auc(meta, s, env, n_boot=1000, seed=0):
    per = []
    for h, n in groups(meta, [env]).values():
        d = s[h][:, None] - s[n][None, :]
        per.append(float((d > 0).mean() + 0.5 * (d == 0).mean()))
    per, rng = np.array(per), np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(per)


def main(a):
    meta = json.load(open(a.meta)) if a.meta else onp.load_acts()[0]
    X = np.load(a.feats) if a.feats else np.asarray(onp.load_acts()[1][:, 0, onp.LAYERS.index(36)], np.float32)
    cat = dict(CATEGORY)
    if a.cats:
        cat.update(json.loads(a.cats))
    envs = [e for e in cat if groups(meta, [e])]
    Sinv = onp.sigma_inv(36)[0].numpy() if not a.no_lda else None
    res = {"loeo": {}, "category": {}, "curve": {}}
    S = lambda w: np.asarray(X @ w, np.float64)

    print("leave one environment out (fit on the others)")
    for t in envs:
        W = fit(meta, X, [e for e in envs if e != t], Sinv)
        res["loeo"][t] = {k: auc(meta, S(w), t) for k, w in W.items()}
        print(f"  {t:14s} {cat[t]:10s} n={res['loeo'][t]['probe'][3]:3d}  " + "  ".join(
            f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in res["loeo"][t].items()), flush=True)
    for k in ("probe", "lda", "dom"):
        if k in res["loeo"][envs[0]]:
            print(f"  mean {k}: {np.mean([res['loeo'][t][k][0] for t in envs]):.3f}")

    print("leave one category out (fit on the other categories)")
    for c in sorted(set(cat[e] for e in envs)):
        tr = [e for e in envs if cat[e] != c]
        W = fit(meta, X, tr, Sinv)
        for t in [e for e in envs if cat[e] == c]:
            res["category"][t] = {k: auc(meta, S(w), t) for k, w in W.items()} | {"fit_envs": tr}
            print(f"  {c:10s} {t:14s} fit on {len(tr)} envs  " + "  ".join(
                f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in res["category"][t].items() if k != "fit_envs"), flush=True)
    for k in ("probe", "lda", "dom"):
        if k in res["category"][envs[0]]:
            print(f"  mean {k}: {np.mean([res['category'][t][k][0] for t in envs]):.3f}")

    print("held-out AUROC against number of fitting environments (probe; mean over targets and subsets)")
    rng = np.random.default_rng(0)
    for k in range(1, len(envs)):
        vals = {"probe": [], "lda": [], "dom": []}
        for t in envs:
            others = [e for e in envs if e != t]
            subs = list(itertools.combinations(others, k))
            for i in rng.permutation(len(subs))[:a.subsets]:
                W = fit(meta, X, list(subs[i]), Sinv)
                for name, w in W.items():
                    vals[name].append((t, auc(meta, S(w), t, n_boot=1)[0]))
        res["curve"][k] = {n: float(np.mean([np.mean([x for tt, x in v if tt == t]) for t in envs])) for n, v in vals.items() if v}
        print(f"  k={k}: " + "  ".join(f"{n} {x:.3f}" for n, x in res["curve"][k].items()), flush=True)
    json.dump(res, open(onp.OUT / a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats")
    ap.add_argument("--meta")
    ap.add_argument("--cats", help="JSON {env: category} to add to or override the built-in table")
    ap.add_argument("--subsets", type=int, default=4)
    ap.add_argument("--no-lda", action="store_true")
    ap.add_argument("--out", default="loeo.json")
    main(ap.parse_args())

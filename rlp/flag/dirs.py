"""Read-out directions from harvested synthetic activations (CPU).   python -m rlp.flag.dirs

All are unit vectors per layer, oriented so that a hack projects higher.

    gf_pool      Goodfire's recipe on Goodfire's data: the mean over their individual contrasts (all four families)
                 of the matched difference of means. gf_<contrast> are the single-contrast probes they select among.
    dom_swe      the same recipe on the three SWE training families only (no held-out family, no non-coding data)
    dom_ours     the same recipe on our training set (three SWE families + one non-coding family)
    probe_ours   L2-regularised logistic regression on our training set
    rnd<i>       random directions

A contrast's difference of means is taken within the contrast (positives and negatives come from the same groups and
the same message position), then contrasts are averaged with equal weight, so family identity cannot leak into the
direction through unequal numbers of positives per family.

The layer for everything downstream is chosen here, from training families only: the layer at which dom_ours
separates validation-split groups of the training families best, by d' (difference of class means over pooled sd).
"""
import json

import numpy as np

from .. import config
from . import synth as S

OUT = config.CACHE / "flag"
N_RANDOM = 100


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]                                  # average ranks over ties
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def label_arrays(recs, contrasts):
    """y[c]: per record, 1 / 0 / -1 (not part of contrast c)."""
    return {c: np.array([r["labels"].get(c, -1) for r in recs]) for c in contrasts}


def any_label(recs, contrasts):
    """Per record: 1 if it is the positive of any individual contrast, 0 if it only appears as a negative, -1 else."""
    y = np.full(len(recs), -1)
    for i, r in enumerate(recs):
        v = [r["labels"][c] for c in contrasts if c in r["labels"]]
        if v:
            y[i] = int(max(v) == 1)
    return y


def dom(X, recs, contrasts, mask):
    """Mean over contrasts of (mean positives - mean negatives). X: [N, d]. -> ([d], per-contrast {name: [d]})"""
    Y = label_arrays(recs, contrasts)
    per = {}
    for c in contrasts:
        p, n = mask & (Y[c] == 1), mask & (Y[c] == 0)
        if p.sum() >= 5 and n.sum() >= 5:
            per[c] = X[p].mean(0) - X[n].mean(0)
    return np.mean(list(per.values()), 0), per


def logreg(X, y, w, C=1.0, iters=200):
    """Weighted L2 logistic regression on standardised features, by L-BFGS. -> weight vector in the ORIGINAL space
    (so that x @ w ranks like the fitted model) and the bias."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float64)
    yt, wt = torch.tensor(y, dtype=torch.float64), torch.tensor(w / w.sum(), dtype=torch.float64)
    beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, b0], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ beta + b0
        loss = (wt * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).sum() \
            + beta.pow(2).sum() / (2 * C * len(y))
        loss.backward()
        return loss
    opt.step(closure)
    wv = beta.detach().numpy() / sd
    return wv, float(b0.detach()) - float(mu @ wv)


def balanced_weights(recs, y, mask):
    """Each (family, label) cell gets the same total weight."""
    w = np.zeros(len(recs))
    fam = np.array([r["family"] for r in recs])
    cells = [(f, v) for f in sorted(set(fam[mask])) for v in (0, 1)]
    for f, v in cells:
        m = mask & (fam == f) & (y == v)
        if m.sum():
            w[m] = 1.0 / m.sum()
    return w


def fit_probe(X, recs, y, mask, grid=(1e-3, 1e-2, 1e-1, 1.0), folds=4, seed=0):
    """Pick C by grouped cross-validation (mean per-family AUROC), refit on everything in mask."""
    groups = np.array([r["group"] for r in recs])
    fam = np.array([r["family"] for r in recs])
    ug = np.array(sorted(set(groups[mask])))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % folds))
    f = np.array([fold_of.get(g, -1) for g in groups])
    score = {}
    for C in grid:
        s = []
        for k in range(folds):
            tr, te = mask & (f != k) & (y >= 0), mask & (f == k) & (y >= 0)
            wv, _ = logreg(X[tr], y[tr], balanced_weights(recs, y, tr)[tr], C)
            for fm in sorted(set(fam[te])):
                m = te & (fam == fm)
                s.append(auroc(X[m & (y == 1)] @ wv, X[m & (y == 0)] @ wv))
        score[C] = float(np.nanmean(s))
    best = max(score, key=score.get)
    m = mask & (y >= 0)
    wv, b = logreg(X[m], y[m], balanced_weights(recs, y, m)[m], best)
    return wv, b, best, score


def dprime(p, n):
    return float((p.mean() - n.mean()) / (np.sqrt((p.var() + n.var()) / 2) + 1e-12))


def build(acts, recs, layers, harvest="close", log=print):
    """-> (dirs {name: [n layers, d]}, info). acts: {'close'|'turn': [N, nL, d]}."""
    C = S.individual(S.load_contrasts())
    fam = np.array([r["family"] for r in recs])
    split = np.array([r["split"] for r in recs])
    tr = split == "train"
    sets = {"gf_pool": [c for c, v in C.items() if v["dataset"] in S.GOODFIRE],
            "dom_swe": [c for c, v in C.items() if v["dataset"] in S.SWE_TRAIN],
            "dom_ours": [c for c, v in C.items() if v["dataset"] in S.TRAIN]}
    y_ours = any_label(recs, sets["dom_ours"])
    m_ours = tr & np.isin(fam, S.TRAIN)
    dirs, info = {}, {"probe_C": {}, "probe_cv": {}, "layer_dprime": {}}
    nL = len(layers)
    d = acts[harvest].shape[-1]
    for k in list(sets) + ["probe_ours"] + [f"gf_{c}" for c in sets["gf_pool"]]:
        dirs[k] = np.zeros((nL, d), np.float32)
    for li, L in enumerate(layers):
        X = acts[harvest][:, li].astype(np.float32)
        for name, cs in sets.items():
            v, per = dom(X, recs, cs, tr & np.isin(fam, [C[c]["dataset"] for c in cs]))
            dirs[name][li] = unit(v)
            if name == "gf_pool":
                for c, pv in per.items():
                    dirs[f"gf_{c}"][li] = unit(pv)
        wv, b, bestC, cv = fit_probe(X, recs, y_ours, m_ours)
        dirs["probe_ours"][li] = unit(wv)
        info["probe_C"][L], info["probe_cv"][L] = bestC, cv
        va = (split == "validation") & np.isin(fam, S.TRAIN)
        s = X @ dirs["dom_ours"][li]
        info["layer_dprime"][L] = dprime(s[va & (y_ours == 1)], s[va & (y_ours == 0)])
        log(f"[dirs] layer {L}: probe C {bestC}, dom_ours validation d' {info['layer_dprime'][L]:.2f}", flush=True)
    info["layer"] = int(max(info["layer_dprime"], key=info["layer_dprime"].get))
    rng = np.random.default_rng(0)
    for i in range(N_RANDOM):
        dirs[f"rnd{i}"] = unit(rng.standard_normal((nL, d))).astype(np.float32)
    return dirs, info


def synth_table(acts, recs, layers, dirs, harvest="close"):
    """AUROC of each direction on validation-split groups, per family (hack positives vs their matched negatives)."""
    C = S.individual(S.load_contrasts())
    fam = np.array([r["family"] for r in recs])
    va = np.array([r["split"] == "validation" for r in recs])
    out = {}
    for f in sorted(set(fam)):
        y = any_label(recs, [c for c, v in C.items() if v["dataset"] == f])
        m = va & (fam == f)
        out[f] = {}
        for li, L in enumerate(layers):
            X = acts[harvest][:, li].astype(np.float32)
            out[f][L] = {k: auroc(X[m & (y == 1)] @ v[li], X[m & (y == 0)] @ v[li]) for k, v in dirs.items()}
    return out


def main():
    z = np.load(OUT / "synth_acts.npz")
    recs = json.load(open(OUT / "synth_meta.json"))
    layers = [int(x) for x in z["layers"]]
    ok = z["ok"]
    recs = [r for r, o in zip(recs, ok) if o]
    hs = [h for h in ("close", "turn", "reason") if h in z.files and np.abs(z[h][ok][:50]).sum() > 0]
    acts = {h: z[h][ok] for h in hs}
    res = {}
    for h in hs:
        dirs, info = build(acts, recs, layers, h)
        np.savez(OUT / f"dirs_{h}.npz", **dirs)
        tab = synth_table(acts, recs, layers, dirs, h)
        res[h] = dict(info=info, synth=tab)
        L = info["layer"]
        print(f"== harvest {h}: chosen layer {L}")
        for f, t in tab.items():
            rn = [t[L][k] for k in t[L] if k.startswith("rnd")]
            print(f"  {f:30s}", {k: round(t[L][k], 3) for k in ("gf_pool", "dom_swe", "dom_ours", "probe_ours")},
                  "random p95", round(float(np.percentile(rn, 95)), 3))
    json.dump(res, open(OUT / "dirs_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()

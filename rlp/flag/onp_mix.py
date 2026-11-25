"""Which mix of fitting environments gives the probe that detects hacks best on environments it has not seen?

Post hoc, on cached layer-36 completion means (the seven environments of results/05 plus three new ones; log_scrub
is left out because it has 12 hacks). A logistic probe (the recipe of onp.closed_form, every environment weighted
equally) is fitted on every set of 1 to 4 environments and scored on every environment outside the set. The question
is whether, at a fixed number of fitting environments, sets that span several hack categories do better than sets
concentrated in one or two.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.onp_mix
"""
import itertools
import json

import numpy as np

from . import onp, onp_loeo as LO, onp_more as M

ENVS = ["length_pad", "hardcode", "fake_cite", "quiz_key", "leak", "save_tamper", "ckpt_swap", "thresh_tamper", "fake_pass", "ref_copy"]


def main(kmax=4):
    meta = json.load(open(onp.OUT / "meta_all.json"))
    X = np.load(onp.OUT / "means_36_all.npy")
    cat = M.CATS
    G = {e: list(LO.groups(meta, [e]).values()) for e in ENVS}

    def auc(s, e):
        return float(np.mean([((s[h][:, None] - s[n][None, :]) > 0).mean() for h, n in G[e]]))

    rows = []
    for k in range(1, kmax + 1):
        for S in itertools.combinations(ENVS, k):
            w = LO.fit(meta, X, list(S), balance=True)["probe"]
            s = np.asarray(X @ w, np.float64)
            for t in ENVS:
                if t not in S:
                    rows.append(dict(k=k, S=list(S), t=t, auc=auc(s, t), ncat=len({cat[e] for e in S}),
                                     same=cat[t] in {cat[e] for e in S}))
        print(f"k={k} done ({len(rows)} set-target pairs)", flush=True)
    json.dump(rows, open(onp.OUT / "mix.json", "w"))
    report(rows)


def tmean(rows):
    """Mean over targets of the mean over sets, so that easy targets do not dominate."""
    by = {}
    for r in rows:
        by.setdefault(r["t"], []).append(r["auc"])
    return (float(np.mean([np.mean(v) for v in by.values()])), len(rows), len(by)) if by else (float("nan"), 0, 0)


def report(rows=None):
    rows = rows or json.load(open(onp.OUT / "mix.json"))
    print("\nmean held-out matched AUROC (mean over targets of the mean over fitting sets); n = set-target pairs / targets")
    for label, f in (("all held-out environments", lambda r: True),
                     ("target's category NOT in the fitting set", lambda r: not r["same"]),
                     ("target's category in the fitting set", lambda r: r["same"])):
        print(f"\n{label}")
        for k in sorted({r["k"] for r in rows}):
            cells = []
            for nc in range(1, k + 1):
                m, n, nt = tmean([r for r in rows if r["k"] == k and r["ncat"] == nc and f(r)])
                if n:
                    cells.append(f"{nc} categories: {m:.3f} (n={n}/{nt})")
            print(f"  {k} fitting environments   " + "   ".join(cells))
    # the same contrast with the targets held fixed: for each target, diverse minus concentrated sets of equal size
    print("\nper target, sets of 3 and 4 environments none of which shares the target's category: most diverse vs least diverse")
    for k in (3, 4):
        d = []
        for t in ENVS:
            rs = [r for r in rows if r["k"] == k and r["t"] == t and not r["same"]]
            hi = [r["auc"] for r in rs if r["ncat"] == max(x["ncat"] for x in rs)]
            lo = [r["auc"] for r in rs if r["ncat"] == min(x["ncat"] for x in rs)]
            d.append((t, float(np.mean(hi)), float(np.mean(lo)), max(x["ncat"] for x in rs), min(x["ncat"] for x in rs)))
        print(f"  k={k}: " + "  ".join(f"{t[:9]} {a:.2f}/{b:.2f}" for t, a, b, _, _ in d)
              + f"   mean difference {np.mean([a - b for _, a, b, _, _ in d]):+.3f} (targets where diverse is better: {sum(a > b for _, a, b, _, _ in d)}/{len(d)})")
    print("\nfitting sets of 4 ranked by mean AUROC over the six environments left out (selection on the same data; descriptive only)")
    by = {}
    for r in rows:
        if r["k"] == 4:
            by.setdefault(tuple(r["S"]), []).append(r["auc"])
    rk = sorted(by.items(), key=lambda x: -np.mean(x[1]))
    for S, v in rk[:5] + rk[-3:]:
        print(f"  {np.mean(v):.3f} (min {min(v):.2f})  {', '.join(S)}  [{len({M.CATS[e] for e in S})} categories]")
    print("  by number of categories among sets of 4: " + "  ".join(
        f"{nc}: mean {np.mean([np.mean(v) for S, v in by.items() if len({M.CATS[e] for e in S}) == nc]):.3f}, "
        f"worst-target {np.mean([min(v) for S, v in by.items() if len({M.CATS[e] for e in S}) == nc]):.3f} "
        f"({sum(len({M.CATS[e] for e in S}) == nc for S in by)} sets)" for nc in (2, 3, 4)))
    one = {r["S"][0]: [] for r in rows if r["k"] == 1}
    for r in rows:
        if r["k"] == 1:
            one[r["S"][0]].append(r["auc"])
    print("\nsingle fitting environment, mean AUROC over the other nine: " + "  ".join(f"{e} {np.mean(v):.2f}" for e, v in sorted(one.items(), key=lambda x: -np.mean(x[1]))))


if __name__ == "__main__":
    main()

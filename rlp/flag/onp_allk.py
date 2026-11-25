"""Post hoc tables of 2026-11-24 (evening): (a) the fitted detector w, steering vector v and other directions as
detectors on every environment; (b) every fitting set of 1 to 9 environments out of ten, scored on the rest.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.onp_allk
"""
import itertools, json
from collections import defaultdict
import numpy as np, torch
from . import onp, onp_loeo as LO, onp_more as M, onp_mix as MX

O = onp.OUT


def detect(meta, X):
    d = torch.load(O / "fitw_w2.pt", map_location="cpu")
    z, c = np.load(O / "dirs_cf_36.npz"), np.load(O / "write_cands_36.npz")
    dirs = {"probe": z["probe"], "fitted w": d["w"].float().numpy(), "fitted v": d["v"].float().numpy(),
            "S.probe": c["Stok_probe"], "lda": z["lda"], "dom": z["dom"]}
    envs = ["save_tamper", "fake_pass", "fake_cite", "ref_copy", "ckpt_swap", "thresh_tamper", "log_scrub", "length_pad", "leak", "quiz_key", "hardcode"]
    res = {n: {e: LO.auc(meta, np.asarray(X @ u, np.float64), e)[:3] for e in envs} for n, u in dirs.items()}
    r = np.array([[LO.auc(meta, np.asarray(X @ z[f"rnd{i}"], np.float64), e, n_boot=1)[0] for e in envs] for i in range(20)])
    res["random"] = {e: [float(r[:, j].min()), float(r[:, j].max())] for j, e in enumerate(envs)}
    json.dump(res, open(O / "fitted_detect_all.json", "w"), indent=1)
    for n in dirs:
        print(n.ljust(9), " ".join(f"{e[:8]} {res[n][e][0]:.2f}" for e in envs), flush=True)


def allk(meta, X):
    G = {e: list(LO.groups(meta, [e]).values()) for e in MX.ENVS}
    auc = lambda s, e: float(np.mean([((s[h][:, None] - s[n][None, :]) > 0).mean() for h, n in G[e]]))
    rows = []
    for k in range(1, 10):
        for S in itertools.combinations(MX.ENVS, k):
            s = np.asarray(X @ LO.fit(meta, X, list(S), balance=True)["probe"], np.float64)
            rows += [dict(k=k, S=list(S), t=t, auc=auc(s, t)) for t in MX.ENVS if t not in S]
    json.dump(rows, open(O / "mix_all_k.json", "w"))
    by = defaultdict(list)
    for r in rows:
        by[(r["k"], tuple(r["S"]))].append(r["auc"])
    out = {}
    for k in range(1, 10):
        bt = defaultdict(list)
        for r in rows:
            if r["k"] == k:
                bt[r["t"]].append(r["auc"])
        sm = [float(np.mean(v)) for (kk, _), v in by.items() if kk == k]
        out[k] = dict(tmean=float(np.mean([np.mean(v) for v in bt.values()])), sets=[round(x, 3) for x in sm],
                      worst=float(np.mean([min(v) for (kk, _), v in by.items() if kk == k])),
                      per_target={t: float(np.mean(v)) for t, v in bt.items()})
        best = max(((np.mean(v), min(v), S) for (kk, S), v in by.items() if kk == k), key=lambda x: x[0])
        print(f"k={k} mean {out[k]['tmean']:.3f}  best {best[0]:.3f} (min {best[1]:.2f}) {', '.join(best[2])}", flush=True)
    eff = {}
    for e in MX.ENVS:
        ds, dx = [], []
        for k in range(2, 7):
            for t in MX.ENVS:
                if t != e:
                    a = np.mean([r["auc"] for r in rows if r["k"] == k and r["t"] == t and e in r["S"]])
                    b = np.mean([r["auc"] for r in rows if r["k"] == k and r["t"] == t and e not in r["S"]])
                    ds.append(a - b)
                    if M.CATS[t] != M.CATS[e]:
                        dx.append(a - b)
        eff[e] = (float(np.mean(ds)), float(np.mean(dx)))
    out["effect"] = eff
    print("effect of including:", {e: round(v[0], 3) for e, v in sorted(eff.items(), key=lambda x: -x[1][0])})
    json.dump(out, open(O / "mix_all_k_summary.json", "w"))


if __name__ == "__main__":
    meta = json.load(open(O / "meta_all.json"))
    X = np.load(O / "means_36_all.npy")
    detect(meta, X)
    allk(meta, X)
    print("ALLK_DONE")

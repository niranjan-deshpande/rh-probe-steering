"""More environments for the detector (2026-11-24): the four new environments sampled on-policy (h36_env2) added to
the seven of results/05.

  a  the original probe (fitted on four environments) scored on the new environments, which it never saw
  b  for each hack category: a probe fitted on the four original fitting environments against a probe fitted on every
     environment outside that category, both scored on the category's environments
  c  pooling: the original directions scored per token and pooled by mean, max, top 5% and best 64-token window

It also writes the combined features and meta for rlp.flag.onp_loeo (leave one environment / category out, and the
curve against the number of fitting environments).

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.onp_more
"""
import json

import numpy as np

from . import onp, onp_loeo as LO

CATS = dict(LO.CATEGORY) | {"ckpt_swap": "Tamper", "thresh_tamper": "Tamper", "log_scrub": "Tamper", "ref_copy": "Plagiarise"}
ORIG_FIT = list(onp.FIT)


def main():
    meta0, means = onp.load_acts()
    X0 = np.asarray(means[:, 0, onp.LAYERS.index(36)], np.float32)
    z = np.load(onp.OUT / "h36_env2.npz")
    meta1 = json.load(open(onp.OUT / "h36_env2.json"))
    meta, X = meta0 + meta1, np.concatenate([X0, z["X"]])
    np.save(onp.OUT / "means_36_all.npy", X)
    json.dump(meta, open(onp.OUT / "meta_all.json", "w"))
    envs = [e for e in CATS if LO.groups(meta, [e])]
    new = [e for e in ("ckpt_swap", "thresh_tamper", "log_scrub", "ref_copy") if e in envs]
    cf = np.load(onp.OUT / "dirs_cf_36.npz")
    S = lambda w: np.asarray(X @ w, np.float64)
    res = {"counts": {}, "a": {}, "b": {}, "c": {}}
    for e in envs:
        idx = [i for i, m in enumerate(meta) if m["beh"] == e and m["y"] is not None]
        res["counts"][e] = dict(hack=int(sum(meta[i]["y"] for i in idx)), honest=int(sum(1 - meta[i]["y"] for i in idx)),
                                prompts_both=len(LO.groups(meta, [e])))
    print("counts (hack / honest / prompts with both):", {e: tuple(c.values()) for e, c in res["counts"].items() if e in new})

    print("\na. directions of results/05 (fitted on length_pad, quiz_key, leak, hardcode) on the new environments")
    for e in new:
        res["a"][e] = {k: LO.auc(meta, S(cf[k]), e) for k in ("probe", "lda", "dom")}
        print(f"  {e:14s} {CATS[e]:10s} n={res['a'][e]['probe'][3]:3d}  " + "  ".join(
            f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in res["a"][e].items()))

    print("\nb. held-out category: probe fitted on the four original environments vs on all environments of other categories")
    for c in sorted(set(CATS[e] for e in envs)):
        tr_all = [e for e in envs if CATS[e] != c]
        tr_orig = [e for e in ORIG_FIT if CATS[e] != c]
        Wa, Wo = LO.fit(meta, X, tr_all)["probe"], LO.fit(meta, X, tr_orig)["probe"]
        for t in [e for e in envs if CATS[e] == c]:
            a, o = LO.auc(meta, S(Wa), t), LO.auc(meta, S(Wo), t)
            res["b"][t] = dict(category=c, all_other=a, orig_other=o, n_all=len(tr_all), n_orig=len(tr_orig))
            print(f"  {c:10s} {t:14s} fit on {len(tr_orig)} original envs: {o[0]:.2f} [{o[1]:.2f}, {o[2]:.2f}]   "
                  f"fit on {len(tr_all)} envs: {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]")
    for k in ("orig_other", "all_other"):
        print(f"  mean over environments, {k}: {np.mean([r[k][0] for r in res['b'].values()]):.3f}")

    print("\nc. pooling of per-token scores (original directions), matched AUROC")
    for tag in ("onp_held", "env2"):
        f = onp.OUT / f"h36_{tag}.npz"
        if not f.exists():
            continue
        zz, mm = np.load(f), json.load(open(onp.OUT / f"h36_{tag}.json"))
        dirs, pools = list(zz["dirs"]), list(zz["pools"])
        for e in sorted({m["beh"] for m in mm}):
            for di, d in enumerate(dirs):
                if d == "dom":
                    continue
                r = {p: LO.auc(mm, zz["pooled"][:, di, pi].astype(np.float64), e, n_boot=200)[0] for pi, p in enumerate(pools)}
                res["c"][f"{e}/{d}"] = r
                print(f"  {e:14s} {d:6s} " + "  ".join(f"{p} {x:.2f}" for p, x in r.items()))
    json.dump(res, open(onp.OUT / "more_envs.json", "w"), indent=1)


if __name__ == "__main__":
    main()

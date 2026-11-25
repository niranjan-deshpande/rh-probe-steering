"""Choosing fitting environments on validation environments and reporting on test environments (post hoc, 2026-11-24).

Ten environments are split into a fitting pool F (6), validation V (2) and test T (2), over all 1,260 such splits. For
each split the fitting set S (any non-empty subset of F) with the best mean matched AUROC on V is chosen and scored on
T. Baselines on the same T: all of F; a random subset of F of the chosen size; the subset that is best on T itself
(an upper bound that no procedure can use). Two recipes: the logistic probe (ours) and the plain mean difference (the
recipe Goodfire uses, here applied to our on-policy environments). Probe scores come from mix_all_k.json.

    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.onp_select
"""
import itertools, json
import numpy as np
from . import onp, onp_loeo as LO, onp_mix as MX, onp_more as M

O = onp.OUT


def tables():
    rows = json.load(open(O / "mix_all_k.json"))
    P = {(frozenset(r["S"]), r["t"]): r["auc"] for r in rows}
    meta = json.load(open(O / "meta_all.json")); X = np.load(O / "means_36_all.npy")
    G = {e: list(LO.groups(meta, [e]).values()) for e in MX.ENVS}
    De = {e: np.mean([X[h].mean(0) - X[n].mean(0) for h, n in G[e]], 0) for e in MX.ENVS}
    auc = lambda s, e: float(np.mean([((s[h][:, None] - s[n][None, :]) > 0).mean() for h, n in G[e]]))
    Dm = {}
    for k in range(1, 7):
        for S in itertools.combinations(MX.ENVS, k):
            s = np.asarray(X @ np.mean([De[e] for e in S], 0), np.float64)
            for t in MX.ENVS:
                if t not in S:
                    Dm[(frozenset(S), t)] = auc(s, t)
    return {"logistic probe": P, "mean difference": Dm}


def run(A, by_cat=False, seed=0):
    rng = np.random.default_rng(seed)
    E = MX.ENVS
    out = []
    for T in itertools.combinations(E, 2):
        rest = [e for e in E if e not in T]
        for V in itertools.combinations(rest, 2):
            F = [e for e in rest if e not in V]
            if by_cat:                      # test categories never appear in fitting or validation
                tc = {M.CATS[t] for t in T}
                F = [e for e in F if M.CATS[e] not in tc]
                if any(M.CATS[v] in tc for v in V) or len(F) < 2:
                    continue
            subs = [frozenset(S) for k in range(1, len(F) + 1) for S in itertools.combinations(F, k)]
            val = np.array([np.mean([A[(S, v)] for v in V]) for S in subs])
            tst = np.array([np.mean([A[(S, t)] for t in T]) for S in subs])
            i = int(val.argmax())
            same = [j for j, S in enumerate(subs) if len(S) == len(subs[i])]
            out.append(dict(chosen=tst[i], allF=tst[-1], rand=tst[rng.choice(same)], oracle=tst.max(), size=len(subs[i]),
                            val=val[i], single=np.mean([tst[j] for j, S in enumerate(subs) if len(S) == 1])))
    return out


if __name__ == "__main__":
    T = tables()
    res = {}
    for by_cat in (False, True):
        print("\n" + ("test CATEGORIES held out of fitting and validation" if by_cat else "test ENVIRONMENTS held out") )
        for name, A in T.items():
            o = run(A, by_cat)
            m = lambda k: float(np.mean([x[k] for x in o]))
            d = np.array([x["chosen"] - x["allF"] for x in o])
            res[f"{name}|{by_cat}"] = {k: m(k) for k in ("chosen", "allF", "rand", "oracle", "single", "size", "val")} | {"n": len(o), "win": float((d > 0).mean())}
            print(f"  {name:16s} splits {len(o):4d}  chosen on validation {m('chosen'):.3f}  all of the pool {m('allF'):.3f}  random same size {m('rand'):.3f}  "
                  f"one environment {m('single'):.3f}  best on test (bound) {m('oracle'):.3f}  mean size chosen {m('size'):.1f}  "
                  f"validation score of chosen {m('val'):.3f}  chosen beats all-of-pool in {100 * (d > 0).mean():.0f}% (mean diff {d.mean():+.3f})")
    json.dump(res, open(O / "select_nested.json", "w"), indent=1)

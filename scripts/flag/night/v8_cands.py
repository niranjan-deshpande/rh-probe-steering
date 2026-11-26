"""Laptop, 26 November: candidate steering vectors that differ only in what the logistic probe was fitted on.

    .venv/bin/python scripts/flag/night/v8_cands.py     ->  cache/flag/night/v8_cands.npz + v8_cands.json

Every candidate is (token covariance of cov_36.pt) times a logistic probe (rlp.flag.mt_select.fit_probe, C 0.01,
environments weighted equally), normalised. A is the vector already in the steering table (write_cands_36.npz).
  A  the probe of 24 November (length_pad, quiz_key, leak, hardcode)          [existing vector, not refitted]
  B  all single-turn environments
  C  the single-turn environments whose hack is tampering with a file (category "Tamper" of mt_select.CATS)
  D  the subset of single-turn environments with the best matched AUROC on rollout families OTHER than save_game and
     release_gate (mean over the usable ones among leak_outputs, answer_key, scorer_copy, ib_lcb)
  E  Goodfire's contrast pairs (each contrast one environment)
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, ".")
from rlp.flag import mt_select as M

O, MT = Path("cache/flag/onp"), Path("cache/flag/mt")
S = torch.load(O / "cov_36.pt", map_location="cpu")["cov"].double().numpy()
unit = lambda v: v / np.linalg.norm(v)
write = lambda w: unit(S @ np.asarray(w, np.float64))

st = M.load_st(O / "means_36_all.npy", O / "meta_all.json")
names = [e.name for e in st]
print("single-turn environments:", [(e.name, e.n_tasks, len(e.y)) for e in st])
A = unit(np.load(O / "write_cands_36.npz")["Stok_probe"].astype(np.float64))
four = [e for e in st if e.name in ("length_pad", "quiz_key", "leak", "hardcode")]
print("check: cov x old probe vs A", float(write(np.load(O / "dirs_cf_36.npz")["probe"]) @ A),
      "| cov x refit on the four vs A", float(write(M.fit_probe(four)) @ A))

runs = [str(p.parent) for p in sorted(MT.glob("*/acts_36.npz"))]
fams, _, counts = M.load_mt(runs)
val = [f for f in fams if f not in ("save_game", "release_gate") and fams[f].n_tasks >= 2]
print("rollout runs with activations:", runs, "| validation families for D:", [(f, fams[f].n_tasks) for f in val])

cands, info = {"A": A}, {"A": dict(envs=["length_pad", "quiz_key", "leak", "hardcode"], note="existing vector")}
cands["B"], info["B"] = write(M.fit_probe(st)), dict(envs=names)
tam = [e for e in st if M.CATS.get(e.name) == "Tamper"]
cands["C"], info["C"] = write(M.fit_probe(tam)), dict(envs=[e.name for e in tam])

best, table = None, []
for r in range(1, len(st) + 1):
    for sub in itertools.combinations(st, r):
        w = M.fit_probe(list(sub))
        per = {f: float(np.mean(fams[f].per_task(w))) for f in val}
        sc = float(np.mean(list(per.values())))
        table.append((sc, [e.name for e in sub], per))
        if best is None or sc > best[0]:
            best = (sc, list(sub), per)
table.sort(key=lambda t: -t[0])
print("D: best subsets by mean validation AUROC:")
for t in table[:5]:
    print("   ", round(t[0], 3), t[1], {k: round(v, 3) for k, v in t[2].items()})
whole = [t for t in table if len(t[1]) == len(st)][0]
print("   whole pool:", round(whole[0], 3), "| the four of A:", [round(t[0], 3) for t in table if sorted(t[1]) == sorted(info["A"]["envs"])])
cands["D"], info["D"] = write(M.fit_probe(best[1])), dict(envs=[e.name for e in best[1]], val_auroc=best[0], per_family=best[2], val_families=val)

_, gf, note = M.load_gf(Path("cache/flag/gf27"))
print("Goodfire contrasts:", len(gf), note)
cands["E"], info["E"] = write(M.fit_probe(gf)), dict(envs=[e.name for e in gf])

k = list(cands)
cos = {a: {b: round(float(cands[a] @ cands[b]), 3) for b in k} for a in k}
sd = {a: round(float(np.sqrt(cands[a] @ S @ cands[a])), 3) for a in k}
print("cosines:"); [print("   ", a, cos[a]) for a in k]
print("sd of activations along each:", sd)
np.savez("cache/flag/night/v8_cands.npz", **{a: cands[a].astype(np.float32) for a in k})
json.dump(dict(info=info, cos=cos, sd_along=sd, d_top5=[(t[0], t[1]) for t in table[:5]]), open("cache/flag/night/v8_cands.json", "w"), indent=1)

"""Steering files for the transformed-detector test (2026-11-24): v = r * unit(Sigma w) for the logistic probe, with
Sigma the token covariance at the layer, and eight more random directions (rnd2..rnd9) for the null.

    python scripts/flag/onp_mkvec2.py 36 20
"""
import sys
import numpy as np, torch
from rlp.flag import onp
L, r = int(sys.argv[1]), float(sys.argv[2])
cf = np.load(onp.OUT / f"dirs_cf_{L}.npz"); D = onp.OUT / "steer"; D.mkdir(exist_ok=True)
cov = torch.load(onp.OUT / f"cov_{L}.pt")["cov"].double()
vecs = {"sprobe": cov @ torch.tensor(cf["probe"]).double()} | {f"r{j}": torch.tensor(cf[f"rnd{j}"]).double() for j in range(2, 10)}
for k, u in vecs.items():
    u = (u / u.norm()).float()
    for s, sg in (("pos", 1.0), ("neg", -1.0)):
        torch.save(dict(v=sg * r * u, layer=L), D / f"{k}{int(r)}_{s}.pt")
torch.save(dict(v=torch.zeros(len(u)), layer=L), D / "base.pt")
print(sorted(p.name for p in D.glob("*.pt")))

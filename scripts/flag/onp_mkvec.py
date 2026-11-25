import numpy as np, torch, sys
from rlp.flag import onp
L, r = int(sys.argv[1]), float(sys.argv[2])
cf = np.load(onp.OUT / f"dirs_cf_{L}.npz"); D = onp.OUT / "steer"; D.mkdir(exist_ok=True)
for k, src in (("probe", "probe"), ("lda", "lda"), ("dom", "dom"), ("rnd", "rnd0")):
    u = torch.tensor(cf[src]).float(); u = u / u.norm()
    for s, sg in (("pos", 1.0), ("neg", -1.0)):
        torch.save(dict(v=sg * r * u, layer=L), D / f"{k}{int(r)}_{s}.pt")
torch.save(dict(v=torch.zeros(len(u)), layer=L), D / "base.pt")
torch.save(dict(v=400 * u, layer=L), D / "huge.pt")
print(sorted(p.name for p in D.glob("*.pt")))

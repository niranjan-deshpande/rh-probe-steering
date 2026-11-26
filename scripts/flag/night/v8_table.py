"""Laptop, 26 November: steering table for the selection run.  -> cache/flag/night/steer_table_v8.pt (+ .json)

Rows (all "subtract": add = -norm * unit vector): candidates A to E of v8_cands.npz at norms 5, 10, 15, 20, Goodfire's
direction at 5, 10, 20, and the self-test rows of last night's table. A at 20 and Goodfire at 20 keep their old names
(sprobe_sub, gf_sub) and are bit-identical to the rows used on 25 November.
"""
import hashlib, json
import numpy as np, torch

old = torch.load("cache/flag/night/steer_table.pt", map_location="cpu")
on = old["names"]
c = np.load("cache/flag/night/v8_cands.npz")
gf = -old["add"][on.index("gf_sub")].double(); gf = gf / gf.norm()
A_old = -old["add"][on.index("sprobe_sub")].double() / 20
assert float(torch.tensor(c["A"]).double() @ A_old / A_old.norm()) > 0.999999
names, add = ["base"], [torch.zeros(5120)]
for k in "ABCDE":
    u = torch.tensor(c[k]).double(); u = u / u.norm()
    for n in (5, 10, 15, 20):
        if k == "A" and n == 20:
            names.append("sprobe_sub"); add.append(old["add"][on.index("sprobe_sub")])
        else:
            names.append(f"{k}_sub_n{n}"); add.append((-n * u).float())
for n in (5, 10):
    names.append(f"gf_sub_n{n}"); add.append((-n * gf).float())
names.append("gf_sub"); add.append(old["add"][on.index("gf_sub")])
add = torch.stack(add); proj = torch.zeros_like(add)
for i, n in enumerate(on):
    if n == "base_dup" or n.startswith("canary"):
        names.append(n); add = torch.cat([add, old["add"][i:i + 1]]); proj = torch.cat([proj, old["proj"][i:i + 1]])
assert len(set(names)) == len(names) and not add[0].any()
P = "cache/flag/night/steer_table_v8.pt"
torch.save(dict(add=add.float().contiguous(), proj=proj.float().contiguous(), layer=36, names=names, kinds=["v8"] * len(names), norm=None), P)
json.dump(dict(sha256=hashlib.sha256(open(P, "rb").read()).hexdigest(), rows={n: i for i, n in enumerate(names)},
               add_norm=[round(float(x), 3) for x in add.double().norm(dim=-1)]), open(P + ".json", "w"), indent=1)
print(json.dumps({n: i for i, n in enumerate(names)}))
print([round(float(x), 2) for x in add.double().norm(dim=-1)])

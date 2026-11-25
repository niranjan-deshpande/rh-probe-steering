"""Teacher-forced steering gain and generic-text KL of unit directions at several norms (no gradients)."""
import json, sys, numpy as np, torch
from rlp.flag import onp
from rlp.flag import q38 as Q
torch.set_num_threads(8)
L = int(sys.argv[1]); norms = [float(x) for x in sys.argv[2].split(",")]; extra = sys.argv[3:]
meta, means = onp.load_acts()
cf = dict(np.load(onp.OUT / f"dirs_cf_{L}.npz"))
dirs = {k: torch.tensor(cf[k]) for k in ("dom", "lda", "probe", "rnd0")}
for t in extra:
    v = torch.load(onp.OUT / f"v_{t}.pt")["v"].float(); dirs[t] = v / v.norm()
model, tok = onp.load_model(Q.MODEL)
recs = {(r["group"], r["k"]): r for r in onp.load_recs("onp")}
toks = [onp.tokenise(tok, recs[(m["group"], m["k"])]) for m in meta]
pairs = onp.steer_pairs(meta, toks, 2000)
rng = np.random.default_rng(0)
sel = []
for e in sorted(pairs):
    for j in rng.permutation(len(pairs[e]))[:5]:
        h, n = pairs[e][j]; sel.append((e, h[rng.integers(len(h))], n[rng.integers(len(n))]))
gen = torch.stack([torch.tensor(s) for s in onp.generic_seqs(tok)[:16]]).to(onp.DEV)
print("pairs", {e: sum(x[0] == e for x in sel) for e in pairs}, flush=True)
with torch.no_grad():
    base = {i: float(onp.completion_lp(model, toks[i][0], toks[i][1])) for _, h, n in sel for i in (h, n)}
    p0 = torch.log_softmax(model(input_ids=gen, use_cache=False).logits[:, 8:].float(), -1)
    out = {}
    for name, d in dirs.items():
        for r in norms:
            for sgn in (1, -1):
                v = (sgn * r * d).to(onp.DEV)
                with onp.add_vec(model, L, v):
                    gh = [float(onp.completion_lp(model, toks[h][0], toks[h][1])) - base[h] for _, h, n in sel]
                    gn = [float(onp.completion_lp(model, toks[n][0], toks[n][1])) - base[n] for _, h, n in sel]
                    p1 = torch.log_softmax(model(input_ids=gen, use_cache=False).logits[:, 8:].float(), -1)
                kl = float((p0.exp() * (p0 - p1)).sum(-1).mean())
                g = np.array(gh) - np.array(gn)
                byenv = {e: round(float(np.mean([x for x, s in zip(g, sel) if s[0] == e])), 3) for e in sorted(pairs)}
                out[f"{name}|{sgn*r}"] = dict(g_hack=float(np.mean(gh)), g_honest=float(np.mean(gn)), gain=float(g.mean()), se=float(g.std() / len(g) ** .5), kl=kl, byenv=byenv)
                print(f"[diag] {name:6s} r {sgn*r:6.1f}  g_hack {np.mean(gh):+.3f} g_honest {np.mean(gn):+.3f} gain {g.mean():+.3f} (se {g.std()/len(g)**.5:.3f}) kl {kl:.3f} {byenv}", flush=True)
json.dump(out, open(onp.OUT / f"diag_{L}_{'_'.join(extra) or 'cf'}.json", "w"), indent=1)
print("DIAG_DONE", flush=True)

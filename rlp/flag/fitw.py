"""Fit one detector w whose transform v = alpha * unit(Sigma w) is the steering vector (2026-11-24 recipe).

    L = L_det(w) + lam_var * w' Sigma w + lam_steer * L_steer(Sigma w) + lam_kl * KL(Sigma w)       with |w| = 1

  L_det     logistic loss of w on prompt-centred completion means of the fitting prompts (cached activations)
  w'Sigma w variance of the detector's readout over tokens (replaces v' Sigma^-1 v, which fought detection)
  L_steer   on-policy signal from prefix resampling (rlp.flag.resample): for a prefix of the model's own rollout from
            which its free continuations sometimes hack and sometimes do not, raise the log-probability of the first
            N tokens of a hack continuation relative to an honest continuation of the SAME prefix under +v, and lower
            it under -v. One hack / honest pair per micro-step, weighted by 4 p (1 - p).
  KL        on unrelated text, against the unsteered model.

    python -m rlp.flag.fitw diag --rs rs                 # does the windowed steering term have signal at all?
    python -m rlp.flag.fitw fit  --rs rs --tag w1
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from . import onp, q38 as Q, resample as RS

DEV = onp.DEV


def unit(x):
    return x / x.norm()


def examples(tok, name, n_tok, max_t, min_class=1):
    """Prefixes with at least min_class hack and honest continuations. -> list of dicts with token ids."""
    out = []
    rows = [((nm, sid, cut), r) for nm in name.split(",") for (sid, cut), r in sorted(RS.load(nm).items())]
    for (nm, sid, cut), r in rows:
        ys = [c["y"] for c in r["conts"] if c["y"] is not None]
        if sum(ys) < min_class or len(ys) - sum(ys) < min_class:
            continue
        p = tok(Q.prompt(tok, r["messages"]), add_special_tokens=False)["input_ids"] + r["prefix_ids"]
        if len(p) + n_tok > max_t:
            continue
        conts = {1: [], 0: []}
        for c in r["conts"]:
            if c["y"] is None:
                continue
            ids = tok(c["cont"], add_special_tokens=False)["input_ids"][:n_tok]
            if len(ids) >= 16:
                conts[c["y"]].append(ids)
        if conts[1] and conts[0]:
            ph = float(np.mean(ys))
            out.append(dict(sid=sid, cut=f"{nm}:{cut}", beh=r["beh"], frac=r["frac"], p=p, hack=conts[1], honest=conts[0], ph=ph, wt=4 * ph * (1 - ph)))
    return out


def window_lp(model, p, c):
    """Mean log-probability of continuation tokens c after prefix p (only those positions are unembedded)."""
    ids = torch.tensor([p + c], device=model.device)
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(p) - 1:-1]
    return -F.cross_entropy(model.lm_head(h).float(), ids[0, len(p):])


def prompt_of(e):
    """sid is group|k with group = environment|condition|item, and one prompt has a hack and an honest source."""
    return e["sid"].rsplit("|", 1)[0]


def split(ex, frac=0.2, seed=0):
    """Hold out whole prompts for the diagnostic (fixed 2026-11-25: it held out source completions, so a held-out
    prefix could share its prompt with a fitting prefix)."""
    prompts = sorted({prompt_of(e) for e in ex})
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(prompts)[:max(1, int(frac * len(prompts)))].tolist())
    return [e for e in ex if prompt_of(e) not in held], [e for e in ex if prompt_of(e) in held]


def gain(model, L, v, ex, base, rng, sign=1.0):
    """Weighted mean over prefixes of the change in (lp hack - lp honest) when sign * v is added. -> mean, SE"""
    vals = []
    for i, e in enumerate(ex):
        h, n = e["hack"][rng.integers(len(e["hack"]))], e["honest"][rng.integers(len(e["honest"]))]
        key = (e["sid"], e["cut"], tuple(h[:8]), tuple(n[:8]))
        with torch.no_grad():
            if key not in base:
                base[key] = float(window_lp(model, e["p"], h) - window_lp(model, e["p"], n))
            with onp.add_vec(model, L, sign * v):
                g = float(window_lp(model, e["p"], h) - window_lp(model, e["p"], n))
        vals.append(e["wt"] * sign * (g - base[key]))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std() / np.sqrt(len(vals)))


def load_common(a):
    model, tok = onp.load_model(a.model)
    cf = np.load(onp.OUT / f"dirs_cf_{a.layer}.npz")
    c = torch.load(onp.OUT / f"cov_{a.layer}.pt")
    Sig = c["cov"].float().to(DEV)
    ex = examples(tok, a.rs, a.n_tok, a.max_t)
    print(f"[fitw] {len(ex)} usable prefixes from {len({e['sid'] for e in ex})} sources; by env "
          f"{ {b: sum(e['beh'] == b for e in ex) for b in sorted({e['beh'] for e in ex})} }; "
          f"mean prefix {np.mean([len(e['p']) for e in ex]):.0f} tokens", flush=True)
    return model, tok, cf, Sig, ex


def diag(a):
    """Windowed teacher-forced gain of fixed directions at norm alpha. If this is flat, the fit has nothing to climb."""
    model, tok, cf, Sig, ex = load_common(a)
    rng = np.random.default_rng(0)
    ex = [ex[i] for i in rng.permutation(len(ex))[:a.n_diag]]
    dirs = {"dom": torch.tensor(cf["dom"]), "sprobe": Sig.cpu() @ torch.tensor(cf["probe"]), "probe": torch.tensor(cf["probe"]),
            "lda": torch.tensor(cf["lda"]), "rnd2": torch.tensor(cf["rnd2"]), "rnd3": torch.tensor(cf["rnd3"])}
    base, res = {}, {}
    for k, d in dirs.items():
        v = (a.alpha * unit(d.float())).to(DEV)
        for sign in (1.0, -1.0):
            m, se = gain(model, a.layer, v, ex, base, np.random.default_rng(1), sign)
            res[f"{k}{'+' if sign > 0 else '-'}"] = (m, se)
            print(f"[diag] {k:7s} {'+' if sign > 0 else '-'}{a.alpha:.0f}: signed gain {m:+.4f} (SE {se:.4f}) nats/token over {len(ex)} prefixes", flush=True)
    json.dump(res, open(onp.OUT / f"fitw_diag_{a.rs.replace(',', '+')}.json", "w"), indent=1)
    print("FITW_DIAG_DONE", flush=True)


def fit(a):
    model, tok, cf, Sig, ex = load_common(a)
    onp.checkpoint_blocks(model, a.layer)
    tr, ho = split(ex)
    meta = json.load(open(onp.OUT / "meta.json"))
    X = np.load(onp.OUT / f"means_{a.layer}.npy")
    rows, ys = [], []
    for k, (h, n) in onp.groups(meta, "fit").items():
        mu = X[h + n].mean(0)
        rows += [X[i] - mu for i in h + n]
        ys += [1.0] * len(h) + [-1.0] * len(n)
    R, ysign = torch.tensor(np.stack(rows), device=DEV), torch.tensor(ys, device=DEV)
    # The detection term is the logistic probe's own objective. beta lives in standardised coordinates (w = beta / sd)
    # on the sphere of the probe's norm; there the probe is a stationary point of the mean logistic loss (its
    # gradient is radial, because the probe minimises loss + |beta|^2 / 2C). So detection starts at its optimum with
    # zero tangential gradient, and only the steering term moves w, against the curvature of the detection loss.
    # (The first version, softplus on unit w with no regulariser, drifted away from the probe and overfitted:
    # validation 0.95 -> 0.98, held-out 0.92 -> 0.87 in 25 steps. Kept as fitw_w1.pt.)
    from sklearn.linear_model import LogisticRegression
    sd = R.std(0) + 1e-6
    Rs = R / sd
    b0 = torch.tensor(LogisticRegression(C=0.01, max_iter=2000, fit_intercept=False).fit(Rs.cpu().numpy(), (ysign > 0).cpu().numpy()).coef_[0],
                      dtype=torch.float32, device=DEV)
    b_norm = b0.norm()
    gen = [torch.tensor(s, device=DEV) for s in onp.generic_seqs(tok, n=8)]
    with torch.no_grad():
        p0 = [torch.log_softmax(model(input_ids=g[None], use_cache=False).logits[0, 8:].float(), -1) for g in gen]
    rng = np.random.default_rng(a.seed)
    u = unit(b0).clone().requires_grad_(True)                       # direction of beta; starts at the probe
    opt = torch.optim.Adam([u], lr=a.lr)
    wvec = lambda: unit(unit(u) / sd)                                  # detector in activation coordinates
    vec = lambda: a.alpha * unit(Sig @ wvec())
    base_ho, hist = {}, []
    dom, probe = torch.tensor(cf["dom"]).to(DEV), torch.tensor(cf["probe"]).to(DEV)

    def report(step):
        w = wvec().detach()
        v = vec().detach()
        t = onp.score_table(meta, X, {"w": w.cpu().numpy()})["w"]
        gp, sp = gain(model, a.layer, v, ho, base_ho, np.random.default_rng(1), 1.0)
        gn, sn = gain(model, a.layer, v, ho, base_ho, np.random.default_rng(1), -1.0)
        return dict(step=step, val=t["val_mean"], held=t["held_mean"], var=float(w @ Sig @ w), cos_w_probe=float(w @ unit(probe)),
                    cos_v_dom=float(unit(v) @ unit(dom)), ho_gain_pos=gp, ho_gain_neg=gn, ho_se=max(sp, sn))

    for step in range(a.steps + 1):
        if step % a.every == 0 or step == a.steps:
            rec = report(step)
            hist.append(rec)
            print("[fitw] " + " ".join(f"{k} {x:.4f}" if isinstance(x, float) else f"{k} {x}" for k, x in rec.items()), flush=True)
            ck = dict(w=wvec().detach().cpu(), v=vec().detach().cpu(), layer=a.layer, step=step, hist=hist,
                      args={k: x for k, x in vars(a).items() if isinstance(x, (int, float, str, bool, type(None)))})
            torch.save(ck, onp.OUT / f"fitw_{a.tag}.pt")
            torch.save(ck, onp.OUT / f"fitw_{a.tag}_step{step}.pt")
            if step == a.steps:
                break
        opt.zero_grad()
        g_tot = 0.0
        for _ in range(a.pairs):
            e = tr[rng.integers(len(tr))]
            h, n = e["hack"][rng.integers(len(e["hack"]))], e["honest"][rng.integers(len(e["honest"]))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            for c, cs in ((h, 1.0), (n, -1.0)):           # two backward passes, so only one graph is alive at a time
                with onp.add_vec(model, a.layer, sign * vec()):
                    lp = window_lp(model, e["p"], c)
                (-a.lam_steer * e["wt"] * sign * cs * lp / a.pairs).backward()
                g_tot += e["wt"] * sign * cs * lp.detach().item() / a.pairs
        w = wvec()
        l_det = F.softplus(-ysign * (Rs @ (b_norm * unit(u)))).mean()
        l_var = w @ Sig @ w
        j = int(rng.integers(len(gen)))
        with onp.add_vec(model, a.layer, vec()):
            p1 = torch.log_softmax(model(input_ids=gen[j][None], use_cache=False).logits[0, 8:].float(), -1)
        l_kl = (p0[j].exp() * (p0[j] - p1)).sum(-1).mean()
        (a.lam_det * l_det + a.lam_var * l_var + a.lam_kl * l_kl).backward()
        gnorm = u.grad.norm().item()
        opt.step()
        with torch.no_grad():
            u.copy_(unit(u))
        if step % 5 == 0:
            print(f"[fitw] step {step} steer_obj {g_tot:+.4f} det {l_det.item():.4f} var {l_var.item():.3f} kl {l_kl.item():.4f} "
                  f"gnorm {gnorm:.3f} gb {torch.cuda.max_memory_allocated() / 2**30 if DEV == 'cuda' else 0:.1f}", flush=True)
    D = onp.OUT / "steer"
    D.mkdir(exist_ok=True)
    # checkpoint rule, fixed before looking at any steering run: the step with the largest held-out-prefix gain
    # (mean of both signs) among steps whose validation AUROC is within 0.01 of step 0
    ok = [h for h in hist if h["val"] >= hist[0]["val"] - 0.01]
    best = max(ok, key=lambda h: h["ho_gain_pos"] + h["ho_gain_neg"])
    v = torch.load(onp.OUT / f"fitw_{a.tag}_step{best['step']}.pt")["v"].float()
    print(f"[fitw] chosen step {best['step']}: " + " ".join(f"{k} {x:.4f}" for k, x in best.items() if isinstance(x, float)), flush=True)
    for s, sg in (("pos", 1.0), ("neg", -1.0)):
        torch.save(dict(v=sg * v, layer=a.layer), D / f"fitw{a.tag}_{s}.pt")
    json.dump(hist, open(onp.OUT / f"fitw_hist_{a.tag}.json", "w"), indent=1)
    print("FITW_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["diag", "fit"])
    ap.add_argument("--rs", default="rs")
    ap.add_argument("--model", default=Q.MODEL)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--alpha", type=float, default=20.0, help="norm of the steering vector (24% of the residual norm at layer 36)")
    ap.add_argument("--n-tok", type=int, default=128, help="continuation tokens scored after each prefix")
    ap.add_argument("--max-t", type=int, default=4096)
    ap.add_argument("--n-diag", type=int, default=60)
    ap.add_argument("--tag", default="w1")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-steer", type=float, default=10.0)
    ap.add_argument("--lam-det", type=float, default=1.0)
    ap.add_argument("--lam-var", type=float, default=0.0)
    ap.add_argument("--lam-kl", type=float, default=0.3)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dict(diag=diag, fit=fit)[a.stage](a)

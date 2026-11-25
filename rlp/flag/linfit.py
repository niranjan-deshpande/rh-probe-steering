"""Linearised steering fit, laptop half (no model). Reads what rlp.flag.lingrad cached under cache/flag/onp/lingrad/.

The steering term of the 24 November fit (rlp.flag.fitw) for a prefix k is
    S_k(v) = wt_k * [ mean log-prob per token of the hack continuations - the same for the honest continuations ],
with v added at the input of block 36 at every position and wt_k = 4 p (1 - p), p the prefix's hack rate. lingrad
stores, per continuation, the gradient of the summed log-probability at v = 0. So to first order S(v) - S(0) = g . v
with g = mean_k wt_k * C_k, C_k = mean per-token gradient of hack continuations minus honest ones, and the best
vector of a given norm is along g. No gradient descent is run.

    python -m rlp.flag.linfit noise     # noise-only control vector            -> cache/flag/onp/noise_vec.pt
    python -m rlp.flag.linfit fit       # after `lingrad --stage win`          -> cache/flag/onp/linfit_vec.pt
    python -m rlp.flag.linfit check     # after `lingrad --stage sec`; prints LINFIT_CHECK_PASS or LINFIT_CHECK_FAIL
    python -m rlp.flag.linfit relin     # fallback, after `lingrad --stage win --point +lin` and `--point -lin`

Outputs (all vectors are float32, norm 20, layer 36, hack-positive, in the format of cache/flag/onp/steer/*.pt:
dict(v=..., layer=...)):
    cache/flag/onp/noise_vec.pt        the 24 November optimiser run from the probe with a pure-noise steering gradient
                                       (isotropic, scale 0.01, 125 Adam steps, seed 0). It moves the way the real fit
                                       moved without any steering information, so it is the control for "the fitted v"
    cache/flag/onp/linfit_vec.pt       the linear-fit vector (the candidate chosen by the rule below)
    cache/flag/onp/linfit/lin_free.pt  20 * unit(g)
    cache/flag/onp/linfit/lin_sub.pt   20 * unit(projection of g on span{Sigma.probe, probe, dom, lda, top-k PCs of Sigma})
    cache/flag/onp/linfit.json, linfit_check.json     the numbers printed

Rules fixed before the run.
  Which candidate. lin_free or lin_sub, whichever has the larger first-order gain on held-out prompts in K-fold
  cross-validation by prompt (a prompt = environment | condition | item; all prefixes of one prompt share a fold).
  First-order check (on the 60 prefixes lingrad scored at +20 and -20 along the vector). PASS needs all three:
    (1) over continuations, corr( g_c . v , [lp_c(+v) - lp_c(-v)] / 2 ) >= 0.7,
    (2) the slope of the measured odd part on the prediction is within [0.5, 1.5],
    (3) the measured steering gain, mean_k wt_k * (odd part of hack - odd part of honest), is positive and at least
        half of the predicted gain.
  If it fails, `relin` takes one step re-linearised at +v and -v; that step is "stable" if its cosine with the vector
  refitted at 0 on the same prefixes is at least 0.5. Otherwise the linear candidate is dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import config

OUT = config.CACHE / "flag" / "onp"
STAGING = config.CACHE / "_staging_night_2026-11-24"


def unit(x):
    return x / (np.linalg.norm(x) + 1e-30)


def _pt(f, key="v"):
    return torch.load(f, map_location="cpu", weights_only=False)[key].double().numpy()


def resolve(name, layer=36, onp=None):
    """Named direction -> raw float64 vector. Names: sprobe (Sigma.probe), any key of dirs_cf_<L>.npz (probe, dom,
    lda, rnd0..), lin, lin_free, lin_sub, noise, fitv / fitw (fitw_w2.pt), relin, a steer/ file stem, or a path to a
    .pt file, optionally 'path.pt:key'."""
    onp = Path(onp or OUT)
    cf = onp / f"dirs_cf_{layer}.npz"
    if name == "sprobe":
        return torch.load(onp / f"cov_{layer}.pt", map_location="cpu")["cov"].double().numpy() @ np.load(cf)["probe"].astype(np.float64)
    table = {"lin": (onp / "linfit_vec.pt", "v"), "noise": (onp / "noise_vec.pt", "v"), "relin": (onp / "linfit_relin_vec.pt", "v"),
             "lin_free": (onp / "linfit" / "lin_free.pt", "v"), "lin_sub": (onp / "linfit" / "lin_sub.pt", "v"),
             "fitv": (onp / "fitw_w2.pt", "v"), "fitw": (onp / "fitw_w2.pt", "w")}
    if name in table:
        return _pt(*table[name])
    if cf.exists() and name in np.load(cf).files:
        return np.load(cf)[name].astype(np.float64)
    path, _, key = name.partition(".pt:")
    for f in (Path(path + ".pt") if key else Path(name), onp / "steer" / f"{name}.pt", onp / "steer" / f"{name}_pos.pt"):
        if f.suffix == ".pt" and f.exists():
            return _pt(f, key or "v")
    raise FileNotFoundError(f"direction {name!r} not found under {onp}")


def save_vec(path, v, layer, alpha, **extra):
    v = alpha * unit(np.asarray(v, np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(v=torch.tensor(v, dtype=torch.float32), layer=layer, **extra), path)
    return v


# ---------------------------------------------------------------- cached gradients

def load(stage, onp=None):
    """-> {prefix key: record} for one lingrad stage directory."""
    D = Path(onp or OUT) / "lingrad" / stage
    return {r["key"]: r for r in (torch.load(f, map_location="cpu", weights_only=False) for f in sorted(D.glob("*.pt")))} if D.exists() else {}


def per_token_grad(r, part="all"):
    g = (r["g_pre"] + r["g_cont"]) if part == "all" else r["g_cont"]
    return g.double().numpy() / np.array(r["n"], float)[:, None]


def contrast(r, part="all"):
    """Per prefix: mean per-token gradient of hack continuations minus honest ones, and the weight 4p(1-p)."""
    y = np.array(r["y"])
    if len(y) == 0 or y.min() == y.max():
        return None, 0.0
    g = per_token_grad(r, part)
    p = float(r.get("ph", y.mean()))
    return g[y == 1].mean(0) - g[y == 0].mean(0), 4 * p * (1 - p)


def stack(R, part="all"):
    """-> keys, C [K, d], W [K], prompt [K] over prefixes with both classes."""
    keys, C, W, prompt = [], [], [], []
    for k in sorted(R):
        c, w = contrast(R[k], part)
        if c is not None:
            keys.append(k), C.append(c), W.append(w), prompt.append(R[k]["prompt"])
    return keys, np.array(C), np.array(W), np.array(prompt)


def clustered(x, groups):
    """Mean of x and its standard error with whole groups (prompts) as the sampling unit."""
    x, groups = np.asarray(x, float), np.asarray(groups)
    m, ug = x.mean(), sorted(set(groups.tolist()))
    if len(ug) < 2:
        return float(m), float("nan")
    r = np.array([(x[groups == g] - m).sum() for g in ug])
    return float(m), float(np.sqrt(len(ug) / (len(ug) - 1) * (r ** 2).sum()) / len(x))


def subspace(layer, onp, k):
    """Orthonormal basis of span{Sigma.probe, probe, dom, lda, top-k principal components of Sigma}."""
    onp = Path(onp)
    S = torch.load(onp / f"cov_{layer}.pt", map_location="cpu")["cov"].double().numpy()
    cf = np.load(onp / f"dirs_cf_{layer}.npz")
    ev, U = np.linalg.eigh(S)
    cols = [S @ cf["probe"]] + [cf[n].astype(np.float64) for n in ("probe", "dom", "lda") if n in cf.files]
    return np.linalg.qr(np.column_stack([unit(c) for c in cols] + [U[:, ::-1][:, :k]]))[0]


def solve(C, W, B, alpha):
    g = (W[:, None] * C).mean(0)
    return {"lin_free": alpha * unit(g), "lin_sub": alpha * unit(B @ (B.T @ g))}


def cross_validate(C, W, prompt, B, alpha, fixed, folds=5, seed=0):
    """Prompt-grouped K-fold: first-order gain (nats per token, weighted as the fit) of every candidate on prefixes
    of held-out prompts. -> {candidate: (mean, clustered SE)}"""
    ups = np.array(sorted(set(prompt.tolist())))
    fold = {p: i % folds for i, p in enumerate(np.random.default_rng(seed).permutation(ups))}
    fid = np.array([fold[p] for p in prompt])
    vals = {}
    for f in range(folds):
        tr, te = fid != f, fid == f
        if not te.any() or not tr.any():
            continue
        for k, v in (dict(fixed) | solve(C[tr], W[tr], B, alpha)).items():
            vals.setdefault(k, np.full(len(C), np.nan))[te] = W[te] * (C[te] @ v)
    return {k: clustered(x, prompt) for k, x in vals.items()}


def fit(a):
    onp = Path(a.onp)
    keys, C, W, prompt = stack(load("win_0", onp), a.part)
    if not len(C):
        raise SystemExit(f"[linfit] no gradients under {onp / 'lingrad' / 'win_0'}")
    print(f"[linfit] {len(C)} prefixes from {len(set(prompt.tolist()))} prompts, width {C.shape[1]}")
    B = subspace(a.layer, onp, a.k)
    fixed = {}
    for n in ("sprobe", "probe", "dom", "lda", "fitv", "noise", "rnd2"):
        try:
            fixed[n] = a.alpha * unit(resolve(n, a.layer, onp))
        except (FileNotFoundError, KeyError):
            pass
    cv = cross_validate(C, W, prompt, B, a.alpha, fixed, a.folds)
    print(f"first-order gain on held-out prompts at norm {a.alpha:g}, nats per token ({a.folds}-fold by prompt; SE clustered by prompt):")
    for k, (m, se) in cv.items():
        print(f"  {k:9s} {m:+.4f} ({se:.4f})")
    kind = max(("lin_free", "lin_sub"), key=lambda k: cv[k][0])
    full = solve(C, W, B, a.alpha)
    ups, boot = np.array(sorted(set(prompt.tolist()))), {}
    for name in ("lin_free", "lin_sub"):
        cs = []
        for b in range(a.boot):
            pick = np.random.default_rng(b).choice(ups, len(ups))
            idx = np.concatenate([np.where(prompt == p)[0] for p in pick])
            cs.append(float(unit(solve(C[idx], W[idx], B, a.alpha)[name]) @ unit(full[name])))
        boot[name] = (float(np.median(cs)), float(np.percentile(cs, 5)))
        print(f"  {name}: cosine of a prompt-bootstrap refit with the full fit, median {boot[name][0]:.2f}, 5th percentile {boot[name][1]:.2f}")
    cos = {n: float(unit(full[kind]) @ unit(v)) for n, v in fixed.items()}
    print(f"[linfit] chosen: {kind}; cosines with it: " + ", ".join(f"{n} {c:+.2f}" for n, c in cos.items()))
    for name, v in full.items():
        save_vec(onp / "linfit" / f"{name}.pt", v, a.layer, a.alpha)
    save_vec(onp / "linfit_vec.pt", full[kind], a.layer, a.alpha, kind=kind, n_prefix=len(C), cv={k: list(x) for k, x in cv.items()})
    res = dict(kind=kind, n_prefix=len(C), n_prompt=len(ups), alpha=a.alpha, part=a.part, k=a.k, folds=a.folds,
               cv={k: list(x) for k, x in cv.items()}, boot=boot, cos_chosen=cos, cos_free_sub=float(unit(full["lin_free"]) @ unit(full["lin_sub"])))
    json.dump(res, open(onp / "linfit.json", "w"), indent=1)
    print(f"[linfit] -> {onp / 'linfit_vec.pt'}", flush=True)
    print("LINFIT_FIT_DONE", flush=True)
    return res


# ---------------------------------------------------------------- first-order check against measured secants

def check_dir(R0, Rs, name, v, part="all"):
    """Compare g . v with the measured odd part at +v / -v for one direction. -> dict of statistics (or None)."""
    pred, odd, even, pg, ag, pr = [], [], [], [], [], []
    for k, r in Rs.items():
        r0 = R0.get(k)
        if r0 is None or f"{name}+" not in (r["lp"][0] if r["lp"] else {}):
            continue
        row = {j: i for i, j in enumerate(r0["j"])}
        if any(j not in row for j in r["j"]):
            continue
        idx = [row[j] for j in r["j"]]
        n = np.array(r["n"], float)
        p = (per_token_grad(r0, part)[idx]) @ v
        o = np.array([(x[f"{name}+"] - x[f"{name}-"]) / 2 for x in r["lp"]]) / n
        e = np.array([(x[f"{name}+"] + x[f"{name}-"]) / 2 - x["0"] for x in r["lp"]]) / n
        pred.append(p), odd.append(o), even.append(e)
        y = np.array(r["y"])
        if y.min() != y.max():
            ph = float(r0.get("ph", np.mean(r0["y"])))
            w = 4 * ph * (1 - ph)
            pg.append(w * (p[y == 1].mean() - p[y == 0].mean()))
            ag.append(w * (o[y == 1].mean() - o[y == 0].mean()))
            pr.append(r0["prompt"])
    if not pred:
        return None
    pred, odd, even = (np.concatenate(x) for x in (pred, odd, even))
    ok = pred.std() > 0 and odd.std() > 0
    am, ase = clustered(ag, pr)
    return dict(n_cont=len(pred), n_prefix=len(pg), corr=float(np.corrcoef(pred, odd)[0, 1]) if ok else float("nan"),
                slope=float(np.polyfit(pred, odd, 1)[0]) if ok else float("nan"),
                even_over_odd=float(np.sqrt((even ** 2).mean() / max((odd ** 2).mean(), 1e-30))),
                even_mean=float(even.mean()), gain_pred=float(np.mean(pg)), gain_meas=am, gain_meas_se=ase)


def passes(s, a):
    return bool(s is not None and s["corr"] >= a.min_corr and a.slope_lo <= s["slope"] <= a.slope_hi
                and s["gain_meas"] > 0 and s["gain_meas"] >= a.min_ratio * s["gain_pred"])


def check(a):
    onp = Path(a.onp)
    R0, Rs = load("win_0", onp), load("sec", onp)
    names = sorted({c[:-1] for r in Rs.values() for x in r["lp"][:1] for c in x if c != "0"})
    res = {}
    print(f"first-order check at norm {a.alpha:g}: prediction g.v at v = 0 against the measured odd part, {len(Rs)} prefixes")
    for n in names:
        s = check_dir(R0, Rs, n, a.alpha * unit(resolve(n, a.layer, onp)), a.part)
        if s is None:
            continue
        res[n] = s
        print(f"  {n:9s} corr {s['corr']:+.2f}  slope {s['slope']:+.2f}  |even|/|odd| {s['even_over_odd']:.2f}  steering gain predicted "
              f"{s['gain_pred']:+.4f}, measured {s['gain_meas']:+.4f} ({s['gain_meas_se']:.4f}) nats/token  [{s['n_cont']} continuations]")
    ok = passes(res.get(a.cand), a)
    json.dump(dict(cand=a.cand, passed=ok, rule=dict(min_corr=a.min_corr, slope=[a.slope_lo, a.slope_hi], min_ratio=a.min_ratio), dirs=res),
              open(onp / "linfit_check.json", "w"), indent=1)
    if a.cand not in res:
        print(f"[linfit] no secants for {a.cand!r} under {onp / 'lingrad' / 'sec'}")
    print("LINFIT_CHECK_PASS" if ok else "LINFIT_CHECK_FAIL", flush=True)
    return ok


def relin(a):
    """One re-linearised step: the gradient of the odd objective [S(v) - S(-v)] / 2 at v = lin is the mean of the
    contrast gradients at +lin and -lin."""
    onp = Path(a.onp)
    Rp, Rm, R0 = load(f"win_+{a.cand}", onp), load(f"win_-{a.cand}", onp), load("win_0", onp)
    common = {k: R0[k] for k in R0 if k in Rp and k in Rm}
    if not common:
        print(f"[linfit] no prefixes with gradients at 0, +{a.cand} and -{a.cand}")
        print("LINFIT_RELIN_UNSTABLE", flush=True)
        return False
    keys, C0, W, _ = stack(common, a.part)
    Cp, Cm = (np.array([contrast(R[k], a.part)[0] for k in keys]) for R in (Rp, Rm))
    kind = json.load(open(onp / "linfit.json"))["kind"] if (onp / "linfit.json").exists() else "lin_free"
    B = subspace(a.layer, onp, a.k)
    v1, v0 = solve((Cp + Cm) / 2, W, B, a.alpha)[kind], solve(C0, W, B, a.alpha)[kind]
    lin = resolve(a.cand, a.layer, onp)
    c_same, c_lin = float(unit(v1) @ unit(v0)), float(unit(v1) @ unit(lin))
    save_vec(onp / "linfit_relin_vec.pt", v1, a.layer, a.alpha, kind=kind, n_prefix=len(keys))
    print(f"[linfit] re-linearised step on {len(keys)} prefixes ({kind}): cosine with the refit at 0 on the same prefixes "
          f"{c_same:+.2f}, with {a.cand} {c_lin:+.2f} -> {onp / 'linfit_relin_vec.pt'}")
    ok = c_same >= a.min_relin_cos
    print("LINFIT_RELIN_STABLE" if ok else "LINFIT_RELIN_UNSTABLE", flush=True)
    return ok


# ---------------------------------------------------------------- noise-only control

def noise_run(S, R, y, seed, alpha=20.0, scale=0.01, steps=125, lr=3e-4, C=0.01):
    """The fitw optimiser (Adam on the unit direction of the standardised probe coefficients, detection term at full
    weight) with the steering gradient replaced by an isotropic random vector each step. -> (w, v) as float64."""
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    u_ = lambda x: x / x.norm()
    S, R, y = (torch.as_tensor(x).double() for x in (S, R, y))
    sd = R.std(0) + 1e-6
    Rs = R / sd
    b0 = torch.tensor(LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(Rs.numpy(), (y > 0).numpy()).coef_[0]).double()
    bn = b0.norm()
    g = torch.Generator().manual_seed(seed)
    u = u_(b0).clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    w = lambda: u_(u_(u) / sd)
    v = lambda: alpha * u_(S @ w())
    for _ in range(steps):
        opt.zero_grad()
        z = torch.randn(S.shape[0], generator=g, dtype=torch.float64)
        (scale * (z @ v()) + F.softplus(-y * (Rs @ (bn * u_(u)))).mean()).backward()
        opt.step()
        with torch.no_grad():
            u.copy_(u_(u))
    return w().detach().numpy(), v().detach().numpy()


def noise(a):
    from . import onp as O
    onp = Path(a.onp)
    S = torch.load(onp / f"cov_{a.layer}.pt", map_location="cpu")["cov"].double()
    meta, X = json.load(open(onp / "meta.json")), np.load(onp / f"means_{a.layer}.npy")
    rows, ys = [], []
    for _, (h, n) in O.groups(meta, "fit").items():          # the fitting rows of the 24 November fit
        mu = X[h + n].mean(0)
        rows += [X[i] - mu for i in h + n]
        ys += [1.0] * len(h) + [-1.0] * len(n)
    R, y = np.stack(rows), np.array(ys)
    for seed in a.seeds:
        w, v = noise_run(S, R, y, seed, a.alpha)
        f = onp / ("noise_vec.pt" if seed == a.seeds[0] else f"noise_vec_s{seed}.pt")
        save_vec(f, v, a.layer, a.alpha, seed=seed, w=torch.tensor(w, dtype=torch.float32))
        msg = []
        for n in ("probe", "sprobe", "fitv", "fitw"):
            try:
                msg.append(f"{'w' if n in ('probe', 'fitw') else 'v'}.{n} {float(unit(w if n in ('probe', 'fitw') else v) @ unit(resolve(n, a.layer, onp))):+.3f}")
            except (FileNotFoundError, KeyError):
                pass
        old = STAGING / f"noisenull{seed}_pos.pt"
        if old.exists():
            msg.append(f"v.(reviewed draft's vector) {float(unit(v) @ unit(_pt(old))):+.4f}")
        print(f"[linfit] noise seed {seed}: cosines " + ", ".join(msg) + f" -> {f}", flush=True)
    print("LINFIT_NOISE_DONE", flush=True)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fit", "check", "relin", "noise"))
    ap.add_argument("--onp", default=str(OUT), help="directory holding lingrad/, cov_<L>.pt and dirs_cf_<L>.npz")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--k", type=int, default=10, help="top principal components of Sigma added to the subspace")
    ap.add_argument("--part", choices=("all", "cont"), default="all", help="gradient positions: all, or continuation only")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--cand", default="lin", help="the direction the pass/fail rule applies to")
    ap.add_argument("--min-corr", type=float, default=0.7)
    ap.add_argument("--slope-lo", type=float, default=0.5)
    ap.add_argument("--slope-hi", type=float, default=1.5)
    ap.add_argument("--min-ratio", type=float, default=0.5)
    ap.add_argument("--min-relin-cos", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    return ap


def args(stage="fit", **over):
    """The command-line defaults as a namespace, for callers inside Python (lingrad --stage all, tests)."""
    a = parser().parse_args([stage])
    for k, v in over.items():
        setattr(a, k, v)
    return a


if __name__ == "__main__":
    a = parser().parse_args()
    {"fit": fit, "check": check, "relin": relin, "noise": noise}[a.stage](a)

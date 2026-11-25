"""On-policy recipe for a reward-hacking vector on the frozen Qwen3.8-27B (started 2026-11-24).

The model is never trained. One vector v at one layer is fitted to

    L = L_steer + lam1 * L_AUC + lam2 * L_KL + lam3 * v' Sigma^-1 v

    L_steer   adding v at every position makes the model's own hack continuation more likely and its own honest
              continuation (same prompt) less likely: -[(lp_hack(v) - lp_hack(0)) - (lp_honest(v) - lp_honest(0))],
              mean log-probability per completion token. Teacher-forced on sampled continuations (gradients need
              fixed text); every evaluation of steering is in free generation (rlp.flag.base_rate on a steered server).
    L_AUC     pairwise logistic surrogate: for hack/honest completions of the same prompt, v . (mean activation of the
              hack completion - mean activation of the honest one) should be positive. Scale-free in v.
    L_KL      KL(p_0 || p_v) per token on unrelated text (data/flag/generic.json).
    Sigma     covariance of layer activations over completion tokens of the fitting data plus the unrelated text.

Data: rlp.flag.base_rate output (several samples per prompt, labelled by program detectors). "Fitting" environments
feed the loss (items 0-29 fit, 30-39 validation); "held-out" environments are only scored.

    python -m rlp.flag.onp harvest            # activations (pooled means) and covariance, all candidate layers
    python -m rlp.flag.onp dirs               # closed-form directions and detection scores -> picks the layer
    python -m rlp.flag.onp train --tag a ...  # the vector
    python -m rlp.flag.onp export --tag a     # steering files for the patched server

Layer convention as in the rest of the repo: index L is the input to decoder block L (hidden_states[L]).
"""
import argparse
import json
import math

import numpy as np
import torch

from .. import config
from ..manif import envs as E
from . import q38 as Q

SRC = config.CACHE / "flag" / "base_rate"
OUT = config.CACHE / "flag" / "onp"
LAYERS = (16, 24, 28, 32, 36, 40, 48)
FIT = {"length_pad": ("neutral", "tempt"), "quiz_key": ("neutral", "tempt"), "leak": ("tempt",), "hardcode": ("neutral", "tempt")}
HELD = {"save_tamper": ("tempt",), "fake_pass": ("tempt",), "chess_key": ("tempt",), "fake_cite": ("neutral",)}
N_FIT = 30                 # items per fitting environment used for fitting; the rest are validation
POOLS = ("all", "reasoning", "answer")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def load_recs(name="onp"):
    """Completions with a label y (1 hack, 0 honest, None unusable) and a role (fit / val / held)."""
    out = []
    for f in sorted((SRC / name).glob("*__*.jsonl")):
        if f.name.endswith(".labels.jsonl"):
            continue
        beh, cond = f.name[:-len(".jsonl")].split("__")
        if cond not in FIT.get(beh, ()) + HELD.get(beh, ()):
            continue
        lab = {(x["uid"], x.get("k", 0)): x for x in map(json.loads, open(str(f).replace(".jsonl", ".labels.jsonl")))}
        idx = {it["uid"]: i for i, it in enumerate(E.items(beh))}
        for r in map(json.loads, open(f)):
            x = lab[(r["uid"], r.get("k", 0))]
            ok = bool(x.get("valid")) and not x["trunc"] and "raw" in r
            role = "held" if beh in HELD else ("fit" if idx[r["uid"]] < N_FIT else "val")
            out.append(dict(beh=beh, cond=cond, uid=r["uid"], k=r.get("k", 0), messages=r["messages"], raw=r.get("raw", ""),
                            y=(int(x["hit"]) if ok else None), role=role, group=f"{beh}|{cond}|{r['uid']}"))
    return out


def tokenise(tok, rec, max_c=None):
    """-> prompt ids, completion ids, index in the completion of the first answer token (after </think>)."""
    p = tok(Q.prompt(tok, rec["messages"]), add_special_tokens=False)["input_ids"]
    c = tok(rec["raw"] + Q.STOP, add_special_tokens=False)["input_ids"]
    close = tok.convert_tokens_to_ids(Q.TH_CLOSE)
    a = c.index(close) + 1 if close in c else len(c)
    if max_c:
        c = c[:max_c]
    return p, c, min(a, len(c))


# ---------------------------------------------------------------- model

def load_model(name=Q.MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not isinstance(name, str):
        return name                                           # (model, tokenizer) passed in by a test
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def blocks_of(model):
    n = getattr(model.config, "text_config", model.config).num_hidden_layers
    best = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n and name.endswith("layers") and "visual" not in name:
            best = m if best is None or "language_model" in name else best
    assert best is not None, "decoder layers not found"
    return best


class _Stop(Exception):
    pass


def read_layers(model, ids, layers):
    """-> {L: [T, d]} for one sequence, without running blocks above max(layers)."""
    got, hooks, top, blks = {}, [], max(layers), blocks_of(model)

    def mk(L):
        def pre(mod, args, kwargs):
            got[L] = (args[0] if args else kwargs["hidden_states"])[0]
            if L == top:
                raise _Stop()
        return pre
    for L in layers:
        hooks.append(blks[L].register_forward_pre_hook(mk(L), with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in hooks:
            h.remove()
    return got


class add_vec:
    """Context manager: add v to the input of block L at every position."""
    def __init__(self, model, L, v):
        self.blk, self.v = blocks_of(model)[L], v

    def __enter__(self):
        def pre(mod, args, kwargs):
            if args:
                return (args[0] + self.v.to(args[0].dtype),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = kwargs["hidden_states"] + self.v.to(kwargs["hidden_states"].dtype)
            return args, kwargs
        self.h = self.blk.register_forward_pre_hook(pre, with_kwargs=True)

    def __exit__(self, *a):
        self.h.remove()


def checkpoint_blocks(model, L):
    """Recompute blocks L.. during backward instead of storing their graphs (memory for long sequences)."""
    from torch.utils.checkpoint import checkpoint
    for blk in blocks_of(model)[L:]:
        if getattr(blk, "_ckpt", False):
            continue
        orig = blk.forward

        def fwd(*args, _orig=orig, **kw):
            if torch.is_grad_enabled() and args and args[0].requires_grad:
                return checkpoint(_orig, *args, use_reentrant=False, **kw)
            return _orig(*args, **kw)
        blk.forward, blk._ckpt = fwd, True


def completion_lp(model, p, c):
    """Mean log-probability per completion token (float32 scalar, differentiable through any active hook)."""
    ids = torch.tensor([p + c], device=model.device)
    if hasattr(model, "lm_head") and hasattr(model, "model"):     # unembed the completion positions only (memory)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(p) - 1:-1]
        logits = model.lm_head(h)
    else:
        logits = model(input_ids=ids, use_cache=False).logits[0, len(p) - 1:-1]
    return -torch.nn.functional.cross_entropy(logits.float(), ids[0, len(p):])


def generic_seqs(tok, n=64, max_len=72):
    out = []
    for s in json.load(open(config.DATA / "flag" / "generic.json")):
        ids = tok(Q.prompt(tok, [{"role": "user", "content": s["instruction"]}]) + "\n</think>\n\n" + s["output"],
                  add_special_tokens=False)["input_ids"]
        if len(ids) >= max_len:
            out.append(ids[:max_len])
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- stage 2a: harvest

def harvest(a):
    model, tok = load_model(a.model)
    recs = [r for r in load_recs(a.name) if r["raw"]]
    toks = [tokenise(tok, r, a.max_c) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: len(toks[i][0]) + len(toks[i][1]))
    d = blocks_of(model)[0].input_layernorm.weight.shape[0]
    means = np.zeros((len(recs), len(POOLS), len(LAYERS), d), np.float32)
    s1 = {L: torch.zeros(d, dtype=torch.float64, device=DEV) for L in LAYERS}
    s2 = {L: torch.zeros(d, d, dtype=torch.float64, device=DEV) for L in LAYERS}
    n_tok, norm_sum = 0, {L: 0.0 for L in LAYERS}

    def cov_add(got, sl):
        nonlocal n_tok
        for L in LAYERS:
            X = got[L][sl].float()
            s1[L] += X.sum(0).double()
            s2[L] += (X.T @ X).double()
            norm_sum[L] += float(X.norm(dim=-1).sum())
        n_tok += got[LAYERS[0]][sl].shape[0]

    for j, i in enumerate(order):
        p, c, ans = toks[i]
        got = read_layers(model, p + c, LAYERS)
        for li, L in enumerate(LAYERS):
            h = got[L][len(p):].float()
            means[i, 0, li] = h.mean(0).cpu().numpy()
            means[i, 1, li] = (h[:ans].mean(0) if ans > 0 else h.mean(0)).cpu().numpy()
            means[i, 2, li] = (h[ans:].mean(0) if ans < len(c) else h.mean(0)).cpu().numpy()
        if recs[i]["role"] == "fit":
            cov_add(got, slice(len(p), None))
        if j % 50 == 0:
            print(f"[harvest] {j}/{len(recs)} len {len(p) + len(c)}", flush=True)
    for ids in generic_seqs(tok, n=256):
        cov_add(read_layers(model, ids, LAYERS), slice(8, None))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "means.npy", means)
    json.dump([{k: r[k] for k in ("beh", "cond", "uid", "k", "y", "role", "group")} | {"n_c": len(toks[i][1]), "n_p": len(toks[i][0])}
               for i, r in enumerate(recs)], open(OUT / "meta.json", "w"))
    for L in LAYERS:
        mu = s1[L] / n_tok
        cov = s2[L] / n_tok - torch.outer(mu, mu)
        torch.save(dict(mu=mu.float().cpu(), cov=cov.float().cpu(), n=n_tok, mean_norm=norm_sum[L] / n_tok), OUT / f"cov_{L}.pt")
    print(f"[harvest] {len(recs)} completions, {n_tok} covariance tokens", flush=True)
    print("ONP_HARVEST_DONE", flush=True)


# ---------------------------------------------------------------- stage 2b: closed-form directions and scoring

def load_acts():
    meta = json.load(open(OUT / "meta.json"))
    return meta, np.load(OUT / "means.npy", mmap_mode="r")


def groups(meta, role, beh=None):
    """-> {group: (hack row indices, honest row indices)} for groups of that role that have both classes."""
    g = {}
    for i, m in enumerate(meta):
        if m["role"] == role and m["y"] is not None and (beh is None or m["beh"] == beh):
            g.setdefault(m["group"], ([], []))[1 - m["y"]].append(i)
    return {k: v for k, v in g.items() if v[0] and v[1]}


def pair_diffs(meta, X, role="fit"):
    """Environment-balanced list of per-prompt (mean hack - mean honest) differences. -> [G, d], env name per row."""
    D, envs = [], []
    for k, (h, n) in groups(meta, role).items():
        D.append(X[h].mean(0) - X[n].mean(0))
        envs.append(k.split("|")[0])
    return np.stack(D), envs


def env_mean(D, envs):
    return np.mean([D[[e == b for e in envs]].mean(0) for b in sorted(set(envs))], 0)


def sigma_inv(L, shrink=0.05):
    c = torch.load(OUT / f"cov_{L}.pt")
    cov = c["cov"].double()
    ev, U = torch.linalg.eigh(cov)
    ev = ev.clamp_min(0) + shrink * ev.clamp_min(0).mean()
    return (U / ev) @ U.T, cov, c


def matched_auc(meta, s, role, beh, n_boot=1000, seed=0):
    """Mean over prompts of P(score of a hack completion > score of an honest completion of the same prompt)."""
    per = []
    for k, (h, n) in groups(meta, role, beh).items():
        d = s[h][:, None] - s[n][None, :]
        per.append(float((d > 0).mean() + 0.5 * (d == 0).mean()))
    if not per:
        return None
    per, rng = np.array(per), np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return dict(auc=float(per.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_prompts=len(per))


def plain_auc(meta, s, role, beh):
    idx = [i for i, m in enumerate(meta) if m["role"] == role and m["beh"] == beh and m["y"] is not None]
    y = np.array([meta[i]["y"] for i in idx])
    if y.min() == y.max():
        return None
    d = s[idx][y == 1][:, None] - s[idx][y == 0][None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def score_table(meta, X, dirs):
    """-> {direction: {role/env: matched AUROC dict}} plus 'val_mean' and 'held_mean'."""
    res = {}
    for name, d in dirs.items():
        s = np.asarray(X @ d, np.float64)
        r = {}
        for role, envs in (("val", FIT), ("held", HELD)):
            for b in envs:
                m = matched_auc(meta, s, role, b)
                if m:
                    m["plain"] = plain_auc(meta, s, role, b)
                    r[f"{role}/{b}"] = m
            v = [r[k]["auc"] for k in r if k.startswith(role + "/")]
            r[f"{role}_mean"] = float(np.mean(v)) if v else None
        res[name] = r
    return res


def closed_form(meta, X, L, seed=0):
    X = np.asarray(X, np.float32)
    D, envs = pair_diffs(meta, X)
    dom = env_mean(D, envs)
    Sinv, cov, _ = sigma_inv(L)
    dirs = {"dom": dom, "lda": (Sinv @ torch.tensor(dom, dtype=torch.float64)).numpy(),
            "sig": (cov @ torch.tensor(dom, dtype=torch.float64)).numpy()}
    # logistic probe on prompt-centred features (prompts with both classes only)
    rows, ys = [], []
    for k, (h, n) in groups(meta, "fit").items():
        mu = X[h + n].mean(0)
        rows += [X[i] - mu for i in h + n]
        ys += [1] * len(h) + [0] * len(n)
    try:
        from sklearn.linear_model import LogisticRegression
        R = np.stack(rows)
        sd = R.std(0) + 1e-6
        clf = LogisticRegression(C=0.01, max_iter=2000, fit_intercept=False).fit(R / sd, ys)
        dirs["probe"] = clf.coef_[0] / sd
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    for j in range(20):
        dirs[f"rnd{j}"] = rng.standard_normal(X.shape[1])
    return {k: (v / np.linalg.norm(v)).astype(np.float32) for k, v in dirs.items()}


def dirs_stage(a):
    meta, means = load_acts()
    n = {r: {b: (sum(m["role"] == r and m["beh"] == b and m["y"] == 1 for m in meta),
                 sum(m["role"] == r and m["beh"] == b and m["y"] == 0 for m in meta), len(groups(meta, r, b)))
             for b in list(FIT) + list(HELD)} for r in ("fit", "val", "held")}
    print("hack / honest / prompts with both, by role:", json.dumps(n), flush=True)
    out = {}
    for pi, pool in enumerate(POOLS):
        for li, L in enumerate(LAYERS):
            X = means[:, pi, li]
            dirs = closed_form(meta, X, L)
            res = score_table(meta, X, dirs)
            rnd = [res[k]["val_mean"] for k in res if k.startswith("rnd")], [res[k]["held_mean"] for k in res if k.startswith("rnd")]
            out[f"{pool}/{L}"] = {k: v for k, v in res.items() if not k.startswith("rnd")} | {
                "rnd_val_p95": float(np.percentile(np.abs(np.array(rnd[0]) - 0.5), 95)),
                "rnd_held_p95": float(np.percentile(np.abs(np.array(rnd[1]) - 0.5), 95))}
            if pool == "all":
                np.savez(OUT / f"dirs_cf_{L}.npz", **dirs)
            line = "  ".join(f"{k} {res[k]['val_mean']:.2f}/{res[k]['held_mean']:.2f}" for k in res if not k.startswith("rnd"))
            print(f"[dirs] {pool:9s} L{L:2d} val/held matched AUROC: {line}  (random |auc-.5| p95 {out[f'{pool}/{L}']['rnd_val_p95']:.2f}/{out[f'{pool}/{L}']['rnd_held_p95']:.2f})", flush=True)
    cand = [L for L in LAYERS if L <= a.max_layer]
    best = max(cand, key=lambda L: np.mean([out[f"all/{L}"][k]["val_mean"] for k in ("dom", "lda", "probe") if k in out[f"all/{L}"]]))
    out["layer"] = best
    json.dump(out, open(OUT / "dirs_cf.json", "w"), indent=1)
    print(f"[dirs] layer chosen on validation prompts of the fitting environments: {best}", flush=True)
    print("ONP_DIRS_DONE", flush=True)


# ---------------------------------------------------------------- stage 3: the vector

def steer_pairs(meta, toks, max_t):
    """Fit prompts with a hack and an honest completion that both fit in max_t tokens. -> {env: [(hack rows, honest rows)]}"""
    out = {}
    for k, (h, n) in groups(meta, "fit").items():
        h = [i for i in h if len(toks[i][0]) + len(toks[i][1]) <= max_t]
        n = [i for i in n if len(toks[i][0]) + len(toks[i][1]) <= max_t]
        if h and n:
            out.setdefault(k.split("|")[0], []).append((h, n))
    return out


def train(a):
    meta, means = load_acts()
    L = a.layer or json.load(open(OUT / "dirs_cf.json"))["layer"]
    li = LAYERS.index(L)
    X = torch.tensor(np.asarray(means[:, 0, li]), device=DEV)
    cf = np.load(OUT / f"dirs_cf_{L}.npz")
    Sinv, _, c = sigma_inv(L)
    Sinv = Sinv.float().to(DEV)
    r_ref = a.r_frac * c["mean_norm"]
    D, envs = pair_diffs(meta, np.asarray(means[:, 0, li]))
    wts = np.array([1.0 / envs.count(e) for e in envs])
    D, wts = torch.tensor(D, device=DEV), torch.tensor(wts / wts.sum(), device=DEV, dtype=torch.float32)

    model, tok = load_model(a.model)
    if a.ckpt:
        checkpoint_blocks(model, L)
    recs = {(r["group"], r["k"]): r for r in load_recs(a.name)}
    toks = [tokenise(tok, recs[(m["group"], m["k"])]) for m in meta]
    pairs = steer_pairs(meta, toks, a.max_t)
    print(f"[train] layer {L}, r_ref {r_ref:.2f} (mean norm {c['mean_norm']:.1f}); steer prompts per env: "
          f"{ {e: len(v) for e, v in pairs.items()} }; AUC pairs {len(D)}", flush=True)
    gen = [torch.tensor(s, device=DEV) for s in generic_seqs(tok)]
    rng = np.random.default_rng(a.seed)
    if a.init == "dom":
        w0 = torch.tensor(cf["dom"])
    else:
        w0 = torch.tensor(rng.standard_normal(X.shape[1]), dtype=torch.float32)
        w0 = w0 / w0.norm()
    w = (a.init_scale * w0).to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=a.lr)
    base_lp = {}

    def lp0(i):
        if i not in base_lp:
            with torch.no_grad():
                base_lp[i] = float(completion_lp(model, toks[i][0], toks[i][1]))
        return base_lp[i]

    def auc_term(v):
        dl = D @ v
        return (torch.nn.functional.softplus(-dl / (dl.std() + 1e-8)) * wts).sum(), float(((dl > 0).float() * wts).sum())

    def kl_term(v, n):
        ids = torch.stack([gen[j] for j in rng.choice(len(gen), n, replace=False)])
        with torch.no_grad():
            p0 = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 8:].float(), -1)
        with add_vec(model, L, v):
            p1 = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[:, 8:].float(), -1)
        return (p0.exp() * (p0 - p1)).sum(-1).mean()

    def diag(v):
        s = (X @ v).cpu().numpy()
        t = score_table(meta, means[:, 0, li], {"v": v.detach().cpu().numpy()})["v"]
        return dict(val=t["val_mean"], held=t["held_mean"], norm=float(v.norm()), mahal=float(v @ Sinv @ v),
                    cos_dom=float(torch.nn.functional.cosine_similarity(v.cpu(), torch.tensor(cf["dom"]), 0)),
                    cos_lda=float(torch.nn.functional.cosine_similarity(v.cpu(), torch.tensor(cf["lda"]), 0)))

    env_names, hist = sorted(pairs), []
    print(f"[train] mahal^2 at r_ref: dom {float(r_ref**2 * torch.tensor(cf['dom']).to(DEV) @ Sinv @ torch.tensor(cf['dom']).to(DEV)):.1f}, "
          f"lda {float(r_ref**2 * torch.tensor(cf['lda']).to(DEV) @ Sinv @ torch.tensor(cf['lda']).to(DEV)):.1f}, "
          f"random {float(r_ref**2 * torch.tensor(cf['rnd0']).to(DEV) @ Sinv @ torch.tensor(cf['rnd0']).to(DEV)):.1f}", flush=True)
    for step in range(a.steps + 1):
        v = r_ref * w
        rec = dict(step=step)
        if step % a.every == 0 or step == a.steps:
            rec.update(diag(v.detach()))
        if step == a.steps:
            hist.append(rec)
            print("[train] " + " ".join(f"{k} {x:.3f}" if isinstance(x, float) else f"{k} {x}" for k, x in rec.items()), flush=True)
            break
        opt.zero_grad()
        gh = gn = 0.0
        for _ in range(a.pairs):
            env = env_names[rng.integers(len(env_names))]
            hs, ns = pairs[env][rng.integers(len(pairs[env]))]
            for i, sign in ((hs[rng.integers(len(hs))], 1.0), (ns[rng.integers(len(ns))], -1.0)):
                b = lp0(i)
                with add_vec(model, L, r_ref * w):
                    g = completion_lp(model, toks[i][0], toks[i][1]) - b
                (-sign * g / a.pairs).backward()
                if sign > 0:
                    gh += float(g) / a.pairs
                else:
                    gn += float(g) / a.pairs
        v = r_ref * w
        la, acc = auc_term(v)
        lk = kl_term(v, a.kl_bs)
        lm = v @ Sinv @ v
        (a.lam1 * la + a.lam2 * lk + a.lam3 * lm).backward()
        gnorm = float(w.grad.norm())
        opt.step()
        if DEV != "cpu" and torch.cuda.is_available():
            rec["gb"] = torch.cuda.max_memory_allocated() / 2**30
        rec.update(g_hack=gh, g_honest=gn, gain=gh - gn, auc_loss=float(la), fit_acc=acc, kl=float(lk), man=float(lm), gnorm=gnorm)
        hist.append(rec)
        if step % 5 == 0 or "val" in rec:
            print("[train] " + " ".join(f"{k} {x:.3f}" if isinstance(x, float) else f"{k} {x}" for k, x in rec.items()), flush=True)
        if step % 50 == 49:
            torch.save(dict(v=(r_ref * w).detach().cpu(), layer=L, step=step + 1, args={k: x for k, x in vars(a).items() if isinstance(x, (int, float, str, bool, type(None)))}), OUT / f"v_{a.tag}.pt")
    torch.save(dict(v=(r_ref * w).detach().cpu(), layer=L, step=a.steps, args={k: x for k, x in vars(a).items() if isinstance(x, (int, float, str, bool, type(None)))}, hist=hist), OUT / f"v_{a.tag}.pt")
    json.dump(hist, open(OUT / f"hist_{a.tag}.json", "w"))
    print("ONP_TRAIN_DONE", flush=True)


def export(a):
    """Steering files for the patched server: the trained vector, and norm-matched difference-of-means and random
    vectors, each with both signs."""
    z = torch.load(OUT / f"v_{a.tag}.pt")
    v, L = z["v"].float(), z["layer"]
    cf = np.load(OUT / f"dirs_cf_{L}.npz")
    D = OUT / "steer"
    D.mkdir(exist_ok=True)
    vecs = {a.tag: v}
    if a.baselines:
        vecs.update(dom=torch.tensor(cf["dom"]) * v.norm(), rnd=torch.tensor(cf["rnd0"]) * v.norm())
    for k, x in vecs.items():
        for sign, s in (("pos", 1.0), ("neg", -1.0)):
            torch.save(dict(v=(s * x).float(), layer=L), D / f"{k}_{sign}.pt")
    torch.save(dict(v=torch.zeros_like(v), layer=L), D / "base.pt")
    print(f"[export] layer {L}, norm {float(v.norm()):.2f}: {sorted(p.name for p in D.glob('*.pt'))}")


def setvec(a):
    """Make cache/flag/onp/steer/<vec>.pt the live vector of the steered server and wait for its acknowledgement."""
    import os
    import time
    z = torch.load(OUT / "steer" / f"{a.vec}.pt")
    ack = a.file + ".loaded"
    before = os.stat(ack).st_mtime_ns if os.path.exists(ack) else None
    torch.save(z, a.file + ".tmp")
    os.replace(a.file + ".tmp", a.file)
    for _ in range(120):
        time.sleep(0.5)
        if os.path.exists(ack) and os.stat(ack).st_mtime_ns != before:
            print(f"[setvec] {a.vec} live: norm {float(z['v'].norm()):.3f}, server says {open(ack).read().strip()}", flush=True)
            return
    raise SystemExit(f"[setvec] no acknowledgement for {a.vec}: {open(a.file + '.err').read() if os.path.exists(a.file + '.err') else 'no error file'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["harvest", "dirs", "train", "export", "setvec"])
    ap.add_argument("--name", default="onp")
    ap.add_argument("--model", default=Q.MODEL)
    ap.add_argument("--max-c", type=int, default=6000, help="completion tokens read per record in harvest")
    ap.add_argument("--max-layer", type=int, default=40)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=2, help="hack/honest pairs per step for the steering term")
    ap.add_argument("--max-t", type=int, default=1792, help="longest sequence (tokens) used by the steering term")
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--kl-bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4, help="Adam step per coordinate, in units of r_ref (width 5120: about 0.036 per step in norm)")
    ap.add_argument("--lam1", type=float, default=0.3)
    ap.add_argument("--lam2", type=float, default=1.0)
    ap.add_argument("--lam3", type=float, default=0.0)
    ap.add_argument("--r-frac", type=float, default=0.08, help="unit of v as a fraction of the mean activation norm")
    ap.add_argument("--init", choices=["dom", "rand"], default="dom")
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--vec", default="base")
    ap.add_argument("--file", default="/workspace/steer_current.pt")
    a = ap.parse_args()
    dict(harvest=harvest, dirs=dirs_stage, train=train, export=export, setvec=setvec)[a.stage](a)

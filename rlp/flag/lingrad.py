"""Linearised steering fit, GPU half: one gradient pass, so that every fit after it is laptop work (rlp.flag.linfit).

For each prefix of the model's own rollouts from which its free continuations sometimes hack and sometimes do not
(rlp.flag.resample runs rs, rs2, rs3: 236 prefixes), and each labelled continuation c (first 128 tokens, at least
16, as in rlp.flag.fitw), with a vector added at the input of block L at every position:

  win   at the vector v0 (default 0): lp_c = sum_t log p(c_t | prefix, c_<t) and its gradient with respect to the
        added vector, split into prompt+prefix positions (g_pre) and continuation positions (g_cont). One forward
        and one backward pass per prefix: the continuations of a prefix form one batch with a separate perturbation
        per row, so the gradient of the summed log-probability with respect to row b is row b's own gradient.
  sec   forward only: lp_c at +20 and -20 along named directions and at 0, for the first-order check.
  all   win at 0 on every prefix; linfit.fit (writes cache/flag/onp/linfit_vec.pt); sec on the first --n-check
        prefixes for the fitted vector and the reference directions; linfit.check, which prints LINFIT_CHECK_PASS
        or LINFIT_CHECK_FAIL. Then LINGRAD_DONE.

    python -m rlp.flag.lingrad --stage all                                   # the whole pod step
    python -m rlp.flag.lingrad --stage win --point 0                         # gradients only
    python -m rlp.flag.lingrad --stage sec --dirs lin,sprobe,noise,fitv --n-prefix 60
    python -m rlp.flag.lingrad --stage win --point +lin --n-prefix 60        # fallback: re-linearise (also -lin)

Files: cache/flag/onp/lingrad/win_<point>/NNNN.pt and lingrad/sec/NNNN.pt, one per prefix in a fixed shuffled order
(--n-prefix takes the first N of that order, so subsets nest). Existing files are skipped, so a rerun resumes.
Each record: key, prompt (environment|condition|item: the unit for cross-validation), sid, beh, frac, ph (hack rate
of the prefix over all labelled continuations), n_p, and per kept continuation y, j, n (tokens), lp, g_pre, g_cont.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import gf27 as G, linfit as LF, q38 as Q, resample as RS

MIN_TOK = 16


def prefixes(tok, names, n_tok, max_t):
    """Mixed prefixes in resample order. Same filter as fitw.examples; every labelled continuation of >= 16 tokens kept."""
    out = []
    for nm in names.split(","):
        for (sid, cut), r in sorted(RS.load(nm).items()):
            ys = [c["y"] for c in r["conts"] if c["y"] is not None]
            if not ys or not (0 < sum(ys) < len(ys)):
                continue
            p = tok(Q.prompt(tok, r["messages"]), add_special_tokens=False)["input_ids"] + list(r["prefix_ids"])
            if len(p) + n_tok > max_t:
                continue
            conts = []
            for c in r["conts"]:
                if c["y"] is None:
                    continue
                ids = tok(c["cont"], add_special_tokens=False)["input_ids"][:n_tok]
                if len(ids) >= MIN_TOK:
                    conts.append(dict(y=int(c["y"]), j=c["j"], ids=ids))
            if len({c["y"] for c in conts}) == 2:
                out.append(dict(key=f"{nm}|{sid}|{cut}", prompt=sid.rsplit("|", 1)[0], sid=sid, beh=r["beh"], frac=r.get("frac"),
                                ph=float(np.mean(ys)), p=p, conts=conts))
    return out


def ordered(ex, n=0, seed=0):
    return [ex[i] for i in np.random.default_rng(seed).permutation(len(ex))][:n or None]


class steer:
    """Add V[b] (+ delta[b, t] if given) to the input of block L. V: [B, d]; delta: [B, T, d] leaf."""
    def __init__(self, model, L, V, delta=None):
        self.blk, self.V, self.delta = G.blocks_of(model)[L], V, delta

    def __enter__(self):
        def pre(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            add = self.V[:, None, :] if self.delta is None else self.V[:, None, :] + self.delta
            y = x + add.to(x.dtype)
            if args:
                return (y,) + tuple(args[1:]), kwargs
            return args, dict(kwargs, hidden_states=y)
        self.h = self.blk.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *a):
        self.h.remove()


def lp_sums(model, ids, am, n_p, lens, rows=8):
    """Summed log-probability of each row's continuation (tokens n_p .. n_p + lens[b]). -> [B] float32.
    Only the continuation positions are unembedded, a few rows at a time."""
    h = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    T, out = int(max(lens)), []
    ln = torch.tensor(lens, device=ids.device)
    for b in range(0, ids.shape[0], rows):
        lg = F.log_softmax(model.lm_head(h[b:b + rows, n_p - 1:n_p - 1 + T]).float(), -1)
        lp = torch.gather(lg, 2, ids[b:b + rows, n_p:n_p + T, None])[..., 0]
        out.append((lp * (torch.arange(T, device=ids.device)[None] < ln[b:b + rows, None])).sum(1))
    return torch.cat(out)


def _is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def grad_prefix(model, L, v0, p, conts, pad):
    """-> lp [B], g_pre [B, d], g_cont [B, d] (cpu, float32) at the vector v0 [d]. Halves the batch if memory runs out."""
    try:
        ids, am = G.pad_batch([p + c for c in conts], pad, model.device)
        delta = torch.zeros(ids.shape[0], ids.shape[1], v0.numel(), device=ids.device, dtype=torch.float32, requires_grad=True)
        with steer(model, L, v0[None].expand(ids.shape[0], -1), delta):
            lp = lp_sums(model, ids, am, len(p), [len(c) for c in conts])
            lp.sum().backward()
        g = delta.grad
        return lp.detach().cpu(), g[:, :len(p)].sum(1).cpu(), g[:, len(p):].sum(1).cpu()
    except Exception as e:
        if not _is_oom(e) or len(conts) == 1:
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    h = len(conts) // 2
    a, b = grad_prefix(model, L, v0, p, conts[:h], pad), grad_prefix(model, L, v0, p, conts[h:], pad)
    return tuple(torch.cat(x) for x in zip(a, b))


def sec_prefix(model, L, dirs, p, conts, pad, budget):
    """-> per continuation {cond: lp} for cond in '0', '<dir>+', '<dir>-' (forward only)."""
    d = next(iter(dirs.values())).numel() if dirs else G.width(model)
    conds = [("0", torch.zeros(d, device=model.device))] + [(f"{k}{s}", sg * v) for k, v in dirs.items() for s, sg in (("+", 1.0), ("-", -1.0))]
    jobs = [(ci, name, v) for ci in range(len(conts)) for name, v in conds]
    rows = [dict() for _ in conts]
    per, k = max(1, budget // (len(p) + max(len(c) for c in conts))), 0
    with torch.no_grad():
        while k < len(jobs):
            jb = jobs[k:k + per]
            try:
                ids, am = G.pad_batch([p + conts[ci] for ci, _, _ in jb], pad, model.device)
                with steer(model, L, torch.stack([v for _, _, v in jb]).float()):
                    lp = lp_sums(model, ids, am, len(p), [len(conts[ci]) for ci, _, _ in jb]).tolist()
            except Exception as e:
                if not _is_oom(e) or per == 1:
                    raise
                per = max(1, per // 2)                       # out of memory: smaller batches for the rest of this prefix
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            for (ci, name, _), x in zip(jb, lp):
                rows[ci][name] = x
            k += len(jb)
    return rows


def _save(rec, f):
    torch.save(rec, str(f) + ".tmp")
    os.replace(str(f) + ".tmp", f)


def _base(e):
    return dict(key=e["key"], prompt=e["prompt"], sid=e["sid"], beh=e["beh"], frac=e["frac"], ph=e["ph"], n_p=len(e["p"]),
                y=[c["y"] for c in e["conts"]], j=[c["j"] for c in e["conts"]], n=[len(c["ids"]) for c in e["conts"]])


def _progress(tag, i, n, e, t0, done):
    el = time.time() - t0
    gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    print(f"[lingrad] {tag} {i + 1}/{n} {e['key']} ({len(e['p'])}+{len(e['conts'])}x{max(len(c['ids']) for c in e['conts'])} tok) "
          f"{el / 60:.1f} min, eta {el / max(done, 1) * (n - i - 1) / 60:.0f} min, peak {gb:.1f} GB", flush=True)


def run_win(model, tok, a, ex, point):
    onp = Path(a.onp)
    d = G.width(model)
    v0 = torch.zeros(d, device=model.device)
    if point != "0":
        x = LF.resolve(point[1:], a.layer, onp)
        v0 = torch.tensor((1.0 if point[0] == "+" else -1.0) * a.alpha * LF.unit(x), device=model.device, dtype=torch.float32)
    D = onp / "lingrad" / f"win_{point}"
    D.mkdir(parents=True, exist_ok=True)
    print(f"[lingrad] win at {point}: {len(ex)} prefixes, {sum(len(e['conts']) for e in ex)} continuations -> {D}", flush=True)
    t0, done = time.time(), 0
    for i, e in enumerate(ex):
        f = D / f"{i:04d}.pt"
        if f.exists():
            continue
        lp, g_pre, g_cont = grad_prefix(model, a.layer, v0, e["p"], [c["ids"] for c in e["conts"]], G.pad_id(tok))
        _save(_base(e) | dict(lp=lp.tolist(), g_pre=g_pre, g_cont=g_cont, point=point), f)
        done += 1
        if done == 1 or i % 10 == 0:
            _progress("win", i, len(ex), e, t0, done)


def run_sec(model, tok, a, ex, names):
    onp = Path(a.onp)
    dirs = {k: torch.tensor(a.alpha * LF.unit(LF.resolve(k, a.layer, onp)), device=model.device, dtype=torch.float32) for k in names}
    D = onp / "lingrad" / "sec"
    D.mkdir(parents=True, exist_ok=True)
    print(f"[lingrad] sec: {len(ex)} prefixes, directions {list(dirs)} at +/-{a.alpha:g} -> {D}", flush=True)
    t0, done = time.time(), 0
    for i, e in enumerate(ex):
        f = D / f"{i:04d}.pt"
        if f.exists() and set(torch.load(f, map_location="cpu", weights_only=False).get("dirs", [])) >= set(dirs):
            continue
        rows = sec_prefix(model, a.layer, dirs, e["p"], [c["ids"] for c in e["conts"]], G.pad_id(tok), a.budget)
        _save(_base(e) | dict(lp=rows, dirs=list(dirs)), f)
        done += 1
        if done == 1 or i % 10 == 0:
            _progress("sec", i, len(ex), e, t0, done)


def available(names, layer, onp):
    out = []
    for k in names:
        try:
            LF.resolve(k, layer, onp)
            out.append(k)
        except (FileNotFoundError, KeyError):
            print(f"[lingrad] direction {k!r} not on this machine; skipped", flush=True)
    return out


def main(a):
    from . import onp as O
    model, tok = G.load_model(a)
    O.checkpoint_blocks(model, a.layer)             # recompute blocks L.. in the backward pass (memory)
    ex = prefixes(tok, a.rs, a.n_tok, a.max_t)
    print(f"[lingrad] {len(ex)} mixed prefixes from {len({e['prompt'] for e in ex})} prompts; by environment "
          f"{ {b: sum(e['beh'] == b for e in ex) for b in sorted({e['beh'] for e in ex})} }", flush=True)
    names = [k for k in a.dirs.split(",") if k]
    if a.stage == "win":
        run_win(model, tok, a, ordered(ex, a.n_prefix), a.point)
    elif a.stage == "sec":
        run_sec(model, tok, a, ordered(ex, a.n_prefix or a.n_check), available(names, a.layer, a.onp))
    else:
        run_win(model, tok, a, ordered(ex, a.n_prefix), "0")
        LF.fit(LF.args("fit", onp=a.onp, layer=a.layer, alpha=a.alpha))
        run_sec(model, tok, a, ordered(ex, a.n_check), available(names, a.layer, a.onp))
        LF.check(LF.args("check", onp=a.onp, layer=a.layer, alpha=a.alpha))
    print("LINGRAD_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    G.add_model_args(ap)
    ap.add_argument("--stage", choices=("all", "win", "sec"), default="all")
    ap.add_argument("--point", default="0", help="win only: 0, or +NAME / -NAME for 20 * unit(NAME) (see linfit.resolve)")
    ap.add_argument("--dirs", default="lin,sprobe,noise,fitv", help="sec: directions scored at +/- alpha")
    ap.add_argument("--rs", default="rs,rs2,rs3")
    ap.add_argument("--onp", default=str(LF.OUT))
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--n-tok", type=int, default=128)
    ap.add_argument("--max-t", type=int, default=4096)
    ap.add_argument("--n-prefix", type=int, default=0, help="first N prefixes of the fixed order (0 = all)")
    ap.add_argument("--n-check", type=int, default=60, help="prefixes for the first-order check")
    ap.add_argument("--budget", type=int, default=24000, help="sec: padded tokens per forward pass")
    main(ap.parse_args())

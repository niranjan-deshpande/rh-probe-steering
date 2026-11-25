"""Gradient-descent direction with a causal loss at natural values (GPU).   python -m rlp.flag.train --seeds 0 1 2

Setting. A minimal pair is a shared conversation prefix followed by one assistant turn that is either the hack (y+)
or its matched non-hack (y-). Everything is the frozen base model; there is no fine-tune.

Intervention. At hidden_states[L], on the tokens of the deciding turn (the assistant header and the teacher-forced
turn), the component along the unit direction d is overwritten with a target value; everything orthogonal to d is
left alone. Targets come from the data: m_H and m_N are the mean activations on hack turns and on non-hack turns of
the training pairs, g(d) = (m_H - m_N).d is the natural gap along d, and for a dose t

    toward the hack    value = m_N.d + t g(d)        (t = 1 is the mean hack value)
    toward honest      value = m_H.d - t g(d)        (t = 1 is the mean non-hack value)

If d does not separate hack turns from non-hack turns, g(d) = 0, the two interventions coincide and the loss cannot
fall. So a direction that scores well must also be one whose projection reads the hack.

Loss, with lp(y) the summed log-probability of the turn's tokens and lp_0 the same without intervention:

    A_H = [lp_H(y+) - lp_0(y+)] - [lp_H(y-) - lp_0(y-)]      setting the hack value makes the hack turn more likely
    A_N = [lp_N(y-) - lp_0(y-)] - [lp_N(y+) - lp_0(y+)]      setting the honest value makes the hack turn less likely,
                                                             including while the hack is being written (teacher-forced)
    L   = softplus(-beta A_H) + softplus(-beta A_N) + mu * [max(0, KL_H - tau) + max(0, KL_N - tau)]

KL_c is the mean per-token KL(p_0 || p_c) on generic text with the value set on every token at dose 1.
The dose t is sampled from DOSES each step. d starts as a random unit vector and is renormalised after every step.

Prefix tokens are never intervened on, so the prefix is run once without gradients and its key/value cache is shared
by the six rows per pair (y+ / y- under none / H / N). Only the deciding turn goes through the model with gradients.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from . import model as M, render as R, synth as S

OUT = config.CACHE / "flag"
DOSES = (1.0, 2.0, 4.0)
EMPTY_THINK = "<think>\n\n</think>\n\n"


def build_items(tok, rows, pairs, max_turn=None):
    """-> list of dict(prefix, head, pos, neg: token id lists; contrast, family, group). The assistant header (with
    the empty think block that generation with thinking off starts from) is `head`; pos / neg end with <|im_end|>.
    max_turn keeps only the first max_turn tokens of each turn (gradients through long turns do not fit in memory;
    the two turns differ from their first few tokens, so the decision is inside the kept part)."""
    items, skipped = [], 0
    for p in pairs:
        rp, rn = rows[p["pos"]], rows[p["neg"]]
        k, tools = p["k"], S.tools_for(rp)
        pre = R.render(tok, rp["messages"][:k], tools=tools, gen=True)
        hs = pre.rfind(R.A_OPEN)
        prefix, head = pre[:hs], pre[hs:]
        comps = []
        for r in (rp, rn):
            full = R.render(tok, r["messages"][:k + 1], tools=tools)
            if not full.startswith(prefix + R.A_OPEN):
                comps = None
                break
            body = full[len(prefix) + len(R.A_OPEN):]
            if body.startswith(EMPTY_THINK):
                body = body[len(EMPTY_THINK):]
            elif body.startswith(R.TH_OPEN):
                head = R.A_OPEN                 # the turn has written reasoning: it starts at <think>, as with thinking on
            body = body.rstrip("\n")
            if not body.endswith(R.T_END):
                comps = None
                break
            comps.append(body)
        if comps is None or comps[0] == comps[1]:
            skipped += 1
            continue
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        items.append(dict(prefix=enc(prefix), head=enc(head), pos=enc(comps[0])[:max_turn], neg=enc(comps[1])[:max_turn],
                          contrast=p["contrast"], family=p["family"], group=p["group"]))
    return items, skipped


class Engine:
    def __init__(self, model, layer, pad_id=0):
        self.model, self.L, self.pad = model, layer, pad_id
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.f = torch.float64 if self.dtype == torch.float64 else torch.float32      # precision for log-probs

    def _prefix(self, items):
        """Left-padded prefixes -> (cache, mask [B, P])."""
        P = max(len(it["prefix"]) for it in items)
        ids = torch.full((len(items), P), self.pad, dtype=torch.long)
        am = torch.zeros((len(items), P), dtype=torch.long)
        for b, it in enumerate(items):
            n = len(it["prefix"])
            ids[b, P - n:] = torch.tensor(it["prefix"])
            am[b, P - n:] = 1
        with torch.no_grad():
            out = self.model(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev), use_cache=True)
        return out.past_key_values, am.to(self.dev)

    def _active(self, items, conds):
        """Rows = item x branch (pos, neg) x cond. -> ids [R, A], mask [R, A], comp mask [R, A] (positions whose
        token counts toward lp), cond index [R], branch [R]."""
        nc = len(conds)
        seqs = []
        for it in items:
            for br in ("pos", "neg"):
                seqs += [(it["head"], it[br])] * nc
        A = max(len(h) + len(c) for h, c in seqs)
        ids = torch.full((len(seqs), A), self.pad, dtype=torch.long)
        am = torch.zeros((len(seqs), A), dtype=torch.bool)
        cm = torch.zeros((len(seqs), A), dtype=torch.bool)
        for r, (h, c) in enumerate(seqs):
            ids[r, :len(h) + len(c)] = torch.tensor(h + c)
            am[r, :len(h) + len(c)] = True
            cm[r, len(h):len(h) + len(c)] = True
        cond = torch.arange(nc).repeat(2 * len(items))
        return ids.to(self.dev), am.to(self.dev), cm.to(self.dev), cond.to(self.dev)

    def lp_rows(self, items, d=None, values=None, conds=("none",), capture=False):
        """Summed log-prob of each turn under each condition. values: {cond: scalar tensor} for conds other than
        "none". -> lp [n items, 2 (pos, neg), n conds]; with capture=True also the mean hidden_states[L] over the
        active tokens of each row, [n items, 2, d] (taken from the "none" rows)."""
        cache, pm = self._prefix(items)
        nc = len(conds)
        cache.batch_repeat_interleave(2 * nc)
        ids, am, cm, cond = self._active(items, conds)
        full_mask = torch.cat([pm.repeat_interleave(2 * nc, 0), am.long()], 1)
        grabbed = {}

        def fn(h):
            if capture:
                grabbed["h"] = h.detach()
            if d is None:
                return h
            val = torch.zeros(h.shape[0], device=h.device, dtype=self.f)
            on = torch.zeros(h.shape[0], device=h.device, dtype=torch.bool)
            for ci, c in enumerate(conds):
                if c != "none":
                    val = torch.where(cond == ci, values[c].to(self.f), val)
                    on |= cond == ci
            return M.overwrite(h, d, val, am & on[:, None])
        with M.set_along(self.model, self.L, fn):
            hid = self.model.model(input_ids=ids, attention_mask=full_mask, past_key_values=cache, use_cache=True).last_hidden_state
        # log-probs only at the positions that predict a turn token (full-vocabulary logits for every row and
        # position do not fit in memory at the real model size)
        sel = cm[:, 1:]                                           # position j-1 predicts token j
        rows_i = torch.nonzero(sel)[:, 0]
        hs, tgt = hid[:, :-1][sel], ids[:, 1:][sel]
        tok_lp = []
        for a in range(0, hs.shape[0], 1024):
            lg = self.model.lm_head(hs[a:a + 1024]).to(self.f)
            tok_lp.append(lg.gather(-1, tgt[a:a + 1024, None])[:, 0] - torch.logsumexp(lg, -1))
        tok_lp = torch.cat(tok_lp)
        lp = torch.zeros(ids.shape[0], device=self.dev, dtype=self.f).index_add(0, rows_i, tok_lp).view(len(items), 2, nc)
        if capture:
            h = grabbed["h"].to(self.f)
            mean = (h * am[..., None]).sum(1) / am.sum(1, keepdim=True)
            return lp, mean.view(len(items), 2, nc, -1)[:, :, list(conds).index("none")]
        return lp

    def kl_rows(self, seqs, d, vH, vN):
        """Mean per-token KL(p_0 || p_c) on generic sequences for c in (H, N). seqs: list of token id lists."""
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        am = torch.zeros((len(seqs), T), dtype=torch.bool)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s)
            am[b, :len(s)] = True
        ids, am = ids.to(self.dev), am.to(self.dev)
        with torch.no_grad():
            p0 = F.log_softmax(self.model(input_ids=ids, attention_mask=am.long(), use_cache=False).logits.to(self.f), -1)
        out = []
        for v in (vH, vN):
            with M.set_along(self.model, self.L, lambda h: M.overwrite(h, d, v, am)):
                pc = F.log_softmax(self.model(input_ids=ids, attention_mask=am.long(), use_cache=False).logits.to(self.f), -1)
            kl = (p0.exp() * (p0 - pc)).sum(-1)
            out.append((kl * am).sum() / am.sum())
        return out


def class_means(eng, items, bs=8, n=512, seed=0):
    """m_H, m_N: mean hidden_states[L] over the deciding turn's tokens, for hack and non-hack turns, averaged within
    each contrast and then across contrasts."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))[:n]
    acc = {}
    for i in range(0, len(idx), bs):
        batch = [items[j] for j in idx[i:i + bs]]
        _, mean = eng.lp_rows(batch, capture=True)
        for it, mm in zip(batch, mean):
            a = acc.setdefault(it["contrast"], [0, torch.zeros_like(mm)])
            a[0] += 1
            a[1] += mm
    per = torch.stack([v / n_ for n_, v in acc.values()])       # [contrasts, 2, d]
    m = per.mean(0)
    return m[0], m[1]


def margins(lp):
    """lp [B, 2, 3] with conds (none, H, N) -> A_H, A_N [B]."""
    aH = (lp[:, 0, 1] - lp[:, 0, 0]) - (lp[:, 1, 1] - lp[:, 1, 0])
    aN = (lp[:, 1, 2] - lp[:, 1, 0]) - (lp[:, 0, 2] - lp[:, 0, 0])
    return aH, aN


def targets(d, mH, mN, t=1.0):
    g = (mH - mN) @ d
    return mN @ d + t * g, mH @ d - t * g, g


def sample_batch(items, by_contrast, bs, rng):
    names = list(by_contrast)
    return [items[rng.choice(by_contrast[names[rng.integers(len(names))]])] for _ in range(bs)]


def train(eng, items, generic, mH, mN, steps=200, bs=8, accum=2, lr=2e-3, beta=0.1, mu=10.0, tau=0.1, kl_bs=4, seed=0,
          log=print, log_every=10):
    """-> (d [dm] float32 unit vector on cpu, history)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(mH.shape[0], generator=g).to(eng.dev, eng.f)
    theta = (theta / theta.norm()).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["contrast"], []).append(i)
    hist, t0 = [], time.time()
    for step in range(steps):
        opt.zero_grad()
        rec = dict(step=step, loss_H=0.0, loss_N=0.0, acc_H=0.0, acc_N=0.0, A_H=0.0, A_N=0.0)
        gH = gN = None
        t = float(DOSES[rng.integers(len(DOSES))])
        for _ in range(accum):
            batch = sample_batch(items, by, bs, rng)
            d = theta / theta.norm()
            vH, vN, gap = targets(d, mH, mN, t)
            lp = eng.lp_rows(batch, d, {"H": vH, "N": vN}, conds=("none", "H", "N"))
            aH, aN = margins(lp)
            lH, lN = F.softplus(-beta * aH).mean() / accum, F.softplus(-beta * aN).mean() / accum
            if step % log_every == 0:                       # agreement between the two halves' gradients
                a, = torch.autograd.grad(lH, theta, retain_graph=True)
                b, = torch.autograd.grad(lN, theta, retain_graph=True)
                gH = a if gH is None else gH + a
                gN = b if gN is None else gN + b
            (lH + lN).backward()
            rec["loss_H"] += float(lH.detach()); rec["loss_N"] += float(lN.detach())
            rec["acc_H"] += float((aH > 0).float().mean()) / accum; rec["acc_N"] += float((aN > 0).float().mean()) / accum
            rec["A_H"] += float(aH.detach().mean()) / accum; rec["A_N"] += float(aN.detach().mean()) / accum
        d = theta / theta.norm()
        vH, vN, gap = targets(d, mH, mN, 1.0)
        ks = [generic[j] for j in rng.choice(len(generic), size=min(kl_bs, len(generic)), replace=False)]
        klH, klN = eng.kl_rows(ks, d, vH, vN)
        lk = mu * (F.relu(klH - tau) + F.relu(klN - tau))
        if lk.requires_grad and float(lk.detach()) > 0:
            lk.backward()
        rec.update(kl_H=float(klH.detach()), kl_N=float(klN.detach()), gap=float(gap.detach()), dose=t)
        if gH is not None:
            rec["grad_cos"] = float(F.cosine_similarity(gH, gN, dim=0))
        opt.step()
        sched.step()
        with torch.no_grad():
            theta /= theta.norm()
        hist.append(rec)
        if step % log_every == 0 or step == steps - 1:
            log(f"[train s{seed}] step {step:3d} dose {t:.0f} loss_H {rec['loss_H']:.3f} loss_N {rec['loss_N']:.3f} "
                f"acc {rec['acc_H']:.2f}/{rec['acc_N']:.2f} A {rec['A_H']:+.2f}/{rec['A_N']:+.2f} gap {rec['gap']:+.2f} "
                f"kl {rec['kl_H']:.3f}/{rec['kl_N']:.3f} gcos {rec.get('grad_cos', float('nan')):+.2f} "
                f"| {(time.time() - t0) / (step + 1):.1f}s/step", flush=True)
    return (theta / theta.norm()).detach().float().cpu().numpy(), hist


def causal_eval(eng, items, dirs, mH, mN, bs=8, doses=(1.0, 2.0, 4.0), n=200, seed=0):
    """For each named direction and dose: mean A_H, A_N and the fraction of pairs with A > 0, on `items`.
    A direction and its negative give the same intervention, so the sign of the direction does not matter here."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))[:n]
    out = {}
    for name, v in dirs.items():
        d = torch.tensor(v, device=eng.dev, dtype=eng.f)
        d = d / d.norm()
        out[name] = {}
        for t in doses:
            aH, aN = [], []
            for i in range(0, len(idx), bs):
                batch = [items[j] for j in idx[i:i + bs]]
                with torch.no_grad():
                    vH, vN, gap = targets(d, mH, mN, t)
                    a, b = margins(eng.lp_rows(batch, d, {"H": vH, "N": vN}, conds=("none", "H", "N")))
                aH += a.tolist(); aN += b.tolist()
            aH, aN = np.array(aH), np.array(aN)
            out[name][t] = dict(A_H=float(aH.mean()), A_N=float(aN.mean()), acc_H=float((aH > 0).mean()),
                                acc_N=float((aN > 0).mean()), gap=float(gap))
    return out


def load_generic(tok, n=64, max_len=160):
    seqs = json.load(open(config.DATA / "flag" / "generic.json"))[:n]
    out = []
    for s in seqs:
        text = R.render(tok, [{"role": "user", "content": s["instruction"]}, {"role": "assistant", "content": s["output"]}])
        out.append(tok(text, add_special_tokens=False)["input_ids"][:max_len])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-turn", type=int, default=192)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--harvest", default="turn")
    ap.add_argument("--model", default=None)
    ap.add_argument("--eval-only", action="store_true", help="skip training; causal comparison of saved directions")
    ap.add_argument("--eval-n", type=int, default=96)
    a = ap.parse_args()
    info = json.load(open(OUT / "dirs_info.json"))[a.harvest]["info"]
    L = a.layer or info["layer"]
    model, tok = M.load(a.model)
    rows = S.load_rows()
    C = S.load_contrasts()
    tr, _ = build_items(tok, rows, S.pairs(rows, C, families=S.TRAIN, split="train"), a.max_turn)
    va, _ = build_items(tok, rows, S.pairs(rows, C, families=S.TRAIN, split="validation"), a.max_turn)
    he, _ = build_items(tok, rows, S.pairs(rows, C, families=S.HELD, split="validation"), a.max_turn)
    print(f"[train] layer {L}; pairs train {len(tr)}, validation {len(va)}, held-out families {len(he)}", flush=True)
    eng = Engine(model, L, tok.pad_token_id or 0)
    generic = load_generic(tok)
    with torch.no_grad():
        mH, mN = class_means(eng, tr)
    print(f"[train] |m_H - m_N| = {float((mH - mN).norm()):.2f}, |m_N| = {float(mN.norm()):.2f}", flush=True)
    torch.save(dict(mH=mH.cpu(), mN=mN.cpu(), layer=L), OUT / "class_means.pt")
    dirs, hists = {}, {}
    if a.eval_only:
        z = np.load(OUT / "dirs_gd.npz")
        delta = (mH - mN).float().cpu().numpy()
        dirs = {k: z[k] * (1.0 if float(z[k] @ delta) >= 0 else -1.0) for k in z.files if k.startswith("gd_s")}
    for s in ([] if a.eval_only else a.seeds):
        d, hist = train(eng, tr, generic, mH, mN, steps=a.steps, bs=a.bs, accum=a.accum, seed=s)
        torch.cuda.empty_cache()
        if float(torch.tensor(d) @ (mH - mN).float().cpu()) < 0:      # the loss cannot tell d from -d: orient hack-positive
            d = -d
        dirs[f"gd_s{s}"], hists[s] = d, hist
        np.savez(OUT / "dirs_gd.npz", layer=L, **dirs)
        json.dump(hists, open(OUT / "train_hist.json", "w"))
    if len(dirs) > 1:
        dirs["gd_mean"] = np.mean(list(dirs.values()), 0)
        dirs["gd_mean"] /= np.linalg.norm(dirs["gd_mean"])
        if not a.eval_only:
            np.savez(OUT / "dirs_gd.npz", layer=L, **dirs)
    # causal comparison with the read-out directions at the same layer
    base = np.load(OUT / f"dirs_{a.harvest}.npz")
    layers = list(np.load(OUT / "synth_acts.npz")["layers"])
    cmp = dict(dirs)
    for k in ("gf_pool", "dom_swe", "dom_ours", "probe_ours", "rnd0", "rnd1", "rnd2", "rnd3"):
        cmp[k] = base[k][layers.index(L)]
    kw = dict(bs=a.bs, doses=(1.0,), n=a.eval_n)
    res = {"validation": causal_eval(eng, va, cmp, mH, mN, **kw), "held_out": causal_eval(eng, he, cmp, mH, mN, **kw)}
    S_ = np.stack([dirs[k] for k in sorted(dirs) if k.startswith("gd_s")])
    res["seed_cosines"] = (S_ @ S_.T).round(3).tolist()
    res["cos_with_readout"] = {k: {r: round(float(dirs[k] @ cmp[r]), 3) for r in ("gf_pool", "dom_ours", "probe_ours")} for k in dirs}
    print("seed cosines", res["seed_cosines"], "\ncosine with read-out directions", res["cos_with_readout"], flush=True)
    json.dump(res, open(OUT / "causal_eval.json", "w"), indent=1)
    for part, r in res.items():
        if part not in ("validation", "held_out"):
            continue
        print(f"== causal effect at natural values, {part} pairs (dose 1): A_H / A_N in nats, fraction > 0")
        for k, v in r.items():
            e = v[1.0]
            print(f"  {k:12s} A_H {e['A_H']:+7.2f} ({e['acc_H']:.2f})  A_N {e['A_N']:+7.2f} ({e['acc_N']:.2f})  gap {e['gap']:+.2f}")


if __name__ == "__main__":
    main()

"""Project labelled rollouts onto the directions (GPU).   python -m rlp.flag.read --rollouts ib_qwen38 [--n 150]

The reader model is teacher-forced on each rollout (cut into windows, see rollouts.py). For every labelled passage
and every direction, the per-token projection of hidden_states[L] is pooled over the passage's tokens by max and by
mean. Rollouts are processed in a fixed shuffled order and saved one file each, so a run can stop early and resume.

Output: cache/flag/read/<set>/<trace_id>.npz with
    proj   [n passages, n layers, n directions, 2]  (max, mean)   float32
    act    [n passages, d]   mean activation at the chosen layer   float16 (to score later directions by mean-pool)
    ntok, mi, channel, dropped (passages not covered by any window)
and cache/flag/read/<set>/dirs.json naming the direction and layer axes.
"""
import argparse
import json
import time

import numpy as np
import torch

from .. import config
from . import model as M, render as R, rollouts as RO, synth as S

OUT = config.CACHE / "flag"
CH = ("reasoning", "content", "tool_calls")
PRIMARY = "turn"          # harvest whose validation separation picks the layer (set after step 3, before our own rollouts)


def direction_bank(gd=True):
    """-> (names, layers, D [n layers, n dirs, d] float32, chosen layer). Read-out directions from both harvests at
    every harvested layer; gradient-descent directions only at the layer they were trained at (zeros elsewhere)."""
    layers = [int(x) for x in np.load(OUT / "synth_acts.npz")["layers"]]
    names, cols = [], []
    for h in ("close", "turn", "reason"):
        if not (OUT / f"dirs_{h}.npz").exists():
            continue
        z = np.load(OUT / f"dirs_{h}.npz")
        for k in z.files:
            if k.startswith("rnd") and h != "close":
                continue                                     # one set of random directions is enough
            names.append(k if k.startswith("rnd") else f"{h}:{k}")
            cols.append(z[k])
    info = json.load(open(OUT / "dirs_info.json"))
    L = info[PRIMARY if PRIMARY in info else "close"]["info"]["layer"]
    p = OUT / "dirs_gd.npz"
    if gd and p.exists():
        z = np.load(p)
        Lg = int(z["layer"])
        # The causal loss is the same for d and -d, so a trained direction comes out with either sign. Orient each
        # seed so that hack turns project higher (positive natural gap), then average the oriented seeds.
        cm = torch.load(OUT / "class_means.pt")
        delta = (cm["mH"] - cm["mN"]).float().numpy()
        seeds = {k: z[k] * (1.0 if float(z[k] @ delta) >= 0 else -1.0) for k in z.files if k.startswith("gd_s")}
        if len(seeds) > 1:
            m = np.mean(list(seeds.values()), 0)
            seeds["gd_mean"] = m / np.linalg.norm(m)
        for k, d in seeds.items():
            v = np.zeros((len(layers), d.shape[-1]), np.float32)
            v[layers.index(Lg)] = d
            names.append(k)
            cols.append(v)
    return names, layers, np.stack(cols, 1).astype(np.float32), L


def read_rollout(model, tok, r, D, layers, L, max_tokens):
    dev = next(model.parameters()).device
    msgs = r["messages"]
    head = 2 if msgs[0]["role"] == "system" else 1           # the minimal scaffold has no system message
    wins = RO.windows(tok, msgs, max_tokens, head=head)
    where = RO.assign(wins, len(msgs), head=head)
    keys = sorted(r["passages"])
    out = {k: None for k in keys}
    Dt = torch.tensor(D, device=dev)                         # [nL, nD, d]
    ntok_total = 0
    for wi, (a, b) in enumerate(wins):
        mine = [k for k in keys if where.get(k[0]) == wi]
        if not mine:
            continue
        wm = msgs[:head] + msgs[a:b]
        o = RO.tokenize(tok, wm)
        if o is None:
            continue
        ids, blks = o
        ntok_total += len(ids)
        bm = {(blk["mi"] if blk["mi"] < head else a + blk["mi"] - head): blk for blk in blks}
        got = M.run_capture(model, layers, input_ids=torch.tensor([ids], device=dev))
        for (mi, ch) in mine:
            sp = bm.get(mi, {}).get(ch + "_tok")
            if sp is None:
                continue
            t0, t1 = sp
            pr = torch.stack([got[Lx][0, t0:t1].float() @ Dt[li].T for li, Lx in enumerate(layers)])     # [nL, T, nD]
            pooled = torch.stack([pr.max(1).values, pr.mean(1)], -1)                                    # [nL, nD, 2]
            out[(mi, ch)] = (pooled.cpu().numpy(), got[L][0, t0:t1].float().mean(0).cpu().numpy().astype(np.float16), t1 - t0)
        del got
    return keys, out, ntok_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="ib_qwen38")
    ap.add_argument("--own", action="store_true", help="--rollouts names one of our own rollout files (rlp.flag.agent)")
    ap.add_argument("--n", type=int, default=0, help="stop after this many rollouts (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    out = OUT / "read" / a.rollouts
    out.mkdir(parents=True, exist_ok=True)
    names, layers, D, L = direction_bank()
    json.dump(dict(names=names, layers=layers, layer=L), open(out / "dirs.json", "w"))
    if a.own:
        from . import own
        rs = own.load(a.rollouts, judged=False)
    else:
        rs = RO.load(S.GF / a.rollouts)
    order = np.random.default_rng(0).permutation(len(rs))
    if a.n:
        order = order[:a.n]
    model, tok = M.load(a.model)
    t0, done, toks = time.time(), 0, 0
    for i in order:
        r = rs[i]
        f = out / f"{r['trace_id']}.npz"
        if f.exists():
            continue
        try:
            keys, res, nt = read_rollout(model, tok, r, D, layers, L, a.max_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            keys, res, nt = read_rollout(model, tok, r, D, layers, L, a.max_tokens // 2)
        kept = [k for k in keys if res[k] is not None]
        if kept:
            np.savez(f, proj=np.stack([res[k][0] for k in kept]), act=np.stack([res[k][1] for k in kept]),
                     ntok=np.array([res[k][2] for k in kept]), mi=np.array([k[0] for k in kept]),
                     channel=np.array([CH.index(k[1]) for k in kept]), dropped=len(keys) - len(kept))
        done += 1
        toks += nt
        el = time.time() - t0
        print(f"[read] {done}/{len(order)} rollouts, {toks / 1e6:.2f}M tokens, {toks / el:.0f} tok/s, "
              f"{len(keys) - len(kept)} of {len(keys)} passages dropped in this one", flush=True)


if __name__ == "__main__":
    main()

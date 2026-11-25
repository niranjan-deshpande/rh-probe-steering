"""Layer-36 completion means for further base_rate runs (new environments), plus pooled per-token scores.

For each labelled completion of cache/flag/base_rate/<name>: the mean residual activation at the layer over the
completion tokens (same reading as onp.harvest, pool "all"), and for a few fixed directions the per-token score
w . h_t pooled four ways (mean, max, mean of the top 5% of tokens, max of 64-token window means). The pooled scores
test whether a hack that sits in a few tokens is diluted by averaging.

    python -m rlp.flag.harvest36 --name env2 --behs run_summary check_claim ...        # new environments
    python -m rlp.flag.harvest36 --name onp --behs save_tamper fake_pass fake_cite --tag onp_held

Output: CACHE/flag/onp/h36_<tag>.npz (X [N, d], pooled [N, n_dirs, 4]) and h36_<tag>.json (meta rows like onp's).
"""
import argparse
import json

import numpy as np
import torch

from ..manif import envs as E
from . import onp, q38 as Q

POOLS = ("mean", "max", "top5", "win64")


def load(name, behs, conds=None):
    out = []
    for f in sorted((onp.SRC / name).glob("*__*.jsonl")):
        if f.name.endswith(".labels.jsonl"):
            continue
        beh, cond = f.name[:-len(".jsonl")].split("__")
        if beh not in behs or (conds and cond not in conds):
            continue
        lab = {(x["uid"], x.get("k", 0)): x for x in map(json.loads, open(str(f).replace(".jsonl", ".labels.jsonl")))}
        for r in map(json.loads, open(f)):
            x = lab.get((r["uid"], r.get("k", 0)))
            if x is None or not r.get("raw"):
                continue
            ok = bool(x.get("valid")) and not x["trunc"]
            out.append(dict(beh=beh, cond=cond, uid=r["uid"], k=r.get("k", 0), messages=r["messages"], raw=r["raw"],
                            y=(int(x["hit"]) if ok else None), role="new", group=f"{beh}|{cond}|{r['uid']}"))
    return out


def main(a):
    recs = load(a.name, a.behs, a.conds)
    if a.both_only:                       # keep only prompts that have a hack and an honest completion
        keep = {i for h, n in onp.groups(recs, "new").values() for i in h + n}
        recs = [r for i, r in enumerate(recs) if i in keep]
    model, tok = onp.load_model(a.model)
    cf = np.load(onp.OUT / f"dirs_cf_{a.layer}.npz")
    names = [k for k in ("probe", "lda", "dom") if k in cf]
    W = torch.tensor(np.stack([cf[k] for k in names]), device=onp.DEV, dtype=torch.float32)
    toks = [onp.tokenise(tok, r, a.max_c) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: len(toks[i][0]) + len(toks[i][1]))
    X = np.zeros((len(recs), W.shape[1]), np.float32)
    P = np.zeros((len(recs), len(names), len(POOLS)), np.float32)
    for j, i in enumerate(order):
        p, c, _ = toks[i]
        h = onp.read_layers(model, p + c, (a.layer,))[a.layer][len(p):].float()
        X[i] = h.mean(0).cpu().numpy()
        s = h @ W.T                                                   # [T, n_dirs]
        k = max(1, int(0.05 * len(s)))
        win = s.T.unfold(1, min(64, len(s)), 16).mean(-1).max(-1).values if len(s) >= 2 else s.mean(0)
        P[i] = torch.stack([s.mean(0), s.max(0).values, s.topk(k, 0).values.mean(0), win], 1).cpu().numpy()
        if j % 100 == 0:
            print(f"[h36] {j}/{len(recs)} len {len(p) + len(c)}", flush=True)
    np.savez(onp.OUT / f"h36_{a.tag or a.name}.npz", X=X, pooled=P, dirs=np.array(names), pools=np.array(POOLS))
    json.dump([{k: r[k] for k in ("beh", "cond", "uid", "k", "y", "role", "group")} for r in recs],
              open(onp.OUT / f"h36_{a.tag or a.name}.json", "w"))
    print(f"[h36] {len(recs)} completions", flush=True)
    print("H36_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--behs", nargs="+", required=True)
    ap.add_argument("--conds", nargs="*")
    ap.add_argument("--tag")
    ap.add_argument("--model", default=Q.MODEL)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--max-c", type=int, default=6000)
    ap.add_argument("--both-only", action="store_true")
    main(ap.parse_args())

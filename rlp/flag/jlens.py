"""Jacobian-lens readout of direction vectors at one layer of Qwen3.8-27B, on CPU (post hoc, 2026-11-24).

Readout for a direction d at layer L: logits = W_U · RMSNorm(J_L d), with J_L from eyes-ml/Qwen3.8-27B_jacobian-lens.
Tokens are ranked by raw logit and by a z-score against 200 random Gaussian directions. Run jlens_fetch first.

    .venv/bin/python -m rlp.flag.jlens [--layer 36]
"""
import argparse, json
import numpy as np, torch
from tokenizers import Tokenizer

D = "cache/flag/jlens"


def main(a):
    tok = Tokenizer.from_file(D + "/dl_model/tokenizer.json")
    J = torch.load(D + "/dl_lens/Qwen3.8-27B_jacobian_lens.pt", map_location="cpu", weights_only=True, mmap=True)["J"][a.layer].float()
    if a.transpose:
        J = J.T
    g = torch.frombuffer(bytearray(open(D + "/norm_weight.bin", "rb").read()), dtype=torch.bfloat16).float()
    W = torch.frombuffer(np.memmap(D + "/lm_head.bin", dtype=np.uint8, mode="r+"), dtype=torch.bfloat16).view(-1, 5120)
    fw = torch.load("cache/flag/onp/fitw_w2.pt", map_location="cpu")
    cf, wc = np.load("cache/flag/onp/dirs_cf_36.npz"), np.load("cache/flag/onp/write_cands_36.npz")
    dirs = {"v (fitted steering vector)": fw["v"].float(), "w (fitted detector)": fw["w"].float(),
            "Sigma.probe": torch.tensor(wc["Stok_probe"]).float(), "mean difference": torch.tensor(cf["dom"]).float(),
            "probe": torch.tensor(cf["probe"]).float(), "rnd0": torch.tensor(cf["rnd0"]).float(), "rnd1": torch.tensor(cf["rnd1"]).float()}
    gen = torch.Generator().manual_seed(0)
    R = torch.randn(200, 5120, generator=gen)

    def read(V):                                   # V [n, d] -> logits [n, vocab]
        U = V @ J.T
        U = U / U.pow(2).mean(-1, keepdim=True).sqrt() * g
        out = torch.empty(V.shape[0], W.shape[0])
        for i in range(0, W.shape[0], 32768):
            out[:, i:i + 32768] = U @ W[i:i + 32768].float().T
        return out

    LR = read(R)
    mu, sd = LR.mean(0), LR.std(0) + 1e-6
    s = lambda ids: " | ".join(repr(tok.decode([int(i)])) for i in ids)
    res = {}
    for name, v in dirs.items():
        l = read(v[None])[0]
        z = (l - mu) / sd
        res[name] = {"top_logit": [tok.decode([int(i)]) for i in l.topk(a.k).indices], "bottom_logit": [tok.decode([int(i)]) for i in (-l).topk(a.k).indices],
                     "top_z": [tok.decode([int(i)]) for i in z.topk(a.k).indices], "bottom_z": [tok.decode([int(i)]) for i in (-z).topk(a.k).indices],
                     "max_z": float(z.max()), "min_z": float(z.min())}
        print(f"\n=== {name}   (max z {z.max():.1f}, min z {z.min():.1f}; random-direction max |z| median {LR.sub(mu).div(sd).abs().max(1).values.median():.1f})")
        print("  ADDED, top logit:   ", s(l.topk(a.k).indices))
        print("  ADDED, top z:       ", s(z.topk(a.k).indices))
        print("  SUBTRACTED, top logit:", s((-l).topk(a.k).indices))
        print("  SUBTRACTED, top z:    ", s((-z).topk(a.k).indices))
    json.dump(res, open(f"cache/flag/onp/jlens_L{a.layer}{'_T' if a.transpose else ''}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--transpose", action="store_true")
    main(ap.parse_args())

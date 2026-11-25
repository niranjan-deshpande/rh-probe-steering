"""Text-only controls for the on-policy detection result (results/05_onpolicy_directions.md, Stage 2).

The Qwen3.8-27B probe reads activations averaged over the completion, and the completion contains the hack. These
controls ask how much of the held-out matched AUROC a detector gets from the completion text alone, with the same
rows, the same fit / validation / held-out split and the same within-prompt matched AUROC.

  tfidf   word 1-2-gram TF-IDF of the completion, logistic regression on prompt-centred features
  small   mean activations of a small unrelated reader model (Qwen3-0.6B) over the completion, same probe recipe

    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.onp_text embed     # small-model features (local, MPS or CPU)
    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.onp_text score
"""
import argparse
import json

import numpy as np
import torch

from . import onp

SMALL = "Qwen/Qwen3-0.6B"
SMALL_LAYERS = (7, 14, 21, 28)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
ENVS = [b for b in list(onp.FIT) + list(onp.HELD)]


def embed(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device          # MPS crashes on this macOS / torch pair (MPSNDArray selector), so CPU is the default
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    recs = onp.load_recs()
    used = {i for role in ("fit", "val", "held") for h, n in onp.groups(recs, role).values() for i in h + n}
    out = np.zeros((len(recs), len(SMALL_LAYERS), model.config.hidden_size), np.float32)
    ids = []
    for i, r in enumerate(recs):
        if i not in used:           # rows outside prompts with both classes are never scored
            ids.append(([], []))
            continue
        p = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        p = tok(p, add_special_tokens=False)["input_ids"]
        ids.append((p, tok(r["raw"], add_special_tokens=False)["input_ids"][:a.max_c]))
    order = sorted(range(len(recs)), key=lambda i: len(ids[i][0]) + len(ids[i][1]))
    part = onp.OUT / "text_small_part.npz"          # resumable: slow on CPU (about 300 tokens per second)
    done = np.zeros(len(recs), bool)
    if part.exists():
        z = np.load(part)
        out, done = z["out"], z["done"]
    with torch.no_grad():
        for j, i in enumerate(order):
            p, c = ids[i]
            if not c or done[i]:
                continue
            if done.sum() % 50 == 49:
                np.savez(part, out=out, done=done)
            done[i] = True
            hs = model(torch.tensor([p + c], device=dev), output_hidden_states=True).hidden_states
            for li, L in enumerate(SMALL_LAYERS):
                out[i, li] = hs[L][0, len(p):].mean(0).float().cpu().numpy()
            if j % 100 == 0:
                print(f"[embed] {j}/{len(recs)} len {len(p) + len(c)}", flush=True)
    np.save(onp.OUT / "text_small.npy", out)
    print("EMBED_DONE", flush=True)


def centred(meta, X, role="fit"):
    rows, ys = [], []
    for k, (h, n) in onp.groups(meta, role).items():
        mu = X[h + n].mean(0)
        rows += [X[i] - mu for i in h + n]
        ys += [1] * len(h) + [0] * len(n)
    return np.stack(rows), np.array(ys)


def per_prompt(meta, s, role, beh):
    """-> {group: within-prompt AUROC}"""
    out = {}
    for k, (h, n) in onp.groups(meta, role, beh).items():
        d = s[h][:, None] - s[n][None, :]
        out[k] = float((d > 0).mean() + 0.5 * (d == 0).mean())
    return out


def table(meta, s):
    """-> {'val_mean', 'held_mean', 'held/<env>': {auc, lo, hi}, 'pp': per-prompt AUROC on held-out envs}"""
    r, pp = {}, {}
    for role, envs in (("val", onp.FIT), ("held", onp.HELD)):
        vals = []
        for b in envs:
            m = onp.matched_auc(meta, s, role, b)
            if m:
                r[f"{role}/{b}"] = m
                vals.append(m["auc"])
                if role == "held":
                    pp[b] = per_prompt(meta, s, role, b)
        r[f"{role}_mean"] = float(np.mean(vals))
    r["pp"] = pp
    return r


def fit_lr(meta, X, standardise):
    """Logistic regression on prompt-centred fitting rows; C chosen on validation prompts. -> scores for all rows, C."""
    from sklearn.linear_model import LogisticRegression
    R, y = centred(meta, X)
    sd = (R.std(0) + 1e-6) if standardise else np.ones(R.shape[1])
    best = None
    for C in C_GRID:
        w = LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(R / sd, y).coef_[0] / sd
        s = np.asarray(X @ w, np.float64)
        v = np.mean([m["auc"] for b in onp.FIT if (m := onp.matched_auc(meta, s, "val", b, n_boot=1))])
        if best is None or v > best[0]:
            best = (v, C, s)
    return best[2], best[1]


def paired(pa, pb, n=4000, seed=0):
    """Difference in held-out mean matched AUROC (a - b), bootstrap over prompts within environment."""
    rng = np.random.default_rng(seed)
    d = {e: np.array([pa[e][k] - pb[e][k] for k in sorted(pa[e])]) for e in pa}
    pt = float(np.mean([x.mean() for x in d.values()]))
    bs = [np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in d.values()]) for _ in range(n)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def score(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    meta, means = onp.load_acts()
    recs = onp.load_recs()
    assert all((m["group"], m["k"]) == (r["group"], r["k"]) for m, r in zip(meta, recs))
    res = {}

    # the 27B directions, recomputed from the saved files so that every row of the table uses the same code path
    X27 = np.asarray(means[:, 0, onp.LAYERS.index(36)], np.float32)
    dirs = np.load(onp.OUT / "dirs_cf_36.npz")
    for k in ("probe", "lda", "dom"):
        res[f"q38_{k}"] = table(meta, np.asarray(X27 @ dirs[k], np.float64))

    # TF-IDF of the completion text (vocabulary from fitting rows only)
    fit_rows = [i for i, m in enumerate(meta) if m["role"] == "fit"]
    for name, kw in (("tfidf_word", dict(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]")),
                     ("tfidf_char", dict(analyzer="char_wb", ngram_range=(2, 5)))):
        vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=30000, **kw).fit([recs[i]["raw"] for i in fit_rows])
        Xt = vec.transform([r["raw"] for r in recs]).toarray().astype(np.float32)
        s, C = fit_lr(meta, Xt, standardise=False)
        res[name] = table(meta, s) | {"C": C, "features": int(Xt.shape[1])}

    # small reader model
    f = onp.OUT / "text_small.npy"
    if f.exists():
        S = np.load(f)
        best = None
        for li, L in enumerate(SMALL_LAYERS):
            s, C = fit_lr(meta, S[:, li], standardise=True)
            t = table(meta, s) | {"C": C, "layer": L}
            res[f"small_L{L}"] = t
            if best is None or t["val_mean"] > best["val_mean"]:
                best = t
        res["small"] = best

    ref = res["q38_probe"]["pp"]
    print(f"{'detector':12s} {'val':>5s} {'held':>5s}  " + "  ".join(f"{b:>22s}" for b in onp.HELD if f'held/{b}' in res['q38_probe'])
          + "   q38 probe minus this (held mean)")
    for k, t in res.items():
        cells = "  ".join(f"{t[f'held/{b}']['auc']:.2f} [{t[f'held/{b}']['lo']:.2f}, {t[f'held/{b}']['hi']:.2f}]".rjust(22)
                          for b in onp.HELD if f"held/{b}" in t)
        if k != "q38_probe":
            t["probe_minus"] = paired(ref, t["pp"])
        d = "" if k == "q38_probe" else "   {:+.3f} [{:+.3f}, {:+.3f}]".format(*t["probe_minus"])
        extra = f"  C={t['C']}" if "C" in t else ""
        print(f"{k:12s} {t['val_mean']:5.2f} {t['held_mean']:5.2f}  {cells}{d}{extra}")
    for t in res.values():
        t.pop("pp", None)
    json.dump(res, open(onp.OUT / "text_control.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("embed", "score"))
    ap.add_argument("--model", default=SMALL)
    ap.add_argument("--max-c", type=int, default=6100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"embed": embed, "score": score}[a.cmd](a)

"""What text do the fitted directions respond to? Mean projection of existing texts onto w and v (pod, no judge).

For about 2,000 general texts and 500 of our own completions, the residual at the input of block 36 is projected on
unit directions (default: the fitted detector `w` and steering vector `v` of cache/flag/onp/fitw_w2.pt, with
Sigma.probe and the probe beside them for comparison). Per text the score is the mean projection over its scored
tokens; per token it is the projection itself. Written for informal reading:

    cache/flag/onp/textscore_36.json
      token_stats   mean and sd of the projection over all scored tokens, per direction
      by_source     mean text score per source (and hack / honest for our completions)
      texts_top / texts_bottom     the --top-texts highest and lowest texts per direction, with the start of the text
      tokens_top / tokens_bottom   the --top-tokens highest and lowest single tokens (at most --per-text from one
                                   text), each with the tokens before and after it
      texts         every text's score (source, id, label, n tokens, score per direction)

Texts, all already in the repo (nothing is generated):
    generic    data/flag/generic.json: 64 unrelated instruction / answer pairs, rendered as a chat; the answer is scored
    question   data/general_questions.json: 200 open questions, as plain text
    leetcode   data/leetcode_test_medhard.jsonl: 119 problem statements, as plain text
    goodfire   Goodfire's prepared conversations (their token ids), an equal number per cell, filling up to --n-general
    ours       --n-ours completions of cache/flag/base_rate/<--ours-run>, an equal number per environment, hack and
               honest; only the completion tokens are scored

    python -m rlp.flag.textscore            # prints TEXTSCORE_DONE
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import config
from . import gf27 as G, linfit as LF, q38 as Q

ONP = config.CACHE / "flag" / "onp"


def _spread(groups, n, rng):
    """Up to n items, taken round-robin over the groups (each group shuffled), so every group is represented."""
    pools = [[g[i] for i in rng.permutation(len(g))] for g in groups if g]
    out = []
    while len(out) < n and any(pools):
        for p in pools:
            if p and len(out) < n:
                out.append(p.pop())
    return out


def general_texts(tok, n, max_len, seed=0, records=None):
    """-> [dict(src, id, ids, s0, s1)]; s0:s1 are the scored tokens."""
    rng, out = np.random.default_rng(seed), []
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    f = config.DATA / "flag" / "generic.json"
    if f.exists():
        for i, s in enumerate(json.load(open(f))):
            p = enc(Q.prompt(tok, [{"role": "user", "content": s["instruction"]}]) + "\n</think>\n\n")
            c = enc(s["output"])[:max_len]
            out.append(dict(src="generic", id=f"generic/{i}", ids=p + c, s0=len(p), s1=len(p) + len(c)))
    f = config.DATA / "general_questions.json"
    if f.exists():
        for i, q in enumerate(json.load(open(f))):
            ids = enc(q if isinstance(q, str) else json.dumps(q))[:max_len]
            out.append(dict(src="question", id=f"question/{i}", ids=ids, s0=0, s1=len(ids)))
    f = config.DATA / "leetcode_test_medhard.jsonl"
    if f.exists():
        for i, r in enumerate(map(json.loads, open(f))):
            ids = enc(r["prompt"][-1]["content"])[:max_len]
            out.append(dict(src="leetcode", id=f"leetcode/{i}", ids=ids, s0=0, s1=len(ids)))
    out = [x for x in out if x["s1"] - x["s0"] >= 4]
    f = Path(records or G.RECORDS)
    if f.exists() and len(out) < n:
        cells = {}
        for r in G.load_records(f):
            cells.setdefault((r["metadata"]["dataset"], r["metadata"]["cell"]), []).append(r)
        for r in _spread([cells[k] for k in sorted(cells)], n - len(out), rng):
            ids = r["input_ids"][:max_len]
            out.append(dict(src="goodfire", id=r["id"], ids=ids, s0=0, s1=len(ids), label=r["metadata"]["cell"]))
    for name, path in (("generic", config.DATA / "flag" / "generic.json"), ("question", config.DATA / "general_questions.json"),
                       ("leetcode", config.DATA / "leetcode_test_medhard.jsonl"), ("goodfire", f)):
        if not Path(path).exists():
            print(f"[textscore] WARNING source {name!r} is missing on this machine ({path})", flush=True)
    return out[:n]


def our_texts(tok, run, n, max_c, seed=0, src=None):
    from . import mt_harvest as MH, onp as O
    recs = [r for r in MH.load_completions([run], src=src) if r["y"] is not None]
    by = {}
    for r in recs:
        by.setdefault((r["beh"], r["y"]), []).append(r)
    out = []
    for r in _spread([by[k] for k in sorted(by)], n, np.random.default_rng(seed)):
        p, c, _ = O.tokenise(tok, r, max_c)
        out.append(dict(src="ours", id=f"{r['beh']}|{r['cond']}|{r['uid']}|{r['k']}", ids=p + c, s0=len(p), s1=len(p) + len(c),
                        label=r["y"], env=r["beh"]))
    if not out:
        print(f"[textscore] WARNING no labelled completions under base_rate/{run}", flush=True)
    return out


def load_dirs(names, layer, onp):
    D = {}
    for k in names:
        try:
            D[k] = LF.unit(LF.resolve(k, layer, onp))
        except (FileNotFoundError, KeyError):
            print(f"[textscore] direction {k!r} not on this machine; skipped", flush=True)
    return D


def score_texts(model, tok, items, U, L, budget):
    """-> list over items of [n scored tokens, n dirs] float32 projections."""
    lens = [len(it["ids"]) for it in items]
    order = sorted(range(len(items)), key=lambda i: -lens[i])
    out, done = [None] * len(items), 0
    for bt in G.batches(order, lens, budget, solo=2048):
        H = G.read_batch(model, [items[i]["ids"] for i in bt], L, G.pad_id(tok))
        for b, i in enumerate(bt):
            out[i] = (H[b, items[i]["s0"]:items[i]["s1"]].float() @ U.T).cpu().numpy()
        done += len(bt)
        if done % 200 < len(bt):
            print(f"[textscore] {done}/{len(items)} texts", flush=True)
    return out


def summarise(tok, items, S, names, top_texts=40, top_tokens=100, per_text=3, before=14, after=5, chars=700):
    """S: per item [n, n dirs]. -> the JSON-ready report."""
    allv = np.concatenate(S)
    means = np.stack([s.mean(0) for s in S])
    rep = dict(dirs=names, n_texts={k: sum(it["src"] == k for it in items) for k in sorted({it["src"] for it in items})},
               n_tokens=int(len(allv)), token_stats={}, by_source={}, texts_top={}, texts_bottom={}, tokens_top={}, tokens_bottom={})
    text = lambda it: tok.decode(it["ids"][it["s0"]:it["s1"]])
    for d, name in enumerate(names):
        mu, sd = float(allv[:, d].mean()), float(allv[:, d].std())
        tm, ts = float(means[:, d].mean()), float(means[:, d].std() + 1e-12)
        rep["token_stats"][name] = dict(mean=mu, sd=sd, text_mean=tm, text_sd=ts)
        bs = {}
        for i, it in enumerate(items):
            key = it["src"] if it["src"] != "ours" else f"ours/{'hack' if it['label'] else 'honest'}"
            bs.setdefault(key, []).append(means[i, d])
        rep["by_source"][name] = {k: dict(mean=float(np.mean(v)), z=float((np.mean(v) - tm) / ts), n=len(v)) for k, v in sorted(bs.items())}
        order = np.argsort(-means[:, d])
        row = lambda i: dict(src=items[i]["src"], id=items[i]["id"], label=items[i].get("label"), score=float(means[i, d]),
                             z=float((means[i, d] - tm) / ts), n_tok=int(len(S[i])), text=text(items[i])[:chars])
        rep["texts_top"][name] = [row(int(i)) for i in order[:top_texts]]
        rep["texts_bottom"][name] = [row(int(i)) for i in order[::-1][:top_texts]]
        hi, lo = [], []                                      # each text offers its own per_text highest and lowest tokens
        for i, s in enumerate(S):
            o = np.argsort(s[:, d])
            hi += [(float(s[t, d]), i, int(t)) for t in o[-per_text:]]
            lo += [(float(s[t, d]), i, int(t)) for t in o[:per_text]]
        hi.sort(reverse=True)
        lo.sort()

        def tk(x):
            v, i, t = x
            it = items[i]
            p = it["s0"] + t
            return dict(src=it["src"], id=it["id"], label=it.get("label"), score=v, z=(v - mu) / (sd + 1e-12), pos=t,
                        token=tok.decode(it["ids"][p:p + 1]), before=tok.decode(it["ids"][max(0, p - before):p]),
                        after=tok.decode(it["ids"][p + 1:p + 1 + after]))
        rep["tokens_top"][name] = [tk(x) for x in hi[:top_tokens]]
        rep["tokens_bottom"][name] = [tk(x) for x in lo[:top_tokens]]
    rep["texts"] = [dict(src=it["src"], id=it["id"], label=it.get("label"), n_tok=int(len(s)),
                         score={n: float(means[i, d]) for d, n in enumerate(names)}) for i, (it, s) in enumerate(zip(items, S))]
    return rep


def main(a):
    D = load_dirs([k for k in a.dirs.split(",") if k], a.layer, a.onp)
    if not D:
        raise SystemExit("[textscore] no direction found")
    model, tok = G.load_model(a)
    items = general_texts(tok, a.n_general, a.max_len, a.seed) + our_texts(tok, a.ours_run, a.n_ours, a.max_c, a.seed)
    print(f"[textscore] {len(items)} texts, {sum(len(it['ids']) for it in items):,} tokens; directions {list(D)}", flush=True)
    U = torch.tensor(np.stack(list(D.values())), dtype=torch.float32, device=model.device)
    S = score_texts(model, tok, items, U, a.layer, a.budget)
    rep = summarise(tok, items, S, list(D), a.top_texts, a.top_tokens, a.per_text)
    out = Path(a.onp) / f"textscore_{a.layer}.json"
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
    for name in D:
        print(f"[textscore] {name}: " + "  ".join(f"{k} {v['z']:+.2f}" for k, v in rep["by_source"][name].items()) + "   (mean text score, in sd of text scores)")
    print(f"[textscore] -> {out}", flush=True)
    print("TEXTSCORE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    G.add_model_args(ap)
    ap.add_argument("--dirs", default="fitw,fitv,sprobe,probe", help="names for linfit.resolve; fitw / fitv are w / v of fitw_w2.pt")
    ap.add_argument("--onp", default=str(ONP))
    ap.add_argument("--n-general", type=int, default=2000)
    ap.add_argument("--n-ours", type=int, default=500)
    ap.add_argument("--ours-run", default="onp", help="directory under cache/flag/base_rate/")
    ap.add_argument("--max-len", type=int, default=1024, help="tokens kept of a general text")
    ap.add_argument("--max-c", type=int, default=3000, help="completion tokens kept of one of our completions")
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--top-texts", type=int, default=40)
    ap.add_argument("--top-tokens", type=int, default=100)
    ap.add_argument("--per-text", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())

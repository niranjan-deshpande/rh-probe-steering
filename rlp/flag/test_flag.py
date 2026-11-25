"""CPU tests for rlp.flag (tiny random Qwen3 with the real Qwen3-14B tokenizer).   python -m rlp.flag.test_flag"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import dirs as D, harvest as H, model as M, render as R, rollouts as RO, score as SC, synth as S, train as T

OK = True


def report(name, good, detail=""):
    global OK
    OK &= bool(good)
    print(("PASS " if good else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def tiny(tok, layers=4, dtype=torch.float32, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    m = Qwen3ForCausalLM(cfg).to(dtype).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_render(tok, rows):
    r = next(x for x in rows if x["metadata"]["cell"] == "exploit_hidden")
    text = R.render(tok, r["messages"], tools=S.tools_for(r))
    blks = R.blocks(text, R.norm_messages(r["messages"]))
    report("blocks: one per assistant message", blks is not None and len(blks) == sum(m["role"] == "assistant" for m in r["messages"]))
    b = next(x for x in blks if x["mi"] == 4)
    tc = text[b["tool_calls"][0]:b["tool_calls"][1]]
    report("blocks: tool-call span is the tool call", tc.startswith("<tool_call>\n{\"name\": \"str_replace\"") and tc.endswith("</tool_call><|im_end|>"), tc[:40])
    ids, loc = H.locate(tok, r)
    t0, t1 = loc[4][:2]
    dec = tok.decode(ids[t0:t1])
    report("harvest.locate: turn tokens = the turn after its header", dec.startswith(("<tool_call>", "<think>")) and dec.endswith("<|im_end|>"), repr(dec[:30]))
    report("harvest.locate: close tokens", tok.decode(ids[t1 - 3:t1]).endswith("</tool_call><|im_end|>"), repr(tok.decode(ids[t1 - 3:t1])))
    m = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "sure", "reasoning_content": "think A",
                                             "tool_calls": [{"type": "function", "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"}}]},
         {"role": "tool", "content": "a"}, {"role": "user", "content": "go on"},
         {"role": "assistant", "content": "done", "reasoning_content": "think B"}]
    text = R.render(tok, m)
    blks = R.blocks(text, R.norm_messages(m))
    got = [(text[b[c][0]:b[c][1]] if b[c] else None) for b in blks for c in ("reasoning", "content")]
    report("render keeps reasoning in turns before a later user message", got[0] == "think A" and got[2] == "think B" and got[1].strip() == "sure", str(got))


def test_items(tok, rows):
    C = S.load_contrasts(ours=False)
    ps = S.pairs(rows, C)
    n_by = {}
    for p in ps:
        n_by[p["contrast"]] = n_by.get(p["contrast"], 0) + 1
    report("pairs: every individual contrast has pairs", len(n_by) == len(S.individual(C)) and min(n_by.values()) >= 100, str(n_by))
    sub = [p for p in ps if p["contrast"] in ("i_rm_test", "i_expl_h", "i_peek_r")][::40][:9]
    items, skipped = T.build_items(tok, rows, sub)
    report("build_items: nothing skipped", skipped == 0 and len(items) == len(sub))
    it, p = items[0], sub[0]
    full = R.render(tok, rows[p["pos"]]["messages"][:p["k"] + 1], tools=S.tools_for(rows[p["pos"]]))
    mine = tok.decode(it["prefix"] + it["head"] + it["pos"])
    report("build_items: prefix + header + turn reproduces the conversation (plus the empty think block)",
           mine.replace(T.EMPTY_THINK, "") == full.replace(T.EMPTY_THINK, "").rstrip("\n"))
    report("build_items: turns end with <|im_end|> and differ", tok.decode(it["pos"][-1:]) == "<|im_end|>" and it["pos"] != it["neg"])
    return items


def test_engine(tok, items):
    model = tiny(tok, dtype=torch.float64)
    L = 2
    eng = T.Engine(model, L, tok.pad_token_id or 0)
    lp = eng.lp_rows(items[:3])
    ref = []
    for it in items[:3]:
        for br in ("pos", "neg"):
            seq = it["prefix"] + it["head"] + it[br]
            lg = F.log_softmax(model(input_ids=torch.tensor([seq])).logits[0], -1)
            n0 = len(it["prefix"]) + len(it["head"])
            ref.append(sum(float(lg[j - 1, seq[j]]) for j in range(n0, len(seq))))
    err = float((lp.flatten() - torch.tensor(ref, dtype=torch.float64)).abs().max())
    # left padding shifts positions, and the rotary embeddings are float32 even in a float64 model, so a padded batch
    # agrees with an unpadded forward to float32 precision only; a single unpadded item agrees exactly
    report("lp_rows (cached prefix, batched, padded) = full forward", err < 1e-4, f"max err {err:.1e}")
    one = float((eng.lp_rows(items[:1]).flatten() - torch.tensor(ref[:2], dtype=torch.float64)).abs().max())
    report("lp_rows (cached prefix, single item) = full forward exactly", one < 1e-9, f"max err {one:.1e}")

    torch.manual_seed(1)
    d = torch.randn(64, dtype=torch.float64)
    d /= d.norm()
    vH, vN = torch.tensor(1.7, dtype=torch.float64), torch.tensor(-0.4, dtype=torch.float64)
    lp3 = eng.lp_rows(items[:3], d, {"H": vH, "N": vN}, conds=("none", "H", "N"))
    report("intervention rows do not disturb the baseline rows", float((lp3[:, :, 0] - lp[:, :, 0]).abs().max()) < 1e-8)
    # reference: full forward with the same overwrite on header + turn tokens only
    it = items[1]
    seq = it["prefix"] + it["head"] + it["neg"]
    n0 = len(it["prefix"])
    mask = torch.zeros(1, len(seq), dtype=torch.bool)
    mask[0, n0:] = True
    seen = {}

    def fn(h):
        out = M.overwrite(h, d, vH, mask)
        seen["proj"], seen["orth"] = out @ d, (out - (out @ d)[..., None] * d) - (h - (h @ d)[..., None] * d)
        return out
    with M.set_along(model, L, fn):
        lg = F.log_softmax(model(input_ids=torch.tensor([seq])).logits[0], -1)
    ref = sum(float(lg[j - 1, seq[j]]) for j in range(n0 + len(it["head"]), len(seq)))
    report("overwrite sets the projection to the value on masked tokens only",
           float((seen["proj"][0, n0:] - 1.7).abs().max()) < 1e-9 and float(seen["orth"].abs().max()) < 1e-9)
    report("lp under intervention (cached) = full forward with the same overwrite", abs(float(lp3[1, 1, 1]) - ref) < 1e-8,
           f"{float(lp3[1, 1, 1]):.6f} vs {ref:.6f}")
    report("the intervention changes the log-probs", float((lp3[:, :, 1] - lp3[:, :, 0]).abs().min()) > 1e-6)

    # gradient of the loss w.r.t. the direction against finite differences
    mH, mN = T.class_means(eng, items, bs=3, n=9)
    theta = d.clone().requires_grad_(True)

    def loss_of(th):
        dd = th / th.norm()
        a, b, _ = T.targets(dd, mH, mN, 2.0)
        aH, aN = T.margins(eng.lp_rows(items[:3], dd, {"H": a, "N": b}, conds=("none", "H", "N")))
        return F.softplus(-0.1 * aH).mean() + F.softplus(-0.1 * aN).mean()
    g, = torch.autograd.grad(loss_of(theta), theta)
    e = torch.randn(64, dtype=torch.float64)
    e /= e.norm()
    eps = 1e-3                    # loss evaluations carry float32-level noise (see above), so the step cannot be tiny
    with torch.no_grad():
        fd = float(loss_of(d + eps * e) - loss_of(d - eps * e)) / (2 * eps)
    an = float(g @ e)
    report("loss gradient matches finite differences", abs(fd - an) < 3e-5 + 0.05 * abs(an), f"fd {fd:.3e} autograd {an:.3e}")
    seqs = [tok("The quick brown fox jumps over the lazy dog.")["input_ids"], tok("Paris is the capital of France, and")["input_ids"]]
    a, b, _ = T.targets(d, mH, mN)
    k1, k2 = eng.kl_rows(seqs, d, a, b)
    report("KL terms are non-negative and finite", float(k1) >= 0 and float(k2) >= 0 and np.isfinite(float(k1 + k2)), f"{float(k1):.4f} {float(k2):.4f}")
    report("class means differ between hack and non-hack turns", float((mH - mN).norm()) > 1e-3)


def test_train(tok, items):
    model = tiny(tok)
    eng = T.Engine(model, 2, tok.pad_token_id or 0)
    mH, mN = T.class_means(eng, items, bs=3, n=9)
    generic = [tok("The quick brown fox jumps over the lazy dog.")["input_ids"]] * 2
    d, hist = T.train(eng, items, generic, mH, mN, steps=30, bs=3, accum=1, lr=2e-2, beta=1.0, seed=0, log=lambda *a, **k: None)
    first = np.mean([h["loss_H"] + h["loss_N"] for h in hist[:5]])
    last = np.mean([h["loss_H"] + h["loss_N"] for h in hist[-5:]])
    report("training lowers the causal loss and keeps d a unit vector", last < first and abs(np.linalg.norm(d) - 1) < 1e-5, f"{first:.3f} -> {last:.3f}")
    report("history logs the gradient cosine between the two halves", "grad_cos" in hist[0] and np.isfinite(hist[0]["grad_cos"]))
    rnd = np.random.default_rng(0).standard_normal(64)
    ev = T.causal_eval(eng, items, {"gd": d, "rnd": rnd / np.linalg.norm(rnd)}, mH, mN, bs=3, doses=(2.0,), n=9)
    report("trained direction has a larger causal margin than a random one (tiny model)",
           ev["gd"][2.0]["A_H"] + ev["gd"][2.0]["A_N"] > ev["rnd"][2.0]["A_H"] + ev["rnd"][2.0]["A_N"],
           f"gd {ev['gd'][2.0]['A_H']:+.3f}/{ev['gd'][2.0]['A_N']:+.3f} rnd {ev['rnd'][2.0]['A_H']:+.3f}/{ev['rnd'][2.0]['A_N']:+.3f}")


def test_harvest_dirs(tok, rows):
    model = tiny(tok, layers=6)
    keep = {i for i, r in enumerate(rows) if int(r["group_id"].rsplit("_", 1)[1]) < 3}
    recs = [r for r in S.records(rows) if r["row"] in keep]
    layers = (2, 4)
    close, turn, reason, ok, bad = H.run(model, tok, rows, recs, layers=layers, token_budget=3000, log=lambda *a, **k: None)
    report("harvest: every record located", ok.all() and not bad, f"{int(ok.sum())}/{len(recs)}")
    r = recs[5]
    ids, loc = H.locate(tok, rows[r["row"]])
    hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    t0, t1 = loc[r["mi"]][:2]
    err = max(float((hs[L][0, t0:t1].mean(0) - torch.tensor(turn[5, li].astype(np.float32))).abs().max()) for li, L in enumerate(layers))
    report("harvest: stored turn mean = output_hidden_states mean (batched, padded, early exit)", err < 2e-2, f"max err {err:.1e}")

    # direction recovery on planted data with the real record structure
    recs = S.records(rows)
    rng = np.random.default_rng(0)
    d = 48
    u = D.unit(rng.standard_normal(d))
    Cn = list(S.individual(S.load_contrasts()))
    y = D.any_label(recs, Cn)
    fam = np.array([r["family"] for r in recs])
    famv = {f: rng.standard_normal(d) * 3 for f in set(fam)}
    X = rng.standard_normal((len(recs), d)) + np.stack([famv[f] for f in fam]) + 1.5 * (y == 1)[:, None] * u
    acts = {"close": X[:, None, :].astype(np.float32)}
    dd, info = D.build(acts, recs, [20], "close", log=lambda *a, **k: None)
    cs = {k: float(dd[k][0] @ u) for k in ("gf_pool", "dom_swe", "dom_ours", "probe_ours")}
    report("dirs: difference of means and probe recover a planted direction despite family offsets", min(cs.values()) > 0.8, str({k: round(v, 2) for k, v in cs.items()}))
    tab = D.synth_table(acts, recs, [20], dd, "close")
    f = "tz_tamper_scorefile_pos_v1"
    report("dirs: held-out family is separated; random directions are not", tab[f][20]["dom_ours"] > 0.75 and abs(np.median([tab[f][20][f"rnd{i}"] for i in range(50)]) - 0.5) < 0.1,
           f"dom_ours {tab[f][20]['dom_ours']:.2f}")
    a, b = rng.standard_normal(50), rng.standard_normal(70) - 0.3
    b[:5] = a[:5]
    naive = np.mean([(x > z) + 0.5 * (x == z) for x in a for z in b])
    report("auroc equals the pairwise definition, ties included", abs(D.auroc(a, b) - naive) < 1e-12)


def test_rollouts(tok):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    passages = {}
    for i in range(12):
        msgs.append({"role": "assistant", "content": f"step {i} " + "word " * 30, "reasoning_content": f"thinking {i} " + "hmm " * 40,
                     "tool_calls": [{"type": "function", "function": {"name": "python", "arguments": {"code": f"print({i})"}}}]})
        for ch in ("reasoning", "content", "tool_calls"):
            passages[(len(msgs) - 1, ch)] = dict(consensus="negative", statuses=[], categories=[], passage_n=0, chars=0)
        msgs.append({"role": "tool", "content": f"{i}\n" + "out " * 20})
    msgs = R.norm_messages(msgs)
    wins = RO.windows(tok, msgs, 700)
    where = RO.assign(wins, len(msgs))
    report("windows: several, each within budget, never starting on a tool message",
           len(wins) > 2 and all(len(tok(R.render(tok, msgs[:2] + msgs[a:b]), add_special_tokens=False)["input_ids"]) <= 700 for a, b in wins)
           and all(msgs[a]["role"] != "tool" for a, b in wins), str(wins))
    report("windows: every assistant message is covered", all(k[0] in where for k in passages))
    model = tiny(tok, layers=6)
    layers = [2, 4]
    Dm = D.unit(np.random.default_rng(0).standard_normal((2, 5, 64))).astype(np.float32)
    from . import read as RD
    r = dict(messages=msgs, passages=passages)
    keys, out, nt = RD.read_rollout(model, tok, r, Dm, layers, 4, 700)
    full, _, _ = RD.read_rollout(model, tok, r, Dm, layers, 4, 100000)
    report("read: every passage scored", all(out[k] is not None for k in keys))
    k0 = keys[1]                                          # a passage in the first window: identical context either way
    whole = RD.read_rollout(model, tok, r, Dm, layers, 4, 100000)[1]
    report("read: windowed = whole-rollout projection for a first-window passage", np.abs(out[k0][0] - whole[k0][0]).max() < 1e-3)
    o = RO.tokenize(tok, msgs)
    ids, blks = o
    hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    b = next(x for x in blks if x["mi"] == k0[0])
    t0, t1 = b[k0[1] + "_tok"]
    ref = (hs[4][0, t0:t1] @ torch.tensor(Dm[1]).T)
    report("read: max and mean pooling match a direct computation",
           np.abs(whole[k0][0][1, :, 0] - ref.max(0).values.detach().numpy()).max() < 1e-3 and np.abs(whole[k0][0][1, :, 1] - ref.mean(0).detach().numpy()).max() < 1e-3)


def test_score():
    rng = np.random.default_rng(0)
    N, nD = 3000, 30
    y = rng.random(N) < 0.1
    ntok = np.exp(rng.normal(4, 1, N) + 0.8 * y).astype(int) + 1
    names = ["close:gf_pool", "close:dom_ours", "len_only"] + [f"rnd{i}" for i in range(nD)]
    P = rng.standard_normal((N, 1, len(names), 2)).astype(np.float32)
    P[:, 0, 0, :] += 1.0 * y[:, None]
    P[:, 0, 1, :] += 2.0 * y[:, None]
    P[:, 0, 2, :] = np.log(ntok)[:, None] + 0.01 * rng.standard_normal((N, 2))
    task = np.array([f"t{i % 60}" for i in range(N)])
    meta = dict(names=names, layers=[20], layer=20)
    res = SC.table(meta, P, y, y & (rng.random(N) < 0.3), task, ntok, 0, names=("close:gf_pool", "close:dom_ours", "len_only"), nboot=200)
    dd = res["dirs"]
    report("score: stronger direction scores higher, with the paired difference excluding zero",
           dd["close:dom_ours"]["auc"] > dd["close:gf_pool"]["auc"] and dd["close:dom_ours"]["diff_vs_gf"][1] > 0, str(dd["close:dom_ours"]["diff_vs_gf"]))
    report("score: a pure length score falls to chance once length is regressed out",
           dd["len_only"]["auc"] > 0.65 and abs(dd["len_only"]["auc_len"] - 0.5) < 0.06, f"{dd['len_only']['auc']:.2f} -> {dd['len_only']['auc_len']:.2f}")
    report("score: random directions sit at chance", abs(res["random"]["p50"] - 0.5) < 0.03 and res["random"]["p95"] < 0.56)


def main():
    from transformers import AutoTokenizer
    from . import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = S.load_rows(ours=False)
    test_render(tok, rows)
    items = test_items(tok, rows)
    test_engine(tok, items)
    test_train(tok, items)
    test_harvest_dirs(tok, rows)
    test_rollouts(tok)
    test_score()
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()

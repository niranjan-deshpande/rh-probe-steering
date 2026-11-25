"""Layer-36 activations of multi-turn rollouts and of new single-turn completions (pod; frozen Qwen3.8-27B).

    python -m rlp.flag.mt_harvest mt --run RUN [RUN ...]          # -> cache/flag/mt/<run>/acts_36.npz + acts_36.json
    python -m rlp.flag.mt_harvest st --names DIR [DIR ...] --tag T   # -> cache/flag/onp/st_<T>_36.npy + st_<T>_36.json
    python -m rlp.flag.mt_harvest mt --run RUN --merge-only       # assemble what the shards hold, no model

mt (Interface 3). Each finished episode of cache/flag/mt/<run>/<family>__<variant>.jsonl (Interface 2) is rendered
once, whole, with the model's chat template: the same call the server made for every turn (tools of the task,
reasoning_content of every assistant turn kept, reasoning_effort as in rlp.flag.q38). The tool schemas are rebuilt
with rlp.flag.mt.make_task + tool_schemas from the episode's family, key, variant and cfg. An assistant message's
tokens are the ones the model generated: after `<|im_start|>assistant\\n<think>\\n` (the generation prompt) up to and
including `<|im_end|>`; the single-turn cache of 24 November used the same convention. Written:
    ep_mean   [E, 5120]  mean over all assistant tokens of the episode (reasoning included; token-weighted)
    step_mean [S, 5120]  mean over the tokens of one assistant message
    ep_ok     [E]        False if the episode could not be rendered or aligned (its y is then None)
    acts_36.json  {"episodes": [{key, family, task, variant, k, cfg, y, ok, cut, n_tok, n_assistant_tok, file}],
                   "steps": [{ep, msg, is_hack_step, n_tok}]}
y is 1 for a hack episode (info.hit), 0 otherwise, None if the episode stopped as truncated or in error, has no
detector verdict, or could not be aligned. is_hack_step marks msg == info.hit_step. A conversation longer than
--max-len tokens is cut at the end (the model is causal, so the kept part is exact); assistant messages that do not
fit are left out and the episode is flagged `cut`.

st. Completions of cache/flag/base_rate/<dir>/ (the format `rlp.flag.base_rate label` reads), tokenised exactly as
rlp.flag.onp did, mean over completion tokens. It writes NEW files only: activations made with the fast kernels are
not bit-identical to the cache of 24 November, so nothing is appended to means_36_all.npy / meta_all.json.

Both modes sort by length, batch under a token budget and write resumable shards keyed by episode, so a rerun after
more rollouts arrive only reads the new ones. The render is checked against the prompts the runner built: for every
assistant message the generation prompt of the conversation before it must be an exact prefix of the whole render;
the count of mismatches is printed (MT_RENDER_OK / MT_RENDER_MISMATCH).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import config
from . import gf27 as G, q38 as Q

MT = config.CACHE / "flag" / "mt"
ONP = config.CACHE / "flag" / "onp"
SRC = config.CACHE / "flag" / "base_rate"
BAD_STOPS = ("truncated", "error")


# ------------------------------------------------------------------ rendering and token alignment

def tools_for(ep, cache=None):
    """Tool schemas the runner gave the model for this episode (rlp.flag.mt). An episode may also carry them itself."""
    if isinstance(ep.get("tools"), list) and ep["tools"] and isinstance(ep["tools"][0], dict):
        return ep["tools"]
    cache = cache if cache is not None else {}
    k = (ep["family"], ep["key"], ep["variant"], json.dumps(ep.get("cfg") or {}, sort_keys=True))
    if k not in cache:
        from . import mt
        try:
            i = int(ep["key"].split("/")[1])
            cache[k] = mt.tool_schemas(mt.make_task(ep["family"], i, ep["variant"], ep.get("cfg") or {}))
        except Exception as e:
            print(f"[mth] WARNING {ep['key']}: task not rebuilt ({type(e).__name__}: {e}); using all four tool schemas", flush=True)
            cache[k] = [mt.TOOL_DEFS[n] for n in ("bash", "read_file", "write_file", "submit")]
    return cache[k]


def render(tok, messages, tools):
    """The whole conversation as the server's chat template writes it (no generation prompt at the end)."""
    return tok.apply_chat_template(Q.norm(messages), tools=tools or None, tokenize=False, add_generation_prompt=False,
                                   reasoning_effort=Q.EFFORT)


def prefix_mismatches(tok, messages, tools, text):
    """Number of assistant messages whose generation prompt (what the server saw) is not a prefix of `text`."""
    return sum(not text.startswith(Q.prompt(tok, messages[:i], tools or None))
               for i, m in enumerate(messages) if m["role"] == "assistant")


class Marks:
    """Token ids of the template's turn markers."""
    def __init__(self, tok):
        self.im_start, self.im_end = (tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", Q.STOP))
        self.header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.gen = tok("<|im_start|>assistant\n<think>\n", add_special_tokens=False)["input_ids"]
        assert self.gen[:len(self.header)] == self.header and self.header[0] == self.im_start


def assistant_spans(ids, mk):
    """-> [(a0, a1)] for every assistant turn, in order: tokens ids[a0:a1] are the generated ones (after the header
    and the opening `<think>` + newline, through <|im_end|>)."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if ids[i] == mk.im_start and ids[i:i + len(mk.header)] == mk.header:
            a0 = i + (len(mk.gen) if ids[i:i + len(mk.gen)] == mk.gen else len(mk.header))
            j = a0
            while j < n and ids[j] != mk.im_end:
                j += 1
            out.append((a0, min(j + 1, n)))
            i = j
        i += 1
    return out


def align(tok, mk, messages, tools, max_len):
    """-> dict(ids, steps=[(message index, a0, a1)], cut, n_tok, mismatch). Raises ValueError if turns do not line up."""
    text = render(tok, messages, tools)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    spans = assistant_spans(ids, mk)
    msgs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(spans) != len(msgs):
        raise ValueError(f"{len(msgs)} assistant messages but {len(spans)} assistant turns in the render")
    if not msgs:
        raise ValueError("no assistant message")
    steps = [(m, a0, a1) for m, (a0, a1) in zip(msgs, spans) if a1 <= max_len and a1 > a0]
    cut = len(ids) > max_len
    if not steps:
        raise ValueError(f"no assistant message fits in {max_len} tokens")
    return dict(ids=ids[:max(a1 for _, _, a1 in steps)] if cut else ids, steps=steps, cut=cut, n_tok=len(ids),
                mismatch=prefix_mismatches(tok, messages, tools, text))


# ------------------------------------------------------------------ shared: sharded, resumable pooled reading

def read_safe(model, seqs, L, pad):
    """read_batch as a list of [T, d] rows. A batch that runs out of memory is read in two halves; a single
    sequence that still does not fit comes back as None (it is reported and left for a later run)."""
    try:
        return list(G.read_batch(model, seqs, L, pad))
    except Exception as e:
        if not (isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()):
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(seqs) == 1:
        return [None]
    h = len(seqs) // 2
    return read_safe(model, seqs[:h], L, pad) + read_safe(model, seqs[h:], L, pad)


def done_ids(sdir):
    out = set()
    for f in sorted(Path(sdir).glob("s*.npz")):
        out |= set(np.load(f, allow_pickle=False)["eid"].tolist())
    return out


def harvest_items(model, tok, items, L, sdir, budget, shard, solo=2048):
    """items: [dict(eid, ids, spans=[(tag, a0, a1)])]. For each item not yet in a shard: the mean over each span and
    the token-weighted mean over all its spans. Shards: sdir/sNNNN.npz."""
    sdir = Path(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    have = done_ids(sdir)
    todo = [it for it in items if it["eid"] not in have]
    nxt = 1 + max([int(f.stem[1:]) for f in sdir.glob("s*.npz")] or [-1])
    lens = [len(it["ids"]) for it in todo]
    order = sorted(range(len(todo)), key=lambda i: -lens[i])
    print(f"[mth] {len(items) - len(todo)} of {len(items)} already read; {len(todo)} to read, {sum(lens):,} tokens, "
          f"longest {max(lens) if lens else 0}", flush=True)
    acc, t0, ntok, oom = [], time.time(), 0, []

    def flush():
        nonlocal acc, nxt
        if not acc:
            return
        G.save_atomic(sdir / f"s{nxt:04d}.npz", eid=np.array([x["eid"] for x in acc]),
                      ep_mean=np.stack([x["ep_mean"] for x in acc]), ep_tok=np.array([x["ep_tok"] for x in acc]),
                      step_ep=np.array([i for i, x in enumerate(acc) for _ in x["tags"]], np.int64),
                      step_tag=np.array([t for x in acc for t in x["tags"]], np.int64),
                      step_tok=np.array([t for x in acc for t in x["toks"]], np.int64),
                      step_mean=np.concatenate([x["step_mean"] for x in acc]))
        acc, nxt = [], nxt + 1
    for bt in G.batches(order, lens, budget, solo):
        H = read_safe(model, [todo[i]["ids"] for i in bt], L, G.pad_id(tok))
        for b, i in enumerate(bt):
            it = todo[i]
            if H[b] is None:
                oom.append(it["eid"])
                print(f"[mth] WARNING out of memory on {it['eid'][:80]} ({lens[i]} tokens); not read", flush=True)
                continue
            sums = torch.stack([H[b][a0:a1].float().sum(0) for _, a0, a1 in it["spans"]])
            n = torch.tensor([a1 - a0 for _, a0, a1 in it["spans"]], device=sums.device)
            acc.append(dict(eid=it["eid"], ep_mean=(sums.sum(0) / n.sum()).cpu().numpy(), ep_tok=int(n.sum()),
                            tags=[t for t, _, _ in it["spans"]], toks=n.tolist(), step_mean=(sums / n[:, None]).cpu().numpy()))
        ntok += sum(lens[i] for i in bt)
        del H
        if len(acc) >= shard:
            flush()
            el = time.time() - t0
            print(f"[mth] {ntok:,}/{sum(lens):,} tokens  {ntok / el:.0f} tok/s  eta {(sum(lens) - ntok) / max(ntok, 1) * el / 60:.0f} min", flush=True)
    flush()
    return oom


def collect(sdir):
    """-> {eid: dict(ep_mean, ep_tok, steps=[(tag, n tokens, mean)])} from all shards."""
    out = {}
    for f in sorted(Path(sdir).glob("s*.npz")):
        z = np.load(f, allow_pickle=False)
        eids = z["eid"].tolist()
        new = {e for e in eids if e not in out}              # an episode is read once; a repeat in a later shard is ignored
        for i, e in enumerate(eids):
            if e in new:
                out[e] = dict(ep_mean=z["ep_mean"][i], ep_tok=int(z["ep_tok"][i]), steps=[])
        for i, t, n, m in zip(z["step_ep"], z["step_tag"], z["step_tok"], z["step_mean"]):
            if eids[i] in new:
                out[eids[i]]["steps"].append((int(t), int(n), m))
    return out


# ------------------------------------------------------------------ multi-turn rollouts

def load_episodes(run_dir):
    """Finished episodes of one run, in file then line order, each with a unique `eid` and its file stem."""
    eps, seen = [], set()
    for f in sorted(Path(run_dir).glob("*__*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            eid = f"{f.stem}|{ep['key']}|{ep['k']}|{json.dumps(ep.get('cfg') or {}, sort_keys=True)}"
            if eid in seen:
                print(f"[mth] WARNING duplicate episode {eid}; the first is kept", flush=True)
                continue
            seen.add(eid)
            eps.append(ep | {"eid": eid, "file": f.stem})
    return eps


def label(ep):
    info = ep.get("info") or {}
    if info.get("stop") in BAD_STOPS or info.get("hit") is None:
        return None
    return int(bool(info["hit"]))


def mt_run(a, run, model=None, tok=None):
    D = Path(a.mt_dir) / run
    eps = load_episodes(D)
    sdir = D / f"acts_{a.layer}_shards"
    side = D / f"acts_{a.layer}_align.json"                 # alignment facts kept across reruns
    notes = json.load(open(side)) if side.exists() else {}
    if not a.merge_only:
        mk, tcache, have, items, bad = Marks(tok), {}, done_ids(sdir) if sdir.exists() else set(), [], 0
        for ep in eps:
            if ep["eid"] in have and ep["eid"] in notes:
                continue
            try:
                al = align(tok, mk, ep["messages"], tools_for(ep, tcache), a.max_len)
            except Exception as e:
                notes[ep["eid"]] = dict(ok=False, error=f"{type(e).__name__}: {e}"[:300])
                bad += 1
                continue
            notes[ep["eid"]] = dict(ok=True, cut=al["cut"], n_tok=al["n_tok"], mismatch=al["mismatch"], n_steps=len(al["steps"]))
            items.append(dict(eid=ep["eid"], ids=al["ids"], spans=al["steps"]))
        json.dump(notes, open(side, "w"))
        mm = sum(n.get("mismatch", 0) for n in notes.values())
        turns = sum(n.get("n_steps", 0) for n in notes.values())
        print(f"[mth] {run}: {len(eps)} episodes, {bad} not aligned, {sum(bool(n.get('cut')) for n in notes.values())} cut at "
              f"{a.max_len} tokens; generation prompt is an exact prefix of the render for {turns - mm} of {turns} assistant turns", flush=True)
        print(f"MT_RENDER_OK {run}" if mm == 0 and bad == 0 else f"MT_RENDER_MISMATCH {run} turns={mm} episodes_not_aligned={bad}", flush=True)
        oom = harvest_items(model, tok, items, a.layer, sdir, a.budget, a.shard, a.solo_len)
        if oom:
            print(f"MT_HARVEST_OOM {run} episodes={len(oom)} (rerun with a smaller --max-len to read their beginnings)", flush=True)
    got = collect(sdir) if sdir.exists() else {}
    d = next(iter(got.values()))["ep_mean"].shape[0] if got else 0
    ep_mean, ep_ok, episodes, steps, step_mean = np.zeros((len(eps), d), np.float32), np.zeros(len(eps), bool), [], [], []
    for ei, ep in enumerate(eps):
        g, info, nt = got.get(ep["eid"]), ep.get("info") or {}, notes.get(ep["eid"], {})
        if g is not None:
            ep_mean[ei], ep_ok[ei] = g["ep_mean"], True
            for msg, n, m in g["steps"]:
                steps.append(dict(ep=ei, msg=msg, is_hack_step=bool(info.get("hit")) and info.get("hit_step") == msg, n_tok=n))
                step_mean.append(m)
        episodes.append({k: ep.get(k) for k in ("key", "family", "task", "variant", "k", "cfg")}
                        | dict(y=label(ep) if g is not None else None, ok=g is not None, cut=bool(nt.get("cut")),
                               n_tok=nt.get("n_tok"), n_assistant_tok=g["ep_tok"] if g else 0, file=ep["file"]))
    G.save_atomic(D / f"acts_{a.layer}.npz", ep_mean=ep_mean, ep_ok=ep_ok,
                  step_mean=np.stack(step_mean).astype(np.float32) if step_mean else np.zeros((0, d), np.float32))
    json.dump(dict(episodes=episodes, steps=steps), open(D / f"acts_{a.layer}.json", "w"))
    ys = [e["y"] for e in episodes]
    print(f"[mth] {run}: {int(ep_ok.sum())}/{len(eps)} episodes read, {len(steps)} assistant steps, "
          f"{sum(s['is_hack_step'] for s in steps)} hack steps; y: {ys.count(1)} hack, {ys.count(0)} not, {ys.count(None)} unusable "
          f"-> {D / f'acts_{a.layer}.npz'}", flush=True)


def mt(a):
    model, tok = (None, None) if a.merge_only else G.load_model(a)
    for run in a.run:
        mt_run(a, run, model, tok)
    print("MT_HARVEST_DONE", flush=True)


# ------------------------------------------------------------------ new single-turn completions

def load_completions(names, behs=None, src=None):
    """Completions of base_rate runs with their program labels. y is None if unlabelled, invalid or cut off."""
    out = []
    for name in names:
        files = [f for f in sorted((Path(src or SRC) / name).glob("*__*.jsonl")) if not f.name.endswith(".labels.jsonl")]
        if not files:
            print(f"[mth] WARNING no completions under {Path(src or SRC) / name}", flush=True)
        for f in files:
            beh, cond = f.name[:-len(".jsonl")].split("__")
            if behs and beh not in behs:
                continue
            lf = Path(str(f).replace(".jsonl", ".labels.jsonl"))
            lab = {(x["uid"], x.get("k", 0)): x for x in map(json.loads, open(lf))} if lf.exists() else {}
            if not lab:
                print(f"[mth] WARNING {lf} missing or empty: y is None for {f.name}", flush=True)
            for r in map(json.loads, open(f)):
                if not r.get("raw"):
                    continue
                x = lab.get((r["uid"], r.get("k", 0)))
                ok = x is not None and bool(x.get("valid")) and not x.get("trunc")
                grp = f"{beh}|{cond}|{r['uid']}" + (f"|{name}" if len(names) > 1 else "")
                out.append(dict(beh=beh, cond=cond, uid=r["uid"], k=r.get("k", 0), messages=r["messages"], raw=r["raw"],
                                y=(int(bool(x["hit"])) if ok else None), role="new", group=grp, run=name))
    return out


def st(a):
    from . import onp as O
    stem = f"st_{a.tag}_{a.layer}"
    recs = load_completions(a.names, a.behs, a.src)
    model, tok = G.load_model(a)
    items, toks = [], []
    for r in recs:
        p, c, _ = O.tokenise(tok, r, a.max_c)
        toks.append((len(p), len(c)))
        items.append(dict(eid=f"{r['run']}|{r['beh']}|{r['cond']}|{r['uid']}|{r['k']}", ids=p + c, spans=[(0, len(p), len(p) + len(c))]))
    out = Path(a.onp)
    harvest_items(model, tok, items, a.layer, out / f"{stem}_shards", a.budget, a.shard, a.solo_len)
    got = collect(out / f"{stem}_shards")
    keep = [i for i, it in enumerate(items) if it["eid"] in got]         # all of them unless one ran out of memory
    recs, toks, items = ([x[i] for i in keep] for x in (recs, toks, items))
    X = np.stack([got[it["eid"]]["ep_mean"] for it in items]).astype(np.float32)
    np.save(out / f"{stem}.npy", X)
    json.dump([{k: r[k] for k in ("beh", "cond", "uid", "k", "y", "role", "group", "run")} | {"n_p": n_p, "n_c": n_c}
               for r, (n_p, n_c) in zip(recs, toks)], open(out / f"{stem}.json", "w"))
    ys = [r["y"] for r in recs]
    print(f"[mth] {len(recs)} completions ({ys.count(1)} hack, {ys.count(0)} honest, {ys.count(None)} unusable) -> {out / f'{stem}.npy'}", flush=True)
    print("ST_HARVEST_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("mt", "st"))
    G.add_model_args(ap)
    ap.add_argument("--run", nargs="+", default=[], help="mt: run directories under cache/flag/mt/")
    ap.add_argument("--mt-dir", default=str(MT))
    ap.add_argument("--merge-only", action="store_true", help="mt: write acts from existing shards without the model")
    ap.add_argument("--max-len", type=int, default=120000, help="mt: longest conversation read, in tokens")
    ap.add_argument("--names", nargs="+", default=[], help="st: directories under cache/flag/base_rate/")
    ap.add_argument("--behs", nargs="*", help="st: only these environments")
    ap.add_argument("--tag", default="night", help="st: output is cache/flag/onp/st_<tag>_<layer>.npy / .json")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--onp", default=str(ONP))
    ap.add_argument("--max-c", type=int, default=6000, help="st: completion tokens read, as on 24 November")
    ap.add_argument("--budget", type=int, default=40000, help="padded tokens per forward pass")
    ap.add_argument("--solo-len", type=int, default=2048, help="sequences longer than this are read one at a time, unpadded")
    ap.add_argument("--shard", type=int, default=32, help="episodes per resumable shard")
    a = ap.parse_args()
    if a.mode == "mt" and not a.run or a.mode == "st" and not a.names:
        ap.error("mt needs --run, st needs --names")
    {"mt": mt, "st": st}[a.mode](a)

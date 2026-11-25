"""Unprompted hack rates of a served model on the single-turn environments from rlp.manif.envs (base-rate pilot).

    python -m rlp.flag.base_rate gen   --name q38 --model Qwen/Qwen3.8-27B --n 30 --conds neutral tempt
    python -m rlp.flag.base_rate label --name q38

For each hack environment the first --n items are run once per condition, with thinking on:

    neutral   the loophole is visible in the prompt and nothing asks for it or forbids it
    tempt     the same, plus a situation note that makes the hack tempting (still not instructed)

`label` runs the environments' program detectors on the answer (the text after the reasoning), so no judge is used.
It executes model-written code, so run it on the pod. Output: CACHE/flag/base_rate/<name>/<beh>__<cond>.jsonl and
summary.json.
"""
import argparse
import concurrent.futures as cf
import json
import zlib

from .. import config
from ..manif import envs as E
from . import MODEL, q38 as Q, render as R

OUT = config.CACHE / "flag" / "base_rate"
MAX_TOKENS = 6000


def one(client, tok, model, it, cond, k=0, max_tokens=MAX_TOKENS):
    msgs = E.messages(it, cond)
    q = Q.is_q38(model)
    p = Q.prompt(tok, msgs) if q else R.render(tok, msgs) + R.A_OPEN
    samp = Q.SAMPLING if q else dict(temperature=0.6, top_p=0.95, top_k=20)
    r = client.completions.create(model=model, prompt=p, max_tokens=max_tokens, temperature=samp["temperature"],
                                  top_p=samp["top_p"], stop=["<|im_end|>"], seed=zlib.crc32((f"{it['uid']}|{cond}" + (f"|{k}" if k else "")).encode()) % 2**31,
                                  extra_body={"top_k": samp["top_k"], "skip_special_tokens": False})
    c = r.choices[0]
    if q:
        m = Q.parse(c.text)
        reasoning, answer = m["reasoning_content"], (c.text.split(Q.TH_CLOSE, 1)[1].strip() if Q.TH_CLOSE in c.text else "")
    else:
        from .agent import parse
        m = parse(c.text)
        reasoning, answer = m["reasoning_content"], m["content"]
    return dict(uid=it["uid"], k=k, beh=it["beh"], cond=cond, messages=msgs, reasoning=reasoning, response=answer, raw=c.text,
                finish_reason=c.finish_reason, completion_tokens=r.usage.completion_tokens)


def gen(a):
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=1800)
    D = OUT / a.name
    D.mkdir(parents=True, exist_ok=True)
    jobs = []
    for beh in (a.behs or E.HACKS):
        for cond in a.conds:
            f = D / f"{beh}__{cond}.jsonl"
            done = {(r["uid"], r.get("k", 0)) for r in map(json.loads, open(f))} if f.exists() else set()
            jobs += [(it, cond, k) for it in E.items(beh)[a.start:a.start + a.n] for k in range(a.samples)
                     if (it["uid"], k) not in done]
    print(f"[base_rate] {len(jobs)} completions to run", flush=True)
    n = tokens = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, client, tok, a.model, it, cond, k, a.max_tokens): (it, cond) for it, cond, k in jobs}
        for fut in cf.as_completed(futs):
            it, cond = futs[fut]
            n += 1
            try:
                r = fut.result()
            except Exception as e:
                print(f"[base_rate] {n}/{len(jobs)} {it['uid']} {cond} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            tokens += r["completion_tokens"]
            with open(D / f"{it['beh']}__{cond}.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            if n % 25 == 0 or n == len(jobs):
                print(f"[base_rate] {n}/{len(jobs)} done, {tokens} completion tokens", flush=True)
    print("BASE_RATE_GEN_DONE", flush=True)


def _label_file(path):
    recs = [json.loads(l) for l in open(path)]
    its = {it["uid"]: it for it in E.items(recs[0]["beh"])} if recs else {}
    out = []
    for r in recs:
        it = its[r["uid"]]
        hit, _ = E.detect(r["beh"], it, r["response"])
        out.append(dict(uid=r["uid"], k=r.get("k", 0), cond=r["cond"], hit=bool(hit), trunc=r["finish_reason"] == "length",
                        empty=not r["response"].strip(), tokens=r["completion_tokens"],
                        **E.extra_flags(r["beh"], it, r["response"]), **E.valid(r["beh"], it, r["response"])))
    with open(str(path).replace(".jsonl", ".labels.jsonl"), "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    beh, cond = path.name[:-len(".jsonl")].split("__")
    n = max(1, len(out))
    row = dict(beh=beh, cond=cond, n=len(out), hit=sum(x["hit"] for x in out), valid=sum(bool(x.get("valid")) for x in out),
               trunc=sum(x["trunc"] for x in out), mean_tokens=round(sum(x["tokens"] for x in out) / n))
    for k in ("solves", "correct"):
        if any(k in x for x in out):
            row[k] = sum(bool(x.get(k)) for x in out)
    return row


def label(a):
    D = OUT / a.name
    files = sorted(p for p in D.glob("*__*.jsonl") if not p.name.endswith(".labels.jsonl"))
    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_label_file, files))
    json.dump(rows, open(D / "summary.json", "w"), indent=1)
    for r in rows:
        extra = " ".join(f"{k}={r[k]}" for k in ("solves", "correct") if k in r)
        print(f"{r['beh']:14s} {r['cond']:8s} n={r['n']:3d} hit={r['hit']:3d} valid={r['valid']:3d} trunc={r['trunc']:3d} "
              f"tok={r['mean_tokens']:5d} {extra}", flush=True)
    print("BASE_RATE_LABEL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "label"])
    ap.add_argument("--name", default="q38")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--samples", type=int, default=1, help="completions per item (sample index k seeds each)")
    ap.add_argument("--start", type=int, default=0, help="index of the first item")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--conds", nargs="+", default=["neutral", "tempt"])
    ap.add_argument("--behs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    a = ap.parse_args()
    (gen if a.stage == "gen" else label)(a)

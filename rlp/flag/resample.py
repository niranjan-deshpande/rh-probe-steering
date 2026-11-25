"""Prefix resampling: where in its own completion does the model decide to hack?

Take completions the frozen model sampled itself (cache/flag/base_rate/<src>), cut each at several token positions,
and let the model continue freely from each cut several times. The detectors label every continued completion, which
gives an on-policy estimate of P(hack | prefix) along the rollout, and, for prefixes where that probability is
intermediate, hack and honest continuations from exactly the same prefix. Nothing is teacher-forced: the prefix is
the model's own sample and so is every continuation.

    python -m rlp.flag.resample plan  --name rs --per-env 16 --cuts 0.15 0.3 0.45 0.6 0.75 0.9
    python -m rlp.flag.resample gen   --name rs --k 6          # served model, raw completions endpoint
    python -m rlp.flag.resample label --name rs                # runs model-written code: on the pod
    python -m rlp.flag.resample table --name rs

Output: CACHE/flag/resample/<name>/{plan.jsonl, gen.jsonl, labels.jsonl, table.json}.
"""
import argparse
import concurrent.futures as cf
import json
import zlib

import numpy as np

from .. import config
from ..manif import envs as E
from . import onp, q38 as Q

OUT = config.CACHE / "flag" / "resample"
ENVS = ("length_pad", "quiz_key", "leak", "hardcode")          # the fitting environments
CAP = {"hardcode": 9000, "leak": 8000}


def plan(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    recs = onp.load_recs(a.src)
    rng = np.random.default_rng(a.seed)
    rows = []
    for beh in a.envs:
        g = {k: v for role in a.roles for k, v in onp.groups(recs, role, beh).items()}
        keys = sorted(g)
        rng.shuffle(keys)
        n = 0
        for key in keys:
            if n >= a.per_env.get(beh, a.per_env["default"]):
                break
            if not (a.min_class <= len(g[key][0]) and a.min_class <= len(g[key][1])):
                continue                      # keep prompts where the outcome is genuinely open
            for cls, idx in zip(("hack", "honest"), g[key]):
                r = recs[idx[int(rng.integers(len(idx)))]]
                ids = tok(r["raw"], add_special_tokens=False)["input_ids"]
                if len(ids) < a.min_tokens or len(ids) > a.max_tokens:
                    continue
                close = tok.convert_tokens_to_ids(Q.TH_CLOSE)
                think = ids.index(close) if close in ids else len(ids)
                base = think if a.within == "reasoning" else len(ids)
                for ci, fr in enumerate(a.cuts):
                    t = max(1, int(round(fr * base)))
                    rows.append(dict(sid=f"{r['group']}|{r['k']}", beh=beh, cond=r["cond"], uid=r["uid"], src_k=r["k"], src_y=r["y"],
                                     cls=cls, cut=ci, frac=fr, cut_t=t, n_t=len(ids), think_t=think,
                                     prefix=tok.decode(ids[:t], skip_special_tokens=False), prefix_ids=ids[:t],
                                     messages=r["messages"]))
                n += 1
    D = OUT / a.name
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "plan.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rem = sum(r["n_t"] - r["cut_t"] for r in rows)
    print(f"[resample] {len({r['sid'] for r in rows})} sources, {len(rows)} prefixes; "
          f"about {rem} tokens per continuation round (if continuations match the source length)", flush=True)


def _one(client, tok, model, row, j):
    # token ids, not text: re-tokenising prompt + prefix as one string could merge tokens differently at the cut
    p = tok(Q.prompt(tok, row["messages"]), add_special_tokens=False)["input_ids"] + row["prefix_ids"]
    cap = CAP.get(row["beh"], 6000)
    r = client.completions.create(model=model, prompt=p, max_tokens=max(256, cap - row["cut_t"]), temperature=Q.SAMPLING["temperature"],
                                  top_p=Q.SAMPLING["top_p"], stop=[Q.STOP],
                                  seed=zlib.crc32(f"rs|{row['sid']}|{row['cut']}|{j}".encode()) % 2**31,
                                  extra_body={"top_k": Q.SAMPLING["top_k"], "skip_special_tokens": False})
    c = r.choices[0]
    raw = row["prefix"] + c.text
    return dict(sid=row["sid"], beh=row["beh"], cond=row["cond"], uid=row["uid"], cut=row["cut"], j=j, cont=c.text,
                response=(raw.split(Q.TH_CLOSE, 1)[1].strip() if Q.TH_CLOSE in raw else ""),
                finish_reason=c.finish_reason, completion_tokens=r.usage.completion_tokens)


def gen(a):
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=3600)
    D = OUT / a.name
    rows = [json.loads(l) for l in open(D / "plan.jsonl")]
    f = D / "gen.jsonl"
    done = {(r["sid"], r["cut"], r["j"]) for r in map(json.loads, open(f))} if f.exists() else set()
    jobs = [(row, j) for row in rows for j in range(a.k) if (row["sid"], row["cut"], j) not in done]
    jobs.sort(key=lambda x: -(x[0]["n_t"] - x[0]["cut_t"]))          # longest first, so the tail of the run is short
    print(f"[resample] {len(jobs)} continuations to run", flush=True)
    n = tokens = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_one, client, tok, a.model, row, j) for row, j in jobs]
        for fut in cf.as_completed(futs):
            n += 1
            try:
                r = fut.result()
            except Exception as e:
                print(f"[resample] {n}/{len(jobs)} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            tokens += r["completion_tokens"]
            with open(f, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if n % 100 == 0 or n == len(jobs):
                print(f"[resample] {n}/{len(jobs)} done, {tokens} completion tokens", flush=True)
    print("RESAMPLE_GEN_DONE", flush=True)


def _label_chunk(recs):
    out, its = [], {}
    for r in recs:
        if r["beh"] not in its:
            its[r["beh"]] = {it["uid"]: it for it in E.items(r["beh"])}
        it = its[r["beh"]][r["uid"]]
        hit, _ = E.detect(r["beh"], it, r["response"])
        v = E.valid(r["beh"], it, r["response"])
        out.append(dict(sid=r["sid"], cut=r["cut"], j=r["j"], hit=bool(hit), trunc=r["finish_reason"] == "length",
                        valid=bool(v.get("valid")), tokens=r["completion_tokens"]))
    return out


def label(a):
    D = OUT / a.name
    recs = [json.loads(l) for l in open(D / "gen.jsonl")]
    chunks = [recs[i::a.workers] for i in range(a.workers)]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        out = [x for ch in ex.map(_label_chunk, chunks) for x in ch]
    with open(D / "labels.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print(f"[resample] labelled {len(out)}", flush=True)
    table(a)
    print("RESAMPLE_LABEL_DONE", flush=True)


def load(name):
    """-> plan rows keyed (sid, cut), each with 'conts': list of dicts (cont text, y in {1, 0, None})."""
    D = OUT / name
    rows = {(r["sid"], r["cut"]): r | {"conts": []} for r in map(json.loads, open(D / "plan.jsonl"))}
    lab = {(x["sid"], x["cut"], x["j"]): x for x in map(json.loads, open(D / "labels.jsonl"))}
    for r in map(json.loads, open(D / "gen.jsonl")):
        x = lab.get((r["sid"], r["cut"], r["j"]))
        if x is None:
            continue
        ok = x["valid"] and not x["trunc"]
        rows[(r["sid"], r["cut"])]["conts"].append(dict(cont=r["cont"], j=r["j"], y=(int(x["hit"]) if ok else None)))
    return rows


def table(a):
    rows = load(a.name)
    out, usable = {}, 0
    for (sid, cut), r in rows.items():
        ys = [c["y"] for c in r["conts"] if c["y"] is not None]
        if not ys:
            continue
        p = float(np.mean(ys))
        usable += 0 < sum(ys) < len(ys)
        out.setdefault(r["beh"], {}).setdefault(r["cls"], {}).setdefault(r["frac"], []).append(p)
    print(f"[resample] prefixes with both a hack and an honest continuation: {usable} of {len(rows)}")
    res = {}
    for beh, d in out.items():
        for cls, e in d.items():
            line = "  ".join(f"{fr:.2f}: {np.mean(v):.2f} (n={len(v)}, mixed {sum(0 < x < 1 for x in v)})" for fr, v in sorted(e.items()))
            res[f"{beh}/{cls}"] = {str(fr): dict(p=float(np.mean(v)), n=len(v), mixed=int(sum(0 < x < 1 for x in v))) for fr, v in e.items()}
            print(f"  {beh:11s} source {cls:6s} P(hack | prefix) by cut fraction: {line}")
    json.dump(res | {"usable": int(usable), "prefixes": len(rows)}, open(OUT / a.name / "table.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "gen", "label", "table"])
    ap.add_argument("--name", default="rs")
    ap.add_argument("--src", default="onp")
    ap.add_argument("--model", default=Q.MODEL)
    ap.add_argument("--envs", nargs="+", default=list(ENVS))
    ap.add_argument("--roles", nargs="+", default=["fit", "val"])
    ap.add_argument("--per-env", type=json.loads, default={"default": 16, "length_pad": 30, "hardcode": 10},
                    help="JSON {env: prompts}; one hack and one honest source completion per prompt")
    ap.add_argument("--min-class", type=int, default=2, help="a prompt needs at least this many hack and honest samples")
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--within", choices=["reasoning", "all"], default="reasoning", help="what the cut fractions are fractions of")
    ap.add_argument("--min-tokens", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    a = ap.parse_args()
    dict(plan=plan, gen=gen, label=label, table=table)[a.stage](a)

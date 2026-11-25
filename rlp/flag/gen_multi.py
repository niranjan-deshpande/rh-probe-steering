"""All steering conditions in one mixed queue against the per-request steering server (patch_vllm_steer_multi.py).

Run on the pod from the repo root, after `python -m rlp.flag.steer_selftest live` has passed:

    python -m rlp.flag.gen_multi run --table T --prefix n1_ --conds @main    --preset main  [--extra-behs puzzle_leak:tempt]
    python -m rlp.flag.gen_multi run --table T --prefix n1_ --conds @fitting --preset fitting
    python -m rlp.flag.gen_multi run --table T --prefix n3_ --conds @pairing --preset pairing
    python -m rlp.flag.gen_multi run --table T --prefix np_ --conds base --groups '[{"beh":"X","cond":"tempt","n":8,"samples":4}]'
    python -m rlp.flag.gen_multi story --table T --prefix n1_ --conds @story
    python -m rlp.flag.gen_multi plan --prefix n1_ --preset main          # only draw and record the prompts

Every (steering condition, prompt, sample k) is one job. All jobs go into one queue, shuffled with a fixed seed and
served by --workers threads (default 256, well above the server's --max-num-seqs), so the server's queue stays full
and no condition waits for its slowest completion. Each request carries extra_body={"vllm_xargs": {"vid": row}}.

Prompts. --groups is a JSON list of {"beh", "cond", "n", "samples" (default 1), "pick"}:
    pick "fresh" (default)   n items drawn without replacement, with random.Random("<seed>|<beh>|<cond>"), from the
                             items of the environment that no earlier run used. "Used" = the index ranges of the runs
                             up to 24 November (USED_BEFORE below, taken from the run scripts) plus every uid found in
                             any existing cache/flag/base_rate/*/<beh>__*.jsonl that does not belong to this prefix.
    pick "range"             items[start : start + n] exactly (the pairing run: the prompts of 24 November).
    pick "uids"              the listed "uids".
The draw is written once to cache/flag/night/plans/<prefix><plan>.json (seed, excluded indices, chosen uids) and read
back on every later call, so a resumed run and every condition use identical prompts. Presets: main (save_tamper,
ckpt_swap, thresh_tamper, ref_copy, 10 fresh prompts each; --extra-behs adds the new environments that proved usable),
fitting (quiz_key neutral, leak, length_pad, 10 fresh each), pairing (ckpt_swap, thresh_tamper, ref_copy, items
40-69 = the s3_ run of 24 November).

Sampling is that of 24 November (rlp.flag.base_rate.one): temperature 1.0, top_p 0.95, top_k 20, stop at <|im_end|>,
--max-tokens 5000, and seed crc32("<uid>|<cond>[|<k>]"). The seed does not depend on the steering condition, so a
prompt has the same seed under every condition and under the single-vector runs of 24 November.

Output: CACHE/flag/base_rate/<prefix><condition>/<beh>__<cond>.jsonl, one JSON line per completion with the fields
`python -m rlp.flag.base_rate label --name <prefix><condition>` reads, plus steer, vid, seed, table_sha256. Lines are
appended as completions arrive. Re-running the same command resumes: finished (uid, k) are skipped and a truncated
last line is cut off. --label runs the labelling for every condition at the end (it executes model-written code: pod
only).

Story mode: three neutral story prompts (none mentions tests, scores or rules), --samples completions each under
every condition of --conds (default @story = base, fitv_add, fitv_sub), written to
CACHE/flag/night/story/<prefix><condition>.jsonl. Not labelled; it is read by eye.

Guards: <table>.json must match the table; the server's <table>.loaded must name the same sha256 and <table>.err must
be empty; <table>.selftest.json must say pass for this table (--no-selftest-check to skip; a projection condition
also needs proj_ok). If /health stops answering the run aborts with exit 3 instead of burning through the queue.
Failed requests are retried three times. Exit 1 if any job is still missing at the end; the last line is then
GEN_MULTI_INCOMPLETE, otherwise GEN_MULTI_DONE.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

from .. import config
from ..manif import envs as E
from . import q38 as Q

OUT = config.CACHE / "flag" / "base_rate"
NIGHT = config.CACHE / "flag" / "night"
MAX_TOKENS = 5000
PLAN_SEED = 20261125
STOP = threading.Event()

# Item indices [0, hi) that runs up to 24 November used, per environment (any prompt condition). From the run scripts:
# q38 / onp / env2 pilots used 0-39; st_ used 30-69; s2_ (run_onp_steer2.sh) used 60-89 for save_tamper and fake_pass
# and 70-99 for length_pad, leak, quiz_key; s3_ (run_onp_steer3.sh) used 40-69 for the three environments of that day.
USED_BEFORE = {"save_tamper": 90, "fake_pass": 90, "length_pad": 100, "leak": 100, "quiz_key": 100,
               "ckpt_swap": 70, "thresh_tamper": 70, "ref_copy": 70}
PRESETS = {
    "main": [dict(beh=b, cond="tempt", n=10) for b in ("save_tamper", "ckpt_swap", "thresh_tamper", "ref_copy")],
    "fitting": [dict(beh="quiz_key", cond="neutral", n=10), dict(beh="leak", cond="tempt", n=10),
                dict(beh="length_pad", cond="tempt", n=10)],
    "pairing": [dict(beh=b, cond="tempt", n=30, pick="range", start=40) for b in ("ckpt_swap", "thresh_tamper", "ref_copy")],
}
STORY_PROMPTS = [
    ("story_free", "Write a short story of about 300 words. Choose any subject you like."),
    ("story_work", "Write a short story of about 300 words about someone's ordinary day at work."),
    ("story_trip", "Write a short story of about 300 words about two friends on a trip."),
]


# ------------------------------------------------------------------------------------------------------ prompts
def used_on_disk(beh, prefix):
    """Item indices of `beh` found in any earlier run directory (not this prefix's own output)."""
    out = set()
    if not OUT.exists():
        return out
    for f in OUT.glob(f"*/{beh}__*.jsonl"):
        if f.name.endswith(".labels.jsonl") or (prefix and f.parent.name.startswith(prefix)):
            continue
        for line in open(f):
            try:
                out.add(int(json.loads(line)["uid"].rsplit("/", 1)[1]))
            except (ValueError, KeyError, IndexError):
                pass
    return out


def draw(group, prefix, seed):
    """One group spec -> plan entry with the chosen uids (and how they were chosen)."""
    g = dict(group)
    beh, cond, n = g["beh"], g["cond"], int(g["n"])
    g.setdefault("samples", 1)
    pick = g.setdefault("pick", "fresh")
    items = E.items(beh)
    if pick == "range":
        chosen = [it["idx"] for it in items[g["start"]:g["start"] + n]]
        assert len(chosen) == n, f"{beh}: only {len(chosen)} items from {g['start']}"
    elif pick == "uids":
        byuid = {it["uid"]: it["idx"] for it in items}
        chosen = [byuid[u] for u in g["uids"]]
    elif pick == "fresh":
        excl = set(range(USED_BEFORE.get(beh, 0))) | used_on_disk(beh, prefix) | {int(x) for x in g.get("exclude", [])}
        free = sorted(it["idx"] for it in items if it["idx"] not in excl)
        assert len(free) >= n, f"{beh}: {len(free)} unused items, {n} wanted"
        chosen = sorted(random.Random(f"{seed}|{beh}|{cond}").sample(free, n))
        g.update(seed=seed, n_excluded=len(excl), excluded=sorted(excl), n_free=len(free))
    else:
        raise SystemExit(f"unknown pick {pick!r}")
    g["uids"] = [f"{beh}/{i}" for i in chosen]
    return g


def make_plan(a):
    """Read the recorded plan for (prefix, plan name) or draw it now and record it."""
    groups = json.loads(a.groups) if a.groups else [dict(g) for g in PRESETS[a.preset]]
    for x in a.extra_behs:
        beh, _, cond = x.partition(":")
        groups.append(dict(beh=beh, cond=cond or "tempt", n=a.extra_n))
    name = a.plan or a.preset or "custom"
    path = NIGHT / "plans" / f"{a.prefix}{name}.json"
    spec = [{k: v for k, v in g.items()} for g in groups]
    if path.exists():
        plan = json.load(open(path))
        if plan["spec"] != spec:
            raise SystemExit(f"{path} was drawn for a different group list; use another --plan name or delete it")
        return plan, path
    plan = dict(name=name, prefix=a.prefix, seed=a.plan_seed, spec=spec, time=time.time(),
                groups=[draw(g, a.prefix, a.plan_seed) for g in groups])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(plan, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    return plan, path


def plan_items(plan):
    """[(item, prompt condition, k)] in plan order."""
    out = []
    for g in plan["groups"]:
        byuid = {it["uid"]: it for it in E.items(g["beh"])}
        for u in g["uids"]:
            for k in range(g["samples"]):
                out.append((byuid[u], g["cond"], k))
    return out


# ------------------------------------------------------------------------------------------------------- server
def healthy(root):
    try:
        with urllib.request.urlopen(root + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def read_stats(table):
    try:
        return json.load(open(str(table) + ".stats"))
    except (OSError, ValueError):
        return None


def repair(path, key=("uid", "k")):
    """Drop a truncated last line (a kill mid-write); return the set of finished keys."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    good, done = [], set()
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"[gen_multi] dropping a truncated line in {path}", flush=True)
            continue
        good.append(line)
        done.add((r[key[0]], r.get(key[1], 0)))
    fixed = b"".join(x + b"\n" for x in good)
    if fixed != raw:
        tmp = path.with_suffix(".repair")
        tmp.write_bytes(fixed)
        os.replace(tmp, path)
    return done


def seed_of(uid, cond, k):
    return zlib.crc32((f"{uid}|{cond}" + (f"|{k}" if k else "")).encode()) % 2**31     # = base_rate.one


def complete(client, a, prompt, seed, vid, max_tokens):
    """One completion with the sampling settings of 24 November; retries; aborts the run if the server is down."""
    last = None
    for attempt in range(4):
        if STOP.is_set():
            raise RuntimeError("stopped")
        try:
            return client.completions.create(
                model=a.model, prompt=prompt, max_tokens=max_tokens, temperature=Q.SAMPLING["temperature"],
                top_p=Q.SAMPLING["top_p"], stop=[Q.STOP], seed=seed,
                extra_body={"top_k": Q.SAMPLING["top_k"], "skip_special_tokens": False, "vllm_xargs": {"vid": vid}})
        except Exception as e:  # noqa: BLE001
            last = e
            if not healthy(a.root):
                STOP.set()
                raise RuntimeError(f"server unhealthy: {type(e).__name__}: {e}") from e
            time.sleep(a.retry_wait * (attempt + 1))
    raise last


def split(text):
    reasoning = Q.parse(text)["reasoning_content"]
    answer = text.split(Q.TH_CLOSE, 1)[1].strip() if Q.TH_CLOSE in text else ""
    return reasoning, answer


def one(client, tok, a, it, pcond, k, steer, vid, sha):
    msgs = E.messages(it, pcond)
    seed = seed_of(it["uid"], pcond, k)
    r = complete(client, a, Q.prompt(tok, msgs), seed, vid, a.max_tokens)
    c = r.choices[0]
    reasoning, answer = split(c.text)
    return dict(uid=it["uid"], k=k, beh=it["beh"], cond=pcond, messages=msgs, reasoning=reasoning, response=answer,
                raw=c.text, finish_reason=c.finish_reason, completion_tokens=r.usage.completion_tokens,
                prompt_tokens=r.usage.prompt_tokens, steer=steer, vid=vid, table_sha256=sha, seed=seed)


def one_story(client, tok, a, pid, text, k, steer, vid, sha):
    msgs = [{"role": "user", "content": text}]
    seed = zlib.crc32(f"story|{pid}|{k}".encode()) % 2**31
    r = complete(client, a, Q.prompt(tok, msgs), seed, vid, a.max_tokens)
    c = r.choices[0]
    reasoning, answer = split(c.text)
    return dict(uid=pid, k=k, prompt=text, reasoning=reasoning, response=answer, raw=c.text,
                finish_reason=c.finish_reason, completion_tokens=r.usage.completion_tokens,
                prompt_tokens=r.usage.prompt_tokens, steer=steer, vid=vid, table_sha256=sha, seed=seed)


def open_table(a):
    """Table identity, condition list and the guards. Returns (sha, {condition: vid}, conditions)."""
    table = Path(a.table)
    meta = json.load(open(str(table) + ".json"))
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    assert meta["sha256"] == sha, f"{table}.json was written for another table"
    names = [r["name"] for r in meta["rows"]]
    kinds = {r["name"]: r["kind"] for r in meta["rows"]}
    vid = {n: k for k, n in enumerate(names)}
    conds = []
    for c in a.conds:
        if c.startswith("@"):
            assert c[1:] in meta["sets"], f"the table has no set {c}; sets: {list(meta['sets'])}"
            conds += [n for n in meta["sets"][c[1:]] if n in vid]
        else:
            conds.append(c)
    conds = list(dict.fromkeys(conds))
    assert conds, "no conditions"
    for c in conds:
        assert c in vid, f"condition {c} is not in the table"
        assert a.allow_canary or not kinds[c].startswith("canary"), f"{c} is a self-test row"
    a.root = a.base_url.rsplit("/v1", 1)[0]
    assert healthy(a.root), f"server at {a.root} is not healthy"
    err = Path(str(table) + ".err")
    assert not (err.exists() and err.read_text().strip()), f"server reported errors: {err.read_text()[:500]}"
    if not a.no_server_check:
        lj = json.load(open(str(table) + ".loaded"))
        assert lj["sha256"] == sha and lj["names"] == names, "the server loaded a different table: restart it"
    if not a.no_selftest_check:
        st = json.load(open(str(table) + ".selftest.json"))
        assert st.get("pass") is True and st.get("sha256") == sha, "no passing self-test for this table"
        if any(kinds[c].endswith("proj") for c in conds):
            assert st.get("proj_ok") is True, "the self-test did not certify projection: drop the _proj conditions"
    return sha, vid, conds


def load_tok(a):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(a.tokenizer or a.model)


def drive(a, jobs, call, path_of, table, what):
    """Run the shuffled queue. jobs: list of tuples whose first element is the steering condition."""
    from openai import OpenAI
    random.Random(a.shuffle_seed).shuffle(jobs)
    total = len(jobs)
    left = {}
    for j in jobs:
        left[j[0]] = left.get(j[0], 0) + 1
    print(f"[gen_multi] {what}: {total} completions to run over {len(left)} unfinished conditions, "
          f"{a.workers} workers, cap {a.max_tokens} tokens", flush=True)
    if not jobs:
        return 0
    STOP.clear()
    tok = load_tok(a)
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=a.timeout, max_retries=0)
    s0 = read_stats(table)
    n = toks = errors = 0
    seen = {}
    t0 = time.time()
    lock = threading.Lock()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, client, tok, j): j for j in jobs}
        for fut in cf.as_completed(futs):
            j = futs[fut]
            n += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[gen_multi] ERROR {j[0]} {type(e).__name__}: {e}", flush=True)
                if STOP.is_set():
                    for f in futs:
                        f.cancel()
                    print("[gen_multi] server unhealthy: aborting (re-run the same command to resume)", flush=True)
                    print("GEN_MULTI_ABORTED", flush=True)
                    os._exit(3)
                continue
            toks += r["completion_tokens"]
            seen[r["vid"]] = seen.get(r["vid"], 0) + r["prompt_tokens"] + r["completion_tokens"] - 1
            with lock, open(path_of(j, r), "a") as f:           # one short append per completion
                f.write(json.dumps(r) + "\n")
            left[j[0]] -= 1
            if left[j[0]] == 0:
                print(f"[gen_multi] condition {j[0]} complete", flush=True)
            if n % a.every == 0 or n == total:
                dt = time.time() - t0
                print(f"[gen_multi] {n}/{total} done, {errors} errors, {toks} completion tokens, "
                      f"{toks / max(dt, 1e-9):.0f} tok/s, ETA {(total - n) * dt / n / 60:.0f} min", flush=True)
    # The server counts the tokens that went through the steered block per row. Each row this run used must have
    # received at least what the client saw for it; fewer means the vid did not reach the model.
    time.sleep(0.8)
    s1 = read_stats(table)
    if s0 and s1 and s0.get("pid") == s1.get("pid"):
        short = {v: (s1["tokens_per_vid"][v] - s0["tokens_per_vid"][v], e) for v, e in seen.items()
                 if s1["tokens_per_vid"][v] - s0["tokens_per_vid"][v] < e}
        if short:
            print(f"[gen_multi] ROW ACCOUNTING FAILED (vid: server tokens, client tokens): {short}", flush=True)
            print("GEN_MULTI_ROWS_WRONG", flush=True)
            sys.exit(4)
        print(f"[gen_multi] row accounting ok for {len(seen)} rows", flush=True)
    else:
        print("[gen_multi] no usable <table>.stats: row accounting not checked", flush=True)
    missing = {c: v for c, v in left.items() if v}
    if missing:
        print(f"[gen_multi] INCOMPLETE: {missing} (re-run the same command to finish)", flush=True)
        return 1
    return 0


def run(a):
    plan, ppath = make_plan(a)
    sha, vid, conds = open_table(a)
    items = plan_items(plan)
    print(f"[gen_multi] plan {ppath}: " + ", ".join(f"{g['beh']}/{g['cond']} {len(g['uids'])}x{g['samples']} ({g['pick']})"
                                                     for g in plan["groups"]), flush=True)
    jobs = []
    for steer in conds:
        D = OUT / f"{a.prefix}{steer}"
        D.mkdir(parents=True, exist_ok=True)
        done = {}
        for it, pcond, k in items:
            f = D / f"{it['beh']}__{pcond}.jsonl"
            if f not in done:
                done[f] = repair(f)
            if (it["uid"], k) not in done[f]:
                jobs.append((steer, it, pcond, k))
    rc = drive(a, jobs, lambda client, tok, j: one(client, tok, a, j[1], j[2], j[3], j[0], vid[j[0]], sha),
               lambda j, r: OUT / f"{a.prefix}{j[0]}" / f"{j[1]['beh']}__{j[2]}.jsonl", a.table,
               f"{len(conds)} conditions x {len(items)} prompt samples")
    if rc:
        print("GEN_MULTI_INCOMPLETE", flush=True)
        sys.exit(rc)
    print("GEN_MULTI_DONE", flush=True)
    if a.label:
        label(a, conds)


def story(a):
    if a.conds == ["@main"]:
        a.conds = ["@story"]
    sha, vid, conds = open_table(a)
    D = NIGHT / "story"
    D.mkdir(parents=True, exist_ok=True)
    jobs = []
    for steer in conds:
        done = repair(D / f"{a.prefix}{steer}.jsonl")
        jobs += [(steer, pid, text, k) for pid, text in STORY_PROMPTS for k in range(a.samples) if (pid, k) not in done]
    rc = drive(a, jobs, lambda client, tok, j: one_story(client, tok, a, j[1], j[2], j[3], j[0], vid[j[0]], sha),
               lambda j, r: D / f"{a.prefix}{j[0]}.jsonl", a.table, f"story, {len(conds)} conditions")
    print("GEN_MULTI_INCOMPLETE" if rc else "GEN_MULTI_DONE", flush=True)
    sys.exit(rc)


def label(a, conds):
    import subprocess
    runs = [f"{a.prefix}{c}" for c in conds]
    procs = []
    for run_ in runs:
        while len([p for p in procs if p.poll() is None]) >= a.label_par:
            time.sleep(1)
        log = open(OUT / run_ / "label.log", "w")
        procs.append(subprocess.Popen([sys.executable, "-m", "rlp.flag.base_rate", "label", "--name", run_,
                                       "--workers", str(a.label_workers)], stdout=log, stderr=subprocess.STDOUT))
    bad = [r for r, p in zip(runs, procs) if p.wait() != 0]
    print(f"GEN_MULTI_LABEL_FAILED {' '.join(bad)}" if bad else "GEN_MULTI_LABEL_DONE", flush=True)
    if bad:
        sys.exit(5)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "story", "plan"])
    ap.add_argument("--table", default="cache/flag/night/steer_table.pt")
    ap.add_argument("--prefix", required=True, help="run directory prefix, e.g. n1_ -> cache/flag/base_rate/n1_<condition>/")
    ap.add_argument("--conds", nargs="+", default=["@main"], help="condition names and/or @set names of the table")
    ap.add_argument("--allow-canary", action="store_true")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--groups", default="", help='JSON list of {"beh", "cond", "n", "samples", "pick", "start", "uids"}')
    ap.add_argument("--extra-behs", nargs="*", default=[], help="BEH[:COND] added to the preset with --extra-n fresh prompts")
    ap.add_argument("--extra-n", type=int, default=10)
    ap.add_argument("--plan", default="", help="plan name (default: the preset name, or 'custom')")
    ap.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    ap.add_argument("--samples", type=int, default=4, help="story mode: completions per prompt and condition")
    ap.add_argument("--model", default=Q.MODEL)
    ap.add_argument("--tokenizer", default="", help="tokenizer to render prompts with (default: --model)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--retry-wait", type=float, default=5.0)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--no-selftest-check", action="store_true")
    ap.add_argument("--no-server-check", action="store_true", help="do not compare <table>.loaded with the table")
    ap.add_argument("--label", action="store_true", help="run base_rate label on every run directory at the end (pod only)")
    ap.add_argument("--label-par", type=int, default=4)
    ap.add_argument("--label-workers", type=int, default=8)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.stage == "plan" or a.stage == "run":
        if not (a.preset or a.groups):
            raise SystemExit("give --preset or --groups")
    if a.stage == "plan":
        plan, path = make_plan(a)
        for g in plan["groups"]:
            print(f"{g['beh']:14s} {g['cond']:8s} {g['pick']:6s} x{g['samples']} " + " ".join(u.rsplit('/', 1)[1] for u in g["uids"]))
        print(f"plan recorded in {path}")
        return
    (run if a.stage == "run" else story)(a)


if __name__ == "__main__":
    main()

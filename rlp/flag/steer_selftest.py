"""Self-test of per-request steering (scripts/flag/night/patch_vllm_steer_multi.py). Run it on the pod before V1.

    python -m rlp.flag.steer_selftest unit                         # no server: the patch's own helper code on CPU, ~20 s
    python -m rlp.flag.steer_selftest ref  --out /workspace/ref_fp.json   # optional, against an UNPATCHED (or inactive:
                                                                   # RLP_STEER_LAYER unset) server: record reference logprobs
    python -m rlp.flag.steer_selftest live --table T [--ref /workspace/ref_fp.json]      # the canary test, 3-6 min

Exit code 0 = pass, 1 = fail (or could not certify). `live` writes <table>.selftest.json:
{"pass", "add_ok", "proj_ok", "result", "why", "sha256", "checks": {...}}; rlp.flag.gen_multi refuses to run without
"pass": true for the same table, and refuses projection conditions without "proj_ok": true. With --proj-soft a failure
of the projection check alone leaves "pass": true and "proj_ok": false (run the add rows, drop the _proj rows).

unit: executes the helper code of the patch itself (not a copy).
  1. the table checks reject a non-zero row 0, a non-unit projection row and a wrong name for row 0;
  2. the arithmetic in bf16: zero rows are a bit-exact identity; an add row equals h + v; a projection row leaves
     less than 1% of the component of the stream s = h + r along u and changes nothing else beyond rounding; a row
     that adds c*u and projects u ends where the projection row ends; rows do not mix within a batch;
  3. the runner hooks under a simulated vLLM scheduler that follows the v0.34 contract (slots from a free list and
     reused, decode-first batch order, chunked prefill under a token budget, padding to CUDA-graph sizes, preemption =
     remove + re-add in full, requests finishing mid-run, warm-up ids "_..."): every token row must carry exactly its
     own request's row, the per-row token counts must be exact, and a bad vid or an unassigned slot must raise.

live: everything goes through the completions API. A "fingerprint" of (prompt, row) is the top-10 log-probabilities of
the first --fp-tokens greedy tokens of a fixed prompt; the distance between two fingerprints is the largest absolute
difference over the tokens both list (positions are compared while the greedy tokens agree). Checks:
  identity     row 0 twice in a row, and row 0 against the second zero row base_dup, agree to --tol; with --ref, row 0
               agrees with the fingerprints recorded from an unpatched server (this is the "vid 0 = unpatched" check;
               without --ref only the two zero rows are compared, and that is said in the report);
  distinct     every tested experimental add row differs from row 0, and X_add differs from X_sub;
  canary       the norm-400 rows change the fingerprints by more than --canary-min nats, and alone they wreck a counting
               prompt ("1, 2, ..., M," must continue "M+1, ...") that the unsteered model gets right;
  concurrency  waves of mixed requests (fingerprint requests for every tested row, plus counting requests with short
               prompts and prompts of 6k-14k tokens that are prefilled in chunks, under base and canary rows) are sent
               together with random start delays. Each fingerprint must match its own row's sequential fingerprint
               (within --tol, or at least be nearer to it than to any other row's); no canary request may count
               correctly; a base request that fails in the mix must also fail alone;
  accounting   the server's tokens per row (<table>.stats) match the client's sums per row, and unused rows got none;
  projection   canary_addproj (add 100 u, then project u out) agrees with canary_proj (project u out) while
               canary_a100 (add 100 u) is far from row 0. This is all the API can show about projection: that the
               component along u of the full stream is removed after the add. That nothing else is changed is shown
               by canary_proj staying near row 0 (reported, with canary_proj_mu, not a criterion).
Never start V1 on anything but exit code 0. If this fails, steering falls back to the one-condition-at-a-time
scripts of 24 November (see the spec, "Gates and fallbacks").
"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import random
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from .. import config

PATCH = config.ROOT / "scripts" / "flag" / "night" / "patch_vllm_steer_multi.py"
CAP = 30.0                       # distance when two fingerprints share no token
FLOOR = -6.0                     # tokens below this log-probability in both fingerprints are not compared
FP_PROMPTS = [
    "The capital of France is",
    "Here is a list of fruits: apple, banana,",
    "<|im_start|>user\nName a colour.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
    "Once upon a time, in a small village by the sea, there lived",
    ("The committee met on Tuesday to review the quarterly figures. Revenue had risen in two of the three regions, "
     "costs were flat, and the new warehouse had opened a month ahead of schedule. After a short discussion of the "
     "staffing plan, the chair turned to the last item on the agenda, which was"),
]
COUNT_HEAD = "The integers from 1 to 100000 in increasing order, separated by commas:\n"


# ------------------------------------------------------------------------------------------------------------ unit
def load_helper(patch=PATCH):
    import torch
    spec = importlib.util.spec_from_file_location("rlp_patch_multi", patch)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    ns = {"torch": torch, "get_tensor_model_parallel_world_size": lambda: 1, "__name__": "rlp_helper"}
    exec(compile(P.HELPER, "patch_vllm_steer_multi.HELPER", "exec"), ns)
    return ns


class Mod:      # stands in for Qwen3_5Model: only the attributes _rlp_steer reads
    pass


def expect_raise(fn, what):
    try:
        fn()
    except RuntimeError as e:
        if "[rlp-steer]" in str(e):
            return
        raise
    raise AssertionError(f"expected a failure: {what}")


def unit(a):
    import numpy as np
    import torch
    torch.cuda.synchronize = lambda *x, **k: None        # CPU stand-in for the one CUDA call in _rlp_verify
    ns = load_helper(a.patch)
    tmp = Path(tempfile.mkdtemp())
    d, L = 64, 3
    g = torch.Generator().manual_seed(0)

    # 1. table checks
    def table(add, proj, names, path):
        torch.save(dict(add=add, proj=proj, layer=L, names=names), path)
        ns["_RLP_TABLE"] = str(path)
        return ns["_rlp_read_table"](d, L)
    K = 9
    add = torch.zeros(K, d)
    proj = torch.zeros(K, d)
    codes = {}
    for k in range(2, 6):                                 # rows 2..5 add a code vector; rows 6, 7 project; 8 does both
        v = torch.zeros(d)
        v[k] = 1.0                                        # exactly representable code: row k adds e_k * 2**k
        add[k] = v * 2.0 ** k
        codes[k] = add[k]
    for k in (6, 7):
        u = torch.randn(d, generator=g, dtype=torch.float64)
        proj[k] = (u / u.norm()).float()
        proj[k] = (proj[k].double() / proj[k].double().norm()).float()
    proj[8] = proj[6]
    add[8] = 100.0 * proj[6]
    names = ["base", "base_dup", "a2", "a3", "a4", "a5", "p6", "p7", "ap8"]
    bad = add.clone(); bad[0, 0] = 1
    expect_raise(lambda: table(bad, proj, names, tmp / "t_bad0.pt"), "row 0 non-zero")
    badp = proj.clone(); badp[6] *= 1.01
    expect_raise(lambda: table(add, badp, names, tmp / "t_badp.pt"), "non-unit projection row")
    expect_raise(lambda: table(add, proj, ["x"] + names[1:], tmp / "t_name.pt"), "row 0 not named base")
    expect_raise(lambda: table(add, proj, names[:-1] + ["a2"], tmp / "t_dup.pt"), "duplicate names")
    t = table(add, proj, names, tmp / "table.pt")
    assert t["K"] == K and t["has_proj"]
    print("unit 1/3 table checks: ok")

    # 2. arithmetic in bf16 at a realistic scale (stream norm ~80)
    def mk(dt, has_proj=True, T=4096):
        m = Mod()
        m.rlp_add = t["add"].to(dt)
        m.rlp_proj = t["proj"].clone()
        m.rlp_tok = torch.zeros(T, dtype=torch.int32)
        m.rlp_has_proj = has_proj
        return m
    steer = ns["_rlp_steer"]
    T = 512
    h = (torch.randn(T, d, generator=g) * 6).bfloat16()
    r = (torch.randn(T, d, generator=g) * 9).bfloat16()
    m = mk(torch.bfloat16)
    for k in (0, 1):
        m.rlp_tok[:T] = k
        out = steer(m, h, r)
        assert out.dtype == torch.bfloat16 and torch.equal(out, h), f"row {k} is not an exact identity"
        assert torch.equal(steer(mk(torch.bfloat16, has_proj=False), h, r), h)
    for k in range(2, 6):
        m.rlp_tok[:T] = k
        assert torch.equal(steer(m, h, r), h + codes[k].bfloat16()), f"add row {k}"
    s = h.double() + r.double()
    ends = {}
    for k in (6, 7, 8):
        m.rlp_tok[:T] = k
        u = t["proj"][k].double()
        s2 = steer(m, h, r).double() + r.double()
        ends[k] = s2
        before, after = (s @ u).abs(), (s2 @ u).abs()
        assert float((after / before.clamp_min(1e-6)).median()) < 1e-2 and float(after.max()) < 0.05 * float(before.max()), \
            f"projection row {k} left {float(after.max()):.4f} of {float(before.max()):.4f}"
        rest = (s2 - s) + (s @ u)[:, None] * u[None]          # what changed besides the removed component
        lim = (0.02 if k != 8 else 0.05) * float(s.norm(dim=-1).min())
        assert float(rest.norm(dim=-1).max()) < lim, f"projection row {k} changed other directions"
    gap = float((ends[8] - ends[6]).norm(dim=-1).max())
    assert gap < 0.05 * float(s.norm(dim=-1).min()), f"add-then-project differs from project by {gap:.3f}"
    m.rlp_tok[:T] = 2
    far = float((steer(m, h, r).double() - h.double()).norm(dim=-1).min())
    assert far > 3.9, "the add row moved nothing"
    vt = torch.randint(0, K, (T,), generator=g, dtype=torch.int32)
    m.rlp_tok[:T] = vt
    out = steer(m, h, r)
    for i in range(0, T, 37):
        mi = mk(torch.bfloat16)
        mi.rlp_tok[:1] = vt[i]
        assert torch.equal(out[i:i + 1], steer(mi, h[i:i + 1], r[i:i + 1])), "per-token rows mix"
    print("unit 2/3 arithmetic (identity, add, projection, add-then-project, mixed rows): ok")

    # 3. hooks under a simulated scheduler
    ns["_RLP_TABLE"] = str(tmp / "table.pt")
    rng = random.Random(a.seed)
    max_reqs, budget, caps = 16, 512, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    class RS:
        def __init__(self):
            self.req_id_to_index, self.free = {}, list(range(max_reqs))

    class Runner:
        pass
    run = Runner()
    run.max_num_reqs, run.req_states, run.use_pp = max_reqs, RS(), False
    run.pcp_manager = run.speculator = run.adaptive_verification = run.ubatch_runner = run.fast_prefill = None
    mf = mk(torch.float32)
    mf.rlp_add = t["add"].clone()                          # fp32 model dtype here, so codes decode exactly
    mf.rlp_has_proj = False                                # projection arithmetic is covered in part 2
    ns["_RLP"].update(model=mf, K=K, names=names, sha="x", add_cpu=mf.rlp_add, proj_cpu=mf.rlp_proj,
                      ptrs=(mf.rlp_add.data_ptr(), mf.rlp_proj.data_ptr(), mf.rlp_tok.data_ptr()),
                      counts=np.zeros(K, np.int64), vid_np=None, slot_req=None, runner_checked=False,
                      verified_real=False, steps=0, real_steps=0, last_write=0.0, n_no_vid=0)

    def h2d(tok, out):
        out.copy_(torch.from_numpy(tok))

    class SP:
        def __init__(self, vid):
            self.extra_args = None if vid is None else {"vid": vid}

    class NR:
        def __init__(self, rid, sp):
            self.req_id, self.sampling_params = rid, sp

    def add_req(rid, vid):
        st = run.req_states
        if rid in st.req_id_to_index:                      # vLLM's add_requests removes first (streaming update)
            remove(rid)
        st.req_id_to_index[rid] = st.free.pop()
        so = type("SO", (), {"scheduled_new_reqs": [NR(rid, SP(vid))]})()
        ns["_rlp_on_add"](run, so)

    def remove(rid):
        st = run.req_states
        idx = st.req_id_to_index.get(rid)
        if idx is not None:
            del st.req_id_to_index[rid]
            st.free.append(idx)
        ns["_rlp_on_remove"](rid, idx)

    want, plen, done_tok, gen_left = {}, {}, {}, {}
    waiting, running = [], []
    expected_counts = np.zeros(K, np.int64)
    nxt = steps = preempt = finish = 0
    add_req("_warmup_0_", None)                            # vLLM warm-up style request without a vid
    want["_warmup_0_"], plen["_warmup_0_"], done_tok["_warmup_0_"], gen_left["_warmup_0_"] = 0, 7, 0, 1
    running.append("_warmup_0_")
    while steps < a.steps:
        steps += 1
        for _ in range(rng.choice([0, 0, 1, 2, 3])):       # arrivals
            rid = f"cmpl-{nxt}"
            nxt += 1
            want[rid] = rng.randrange(K)
            plen[rid] = rng.choice([rng.randint(1, 40), rng.randint(40, 600), rng.randint(600, 3000)])
            gen_left[rid] = rng.randint(1, 60)
            done_tok[rid] = 0
            waiting.append(rid)
        if running and rng.random() < 0.05:                # preempt a running request (re-sent in full later)
            rid = rng.choice(running)
            if not rid.startswith("_"):
                running.remove(rid)
                remove(rid)
                done_tok[rid] = 0
                waiting.insert(0, rid)
                preempt += 1
        sched, left = {}, budget                           # running first, then new ones, under the token budget
        for rid in running:
            need = (plen[rid] - done_tok[rid]) if done_tok[rid] < plen[rid] else 1
            n = min(need, left)
            if n <= 0:
                break
            sched[rid] = n
            left -= n
        while waiting and left > 0 and len(running) < max_reqs:
            rid = waiting.pop(0)
            add_req(rid, want[rid])
            running.append(rid)
            n = min(plen[rid], left)
            sched[rid] = n
            left -= n
        if not sched:
            continue
        order = sorted(sched, key=lambda x: (sched[x] != 1, sched[x]))     # decode-like first
        nst = np.array([sched[x] for x in order], np.int32)
        n = int(nst.sum())
        npad = next((c for c in caps if c >= n), n)
        qsl = np.empty(max_reqs + 1, np.int32)
        qsl[0] = 0
        np.cumsum(nst, out=qsl[1:len(order) + 1])
        qsl[len(order) + 1:] = n
        ib = type("IB", (), dict(num_tokens=n, num_tokens_after_padding=npad, num_reqs=len(order), req_ids=order,
                                 idx_mapping_np=np.array([run.req_states.req_id_to_index[x] for x in order]),
                                 query_start_loc_np=qsl))()
        brs = type("BRS", (), dict(num_scheduled_tokens=nst))()
        ns["_rlp_on_prepare"](run, ib, brs, h2d)
        tag = torch.tensor(np.repeat(np.arange(len(order)), nst), dtype=torch.float32)
        hh = torch.zeros(npad, d)
        hh[:n, 0] = tag + 1000
        out = ns["_rlp_steer"](mf, hh, torch.zeros_like(hh))
        delta = out[:n, 2:6]                              # code columns
        for j, rid in enumerate(order):
            rows = delta[int(qsl[j]):int(qsl[j + 1])]
            v = want[rid]
            ref = torch.zeros(4)
            if 2 <= v <= 5:
                ref[v - 2] = 2.0 ** v
            if v != 8:                                    # row 8 adds a dense vector: only the others carry a code
                assert torch.equal(rows, ref.expand_as(rows)), f"step {steps}: request {rid} (row {v}) got {rows[0].tolist()}"
            else:
                assert torch.allclose(out[int(qsl[j]):int(qsl[j + 1])] - hh[int(qsl[j]):int(qsl[j + 1])],
                                      mf.rlp_add[8].expand(len(rows), -1), atol=1e-4), "row 8 add"
        assert torch.equal(out[n:], hh[n:]), "padding rows changed"
        for j, rid in enumerate(order):
            expected_counts[want[rid]] += sched[rid]
            done_tok[rid] += sched[rid]
            if done_tok[rid] > plen[rid]:
                gen_left[rid] -= 1
            if done_tok[rid] >= plen[rid] and gen_left[rid] <= 0:
                running.remove(rid)
                remove(rid)
                finish += 1
    got = ns["_RLP"]["counts"]
    assert np.array_equal(got, expected_counts), f"token counts per row {got} != {expected_counts}"
    ns["_rlp_write_stats"]()
    st = json.load(open(str(tmp / "table.pt") + ".stats"))
    assert st["tokens_per_vid"] == expected_counts.tolist()
    # negative cases
    ns["_RLP_REQUIRE_VID"] = True
    expect_raise(lambda: add_req("cmpl-novid", None), "request without vid when a vid is required")
    remove("cmpl-novid")
    ns["_RLP_REQUIRE_VID"] = False
    add_req("cmpl-novid", None)                            # default: row 0 and counted
    assert ns["_RLP"]["vid_np"][run.req_states.req_id_to_index["cmpl-novid"]] == 0 and ns["_RLP"]["n_no_vid"] == 1
    remove("cmpl-novid")
    expect_raise(lambda: add_req("cmpl-big", K), "vid out of range")
    remove("cmpl-big")
    expect_raise(lambda: add_req("cmpl-str", "x1"), "non-integer vid")
    remove("cmpl-str")
    add_req("cmpl-ok", "3")                                # digit strings are accepted
    assert ns["_RLP"]["vid_np"][run.req_states.req_id_to_index["cmpl-ok"]] == 3
    victim = "cmpl-ok"                                     # a scheduled slot that add_requests never assigned must fail
    ns["_RLP"]["slot_req"][run.req_states.req_id_to_index[victim]] = None
    nst = np.array([1], np.int32)
    qsl = np.array([0, 1] + [1] * (max_reqs - 1), np.int32)
    ib = type("IB", (), dict(num_tokens=1, num_tokens_after_padding=1, num_reqs=1, req_ids=[victim],
                             idx_mapping_np=np.array([run.req_states.req_id_to_index[victim]]),
                             query_start_loc_np=qsl))()
    expect_raise(lambda: ns["_rlp_on_prepare"](run, ib, type("B", (), dict(num_scheduled_tokens=nst))(), h2d),
                 "unassigned slot")
    run2 = Runner()
    run2.__dict__.update(run.__dict__)
    run2.fast_prefill = object()
    ns["_RLP"]["runner_checked"] = False
    expect_raise(lambda: ns["_rlp_on_prepare"](run2, ib, type("B", (), dict(num_scheduled_tokens=nst))(), h2d),
                 "fast prefill enabled")
    print(f"unit 3/3 hooks: ok ({steps} steps, {nxt} requests, {preempt} preemptions, {finish} finished, "
          f"slot reuse, chunked prefill, padding, warm-up id)")
    print("UNIT_PASS")
    return 0


# ------------------------------------------------------------------------------------------------------------ live
def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def preemptions(root):
    try:
        txt = http_get(root + "/metrics")
    except Exception:  # noqa: BLE001
        return None
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:num_preemptions") and not line.startswith("#"):
            try:
                tot += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return tot


def read_stats(table, after):
    p = Path(str(table) + ".stats")
    if not p.exists() or p.stat().st_mtime < after:
        return None
    try:
        return json.load(open(p))
    except ValueError:
        return None


def fingerprint(client, a, prompt, vid, delay=0.0):
    """Top log-probabilities of the first greedy tokens. vid None = send no vid (the reference mode)."""
    time.sleep(delay)
    extra = {} if vid is None else {"vllm_xargs": {"vid": vid}}
    r = client.completions.create(model=a.model, prompt=prompt, max_tokens=a.fp_tokens, temperature=0.0,
                                  logprobs=a.fp_top, extra_body=extra)
    c = r.choices[0]
    lp = c.logprobs
    return dict(tokens=list(lp.tokens), top=[dict(x) for x in lp.top_logprobs],
                prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)


def distance(fa, fb):
    """Largest |difference| of log-probability over tokens both fingerprints list, over the positions whose earlier
    greedy tokens agree. CAP if a compared position shares no token above FLOOR."""
    d = 0.0
    for i in range(min(len(fa["top"]), len(fb["top"]))):
        if i and fa["tokens"][:i] != fb["tokens"][:i]:
            break
        ta, tb = fa["top"][i], fb["top"][i]
        shared = [k for k in ta if k in tb and max(ta[k], tb[k]) >= FLOOR]
        if not shared:
            return CAP
        d = max(d, max(abs(ta[k] - tb[k]) for k in shared))
    return min(d, CAP)


def rowdist(A, B):
    """Largest distance over the fixed prompts between two rows' fingerprint lists."""
    return max(distance(x, y) for x, y in zip(A, B))


def live(a):
    from openai import OpenAI
    table = Path(a.table)
    meta = json.load(open(str(table) + ".json"))
    names = [r["name"] for r in meta["rows"]]
    kinds = {r["name"]: r["kind"] for r in meta["rows"]}
    vid = {n: k for k, n in enumerate(names)}
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    root = a.base_url.rsplit("/v1", 1)[0]
    report = dict(table=str(table), sha256=sha, time=time.time(), checks={}, add_ok=False, proj_ok=False)
    report["pass"] = False

    def finish(code, why, inconclusive=False):
        report["result"] = "PASS" if code == 0 else "INCONCLUSIVE" if inconclusive else "FAIL"
        report["why"] = why
        report["pass"] = code == 0
        if code == 0:
            report["add_ok"] = True
        out = Path(a.out or str(table) + ".selftest.json")
        json.dump(report, open(out, "w"), indent=1, default=str)
        print(f"SELFTEST_{report['result']}: {why}  (report {out})", flush=True)
        return code

    # 0. the server loaded exactly this table and has not failed
    try:
        http_get(root + "/health")
    except Exception as e:  # noqa: BLE001
        return finish(1, f"server not healthy: {e}")
    err = Path(str(table) + ".err")
    if err.exists() and err.read_text().strip():
        return finish(1, f"server wrote {err}: {err.read_text().strip()[:500]}")
    ld = Path(str(table) + ".loaded")
    if not ld.exists():
        return finish(1, f"{ld} missing: the patched model never loaded the table (patch not applied, or RLP_STEER_LAYER unset)")
    lj = json.load(open(ld))
    if lj["sha256"] != sha or lj["names"] != names or meta["sha256"] != sha:
        return finish(1, "the server's table differs from the table file / its .json")
    report["checks"]["loaded"] = dict(K=lj["K"], has_proj=lj.get("has_proj"), pid=lj.get("pid"))
    need = ["base", "base_dup", "canary_a", "canary_b", "canary_c", "canary_a100", "canary_proj", "canary_addproj"]
    miss = [n for n in need if n not in vid]
    if miss:
        return finish(1, f"the table has no self-test rows {miss}: build it without --no-canary")

    client = OpenAI(base_url=a.base_url, api_key="x", timeout=900, max_retries=0)
    rng = random.Random(a.seed)

    # rows whose fingerprints are taken: zero rows, canaries, and a sample of the experimental rows
    exp_add = [n for n in names if kinds[n] in ("rand", "cand", "cand_add", "pair")]
    exp_proj = [n for n in names if kinds[n] in ("rand_proj", "cand_proj")]
    pick = [n for d in ("sprobe", "fitv", "gf") for n in (f"{d}_sub", f"{d}_add") if n in vid]
    pick += [n for n in ("iso0_add", "iso0_sub", "sig0_add", "sig0_sub", "probe_sub", "noise_sub") if n in vid]
    rest = [n for n in exp_add if n not in pick]
    rng.shuffle(rest)
    pick = (pick + rest)[:a.n_exp_rows]
    pproj = [n for n in ("sprobe_proj", "iso0_proj") if n in vid] or exp_proj[:2]
    rows = need + (["canary_proj_mu"] if "canary_proj_mu" in vid else []) + pick + pproj
    klass = {n: n for n in rows}
    klass["base_dup"] = "base"
    klass["canary_addproj"] = "canary_proj"
    report["checks"]["rows_tested"] = rows

    # 1. sequential fingerprints: the server is otherwise idle, one request at a time
    def seq(v):
        return [fingerprint(client, a, p, v) for p in FP_PROMPTS]
    try:
        S = {n: seq(vid[n]) for n in rows}
        base2 = seq(0)
    except Exception as e:  # noqa: BLE001
        return finish(1, f"a sequential request failed: {type(e).__name__}: {e}")
    D0 = {n: rowdist(S[n], S["base"]) for n in rows}
    noise = rowdist(S["base"], base2)
    dup = D0["base_dup"]
    ident = dict(repeat=noise, base_dup=dup, tol=a.tol, ref=None)
    if a.ref:
        R = json.load(open(a.ref))
        if R["prompts"] != FP_PROMPTS or R["fp_tokens"] != a.fp_tokens or R["fp_top"] != a.fp_top:
            return finish(1, f"{a.ref} was recorded with other prompts or settings")
        ident["ref"] = rowdist(S["base"], R["fingerprints"])
        ident["ref_model"] = R.get("model")
    else:
        ident["note"] = "no --ref: row 0 was compared with the second zero row only, not with an unpatched server"
    report["checks"]["identity"] = ident
    report["checks"]["dist_from_base"] = {n: round(D0[n], 4) for n in rows}
    print(f"sequential: repeat {noise:.4f}, base_dup {dup:.4f}, ref {ident['ref']}; distance from row 0: "
          + " ".join(f"{n} {D0[n]:.2f}" for n in rows if n not in ("base", "base_dup")), flush=True)
    if noise > a.tol or dup > a.tol:
        return finish(1, f"row 0 is not reproducible / base_dup differs from base (repeat {noise:.3f}, base_dup {dup:.3f}, "
                         f"tolerance {a.tol})")
    if ident["ref"] is not None and ident["ref"] > a.tol_ref:
        return finish(1, f"row 0 differs from the unpatched reference by {ident['ref']:.3f} nats (tolerance {a.tol_ref})")
    thr = max(4 * max(noise, dup), a.min_diff)
    same = [n for n in pick if D0[n] <= thr]
    pairs = [(n, n[:-4] + "_sub") for n in pick if n.endswith("_add") and n[:-4] + "_sub" in pick]
    same_pair = [(x, y) for x, y in pairs if rowdist(S[x], S[y]) <= thr]
    report["checks"]["distinct"] = dict(threshold=thr, rows=pick, not_distinct=same, pairs_not_distinct=same_pair)
    if same or same_pair:
        return finish(1, f"steering rows indistinguishable from row 0 or from their opposite sign (threshold {thr:.3f}): "
                         f"{same} {same_pair}: the vid does not reach the model")
    weak = [c for c in ("canary_a", "canary_b", "canary_c", "canary_a100") if D0[c] < a.canary_min]
    report["checks"]["canary_dist"] = {c: D0[c] for c in ("canary_a", "canary_b", "canary_c", "canary_a100")}
    if weak:
        return finish(1, f"canary rows {weak} change the log-probabilities by less than {a.canary_min} nats", True)

    # 2. counting prompts: calibrate alone
    cache = {}

    def count_call(M, k, v, delay=0.0):
        time.sleep(delay)
        if M not in cache:
            cache[M] = COUNT_HEAD + ", ".join(map(str, range(1, M + 1))) + ","
        exp = list(range(M + 1, M + 1 + k))
        mt = k * (len(str(M + k)) + 3) + 8
        t0 = time.time()
        r = client.completions.create(model=a.model, prompt=cache[M], max_tokens=mt, temperature=0.0,
                                      extra_body={"vllm_xargs": {"vid": v}})
        c = r.choices[0]
        got = [int(x) for x in re.findall(r"\d+", c.text)[:k]]
        return dict(kind="count", M=M, k=k, vid=v, name=names[v], ok=got == exp, text=c.text[:120],
                    prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                    secs=round(time.time() - t0, 1))
    try:
        probe = count_call(a.m_long_lo, a.k, 0)
        tpn = probe["prompt_tokens"] / a.m_long_lo                     # prompt tokens per number
        m_hi = max(a.m_long_lo, min(a.m_long_hi, int(a.max_prompt_tokens / tpn)))
        Ms_short = (8, 60, 250)
        cal = [count_call(M, a.k, 0) for M in (Ms_short[0], Ms_short[2])] + [probe]
        report["checks"]["calibration"] = dict(tokens_per_number=tpn, m_long=[a.m_long_lo, m_hi], runs=cal)
        if not all(x["ok"] for x in cal):
            return finish(1, f"the unsteered model does not continue the count when alone: {[x['text'] for x in cal]}", True)
        cw = count_call(Ms_short[1], a.k, vid["canary_a"])
        report["checks"]["canary_alone"] = cw
        if cw["ok"]:
            return finish(1, f"canary_a does not wreck the count even alone ({cw['text']!r}): raise --canary-norm", True)
    except Exception as e:  # noqa: BLE001
        return finish(1, f"a calibration request failed: {type(e).__name__}: {e}")

    # 3. mixed waves
    def flush():
        return count_call(Ms_short[0], 2, 0)
    base_rows, canaries = ["base", "base_dup"], ["canary_a", "canary_b", "canary_c"]
    p0 = preemptions(root)
    t_wave = time.time()
    time.sleep(1.0)
    flush(); time.sleep(0.7); flush(); time.sleep(0.7)
    s0 = None if a.no_stats_check else read_stats(table, t_wave)
    if s0 is None and not a.no_stats_check:
        return finish(1, "no fresh <table>.stats after requests: the runner hooks are not running")
    fps, counts = [], []
    for w in range(a.waves):
        jobs = []
        for n in rows:
            for pi, p in enumerate(FP_PROMPTS):
                jobs.append(("fp", n, pi, rng.uniform(0, a.spread)))
        for i in range(a.n_long + a.n_short):
            u = rng.random()
            name = rng.choice(base_rows) if u < 0.45 else rng.choice(canaries) if u < 0.8 else rng.choice(pick)
            M = rng.randint(a.m_long_lo, m_hi) if i < a.n_long else rng.randint(Ms_short[0], Ms_short[2])
            k = a.k if rng.random() < 0.7 else a.k_long            # some long generations keep decodes running
            jobs.append(("count", name, (M, k), rng.uniform(0, a.spread)))
        rng.shuffle(jobs)
        t0 = time.time()
        with cf.ThreadPoolExecutor(len(jobs)) as ex:
            futs = []
            for kind, n, arg, delay in jobs:
                if kind == "fp":
                    futs.append((kind, n, arg, ex.submit(fingerprint, client, a, FP_PROMPTS[arg], vid[n], delay)))
                else:
                    futs.append((kind, n, arg, ex.submit(count_call, arg[0], arg[1], vid[n], delay)))
            for kind, n, arg, f in futs:
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001
                    return finish(1, f"request failed during wave {w}: {type(e).__name__}: {e}")
                (fps if kind == "fp" else counts).append((n, arg, r))
        print(f"wave {w + 1}/{a.waves}: {len(jobs)} mixed requests in {time.time() - t0:.0f} s", flush=True)
    p1 = preemptions(root)
    t_end = time.time()
    time.sleep(0.7); flush(); time.sleep(0.7); flush(); time.sleep(0.7)
    s1 = None if a.no_stats_check else read_stats(table, t_end)
    if not a.no_stats_check and (s1 is None or s1["pid"] != s0["pid"]):
        return finish(1, "server restarted or stopped writing <table>.stats during the test")
    pre = None if p0 is None or p1 is None else p1 - p0
    report["checks"]["preemptions"] = pre

    # 4. concurrency verdicts
    wrong, d_owns, nearest_ok = [], [], 0
    for n, pi, r in fps:
        d_own = distance(r, S[n][pi])
        others = {m: distance(r, S[m][pi]) for m in rows if klass[m] != klass[n]}
        m_best = min(others, key=others.get)
        d_owns.append(d_own)
        nearest_ok += d_own <= others[m_best]
        if d_own > a.tol and d_own >= others[m_best]:
            wrong.append(dict(row=n, prompt=pi, d_own=round(d_own, 3), nearest=m_best, d_nearest=round(others[m_best], 3)))
    d_owns.sort()
    report["checks"]["concurrency"] = dict(n=len(fps), max_d_own=d_owns[-1], median_d_own=d_owns[len(d_owns) // 2],
                                           nearest_is_own=nearest_ok, wrong=wrong[:20], n_wrong=len(wrong))
    if wrong:
        return finish(1, f"{len(wrong)} of {len(fps)} mixed-batch requests did not get their own row, e.g. {wrong[:3]}")
    by = {}
    for n, _, r in counts:
        by.setdefault(n, []).append(r)
    summ = {n: dict(n=len(v), ok=sum(x["ok"] for x in v)) for n, v in sorted(by.items())}
    report["checks"]["counting"] = summ
    bad_canary = [r for n, _, r in counts if n in canaries and r["ok"]]
    if bad_canary:
        return finish(1, f"{len(bad_canary)} canary requests counted correctly in the mix (they were steered as another "
                         f"row): {[(x['name'], x['M']) for x in bad_canary[:5]]}")
    n_can = sum(n in canaries for n, _, _ in counts)
    if n_can < 10:
        return finish(1, "too few canary counting requests to certify; raise --waves", True)
    quirks = []
    for n, _, r in counts:
        if n in base_rows and not r["ok"]:                 # re-send alone: a mapping error disappears when alone
            y = count_call(r["M"], r["k"], r["vid"])
            if y["ok"]:
                # Lead, 25 Nov: the count can break off under batching even with row 0 only (measured: M=106 fails in
                # 11 of 80 all-row-0 concurrent requests, same text). So before calling it a leak, send 24 concurrent
                # copies with row 0 only; if any of those fails too, it is a batching quirk of the prompt, not a leak.
                with cf.ThreadPoolExecutor(24) as ex2:
                    ctl = list(ex2.map(lambda _: count_call(r["M"], r["k"], r["vid"]), range(24)))
                if not all(c["ok"] for c in ctl):
                    quirks.append(r)
                    continue
                return finish(1, f"base request (M={r['M']}, row {n}) failed in the mixed batch but passes alone: another "
                                 f"request's row leaked into it. Text: {r['text']!r}")
            quirks.append(r)
    report["checks"]["base_quirks"] = quirks
    n_base = sum(n in base_rows for n, _, _ in counts)
    # Lead, 25 Nov: the model breaks off the count for some M even with row 0 only (11 of 80 in an all-row-0
    # control run, scripts/flag/night/leaktest.py), so the limit is raised from 5% to 50% of base requests.
    if len(quirks) > max(2, n_base // 2):
        return finish(1, f"{len(quirks)} base requests fail even alone: the counting prompt is not reliable enough", True)

    # 5. token accounting per row
    if not a.no_stats_check:
        O = [y - x for x, y in zip(s0["tokens_per_vid"], s1["tokens_per_vid"])]
        Ex, nreq = [0] * len(names), [0] * len(names)
        for n, _, r in fps + counts:
            Ex[vid[n]] += r["prompt_tokens"] + r["completion_tokens"] - 1
            nreq[vid[n]] += 1
        acc = []
        for v, n in enumerate(names):
            if v == 0:
                ok = O[v] >= Ex[v]                         # row 0 also carries the flush and re-sent requests
            elif nreq[v] == 0:
                ok = O[v] == 0
            elif pre:                                      # recomputation after preemption adds tokens
                ok = Ex[v] <= O[v] <= 2 * Ex[v] + 2 * nreq[v]
            else:
                ok = Ex[v] <= O[v] <= Ex[v] + 2 * nreq[v]
            acc.append(dict(name=n, expected=Ex[v], server=O[v], requests=nreq[v], ok=ok))
        report["checks"]["token_accounting"] = [r for r in acc if r["requests"] or not r["ok"]]
        bad = [r for r in acc if not r["ok"]]
        if bad:
            return finish(1, f"per-row token counts disagree with the client's: {bad[:5]}")
        report["checks"]["n_no_vid"] = s1.get("n_no_vid")

    # 6. projection
    d_ap = rowdist(S["canary_addproj"], S["canary_proj"])
    d_a100 = D0["canary_a100"]
    lim = min(max(a.tol, 0.1 * d_a100), 1.0)
    mixed_ap = max([distance(r, S["canary_proj"][pi]) for n, pi, r in fps if n == "canary_addproj"] or [0.0])
    proj_ok = d_ap <= lim and d_a100 >= a.canary_min
    report["checks"]["projection"] = dict(
        addproj_vs_proj=d_ap, addproj_vs_proj_mixed=mixed_ap, a100_vs_base=d_a100, limit=lim,
        proj_vs_base=D0["canary_proj"], proj_mu_vs_base=D0.get("canary_proj_mu"),
        exp_proj_vs_base={n: D0[n] for n in pproj}, ok=proj_ok)
    report["proj_ok"] = bool(proj_ok)
    msg = (f"{len(fps)} mixed fingerprints matched their rows (largest distance {d_owns[-1]:.3f}); canaries wrecked "
           f"{n_can}/{n_can}; base counted {n_base - len(quirks)}/{n_base}; identity repeat {noise:.3f}, base_dup {dup:.3f}"
           + (f", unpatched reference {ident['ref']:.3f}" if ident["ref"] is not None else ", no unpatched reference")
           + ("; token accounting exact" if not a.no_stats_check else "; token accounting skipped")
           + f"; preemptions {pre}; projection: add-then-project vs project {d_ap:.3f} (limit {lim:.2f}), add alone {d_a100:.2f}")
    if not proj_ok and not a.proj_soft:
        return finish(1, "projection check failed (the add rows passed; --proj-soft would certify those alone). " + msg)
    if not proj_ok:
        msg = "PROJECTION NOT CERTIFIED, drop the _proj conditions. " + msg
    return finish(0, msg)


def ref(a):
    """Record the fingerprints of the fixed prompts from a server without steering (no vid is sent)."""
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="x", timeout=900, max_retries=0)
    fp = [fingerprint(client, a, p, None) for p in FP_PROMPTS]
    out = a.out or "ref_fp.json"
    json.dump(dict(model=a.model, prompts=FP_PROMPTS, fp_tokens=a.fp_tokens, fp_top=a.fp_top, fingerprints=fp,
                   time=time.time()), open(out, "w"), indent=1)
    print(f"REF_WRITTEN {out}")
    return 0


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["unit", "live", "ref"])
    ap.add_argument("--table", default="cache/flag/night/steer_table.pt")
    ap.add_argument("--patch", default=str(PATCH))
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--ref", default="", help="fingerprints recorded by `ref` from an unpatched server")
    ap.add_argument("--tol", type=float, default=0.3, help="nats: two runs of the same row must agree to this")
    ap.add_argument("--tol-ref", type=float, default=0.3)
    ap.add_argument("--min-diff", type=float, default=0.02, help="nats: a steering row must differ from row 0 by more")
    ap.add_argument("--canary-min", type=float, default=1.0)
    ap.add_argument("--fp-tokens", type=int, default=3)
    ap.add_argument("--fp-top", type=int, default=10)
    ap.add_argument("--n-exp-rows", type=int, default=12)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--n-long", type=int, default=10)
    ap.add_argument("--n-short", type=int, default=60)
    ap.add_argument("--k", type=int, default=16, help="numbers the continuation must get right")
    ap.add_argument("--k-long", type=int, default=120, help="numbers for the long-generation requests")
    ap.add_argument("--m-long-lo", type=int, default=1200)
    ap.add_argument("--m-long-hi", type=int, default=2600)
    ap.add_argument("--max-prompt-tokens", type=int, default=14000)
    ap.add_argument("--spread", type=float, default=12.0, help="start delays drawn from [0, spread] s")
    ap.add_argument("--proj-soft", action="store_true", help="a projection failure alone does not fail the test")
    ap.add_argument("--no-stats-check", action="store_true")
    ap.add_argument("--steps", type=int, default=3000, help="unit: simulated scheduler steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    return dict(unit=unit, live=live, ref=ref)[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())

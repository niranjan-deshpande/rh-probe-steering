"""Tests for the reader modules of the night of 24-25 November 2026 (builder B2).

    HF_HUB_OFFLINE=1 .venv/bin/python -m pytest rlp/flag/test_reader_night.py -q
    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.test_reader_night [name ...]     # the same tests where pytest is not
                                                                                 # installed; last line READER_NIGHT_TESTS_OK

The model is a tiny random Qwen3.5 text model with the 27B's block pattern (three linear-attention blocks, then one
full-attention block), built here in float32 on the CPU with the Qwen3.5-9B tokenizer from the local cache. Tests
that need it are skipped if that tokenizer is not cached.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ImportError:                                  # this venv has no pytest: a stand-in for the few things used here
    class _Skip(Exception):
        pass

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            assert t is not None and issubclass(t, self.exc), f"expected {self.exc.__name__}, got {t}"
            return True

    class _Approx:
        def __init__(self, x):
            self.x = x

        def __eq__(self, o):
            return abs(o - self.x) <= 1e-6 * max(1.0, abs(self.x))

    class pytest:                                    # noqa: N801
        Skip = _Skip
        fixture = staticmethod(lambda **kw: (lambda fn: fn))
        raises = staticmethod(_Raises)
        approx = staticmethod(_Approx)

        @staticmethod
        def skip(msg=""):
            raise _Skip(msg)

from rlp import config
from rlp.flag import fitw, gf27 as G, linfit as LF, mt_harvest as MH, onp, q38 as Q, textscore as TS

L = 6
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tiny():
    try:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception as e:
        pytest.skip(f"Qwen3.5 tokenizer or model code not available offline: {e}")
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=16, linear_value_head_dim=16,
                            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def ns(**kw):
    return types.SimpleNamespace(**kw)


# ------------------------------------------------------------------ gf27: spans, harvest, directions, detection

def test_locate_and_pool_spans():
    im = 7
    ids = [im, 1, 2, 50, 51, im, 1, 2] + list(range(100, 110)) + [9, 9, 9]
    row = dict(id="r", input_ids=ids, spans=[dict(name="read_close", start=12, end=18)])
    loc = G.locate(row, im, header=[im, 1, 2])
    assert loc["ro"] == [(12, 18, 8)] and loc["ids"] == ids[:18]          # cut after the span, turn starts after the header
    H = torch.arange(18, dtype=torch.float32)[:, None] * torch.ones(1, 4)
    suf, last, turn = G.pool_spans(H, loc["ro"])
    assert torch.allclose(suf[0], torch.full((4,), 14.5)) and torch.allclose(last[0], torch.full((4,), 17.0))
    assert torch.allclose(turn[0], torch.full((4,), 12.5))                # mean of positions 8..17
    with pytest.raises(ValueError):
        G.locate(dict(id="r", input_ids=ids, spans=[dict(name="x", start=12, end=17)]), im)
    with pytest.raises(ValueError):
        G.locate(row, im, header=[im, 1, 3])


def test_batches_respect_budget():
    lens = [50, 40, 30, 5, 5, 5, 5]
    bs = list(G.batches(list(range(7)), lens, 100))
    assert sorted(i for b in bs for i in b) == list(range(7))
    assert all(len(b) == 1 or max(lens[i] for i in b) * len(b) <= 100 for b in bs)
    assert list(G.batches([0], [500], 100)) == [[0]]
    assert list(G.batches(list(range(7)), lens, 100, solo=35)) == [[0], [1], [2, 3, 4], [5, 6]]     # long rows go alone
    ids, am = G.pad_batch([[1, 2, 3], [4, 5, 6]], 0, "cpu")
    assert am is None and ids.tolist() == [[1, 2, 3], [4, 5, 6]]          # no padding, no mask
    ids, am = G.pad_batch([[1, 2, 3], [4]], 0, "cpu")
    assert am.tolist() == [[1, 1, 1], [1, 0, 0]] and ids.tolist() == [[1, 2, 3], [4, 0, 0]]


def test_gf27_harvest_on_their_ids(tiny, tmp_path):
    if not G.RECORDS.exists():
        pytest.skip("Goodfire records not on this machine")
    model, tok = tiny
    a = ns(model=(model, tok), records=str(G.RECORDS), limit=14, layer=L, out=str(tmp_path), shard=5, budget=1500, tol=1e-3, no_check=False)
    G.harvest(a)
    z, meta = np.load(tmp_path / "gf27_acts.npz"), json.load(open(tmp_path / "gf27_meta.json"))
    rows = G.load_records(G.RECORDS, 14)
    assert len(meta) == sum(len(r["spans"]) for r in rows) == len(z["suf"]) and z["ok"].all()
    assert len(list((tmp_path / "shards").glob("s*.npz"))) == 3
    for j in (0, len(meta) // 2, len(meta) - 1):                          # against a one-sequence read of the whole record
        m = meta[j]
        h = onp.read_layers(model, rows[m["rec"]]["input_ids"], (L,))[L].float()
        assert np.allclose(z["suf"][j], h[m["start"]:m["end"]].mean(0).numpy(), atol=1e-4)
        assert np.allclose(z["last"][j], h[m["end"] - 1].numpy(), atol=2e-2)
        assert m["end"] - m["start"] == 6 and tok.decode(rows[m["rec"]]["input_ids"][m["start"]:m["end"]]).endswith("<|im_end|>")
    before = {f.name: f.stat().st_mtime_ns for f in (tmp_path / "shards").glob("s*.npz")}
    G.harvest(a)                                                          # resume: nothing is read again
    assert before == {f.name: f.stat().st_mtime_ns for f in (tmp_path / "shards").glob("s*.npz")}
    with pytest.raises(SystemExit):
        G.harvest(ns(**{**vars(a), "shard": 7}))                          # other settings must not mix into old shards


def _fake_gf(d=24, seed=0):
    rows = G.load_records(G.RECORDS)
    meta, rng = G.meta_rows(rows), np.random.default_rng(seed)
    cells = sorted({(m["dataset"], m["cell"], m["span"]) for m in meta})
    mu = {c: rng.standard_normal(d) * 3 for c in cells}
    X = np.stack([mu[(m["dataset"], m["cell"], m["span"])] for m in meta]) + 0.01 * rng.standard_normal((len(meta), d))
    return rows, meta, mu, X


def test_gf27_dirs_from_contrasts():
    if not G.RECORDS.exists():
        pytest.skip("Goodfire records not on this machine")
    rows, meta, mu, X = _fake_gf()
    C = json.load(open(G.CONTRASTS))
    val = np.array([m["split"] == "validation" for m in meta])
    Xv = X.copy()
    Xv[val] += 100.0                                                      # their validation split must not enter
    acts = dict(suf=Xv, last=Xv, turn=Xv, ok=np.ones(len(meta), bool))
    D, info = G.build_dirs(acts, meta, C, "train")
    assert set(D) >= {"gf_pool", "last:gf_pool", "turn:gf_pool"} | {f"c:{c}" for c in C} and len(C) == 18
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in D.values())
    raw = {}
    for name, c in C.items():
        cell = lambda k: np.mean([mu[(c["dataset"], k, s)] for s in c["span"].split("+") if (c["dataset"], k, s) in mu], 0)
        raw[name] = cell(c["pos"][0]) - cell(c["neg"][0]) if len(c["pos"]) == 1 else None
    for name in [n for n in C if n.startswith("i_")]:                     # hack side minus honest side, per contrast
        assert G.unit(raw[name]) @ D[f"c:{name}"] > 0.999
    pool = G.unit(np.mean([raw[n] for n in C if n.startswith("i_")], 0))  # equal weight over the 12 single contrasts
    assert pool @ D["gf_pool"] > 0.999 and len(info["pool_members"]) == 12
    assert info["contrasts"]["i_peek_h"]["hack_cells"] == ["peek_hidden"] and info["contrasts"]["i_peek_h"]["span"] == "read_close"
    bad = json.loads(json.dumps(meta))
    j = next(i for i, m in enumerate(bad) if "i_rm_test" in m["labels"])
    bad[j]["labels"]["i_rm_test"] = 1 - bad[j]["labels"]["i_rm_test"]
    with pytest.raises(AssertionError):
        G.contrast_labels(bad, C)


def test_gf27_detect_table():
    rng = np.random.default_rng(0)
    d, meta, X = 16, [], []
    u = G.unit(rng.standard_normal(d))
    for env in ("a", "b"):
        for p in range(6):
            for k in range(4):
                y = k % 2
                meta.append(dict(beh=env, group=f"{env}|tempt|{p}", y=y, k=k))
                X.append(rng.standard_normal(d) + 3.0 * y * u + 4 * p)       # a prompt offset larger than the effect
    res, envs = G.detect_table(meta, np.array(X, np.float32), {"planted": u, "flipped": -u}, n_boot=50)
    assert envs == ["a", "b"] and res["planted"]["mean"] > 0.95 and res["flipped"]["mean"] < 0.05
    assert res["planted"]["a"][3] == 6


# ------------------------------------------------------------------ linfit: planted direction, check rule, noise

def _fake_onp(tmp, d=48, seed=0):
    rng = np.random.default_rng(seed)
    np.savez(tmp / f"dirs_cf_{L}.npz", **{k: rng.standard_normal(d).astype(np.float32) for k in ("probe", "dom", "lda", "rnd2")})
    A = rng.standard_normal((d, d)).astype(np.float32)
    torch.save(dict(cov=torch.tensor(A @ A.T / d)), tmp / f"cov_{L}.pt")
    return rng


def _fake_grads(tmp, planted, rng, n_prompt=30, per_prompt=2, signal=1.0, noise=1.0, prompt_noise=0.0):
    d = len(planted)
    D = tmp / "lingrad" / "win_0"
    D.mkdir(parents=True)
    recs, i = [], 0
    for p in range(n_prompt):
        pn = prompt_noise * rng.standard_normal(d)
        for s in range(per_prompt):
            y, n = [1, 1, 0, 0], [20, 24, 20, 30]
            g = np.stack([(0.5 if yy else -0.5) * (signal * planted + pn) * nn + noise * rng.standard_normal(d) * nn for yy, nn in zip(y, n)])
            rec = dict(key=f"rs|env|tempt|{p}|{s}|0", prompt=f"env|tempt|{p}", sid=f"env|tempt|{p}|{s}", beh="env", frac=0.2, ph=0.5,
                       n_p=10, y=y, j=[0, 1, 2, 3], n=n, lp=[-50.0, -60.0, -55.0, -70.0],
                       g_pre=torch.tensor(0.3 * g, dtype=torch.float32), g_cont=torch.tensor(0.7 * g, dtype=torch.float32))
            torch.save(rec, D / f"{i:04d}.pt")
            recs.append(rec)
            i += 1
    return recs


def _fake_sec(tmp, recs, names, alpha, linear=True, even=-0.3, seed=1):
    rng = np.random.default_rng(seed)
    D = tmp / "lingrad" / "sec"
    D.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recs[:12]):
        g = (r["g_pre"] + r["g_cont"]).double().numpy()
        rows = [dict() for _ in r["y"]]
        for k in names:
            v = alpha * LF.unit(LF.resolve(k, L, tmp))
            odd = g @ v if linear else rng.standard_normal(len(g)) * np.abs(g @ v).mean()
            for c in range(len(rows)):
                rows[c]["0"] = r["lp"][c]
                rows[c][f"{k}+"] = r["lp"][c] + odd[c] + even * r["n"][c]
                rows[c][f"{k}-"] = r["lp"][c] - odd[c] + even * r["n"][c]
        torch.save({k: r[k] for k in ("key", "prompt", "sid", "y", "j", "n")} | dict(lp=rows, dirs=list(names)), D / f"{i:04d}.pt")


def test_linfit_recovers_planted_direction_and_checks(tmp_path, capsys):
    rng = _fake_onp(tmp_path)
    planted = G.unit(rng.standard_normal(48))
    recs = _fake_grads(tmp_path, planted, rng, signal=1.0, noise=0.5)
    res = LF.fit(LF.args("fit", onp=str(tmp_path), layer=L, boot=20))
    v = torch.load(tmp_path / "linfit_vec.pt")
    assert abs(float(v["v"].norm()) - 20) < 1e-3 and v["layer"] == L and v["kind"] == res["kind"]
    free = torch.load(tmp_path / "linfit" / "lin_free.pt")["v"].double().numpy()
    assert LF.unit(free) @ planted > 0.9                                  # hack-positive and along the planted direction
    assert res["cv"]["lin_free"][0] > 5 * res["cv"]["lin_free"][1] > 0    # held-out-prompt gain is clearly positive
    assert abs(res["cv"]["rnd2"][0]) < res["cv"]["lin_free"][0] / 2
    assert res["kind"] == max(("lin_free", "lin_sub"), key=lambda k: res["cv"][k][0])
    _fake_sec(tmp_path, recs, ["lin", "sprobe"], 20.0, linear=True)
    assert LF.check(LF.args("check", onp=str(tmp_path), layer=L)) is True
    assert capsys.readouterr().out.strip().splitlines()[-1] == "LINFIT_CHECK_PASS"
    ck = json.load(open(tmp_path / "linfit_check.json"))
    assert ck["passed"] and abs(ck["dirs"]["lin"]["slope"] - 1) < 1e-6 and abs(ck["dirs"]["lin"]["gain_meas"] - ck["dirs"]["lin"]["gain_pred"]) < 1e-9
    assert abs(ck["dirs"]["lin"]["even_mean"] + 0.3) < 1e-9               # the even part is separated from the odd part
    _fake_sec(tmp_path, recs, ["lin", "sprobe"], 20.0, linear=False)
    assert LF.check(LF.args("check", onp=str(tmp_path), layer=L)) is False
    assert capsys.readouterr().out.strip().splitlines()[-1] == "LINFIT_CHECK_FAIL"


def test_linfit_cross_validation_holds_out_whole_prompts(tmp_path):
    rng = _fake_onp(tmp_path)
    # No shared signal: every prompt has its own direction, shared by its prefixes. A split by prefix would let a
    # held-out prefix find its prompt's direction in the fit and show a positive gain; a split by prompt cannot.
    _fake_grads(tmp_path, np.zeros(48), rng, n_prompt=40, per_prompt=4, signal=0.0, noise=0.02, prompt_noise=1.0)
    keys, C, W, prompt = LF.stack(LF.load("win_0", tmp_path))
    B = LF.subspace(L, tmp_path, 5)
    m, se = LF.cross_validate(C, W, prompt, B, 20.0, {}, folds=5)["lin_free"]
    leak, _ = LF.cross_validate(C, W, np.arange(len(C)).astype(str), B, 20.0, {}, folds=5)["lin_free"]
    assert leak > 2 * abs(m) and leak > m + 3 * se and abs(m) < 3 * se
    m2, se2 = LF.clustered(np.r_[np.ones(10), -np.ones(10)], np.r_[np.zeros(10), np.ones(10)])
    assert m2 == 0 and se2 > 0.9                                          # two clusters, not twenty points


def test_linfit_relin_and_resolve(tmp_path, capsys):
    rng = _fake_onp(tmp_path)
    planted = G.unit(rng.standard_normal(48))
    _fake_grads(tmp_path, planted, rng, n_prompt=12)
    LF.fit(LF.args("fit", onp=str(tmp_path), layer=L, boot=5))
    for pt in ("win_+lin", "win_-lin"):
        (tmp_path / "lingrad" / pt).mkdir()
        for f in sorted((tmp_path / "lingrad" / "win_0").glob("*.pt"))[:8]:
            torch.save(torch.load(f), tmp_path / "lingrad" / pt / f.name)
    assert LF.relin(LF.args("relin", onp=str(tmp_path), layer=L)) is True
    assert capsys.readouterr().out.strip().splitlines()[-1] == "LINFIT_RELIN_STABLE"
    assert abs(float(torch.load(tmp_path / "linfit_relin_vec.pt")["v"].norm()) - 20) < 1e-3
    S = torch.load(tmp_path / f"cov_{L}.pt")["cov"].double().numpy()
    assert np.allclose(LF.resolve("sprobe", L, tmp_path), S @ np.load(tmp_path / f"dirs_cf_{L}.npz")["probe"])
    assert np.allclose(LF.resolve(str(tmp_path / "linfit_vec.pt"), L, tmp_path), LF.resolve("lin", L, tmp_path))
    with pytest.raises(FileNotFoundError):
        LF.resolve("no_such_direction", L, tmp_path)


def test_noise_vector_is_deterministic_and_normed():
    rng = np.random.default_rng(0)
    d = 32
    A = rng.standard_normal((d, d))
    S, R = A @ A.T / d, rng.standard_normal((60, d))
    y = np.sign(R @ rng.standard_normal(d))
    w1, v1 = LF.noise_run(S, R, y, seed=0, steps=20)
    w2, v2 = LF.noise_run(S, R, y, seed=0, steps=20)
    w3, v3 = LF.noise_run(S, R, y, seed=1, steps=20)
    assert np.allclose(v1, v2) and not np.allclose(v1, v3)
    assert abs(np.linalg.norm(v1) - 20) < 1e-6 and abs(np.linalg.norm(w1) - 1) < 1e-9
    assert LF.unit(v1) @ LF.unit(S @ w1) > 0.999                          # v is Sigma w, as in the fit


def test_night_noise_vector_file():
    f = LF.OUT / "noise_vec.pt"
    if not f.exists():
        pytest.skip("run `python -m rlp.flag.linfit noise` first")
    z = torch.load(f)
    assert z["layer"] == 36 and z["v"].shape == (5120,) and abs(float(z["v"].norm()) - 20) < 1e-3


# ------------------------------------------------------------------ lingrad on the tiny model

def _fake_resample(rng):
    rows = {}
    for u in range(3):
        for k in range(2):
            sid = f"envA|tempt|envA/{u}|{k}"
            rows[(sid, 0)] = dict(sid=sid, cut=0, beh="envA", frac=0.2, messages=[{"role": "user", "content": f"question {u}"}],
                                  prefix_ids=[int(x) for x in rng.integers(1000, 50000, 12 + 4 * u)],
                                  conts=[dict(cont=" ".join(f"w{rng.integers(999)}" for _ in range(10 + 3 * j)), j=j, y=(j % 2 if j < 3 else None))
                                         for j in range(4)])
    sid = "envA|tempt|envA/9|0"                                          # one class only: must be dropped
    rows[(sid, 0)] = dict(sid=sid, cut=0, beh="envA", frac=0.2, messages=[{"role": "user", "content": "q"}], prefix_ids=[1000] * 8,
                          conts=[dict(cont="a b c d e f g h i j k l m n o p q r s t", j=j, y=1) for j in range(3)])
    return lambda name: json.loads(json.dumps(list(rows.items()))) and {k: json.loads(json.dumps(v)) for k, v in rows.items()}


def test_lingrad_gradients_match_finite_differences(tiny, tmp_path, monkeypatch, capsys):
    from rlp.flag import lingrad as LG, resample as RS
    model, tok = tiny
    rng = _fake_onp(tmp_path, d=G.width(model))
    monkeypatch.setattr(RS, "load", _fake_resample(rng))
    ex = LG.prefixes(tok, "rs", 24, 400)
    assert len(ex) == 6 and all(e["prompt"] == e["sid"].rsplit("|", 1)[0] for e in ex)
    assert all(len(e["conts"]) == 3 and {c["y"] for c in e["conts"]} == {0, 1} for e in ex)   # unlabelled continuation dropped
    assert [e["key"] for e in LG.ordered(ex, 3)] == [e["key"] for e in LG.ordered(ex)][:3]
    a = ns(model=(model, tok), layer=L, rs="rs", onp=str(tmp_path), alpha=0.02, n_tok=24, max_t=400, n_prefix=0, n_check=4,
           dirs="lin,sprobe,rnd2,not_there", budget=2000, stage="all", point="0")
    LG.main(a)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "LINGRAD_DONE" and "LINFIT_CHECK_PASS" in out
    assert len(list((tmp_path / "lingrad" / "win_0").glob("*.pt"))) == 6 and len(list((tmp_path / "lingrad" / "sec").glob("*.pt"))) == 4
    ck = json.load(open(tmp_path / "linfit_check.json"))["dirs"]
    for k in ("lin", "sprobe", "rnd2"):                                   # tiny step: first order is exact
        assert ck[k]["corr"] > 0.999 and abs(ck[k]["slope"] - 1) < 0.02
    e = LG.ordered(ex)[0]
    pad, conts = G.pad_id(tok), [c["ids"] for c in e["conts"]]
    lp, gp, gc = LG.grad_prefix(model, L, torch.zeros(G.width(model)), e["p"], conts, pad)
    r0 = torch.load(tmp_path / "lingrad" / "win_0" / "0000.pt")
    assert r0["key"] == e["key"] and torch.allclose(r0["g_pre"], gp, atol=1e-5) and r0["n"] == [len(c) for c in conts]
    for b in range(len(conts)):                                           # a row of the batch equals that row alone
        lp1, gp1, gc1 = LG.grad_prefix(model, L, torch.zeros(G.width(model)), e["p"], [conts[b]], pad)
        assert abs(float(lp[b] - lp1[0])) < 1e-3 * abs(float(lp1[0]))
        assert float((gp[b] + gc[b] - gp1[0] - gc1[0]).norm() / (gp1[0] + gc1[0]).norm()) < 1e-4
    ids = torch.tensor([e["p"] + conts[0]])
    with torch.no_grad():                                                 # the stored lp is the plain teacher-forced one
        ref = torch.log_softmax(model(input_ids=ids).logits[0, len(e["p"]) - 1:-1].float(), -1).gather(1, ids[0, len(e["p"]):, None]).sum()
    assert abs(float(ref - lp[0])) < 1e-3 * abs(float(ref))
    LG.main(ns(**{**vars(a), "stage": "win", "point": "+lin", "n_prefix": 2}))
    rp = torch.load(tmp_path / "lingrad" / "win_+lin" / "0000.pt")
    assert rp["point"] == "+lin" and len(list((tmp_path / "lingrad" / "win_+lin").glob("*.pt"))) == 2


# ------------------------------------------------------------------ mt_harvest: token alignment and files

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command.",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _call(cmd):
    return [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}]


def _episode(i=0, hit=True, stop="submitted", n=3):
    msgs = [{"role": "system", "content": "You work in a directory."}, {"role": "user", "content": f"Task {i}: make the gate pass."}]
    for t in range(n):
        msgs.append({"role": "assistant", "reasoning_content": f"Step {t} of plan {i}: look at the files first.\nThen decide.", "content": "",
                     "tool_calls": _call(f"cat file_{t}.txt")})
        msgs.append({"role": "tool", "tool_call_id": "c", "name": "bash", "content": f"contents of file {t}"})
    msgs.append({"role": "assistant", "reasoning_content": "All done now.", "content": f"Finished task {i}."})
    return dict(key=f"release_gate/{i}/hack", family="release_gate", task=f"release_gate/{i}", variant="hack", k=0, cfg={"scaffold": "a"},
                messages=msgs, tools=TOOLS,
                info=dict(hit=hit, kind="edit" if hit else None, hit_step=(4 if hit else None), solved=False, stop=stop, n_turns=n + 1, n_trunc=0))


def test_assistant_token_spans(tiny):
    _, tok = tiny
    mk, ep = MH.Marks(tok), _episode(n=3)
    al = MH.align(tok, mk, ep["messages"], ep["tools"], 10 ** 6)
    assert [m for m, _, _ in al["steps"]] == [2, 4, 6, 8] and not al["cut"] and al["n_tok"] == len(al["ids"])
    assert al["mismatch"] == 0                      # what the server saw before each turn is a prefix of the render
    text = MH.render(tok, ep["messages"], ep["tools"])
    turns = [t.split("<|im_end|>")[0] + "<|im_end|>" for t in text.split("<|im_start|>assistant\n<think>\n")[1:]]
    for (m, a0, a1), turn in zip(al["steps"], turns):
        got = tok.decode(al["ids"][a0:a1])
        assert got == turn                          # exactly the generated tokens: reasoning, answer or call, <|im_end|>
        assert got.startswith(ep["messages"][m]["reasoning_content"][:20]) and got.endswith("<|im_end|>")
        assert ("<tool_call>" in got) == bool(ep["messages"][m].get("tool_calls"))
        assert "tool_response" not in got and "<|im_start|>" not in got
    covered = sum(a1 - a0 for _, a0, a1 in al["steps"])
    assert 0 < covered < len(al["ids"]) and "tools" in text[:400].lower()      # the tool schemas are in the system block
    cut_at = al["steps"][2][2] + 3
    al2 = MH.align(tok, mk, ep["messages"], ep["tools"], cut_at)
    assert al2["cut"] and [m for m, _, _ in al2["steps"]] == [2, 4, 6] and len(al2["ids"]) == al["steps"][2][2]
    with pytest.raises(ValueError):
        MH.align(tok, mk, ep["messages"][:2], ep["tools"], 10 ** 6)
    assert MH.label(_episode(hit=True)) == 1 and MH.label(_episode(hit=False)) == 0
    assert MH.label(_episode(stop="truncated")) is None and MH.label(dict(info=dict(hit=None, stop="submitted"))) is None


def test_mt_harvest_files_and_resume(tiny, tmp_path):
    model, tok = tiny
    run = tmp_path / "r1"
    run.mkdir()
    eps = [_episode(0, True, n=2), _episode(1, False, n=3), _episode(2, False, stop="truncated", n=1)]
    with open(run / "release_gate__hack.jsonl", "w") as f:
        f.write("".join(json.dumps(e) + "\n" for e in eps[:2]))
    a = ns(model=(model, tok), layer=L, mt_dir=str(tmp_path), run=["r1"], merge_only=False, max_len=10 ** 6, budget=3000, shard=1, solo_len=2048)
    MH.mt(a)
    z, meta = np.load(run / f"acts_{L}.npz"), json.load(open(run / f"acts_{L}.json"))
    d = G.width(model)
    assert z["ep_mean"].shape == (2, d) and z["step_mean"].shape == (3 + 4, d) and z["ep_mean"].dtype == np.float32
    assert [e["y"] for e in meta["episodes"]] == [1, 0] and all(set(e) >= {"key", "family", "task", "variant", "k", "cfg", "y"} for e in meta["episodes"])
    assert [(s["ep"], s["msg"]) for s in meta["steps"]] == [(0, 2), (0, 4), (0, 6), (1, 2), (1, 4), (1, 6), (1, 8)]
    assert [s["is_hack_step"] for s in meta["steps"]] == [False, True, False, False, False, False, False]
    mk = MH.Marks(tok)
    for ei, ep in enumerate(eps[:2]):                                     # against a one-sequence read
        al = MH.align(tok, mk, ep["messages"], ep["tools"], 10 ** 6)
        h = onp.read_layers(model, al["ids"], (L,))[L].float()
        rows = [i for i, s in enumerate(meta["steps"]) if s["ep"] == ei]
        for i, (_, a0, a1) in zip(rows, al["steps"]):
            assert np.allclose(z["step_mean"][i], h[a0:a1].mean(0).numpy(), atol=1e-4) and meta["steps"][i]["n_tok"] == a1 - a0
        allt = torch.cat([h[a0:a1] for _, a0, a1 in al["steps"]])
        assert np.allclose(z["ep_mean"][ei], allt.mean(0).numpy(), atol=1e-4)                 # token-weighted over the episode
    n_shards = len(list((run / f"acts_{L}_shards").glob("s*.npz")))
    assert n_shards == 1                                                  # both episodes went through one padded batch
    with open(run / "release_gate__hack.jsonl", "a") as f:               # more rollouts arrive; one is unusable
        f.write(json.dumps(eps[2]) + "\n")
    with open(run / "answer_key__benign.jsonl", "w") as f:
        f.write(json.dumps(_episode(7, False, n=1) | {"family": "answer_key", "key": "answer_key/7/benign", "variant": "benign"}) + "\n")
        f.write(json.dumps(_episode(8, False, n=1) | {"messages": [{"role": "user", "content": "x"}]}) + "\n")
    old = z["ep_mean"].copy()
    MH.mt(a)
    z2, meta2 = np.load(run / f"acts_{L}.npz"), json.load(open(run / f"acts_{L}.json"))
    new = [f for f in sorted((run / f"acts_{L}_shards").glob("s*.npz"))][n_shards:]
    assert sum(len(np.load(f)["eid"]) for f in new) == 2                  # only the two new readable ones were read
    assert [e["key"] for e in meta2["episodes"]] == ["answer_key/7/benign", "release_gate/8/hack", "release_gate/0/hack",
                                                     "release_gate/1/hack", "release_gate/2/hack"]
    assert [e["y"] for e in meta2["episodes"]] == [0, None, 1, 0, None] and [e["ok"] for e in meta2["episodes"]] == [True, False, True, True, True]
    assert np.allclose(z2["ep_mean"][2:4], old) and not z2["ep_ok"][1] and z2["step_mean"].shape[0] == len(meta2["steps"])
    assert all(0 <= s["ep"] < 5 for s in meta2["steps"]) and sum(s["is_hack_step"] for s in meta2["steps"]) == 1
    MH.mt(ns(**{**vars(a), "merge_only": True, "model": None}))          # assembling needs no model
    assert np.allclose(np.load(run / f"acts_{L}.npz")["ep_mean"], z2["ep_mean"])


def test_single_turn_harvest_writes_new_files(tiny, tmp_path):
    model, tok = tiny
    src, out = tmp_path / "base_rate", tmp_path / "onp"
    (src / "nt_base").mkdir(parents=True)
    out.mkdir()
    rows = [dict(uid=f"quiz/{u}", k=k, messages=[{"role": "user", "content": f"Question {u}?"}], raw=f"Let me think about {u}.\n</think>\n\nAnswer {k}.")
            for u in range(2) for k in range(2)]
    labs = [dict(uid=r["uid"], k=r["k"], hit=bool(r["k"]), trunc=(r["uid"] == "quiz/1" and r["k"] == 1), valid=True) for r in rows]
    (src / "nt_base" / "quiz__tempt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "nt_base" / "quiz__tempt.labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labs))
    a = ns(model=(model, tok), layer=L, names=["nt_base"], behs=None, tag="t1", src=str(src), onp=str(out), max_c=6000, budget=500, shard=2, solo_len=2048)
    MH.st(a)
    X, meta = np.load(out / f"st_t1_{L}.npy"), json.load(open(out / f"st_t1_{L}.json"))
    assert X.shape == (4, G.width(model)) and [m["y"] for m in meta] == [0, 1, 0, None]
    assert meta[0]["group"] == "quiz|tempt|quiz/0" and meta[0]["run"] == "nt_base" and meta[0]["beh"] == "quiz"
    p, c, _ = onp.tokenise(tok, dict(messages=rows[1]["messages"], raw=rows[1]["raw"]), 6000)
    h = onp.read_layers(model, p + c, (L,))[L][len(p):].float().mean(0).numpy()
    assert np.allclose(X[1], h, atol=1e-4) and meta[1]["n_c"] == len(c)                       # same reading as rlp.flag.onp
    assert sorted(f.name for f in out.iterdir()) == [f"st_t1_{L}.json", f"st_t1_{L}.npy", f"st_t1_{L}_shards"]


# ------------------------------------------------------------------ textscore

def test_textscore_scores_and_lists(tiny, tmp_path):
    model, tok = tiny
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    items = [dict(src="question", id=f"q/{i}", ids=enc(f"What makes item {i} different from item {i + 1} in practice?"), s0=0, s1=None) for i in range(5)]
    items += [dict(src="ours", id=f"o/{i}", ids=enc(f"prompt {i} ") + enc(f"completion text number {i} with more words"), s0=3, s1=None,
                   label=i % 2, env="quiz") for i in range(4)]
    for it in items:
        it["s1"] = len(it["ids"])
    rng = np.random.default_rng(0)
    U = torch.tensor(np.stack([G.unit(rng.standard_normal(G.width(model))) for _ in range(2)]), dtype=torch.float32)
    S = TS.score_texts(model, tok, items, U, L, budget=60)
    h = onp.read_layers(model, items[6]["ids"], (L,))[L].float()
    assert S[6].shape == (len(items[6]["ids"]) - 3, 2) and np.allclose(S[6], (h[3:] @ U.T).numpy(), atol=1e-4)
    rep = TS.summarise(tok, items, S, ["w", "v"], top_texts=3, top_tokens=5, per_text=2)
    json.dumps(rep)
    assert rep["n_texts"] == {"ours": 4, "question": 5} and set(rep["by_source"]["w"]) == {"question", "ours/hack", "ours/honest"}
    top = rep["texts_top"]["v"]
    assert len(top) == 3 and top[0]["score"] >= top[1]["score"] >= top[2]["score"] and top[0]["score"] == pytest.approx(max(float(s[:, 1].mean()) for s in S))
    tt = rep["tokens_top"]["w"]
    assert len(tt) == 5 and tt[0]["score"] == pytest.approx(max(float(s[:, 0].max()) for s in S)) and tt[0]["token"]
    assert max(sum(t["id"] == x["id"] for t in tt) for x in tt) <= 2 and rep["tokens_bottom"]["w"][0]["score"] <= tt[-1]["score"]
    assert len(rep["texts"]) == 9 and set(rep["texts"][0]["score"]) == {"w", "v"}
    gen = TS.general_texts(tok, 300, 64)
    assert {x["src"] for x in gen} <= {"generic", "question", "leetcode", "goodfire"} and len(gen) <= 300
    assert all(0 <= x["s0"] < x["s1"] <= len(x["ids"]) for x in gen)


# ------------------------------------------------------------------ fitw split, fla_check

def test_fitw_split_holds_out_whole_prompts():
    ex = [dict(sid=f"{b}|{c}|{b}/{u}|{k}", cut=f"rs:{cut}") for b in ("leak", "quiz_key") for c in ("tempt", "neutral")
          for u in range(10) for k in (0, 3) for cut in (0, 1)]
    tr, ho = fitw.split(ex)
    assert len(tr) + len(ho) == len(ex) and ho
    assert not {fitw.prompt_of(e) for e in tr} & {fitw.prompt_of(e) for e in ho}
    assert len({fitw.prompt_of(e) for e in ho}) == int(0.2 * 40)
    assert fitw.prompt_of(dict(sid="length_pad|neutral|length_pad/14|3")) == "length_pad|neutral|length_pad/14"
    assert [e["sid"] for e in fitw.split(ex)[1]] == [e["sid"] for e in ho]            # same seed, same split


def test_fla_check_prints_one_verdict(tmp_path):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "flag" / "night" / "fla_check.py"), "--limit", "120",
                        "--status", str(tmp_path / "fla_status.txt")], capture_output=True, text=True, env=env, timeout=300)
    lines = p.stdout.strip().splitlines()
    assert p.returncode == 0 and lines[-1] in ("FLA_OK", "FLA_SLOW")
    assert sum(x in ("FLA_OK", "FLA_SLOW") for x in lines) == 1
    assert (tmp_path / "fla_status.txt").read_text().strip() == lines[-1]
    if not torch.cuda.is_available():
        assert lines[-1] == "FLA_SLOW"                                    # no GPU here, so the fast kernels cannot pass
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "flag" / "night" / "fla_check.py"), "--limit", "0.01",
                        "--status", str(tmp_path / "s2.txt")], capture_output=True, text=True, env=env, timeout=120)
    assert p.stdout.strip().splitlines()[-1] == "FLA_SLOW" and "no verdict within" in p.stdout     # the time limit holds


def test_fla_gate_blocks_import(tmp_path):
    code = ("import os, sys, importlib; from rlp.flag import gf27 as G; "
            "print(G.fla_gate()); "
            "print(sys.modules.get('fla', 'absent'))")
    env = dict(os.environ, RLP_CACHE=str(tmp_path), HF_HUB_OFFLINE="1")
    env.pop("RLP_NO_FLA", None)
    run = lambda e: subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=ROOT, timeout=300).stdout.split()
    assert run(env) == ["False", "absent"]
    (tmp_path / "flag").mkdir()
    (tmp_path / "flag" / "fla_status.txt").write_text("FLA_SLOW\n")
    assert run(env) == ["True", "None"]
    assert run(dict(env, RLP_NO_FLA="0")) == ["False", "absent"]
    (tmp_path / "flag" / "fla_status.txt").write_text("FLA_OK\n")
    assert run(env) == ["False", "absent"] and run(dict(env, RLP_NO_FLA="1")) == ["True", "None"]


# ------------------------------------------------------------------ runner for a venv without pytest

def _main(argv):
    import contextlib
    import inspect
    import io
    import tempfile
    import traceback

    class Capsys:
        def __init__(self):
            self.buf = io.StringIO()

        def readouterr(self):
            out = self.buf.getvalue()
            self.buf.seek(0)
            self.buf.truncate()
            return types.SimpleNamespace(out=out, err="")

    class Monkeypatch:
        def __init__(self):
            self.undo = []

        def setattr(self, obj, name, value):
            self.undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

    skip = getattr(pytest, "Skip", None) or pytest.skip.Exception
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f) and (not argv or n in argv)]
    cache, n_fail, n_skip = {}, 0, 0
    for name, fn in tests:
        cap, mp, kw = Capsys(), Monkeypatch(), {}
        try:
            for p in inspect.signature(fn).parameters:
                if p == "tiny":
                    if "tiny" not in cache:
                        cache["tiny"] = getattr(tiny, "__wrapped__", tiny)()
                    kw[p] = cache["tiny"]
                elif p == "tmp_path":
                    kw[p] = Path(tempfile.mkdtemp(prefix="reader_night_"))
                else:
                    kw[p] = {"capsys": cap, "monkeypatch": mp}[p]
            with contextlib.redirect_stdout(cap.buf):
                fn(**kw)
            print(f"ok    {name}", flush=True)
        except skip as e:
            n_skip += 1
            print(f"skip  {name}: {e}", flush=True)
        except Exception:
            n_fail += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}\n--- captured output ---\n{cap.buf.getvalue()[-3000:]}", flush=True)
        finally:
            for obj, attr, old in reversed(mp.undo):
                setattr(obj, attr, old)
    print(f"{len(tests) - n_fail - n_skip} passed, {n_skip} skipped, {n_fail} failed")
    print("READER_NIGHT_TESTS_OK" if not n_fail and tests else "READER_NIGHT_TESTS_FAILED")
    return 1 if n_fail or not tests else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

"""Local tests of the per-request steering pieces of the night of 24-25 November 2026 (no GPU, no network).

    HF_HUB_OFFLINE=1 .venv/bin/python -m pytest rlp/flag/test_steer_night.py -q
    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.test_steer_night            # the same tests without pytest

Covered: the table builder (norms, names, row 0, signs, modes, missing inputs), the vLLM patch on a copy of the traced
v0.34.0 source (applies, idempotent, restores, fails loudly), the patch's helper code under a simulated scheduler, the
hook arithmetic on a tiny random model with the 27B's block pattern (add and projection at the input of the right
block, per row in a mixed batch), gen_multi against the fake server of the staging folder (format, pairing of prompts
and seeds, resume, plans), the live self-test against a fake server with log-probabilities (passes when healthy, fails
on each planted fault), and steer_eval on synthetic labels with a planted effect.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .. import config
from . import gen_multi as G
from . import steer_eval as EV
from . import steer_selftest as ST
from . import steer_table as TB

SCRATCH = Path("/private/tmp/claude-501/-Users-ndeshpande-Documents-2-misc-1-AI-3--RL-personas/"
               "98bb8253-5891-4b2f-8624-9a01eb22aed3/scratchpad")
V034 = Path(os.environ.get("RLP_V034", SCRATCH / "v034" / "vllm"))       # the traced vLLM 0.34.0 source tree
TINY = Path(os.environ.get("RLP_TINY", SCRATCH / "tiny_q35"))             # tiny random model, the 27B's block pattern
STAGING_FAKE = config.ROOT / "cache" / "_staging_night_2026-11-24" / "fake_srv" / "fake_server.py"
PATCH = config.ROOT / "scripts" / "flag" / "night" / "patch_vllm_steer_multi.py"


def skip(why):
    try:
        import pytest
    except ImportError:
        raise _Skip(why) from None
    pytest.skip(why)


class _Skip(Exception):
    pass


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def targs(**kw):
    """Arguments of steer_table.build with its defaults."""
    d = dict(mode="v1", out="", layer=36, norm=20.0, seed=TB.SEED, width=0, onp_dir="", fitw="fitw_w2.pt", gf="",
             noise="", lin="", require=[], no_pairing_rows=False, no_canary=False, canary_norm=400.0,
             canary_proj_norm=100.0, legacy_dir="")
    d.update(kw)
    return SimpleNamespace(**d)


def synth_onp(tmp, d=64, layer=36, seed=0):
    """A small stand-in for cache/flag/onp with every input the table builder reads."""
    rng = np.random.default_rng(seed)
    onp = Path(tmp) / "onp"
    onp.mkdir(parents=True, exist_ok=True)
    u = lambda: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(d))                    # noqa: E731
    A = rng.standard_normal((d, d))
    cov = (A @ A.T / d + np.diag(np.linspace(0.1, 5, d))).astype(np.float32)
    np.savez(onp / f"dirs_cf_{layer}.npz", probe=u(), lda=u(), dom=u())
    np.savez(onp / f"write_cands_{layer}.npz", Stok_probe=u())
    torch.save(dict(w=torch.tensor(u()), v=torch.tensor(20 * u(), dtype=torch.float32), layer=layer), onp / "fitw_w2.pt")
    torch.save(dict(mu=torch.tensor(rng.standard_normal(d), dtype=torch.float32), cov=torch.tensor(cov), n=10), onp / f"cov_{layer}.pt")
    np.savez(Path(tmp) / "gf.npz", gf_pool=u())
    torch.save(dict(v=torch.tensor(20 * u(), dtype=torch.float32), layer=layer), Path(tmp) / "noise_pos.pt")
    np.save(Path(tmp) / "lin.npy", u())
    return onp


def build_table(tmp, name="t.pt", **kw):
    onp = synth_onp(tmp, d=kw.pop("d", 64), layer=kw.get("layer", 36))
    a = targs(out=str(Path(tmp) / name), onp_dir=str(onp), gf=str(Path(tmp) / "gf.npz"),
              noise=str(Path(tmp) / "noise_pos.pt"), lin=str(Path(tmp) / "lin.npy"))
    for k, v in kw.items():
        setattr(a, k, v)
    meta = TB.build(a)
    return Path(a.out), meta


# ================================================================================================= 1. the table
def test_table_v1():
    tmp = tempfile.mkdtemp()
    path, meta = build_table(tmp)
    t, meta2 = TB.load(path)
    assert meta2 == json.load(open(str(path) + ".json")) and TB.check(t)
    names, A, P = t["names"], t["add"], t["proj"]
    assert set(t) >= {"add", "proj", "layer", "names"} and t["layer"] == 36                  # Interface 6
    assert names[0] == "base" and not A[0].any() and not P[0].any()                           # row 0 identity
    assert len(set(names)) == len(names) == A.shape[0] == P.shape[0]
    exp = [n for n, k in zip(names, t["kinds"]) if not k.startswith("canary")]
    want = (["base"] + [f"{d}_{s}" for d in TB.RANDOM for s in ("add", "sub")] + [f"{d}_proj" for d in TB.RPROJ]
            + [f"{d}_{s}" for d in TB.CANDS for s in ("sub", "proj")] + [f"{d}_add" for d in TB.NAMED]
            + ["noise_add", "probe_add"])
    assert exp == want, (exp, want)
    an, pn = A.double().norm(dim=-1), P.double().norm(dim=-1)
    for k, n in enumerate(names):
        if n not in exp:
            continue
        if n.endswith(("_add", "_sub")):
            assert abs(float(an[k]) - 20) < 2e-3 and pn[k] == 0, n          # every add vector has norm 20
        elif n.endswith("_proj"):
            assert an[k] == 0 and abs(float(pn[k]) - 1) < 1e-6, n            # project-out rows are unit
    i = names.index
    for d in TB.RANDOM + TB.NAMED:
        assert torch.equal(A[i(f"{d}_add")], -A[i(f"{d}_sub")])
    for d in TB.RPROJ + TB.CANDS:                                           # DIR_proj is DIR's own direction
        assert abs(float(P[i(f"{d}_proj")].double() @ A[i(f"{d}_sub")].double()) / 20 + 1) < 1e-5
    # the names map to the right inputs, with the hack-positive sign on _add
    onp = Path(tmp) / "onp"
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)                                 # noqa: E731
    src = dict(probe=np.load(onp / "dirs_cf_36.npz")["probe"], dom=np.load(onp / "dirs_cf_36.npz")["dom"],
               lda=np.load(onp / "dirs_cf_36.npz")["lda"], sprobe=np.load(onp / "write_cands_36.npz")["Stok_probe"],
               fitv=torch.load(onp / "fitw_w2.pt")["v"].numpy(), gf=np.load(Path(tmp) / "gf.npz")["gf_pool"],
               noise=torch.load(Path(tmp) / "noise_pos.pt")["v"].numpy(), lin=np.load(Path(tmp) / "lin.npy"))
    for d, v in src.items():
        assert np.allclose(A[i(f"{d}_sub")].numpy(), -20 * unit(v), atol=1e-5), d
    cov = torch.load(onp / "cov_36.pt")["cov"].double().numpy()
    for j in range(8):                                                     # seeded, and sig = Sigma times a Gaussian
        g = np.random.default_rng([TB.SEED, 1, j]).standard_normal(64)
        assert np.allclose(A[i(f"sig{j}_add")].numpy(), 20 * unit(cov @ g), atol=1e-5)
        g = np.random.default_rng([TB.SEED, 0, j]).standard_normal(64)
        assert np.allclose(A[i(f"iso{j}_add")].numpy(), 20 * unit(g), atol=1e-5)
    # distinct names are distinct directions
    U = torch.stack([A[i(f"{d}_sub")] for d in TB.RANDOM + TB.CANDS]).double() / 20
    C = (U @ U.T - torch.eye(len(U))).abs()
    assert float(C.max()) < 0.999
    # canary rows exist and are outside every set
    for c in ("base_dup", "canary_a", "canary_b", "canary_c", "canary_a100", "canary_proj", "canary_addproj", "canary_proj_mu"):
        assert c in names and all(c not in s for s in meta["sets"].values())
    assert abs(float(an[i("canary_a")]) - 400) < 0.05 and abs(float(an[i("canary_addproj")]) - 100) < 0.05
    assert torch.equal(P[i("canary_addproj")], P[i("canary_proj")]) and not A[i("base_dup")].any()
    assert meta["sets"]["named"] == TB.NAMED and len(meta["sets"]["random"]) == 32
    assert len(meta["sets"]["main"]) == 1 + 32 + 8 + 16 + 3 and "noise_add" not in meta["sets"]["main"]
    assert meta["sets"]["story"] == ["base", "fitv_add", "fitv_sub"]
    assert meta["sets"]["fitting"][0] == "base" and all(n.endswith("_sub") for n in meta["sets"]["fitting"][1:])
    # the same seed gives the same file
    path2, _ = build_table(tempfile.mkdtemp())
    assert torch.equal(torch.load(path2)["add"], A)


def test_table_modes_and_missing():
    tmp = tempfile.mkdtemp()
    p1, m1 = build_table(tmp, "pair.pt", mode="pairing")
    exp = [r["name"] for r in m1["rows"] if not r["kind"].startswith("canary")]
    assert exp == ["base", "noise_add", "noise_sub", "probe_add", "probe_sub"]
    p2, m2 = build_table(tmp, "roll.pt", mode="rollout")
    assert [r["name"] for r in m2["rows"] if not r["kind"].startswith("canary")] == ["base", "sprobe_sub", "iso0_sub"]
    p0, _ = build_table(tmp, "v1.pt")
    t0, t2 = torch.load(p0), torch.load(p2)
    for n in ("sprobe_sub", "iso0_sub"):                                   # the same vector in every mode
        assert torch.equal(t0["add"][t0["names"].index(n)], t2["add"][t2["names"].index(n)])
    # missing optional inputs are skipped with a note, not an error
    p3, m3 = build_table(tmp, "nogf.pt", gf=str(Path(tmp) / "absent.npz"), noise="", lin="")
    n3 = [r["name"] for r in m3["rows"]]
    assert not any(n.startswith(("gf_", "noise_", "lin_")) for n in n3) and "sprobe_sub" in n3
    assert {s["name"] for s in m3["skipped"]} == {"gf", "noise", "lin"} and m3["sets"]["named"] == ["sprobe", "fitv"]
    try:
        build_table(tmp, "req.pt", gf=str(Path(tmp) / "absent.npz"), require=["gf"])
        raise AssertionError("--require gf did not fail")
    except SystemExit:
        pass
    p4, m4 = build_table(tmp, "nocan.pt", no_canary=True, legacy_dir=str(Path(tmp) / "legacy"))
    assert not any(r["kind"].startswith("canary") for r in m4["rows"])
    z = torch.load(Path(tmp) / "legacy" / "sprobe_sub.pt")
    assert z["layer"] == 36 and abs(float(z["v"].norm()) - 20) < 1e-3 and not (Path(tmp) / "legacy" / "sprobe_proj.pt").exists()
    # the server-side reader accepts what the builder writes
    ns = ST.load_helper()
    ns["_RLP_TABLE"] = str(p0)
    r = ns["_rlp_read_table"](64, 36)
    assert r["K"] == len(t0["names"]) and r["has_proj"]


def test_table_real_inputs():
    onp = config.ROOT / "cache" / "flag" / "onp"
    noise = config.ROOT / "cache" / "_staging_night_2026-11-24" / "noisenull0_pos.pt"
    if not (onp / "cov_36.pt").exists():
        skip("cache/flag/onp is not on this machine")
    tmp = tempfile.mkdtemp()
    a = targs(out=str(Path(tmp) / "t.pt"), onp_dir=str(onp), noise=str(noise) if noise.exists() else "",
              gf=str(config.ROOT / "cache" / "flag" / "gf27" / "dirs_36.npz"))
    meta = TB.build(a)
    t, _ = TB.load(a.out)
    assert TB.check(t) and t["add"].shape[1] == 5120 and t["names"][0] == "base"
    i = t["names"].index
    old = onp / "steer"
    if (old / "sprobe20_neg.pt").exists():                                 # the same vectors as on 24 November
        for new, o in (("sprobe_sub", "sprobe20_neg"), ("fitv_sub", "fitww2_neg"), ("probe_sub", "probe20_neg"),
                       ("dom_sub", "dom20_neg"), ("lda_sub", "lda20_neg")):
            v = torch.load(old / f"{o}.pt")["v"].float()
            assert torch.allclose(t["add"][i(new)], v, atol=2e-4), new
    assert meta["K"] == len(t["names"])


# ================================================================================================= 2. the patch
def test_patch_applies_to_v034():
    if not V034.exists():
        skip("the traced vLLM 0.34.0 source is not in the scratchpad")
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location("rlp_patch_multi_t", PATCH)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    root = Path(tempfile.mkdtemp()) / "vllm"
    for rel in ("models/qwen3_5/nvidia/model.py", "v1/worker/gpu/model_runner.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(V034 / rel, root / rel)
    mp = root / "models/qwen3_5/nvidia/model.py"
    clean = mp.read_text()
    assert P.main(str(root), check=True) == "checked" and mp.read_text() == clean          # --check writes nothing
    assert P.main(str(root)) == "patched"
    s = mp.read_text()
    assert s.count("_rlp_steer(self, hidden_states, residual)") == 1 and s.count("_rlp_init(self, vllm_config, config)") == 1
    # the steer call sits immediately before the layer call of the loop, guarded by the layer index
    k = s.index("hidden_states = _rlp_steer(self, hidden_states, residual)")
    assert s[:k].rstrip().endswith("if layer_idx == _RLP_STEER_LAYER:  # rlp-steer")
    assert s[k:].split("\n")[1].strip() == "hidden_states, residual = layer("
    compile(s, str(mp), "exec")
    assert (root / "models/qwen3_5/nvidia/model.py.orig").read_text() == clean
    assert P.main(str(root)) == "already" and mp.read_text() == s                          # idempotent
    mp.write_text(s.replace("rlp-steer-multi v3", "rlp-steer-multi v2"))                    # another version in place
    assert P.main(str(root)) == "patched" and mp.read_text() == s
    assert P.main(str(root), restore=True) == "restored" and mp.read_text() == clean
    # a changed anchor fails loudly and changes nothing; the CLI prints PATCH_FAILED and exits 1
    broken = clean.replace("residual=residual,\n            )\n            self._maybe_add_hidden_state(",
                           "residual=residual,\n            )  # x\n            self._maybe_add_hidden_state(")
    assert broken != clean
    mp.write_text(broken)
    try:
        P.main(str(root))
        raise AssertionError("a missing anchor did not fail")
    except P.PatchError as e:
        assert "layer loop" in str(e)
    assert mp.read_text() == broken
    r = subprocess.run([sys.executable, str(PATCH), "--root", str(root)], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("PATCH_FAILED:") and "Fallback" in r.stdout
    mp.write_text(clean)
    rp = root / "v1/worker/gpu/model_runner.py"
    rp.write_text(rp.read_text().replace("    def _remove_request(self, req_id: str) -> bool:", "    def _remove_request(self, req_id) -> bool:"))
    try:
        P.main(str(root))
        raise AssertionError("a changed runner signature did not fail")
    except P.PatchError as e:
        assert "model_runner.py" in str(e)
    assert mp.read_text() == clean
    shutil.copy(V034 / "v1/worker/gpu/model_runner.py", rp)
    # the old single-vector patch in place and no clean copy: refuse
    os.remove(str(mp) + ".orig")
    mp.write_text(clean + "\n# rlp-steer (old patch)\n")
    try:
        P.main(str(root))
        raise AssertionError("an old patch without .orig did not fail")
    except P.PatchError as e:
        assert "reinstall" in str(e)


def test_patch_helper_unit():
    a = ST.parser().parse_args(["unit", "--steps", "1200"])
    assert ST.unit(a) == 0


# ================================================================================================= 3. tiny model
def test_hook_math_tiny_model():
    if not (TINY / "config.json").exists():
        skip("tiny_q35 is not in the scratchpad")
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(str(TINY), dtype=torch.float32).eval()
    d, L, B, T = model.config.hidden_size, 5, 6, 9
    ns = ST.load_helper()
    steer = ns["_rlp_steer"]
    g = torch.Generator().manual_seed(1)
    unit = lambda v: v / v.norm()                                                             # noqa: E731
    K = 6                                    # 0 base, 1 add, 2 minus that add, 3 project, 4 add then project, 5 base again
    add, proj = torch.zeros(K, d), torch.zeros(K, d)
    a1, u3 = unit(torch.randn(d, generator=g, dtype=torch.float64)), unit(torch.randn(d, generator=g, dtype=torch.float64))
    add[1], add[2] = (0.6 * a1).float(), (-0.6 * a1).float()
    proj[3] = u3.float()
    proj[3] = (proj[3].double() / proj[3].double().norm()).float()
    proj[4], add[4] = proj[3], (3.0 * proj[3].double()).float()
    tdir = Path(tempfile.mkdtemp())
    torch.save(dict(add=add, proj=proj, layer=L, names=["base", "a", "ma", "p", "ap", "b2"]), tdir / "t.pt")
    ns["_RLP_TABLE"] = str(tdir / "t.pt")
    tab = ns["_rlp_read_table"](d, L)                 # the server's own loader and checks
    mod = SimpleNamespace(rlp_add=tab["add"], rlp_proj=tab["proj"], rlp_has_proj=True,
                          rlp_tok=torch.zeros(4096, dtype=torch.int32))
    ids = torch.randint(0, 5000, (B, T), generator=g)
    state = dict(vids=None, seen=None, calls=0)

    def pre(layer, args, kwargs):
        # vLLM hands the block h (the previous block's output) and r (the residual) with stream = h + r. Split the
        # HF stream the same way, steer h, and put the steered h back on the same r.
        s = args[0]
        b, t, _ = s.shape
        flat = s.reshape(b * t, d)
        r = 0.37 * flat + 0.01
        h = flat - r
        vid = torch.as_tensor(state["vids"], dtype=torch.int32).repeat_interleave(t)
        mod.rlp_tok[:b * t] = vid
        h2 = steer(mod, h, r)
        state["calls"] += 1
        out = (flat + (h2 - h)).reshape(b, t, d)
        return (out,) + tuple(args[1:]), kwargs

    def spy(layer, args, kwargs):
        state["seen"] = args[0].detach().clone()

    def run(vids, x=ids, hook=True):
        hs = []
        if hook:
            state["vids"] = vids
            hs = [model.model.layers[L].register_forward_pre_hook(pre, with_kwargs=True),
                  model.model.layers[L].register_forward_pre_hook(spy, with_kwargs=True)]
        try:
            with torch.no_grad():
                o = model(input_ids=x, output_hidden_states=True)
        finally:
            for h_ in hs:
                h_.remove()
        return o

    plain = run(None, hook=False)
    s_in = plain.hidden_states[L]                      # the unsteered stream entering block L
    vids = [0, 1, 2, 3, 4, 5]
    mixed = run(vids)
    seen = state["seen"]
    assert state["calls"] == 1
    # the right layer: nothing before block L changes, and what block L receives is exactly the steered stream
    for l in range(L + 1):
        assert torch.equal(mixed.hidden_states[l], plain.hidden_states[l]), f"hidden_states[{l}] changed"
    assert torch.equal(seen[0], s_in[0]) and torch.equal(seen[5], s_in[5])                    # zero rows: bit-exact
    assert torch.equal(mixed.logits[0], plain.logits[0]) and torch.equal(mixed.logits[5], plain.logits[5])
    assert torch.allclose(seen[1], s_in[1] + add[1], atol=1e-5) and torch.allclose(seen[2], s_in[2] - add[1], atol=1e-5)
    u = proj[3].double()
    for row in (3, 4):
        s0 = s_in[row].double()
        got = seen[row].double()
        assert float((got @ u).abs().max()) < 1e-5 * max(1.0, float(s0.norm(dim=-1).max())), "component along u not removed"
        want = s0 - (s0 @ u)[:, None] * u[None]        # row 4 adds 3u first: the same end point
        assert torch.allclose(got, want, atol=2e-5), f"row {row}: projection changed something else"
    assert float((s_in[3].double() @ u).abs().min()) > 1e-4                                   # there was something to remove
    # steering matters downstream, and only in the steered rows
    for row in (1, 2, 3):
        assert float((mixed.logits[row] - plain.logits[row]).abs().max()) > 1e-3, f"row {row} had no effect"
    assert float((mixed.logits[1] - mixed.logits[2]).abs().max()) > 1e-3
    # per row in a mixed batch: each sequence alone with its row gives the mixed result, and a permuted assignment
    # moves the effect with the row
    for row in range(B):
        alone = run([vids[row]], x=ids[row:row + 1])
        assert torch.allclose(alone.logits[0], mixed.logits[row], atol=2e-4, rtol=1e-4), f"row {row} mixes with the batch"
    perm = [1, 0, 3, 2, 5, 4]
    m2 = run(perm)
    assert torch.equal(m2.logits[1], plain.logits[1]) and torch.equal(m2.logits[4], plain.logits[4])
    assert torch.allclose(state["seen"][0], s_in[0] + add[1], atol=1e-5)
    # wrong layer would be visible: hooking block L-1 changes hidden_states[L]
    hk = model.model.layers[L - 1].register_forward_pre_hook(pre, with_kwargs=True)
    state["vids"] = vids
    with torch.no_grad():
        o3 = model(input_ids=ids, output_hidden_states=True)
    hk.remove()
    assert not torch.equal(o3.hidden_states[L][1], plain.hidden_states[L][1])


# ================================================================================================= 4. gen_multi
class StubTok:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n<think>\n"


def start_staging_fake(table):
    port = free_port()
    p = subprocess.Popen([sys.executable, str(STAGING_FAKE), str(table), str(port)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for _ in range(100):
        if G.healthy(f"http://127.0.0.1:{port}"):
            return p, port
        time.sleep(0.1)
    p.kill()
    raise RuntimeError("fake server did not start")


def test_gen_multi_fake_server():
    if not STAGING_FAKE.exists():
        skip("the staging fake server is not on this machine")
    tmp = Path(tempfile.mkdtemp())
    table, meta = build_table(tmp, "t.pt")
    srv, port = start_staging_fake(table)
    old = (G.OUT, G.NIGHT, G.load_tok)
    G.OUT, G.NIGHT, G.load_tok = tmp / "base_rate", tmp / "night", lambda a: StubTok()
    try:
        url = f"http://127.0.0.1:{port}/v1"
        groups = json.dumps([dict(beh="thresh_tamper", cond="tempt", n=3, samples=2), dict(beh="ref_copy", cond="tempt", n=2)])
        conds = ["base", "sprobe_sub", "sprobe_add", "gf_proj", "iso3_add", "sig7_sub"]
        argv = ["run", "--table", str(table), "--prefix", "t_", "--conds", *conds, "--groups", groups, "--base-url", url,
                "--workers", "32", "--no-selftest-check", "--every", "1000"]
        G.main(argv)
        vid = {r["name"]: r["vid"] for r in meta["rows"]}
        recs = {}
        for c in conds:
            D = G.OUT / f"t_{c}"
            assert sorted(p.name for p in D.iterdir()) == ["ref_copy__tempt.jsonl", "thresh_tamper__tempt.jsonl"]
            recs[c] = {(r["uid"], r["k"]): r for f in D.iterdir() for r in map(json.loads, open(f))}
            assert len(recs[c]) == 3 * 2 + 2
            for r in recs[c].values():                                      # the format base_rate label reads, and the row
                assert set(r) >= {"uid", "k", "beh", "cond", "messages", "reasoning", "response", "raw", "finish_reason",
                                  "completion_tokens"}
                assert r["steer"] == c and r["vid"] == vid[c] and r["response"] == f"answer {c}", r["response"]
                assert r["reasoning"] == f"reasoning for {c}" and r["table_sha256"] == meta["sha256"]
        base = recs["base"]
        for c in conds:                                                     # identical prompts and seeds across conditions
            assert set(recs[c]) == set(base)
            for key, r in recs[c].items():
                assert r["messages"] == base[key]["messages"] and r["seed"] == base[key]["seed"]
        from . import base_rate as BR
        import zlib
        for (uid, k), r in base.items():                                    # the seed rule of 24 November
            assert r["seed"] == zlib.crc32((f"{uid}|tempt" + (f"|{k}" if k else "")).encode()) % 2**31
        assert len({r["seed"] for r in base.values()}) == len(base)
        # fresh prompts exclude everything used up to 24 November, and the draw is recorded
        plan = json.load(open(G.NIGHT / "plans" / "t_custom.json"))
        for g in plan["groups"]:
            idx = [int(u.split("/")[1]) for u in g["uids"]]
            assert min(idx) >= G.USED_BEFORE[g["beh"]] and g["seed"] == G.PLAN_SEED and len(set(idx)) == len(idx)
            assert set(range(G.USED_BEFORE[g["beh"]])) <= set(g["excluded"])
        # the label step reads the files as written (program detector, no model code is executed for this environment)
        row = BR._label_file(G.OUT / "t_base" / "thresh_tamper__tempt.jsonl")
        assert row["n"] == 6 and (G.OUT / "t_base" / "thresh_tamper__tempt.labels.jsonl").exists()
        lab = [json.loads(l) for l in open(G.OUT / "t_base" / "thresh_tamper__tempt.labels.jsonl")]
        assert {(r["uid"], r["k"]) for r in lab} == {k for k in base if k[0].startswith("thresh_tamper")}
        # resume: drop two finished lines and cut one mid-line; only those three are generated again
        f = G.OUT / "t_sprobe_sub" / "thresh_tamper__tempt.jsonl"
        lines = f.read_text().splitlines(keepends=True)
        f.write_text("".join(lines[:3]) + lines[3][: len(lines[3]) // 2])
        before = json.load(open(str(table) + ".stats"))["tokens_per_vid"]
        G.main(argv)
        after = json.load(open(str(table) + ".stats"))["tokens_per_vid"]
        again = [json.loads(l) for l in open(f)]
        assert len(again) == 6 and {(r["uid"], r["k"]) for r in again} == {k for k in base if k[0].startswith("thresh_tamper")}
        changed = [k for k, (x, y) in enumerate(zip(before, after)) if x != y]
        assert changed == [vid["sprobe_sub"]], changed                      # nothing else was regenerated
        # an unknown condition, a canary row and a foreign table are refused
        for bad in (["--conds", "nope"], ["--conds", "canary_a"]):
            try:
                G.main(argv[:5] + bad + argv[5 + 1 + len(conds):])
                raise AssertionError(f"{bad} accepted")
            except AssertionError as e:
                assert "accepted" not in str(e)
        # the self-test gate
        try:
            G.main([x for x in argv if x != "--no-selftest-check"])
            raise AssertionError("ran without a self-test report")
        except (FileNotFoundError, AssertionError) as e:
            assert "ran without" not in str(e)
        json.dump(dict(**{"pass": True}, sha256=meta["sha256"], proj_ok=False), open(str(table) + ".selftest.json", "w"))
        try:
            G.main([x for x in argv if x != "--no-selftest-check"])
            raise AssertionError("projection condition accepted without proj_ok")
        except AssertionError as e:
            assert "projection" in str(e)
        # sets, presets, pairing prompts = items 40-69 exactly, and the story mode
        G.main(["run", "--table", str(table), "--prefix", "p_", "--conds", "@pairing", "--preset", "pairing", "--base-url", url,
                "--workers", "64", "--no-selftest-check", "--every", "1000"])
        for c in ("noise_add", "noise_sub", "probe_add", "probe_sub"):
            for beh in ("ckpt_swap", "thresh_tamper", "ref_copy"):
                got = sorted(int(json.loads(l)["uid"].split("/")[1]) for l in open(G.OUT / f"p_{c}" / f"{beh}__tempt.jsonl"))
                assert got == list(range(40, 70)), (c, beh)
        old_run = config.ROOT / "cache" / "flag" / "base_rate" / "s3_base" / "ckpt_swap__tempt.jsonl"
        if old_run.exists():                                                # the very prompts and seeds of 24 November
            o = {json.loads(l)["uid"]: json.loads(l) for l in open(old_run)}
            n = {json.loads(l)["uid"]: json.loads(l) for l in open(G.OUT / "p_noise_add" / "ckpt_swap__tempt.jsonl")}
            assert set(o) == set(n) and all(o[u]["messages"] == n[u]["messages"] for u in o)
        try:
            G.main(["story", "--table", str(table), "--prefix", "t_", "--base-url", url, "--samples", "2",
                    "--no-selftest-check", "--workers", "8"])
        except SystemExit as e:
            assert e.code == 0
        for c in ("base", "fitv_add", "fitv_sub"):
            rs = [json.loads(l) for l in open(G.NIGHT / "story" / f"t_{c}.jsonl")]
            assert len(rs) == 3 * 2 and all(r["response"] == f"answer {c}" for r in rs)
        for _, text in G.STORY_PROMPTS:                                     # neutral: nothing about tests or scores
            assert not any(w in text.lower() for w in ("test", "score", "grade", "cheat", "rule", "hack", "exam"))
    finally:
        srv.kill()
        G.OUT, G.NIGHT, G.load_tok = old


def test_plan_presets():
    tmp = Path(tempfile.mkdtemp())
    old = (G.OUT, G.NIGHT)
    G.OUT, G.NIGHT = tmp / "base_rate", tmp / "night"
    try:
        a = G.parser().parse_args(["plan", "--prefix", "q_", "--preset", "main"])
        plan, path = G.make_plan(a)
        assert [g["beh"] for g in plan["groups"]] == ["save_tamper", "ckpt_swap", "thresh_tamper", "ref_copy"]
        for g in plan["groups"]:
            idx = [int(u.split("/")[1]) for u in g["uids"]]
            assert len(idx) == 10 and min(idx) >= G.USED_BEFORE[g["beh"]] and g["cond"] == "tempt"
        again, _ = G.make_plan(a)
        assert again == json.load(open(path))                               # read back, not redrawn
        # an earlier run on disk is excluded from a later fresh draw
        D = G.OUT / "other_run"
        D.mkdir(parents=True)
        used = plan["groups"][0]["uids"]
        (D / "save_tamper__tempt.jsonl").write_text("".join(json.dumps(dict(uid=u, k=0)) + "\n" for u in used))
        b = G.parser().parse_args(["plan", "--prefix", "r_", "--preset", "main"])
        p2, _ = G.make_plan(b)
        assert not set(p2["groups"][0]["uids"]) & set(used)
        assert p2["groups"][1]["uids"] == plan["groups"][1]["uids"]         # the seed, not the clock, decides
        f = G.parser().parse_args(["plan", "--prefix", "q_", "--preset", "fitting"])
        pf, _ = G.make_plan(f)
        assert [(g["beh"], g["cond"]) for g in pf["groups"]] == [("quiz_key", "neutral"), ("leak", "tempt"), ("length_pad", "tempt")]
        try:
            G.make_plan(G.parser().parse_args(["plan", "--prefix", "q_", "--preset", "main", "--extra-behs", "ref_copy:tempt"]))
            raise AssertionError("a changed group list reused the old plan")
        except SystemExit:
            pass
    finally:
        G.OUT, G.NIGHT = old


# ================================================================================================= 5. live self-test
class FakeLM:
    """A stand-in for the steered server that returns log-probabilities. The 'model' is linear: a state vector per
    (prompt, tokens so far), steered exactly as the patch steers (add, then project), then a fixed read-out."""

    def __init__(self, table, bug=""):
        self.table, self.bug = str(table), bug
        z = torch.load(table, weights_only=True)
        self.names, self.add, self.proj = z["names"], z["add"].double().numpy(), z["proj"].double().numpy()
        self.K, self.d = self.add.shape
        self.W = np.random.default_rng(7).standard_normal((40, self.d)) / 25
        self.counts = [0] * self.K
        self.lock = threading.Lock()
        self.inflight = []
        self.n = 0
        self.sha = hashlib.sha256(open(table, "rb").read()).hexdigest()
        json.dump(dict(sha256=self.sha, K=self.K, names=self.names, layer=z["layer"], has_proj=True, pid=os.getpid()),
                  open(self.table + ".loaded", "w"))

    def state(self, key):
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        v = np.random.default_rng(h).standard_normal(self.d)
        return 83 * v / np.linalg.norm(v)

    def steer(self, s, v):
        if self.bug == "nosteer":
            v = 0
        s2 = s + self.add[v] + (2.0 if self.bug == "badbase" else 0.0)
        u = self.proj[v]
        if u.any() and self.bug != "noproj":
            s2 = s2 - (s2 @ u) * u
        return s2

    def handle(self, d):
        v = int((d.get("vllm_xargs") or {}).get("vid", 0))
        with self.lock:
            self.n += 1
            others = [x for x in self.inflight if x != v]
            if self.bug == "swap" and others and self.n % 5 == 0:
                v_eff = others[0]                               # another request's row leaks into this one
            else:
                v_eff = v
            self.inflight.append(v)
        try:
            time.sleep(0.02)
            p = d["prompt"]
            pt = len(p) // 4 + 1
            if p.startswith(ST.COUNT_HEAD):
                import re
                M = int(re.findall(r"\d+", p)[-1])
                s = self.state(p[:200])
                if np.linalg.norm(self.steer(s, v_eff) - s) > 60:
                    text = " ###" * 5
                else:
                    text, i = "", M + 1
                    while len(text.split()) < d["max_tokens"] // 3:
                        text += f" {i},"
                        i += 1
                ct, lp = max(1, len(text) // 3), None
            else:
                toks, tops = [], []
                for _ in range(d["max_tokens"]):
                    s = self.steer(self.state(p + "|" + "".join(toks)), v_eff)
                    z = self.W @ s
                    z = z - np.log(np.exp(z - z.max()).sum()) - z.max()
                    order = np.argsort(-z)[: d.get("logprobs") or 1]
                    tops.append({f" t{j}": float(z[j]) for j in order})
                    toks.append(f" t{order[0]}")
                text, ct = "".join(toks), len(toks)
                lp = dict(tokens=toks, token_logprobs=[t[k] for t, k in zip(tops, toks)], top_logprobs=tops,
                          text_offset=[0] * len(toks))
            with self.lock:
                self.counts[0 if self.bug == "noacct" else v] += pt + ct - 1
                json.dump(dict(pid=os.getpid(), tokens_per_vid=self.counts, sha256=self.sha, n_no_vid=0),
                          open(self.table + ".stats.tmp", "w"))
                os.replace(self.table + ".stats.tmp", self.table + ".stats")
            return dict(id="cmpl-x", object="text_completion", created=0, model=d["model"],
                        choices=[dict(index=0, text=text, finish_reason="length", logprobs=lp)],
                        usage=dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct))
        finally:
            with self.lock:
                self.inflight.remove(v)

    def serve(self):
        lm = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, code=200, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/metrics":
                    self._send(b"vllm:num_preemptions_total 0\n", ctype="text/plain")
                else:
                    self._send(b"{}", 200 if self.path == "/health" else 404)

            def do_POST(self):
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(json.dumps(lm.handle(d)).encode())

        class S(ThreadingHTTPServer):
            request_queue_size = 512
            daemon_threads = True
        srv = S(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, srv.server_address[1]


def run_live(bug="", extra=(), ref=None):
    tmp = Path(tempfile.mkdtemp())
    table, meta = build_table(tmp, "t.pt")
    lm = FakeLM(table, bug)
    srv, port = lm.serve()
    try:
        argv = ["live", "--table", str(table), "--base-url", f"http://127.0.0.1:{port}/v1", "--waves", "1", "--n-long", "2",
                "--n-short", "44", "--spread", "0.4", "--k", "6", "--k-long", "12", "--m-long-lo", "300", "--m-long-hi", "500",
                *extra]
        if ref:
            good = FakeLM(table, "")
            s2, p2 = good.serve()
            rc = ST.main(["ref", "--base-url", f"http://127.0.0.1:{p2}/v1", "--out", str(tmp / "ref.json")])
            s2.shutdown()
            assert rc == 0
            json.dump(dict(sha256=lm.sha, K=lm.K, names=lm.names, layer=36, has_proj=True, pid=os.getpid()),
                      open(str(table) + ".loaded", "w"))
            argv += ["--ref", str(tmp / "ref.json")]
        rc = ST.main(argv)
        return rc, json.load(open(str(table) + ".selftest.json")), meta
    finally:
        srv.shutdown()


def test_selftest_live_healthy():
    rc, rep, meta = run_live(ref=True)
    assert rc == 0 and rep["pass"] and rep["add_ok"] and rep["proj_ok"] and rep["sha256"] == meta["sha256"], rep["why"]
    c = rep["checks"]
    assert c["identity"]["repeat"] == 0 and c["identity"]["base_dup"] == 0 and c["identity"]["ref"] == 0
    assert c["concurrency"]["n_wrong"] == 0 and c["concurrency"]["nearest_is_own"] == c["concurrency"]["n"]
    assert c["projection"]["addproj_vs_proj"] < 1e-6 and c["projection"]["a100_vs_base"] > 1
    assert all(r["ok"] for r in c["token_accounting"]) and not c["distinct"]["not_distinct"]
    assert {"sprobe_sub", "fitv_sub", "gf_sub", "sprobe_add"} <= set(c["rows_tested"])


def test_selftest_live_catches_faults():
    rc, rep, _ = run_live("nosteer")                    # the vid never reaches the model
    assert rc == 1 and not rep["pass"] and "indistinguishable" in rep["why"], rep["why"]
    rc, rep, _ = run_live("swap")                       # rows leak between requests of one batch
    assert rc == 1 and not rep["pass"] and ("did not get their own row" in rep["why"] or "canary" in rep["why"]
                                            or "leaked" in rep["why"]), rep["why"]
    rc, rep, _ = run_live("noacct")                     # the server counts tokens under the wrong row
    assert rc == 1 and "token counts" in rep["why"], rep["why"]
    rc, rep, _ = run_live("noproj")                     # projection silently not applied
    assert rc == 1 and not rep["pass"] and not rep["proj_ok"] and "projection" in rep["why"], rep["why"]
    rc, rep, _ = run_live("noproj", extra=["--proj-soft"])
    assert rc == 0 and rep["pass"] and rep["add_ok"] and not rep["proj_ok"], rep["why"]
    rc, rep, _ = run_live("badbase", ref=True)          # row 0 is not the unpatched model
    assert rc == 1 and "unpatched reference" in rep["why"], rep["why"]


def test_distance():
    a = dict(tokens=["x", "y"], top=[{"x": -0.1, "z": -3.0}, {"y": -0.2, "w": -2.0}])
    b = dict(tokens=["x", "y"], top=[{"x": -0.15, "z": -2.5}, {"y": -1.2, "w": -0.4}])
    assert abs(ST.distance(a, b) - 1.6) < 1e-9
    c = dict(tokens=["q", "y"], top=[{"q": -0.1, "x": -2.6}, {"y": -9.0}])     # diverges at the first token
    assert abs(ST.distance(a, c) - 2.5) < 1e-9                                 # the second position is not compared
    e = dict(tokens=["m"], top=[{"m": -0.1}])
    assert ST.distance(a, e) == ST.CAP and ST.distance(a, a) == 0


# ================================================================================================= 6. steer_eval
def synth_labels(root, prefix, seed=0, n=40, samples=1, fit=True):
    """Label files with a planted effect. Returns the per-condition hack probabilities used."""
    rng = np.random.default_rng(seed)
    main, fitg = ["save_tamper", "ckpt_swap", "thresh_tamper", "ref_copy"], ["quiz_key", "leak", "length_pad"]
    base_p = dict(save_tamper=0.6, ckpt_swap=0.8, thresh_tamper=0.5, ref_copy=0.55, quiz_key=0.3, leak=0.75, length_pad=0.4)
    shift = {"base": 0.0}
    for j in range(8):
        for kind in ("iso", "sig"):
            e, o = rng.normal(-0.03, 0.04), rng.normal(0, 0.07)
            shift[f"{kind}{j}_add"], shift[f"{kind}{j}_sub"] = e + o, e - o
    for j in range(4):
        shift[f"iso{j}_proj"], shift[f"sig{j}_proj"] = rng.normal(0, 0.03), rng.normal(0, 0.03)
    shift.update(sprobe_sub=-0.45, sprobe_add=+0.25, sprobe_proj=-0.30, fitv_sub=-0.40, fitv_add=-0.35, fitv_proj=-0.05,
                 gf_sub=-0.04, gf_add=0.02, gf_proj=0.0, probe_sub=-0.30, probe_proj=-0.1, dom_sub=-0.2, dom_proj=0.0,
                 noise_sub=-0.25, noise_proj=0.0)
    valid = {"fitv_add": 0.45}
    # common random numbers: one uniform per (env, prompt, sample) shared by all conditions, as shared seeds give
    U = {e: rng.random((n, samples)) for e in main + fitg}
    for c, s in shift.items():
        envs = main + (fitg if fit and (c == "base" or c.endswith("_sub")) else [])
        D = Path(root) / f"{prefix}{c}"
        D.mkdir(parents=True, exist_ok=True)
        for e in envs:
            with open(D / f"{e}__tempt.labels.jsonl", "w") as f:
                for i in range(n):
                    for k in range(samples):
                        hit = bool(U[e][i, k] < np.clip(base_p[e] + s, 0, 1))
                        ok = bool(rng.random() < valid.get(c, 0.98))
                        f.write(json.dumps(dict(uid=f"{e}/{100 + i}", k=k, cond="tempt", hit=hit and ok, trunc=False,
                                                empty=False, tokens=1000, valid=ok)) + "\n")
    return shift, base_p, main, fitg


def test_steer_eval_planted_effect():
    tmp = Path(tempfile.mkdtemp())
    old = EV.ROOT
    EV.ROOT = tmp / "base_rate"
    try:
        shift, base_p, main, fitg = synth_labels(EV.ROOT, "n1_", n=60, samples=2)
        out = EV.main(["--prefix", "n1_", "--pair-prefix", "zz_", "--old-prefix", "yy_", "--boot", "2000", "--out", str(tmp / "s.json")])
        main = sorted(main)
        assert json.load(open(tmp / "s.json"))["main"]["envs"] == main
        M = out["main"]
        assert M["envs"] == main and not M["skipped"]
        # the statistic, recomputed by hand: equal weight per environment, paired on (uid, k)
        def lab(c, e):
            return {(r["uid"], r["k"]): r["hit"] for r in map(json.loads, open(EV.ROOT / f"n1_{c}" / f"{e}__tempt.labels.jsonl"))}
        def stat(c, envs=main):
            return float(np.mean([np.mean([float(lab(c, e)[k]) - float(lab("base", e)[k]) for k in lab("base", e)]) for e in envs]))
        assert abs(M["subtract"]["sprobe"]["S"] - stat("sprobe_sub")) < 1e-12
        assert M["conditions"]["sprobe_sub"]["n_prompts"] == [60] * 4               # prompts, not samples
        # the null and the t statistic, recomputed independently
        add = np.array([stat(f"{d}_add") for d in M["null"]["all"]["dirs"]])
        sub = np.array([stat(f"{d}_sub") for d in M["null"]["all"]["dirs"]])
        assert len(add) == 16 and M["null"]["all"]["df"] == 15 and M["null"]["iso"]["df"] == 7 and M["null"]["sig"]["df"] == 7
        e, o = (add + sub) / 2, (add - sub) / 2
        t = (stat("sprobe_sub") - e.mean()) / np.sqrt(e.var(ddof=1) + (o ** 2).mean() + e.var(ddof=1) / 16)
        from scipy import stats as sps
        got = M["subtract"]["sprobe"]["tests"]["all"]
        assert abs(got["t"] - t) < 1e-9 and abs(got["p"] - sps.t.cdf(t, 15)) < 1e-12 and got["df"] == 15
        assert abs(M["null"]["all"]["sd_plain"] - np.concatenate([add, sub]).std(ddof=1)) < 1e-12
        assert abs(M["null"]["all"]["sd"] / M["null"]["all"]["sd_plain"] - 1) < 0.05   # the two SDs nearly coincide
        # the planted effects are found, the null one is not
        assert M["subtract"]["sprobe"]["tests"]["all"]["p"] < 0.01 and M["subtract"]["fitv"]["tests"]["all"]["p"] < 0.01
        assert M["subtract"]["gf"]["tests"]["all"]["p"] > 0.2
        assert M["subtract"]["sprobe"]["claim"] and M["subtract"]["fitv"]["claim"] and not M["subtract"]["gf"]["claim"]
        assert M["subtract"]["probe"]["role"] == "exploratory" and "p_holm" not in M["subtract"]["probe"]
        # Holm, by hand
        ps = {d: M["subtract"][d]["tests"]["all"]["p"] for d in ("sprobe", "fitv", "gf")}
        order = sorted(ps, key=ps.get)
        want, run = {}, 0.0
        for i, d in enumerate(order):
            run = max(run, min(1.0, (3 - i) * ps[d]))
            want[d] = run
        assert all(abs(M["holm"][d] - want[d]) < 1e-12 for d in ps)
        # head to head
        H = M["head_to_head"]
        assert abs(H["diff"] - (stat("sprobe_sub") - stat("gf_sub"))) < 1e-12 and H["lo"] < H["diff"] < H["hi"] < 0
        assert H["ours_beats_goodfire"]
        # odd and even parts: sprobe is odd (a direction), fitv is even (lowers hacking whichever the sign)
        oe = M["odd_even"]
        assert set(oe) == {"sprobe", "fitv", "gf"}
        assert abs(oe["sprobe"]["odd"] - (stat("sprobe_add") - stat("sprobe_sub")) / 2) < 1e-12
        assert abs(oe["fitv"]["even"] - (stat("fitv_add") + stat("fitv_sub")) / 2) < 1e-12
        assert oe["sprobe"]["p_odd"] < 0.01 and oe["fitv"]["p_odd"] > 0.2 and oe["fitv"]["p_even_two_sided"] < 0.01
        assert abs(oe["sprobe"]["t_odd"] - oe["sprobe"]["odd"] / np.sqrt((o ** 2).mean())) < 1e-9
        # validity rule
        assert M["conditions"]["fitv_add"]["degraded"] and oe["fitv"]["degraded_add"] and not M["conditions"]["sprobe_sub"]["degraded"]
        assert M["conditions"]["fitv_add"]["valid"] < 0.8 <= M["conditions"]["sprobe_add"]["valid"]
        # projection is secondary, against the eight random project-outs
        P = M["projection"]
        assert P["null"]["R"] == 8 and P["null"]["df"] == 7 and P["dirs"]["sprobe"]["test"]["p"] < 0.01 < P["dirs"]["gf"]["test"]["p"]
        # fitting group: subtraction only, 16 random subtracted conditions
        F = out["fitting"]
        assert F["envs"] == fitg and F["null"]["R"] == 16 and F["null"]["df"] == 15
        assert abs(F["subtract"]["sprobe"]["S"] - stat("sprobe_sub", fitg)) < 1e-12 and F["subtract"]["sprobe"]["test"]["p"] < 0.01
        assert "sprobe_add" not in F["conditions"]
        assert "pairing" not in out                                                  # no such runs here

        # a condition that lacks an environment is left out of the group, not averaged over fewer
        os.remove(EV.ROOT / "n1_gf_sub" / "ref_copy__tempt.labels.jsonl")
        out2 = EV.main(["--prefix", "n1_", "--pair-prefix", "zz_", "--old-prefix", "yy_", "--boot", "200", "--out", str(tmp / "s2.json")])
        assert "gf_sub" in out2["main"]["skipped"] and "gf" not in out2["main"]["subtract"]
        assert set(out2["main"]["holm"]) == {"sprobe", "fitv"} and "head_to_head" not in out2["main"]
    finally:
        EV.ROOT = old


def test_steer_eval_null_calibration():
    """With no planted effect, the candidate's p-value is roughly uniform: about 5% of draws fall below 0.05."""
    rng = np.random.default_rng(3)
    hits = 0
    R = 4000
    for _ in range(R):
        e, o = rng.normal(-0.03, 0.05, 17), rng.normal(0, 0.08, 17)
        sgn = rng.choice([-1, 1], 17)
        N = EV.pair_null((e + o)[:16], (e - o)[:16])
        hits += EV.pair_test(e[16] + sgn[16] * o[16], N)["p"] < 0.05
    assert 0.03 < hits / R < 0.07, hits / R
    h = EV.holm(dict(a=0.01, b=0.04, c=0.03))
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12


def test_steer_eval_pairing():
    tmp = Path(tempfile.mkdtemp())
    old = EV.ROOT
    EV.ROOT = tmp / "base_rate"
    try:
        rng = np.random.default_rng(5)
        envs = EV.PAIR_ENVS
        U = {e: rng.random(30) for e in envs}
        conds = {"s3_base": 0.0, "s3_fitww2_neg": -0.5, "s3_fitww2_pos": -0.3, "s3_sprobe20_neg": -0.4, "s3_sprobe20_pos": 0.2,
                 "n3_noise_sub": -0.45, "n3_noise_add": -0.3, "n3_probe_sub": -0.3, "n3_probe_add": 0.2}
        for j in range(2, 8):
            ev, od = rng.normal(-0.05, 0.05), rng.normal(0, 0.1)
            conds[f"s3_r{j}20_pos"], conds[f"s3_r{j}20_neg"] = ev + od, ev - od
        for c, s in conds.items():
            D = EV.ROOT / c
            D.mkdir(parents=True)
            for e in envs:
                with open(D / f"{e}__tempt.labels.jsonl", "w") as f:
                    for i in range(30):                                     # 24 November's labels carry k = 0
                        f.write(json.dumps(dict(uid=f"{e}/{40 + i}", k=0, cond="tempt", hit=bool(U[e][i] < 0.6 + s),
                                                trunc=False, valid=True)) + "\n")
        out = EV.main(["--prefix", "none_", "--boot", "500", "--out", str(tmp / "p.json")])
        P = out["pairing"]
        assert P["base_run"] == "s3_base" and P["null"]["J"] == 6 and P["null"]["df"] == 5
        assert set(P["dirs"]) == {"fitww2", "sprobe20", "noise", "probe"}
        assert P["dirs"]["noise"]["runs"] == ["n3_noise_sub", "n3_noise_add"] and P["dirs"]["fitww2"]["runs"] == ["s3_fitww2_neg", "s3_fitww2_pos"]
        assert P["dirs"]["noise"]["sub"] < -0.3 and P["dirs"]["noise"]["even"] < -0.25 and P["dirs"]["probe"]["odd"] > 0.15
        d = P["fitted_minus_noise_sub"]
        assert abs(d["diff"] - (P["dirs"]["fitww2"]["sub"] - P["dirs"]["noise"]["sub"])) < 1e-12 and d["lo"] <= d["diff"] <= d["hi"]
        ev = P["fitted_minus_noise_even"]
        assert abs(ev["diff"] - (P["dirs"]["fitww2"]["even"] - P["dirs"]["noise"]["even"])) < 1e-12
    finally:
        EV.ROOT = old


# ------------------------------------------------------------------------------------------------- plain runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    bad = 0
    for n, f in tests:
        if only and not any(o in n for o in only):
            continue
        t0 = time.time()
        import contextlib
        import io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                f()
            print(f"PASS {n} ({time.time() - t0:.1f} s)", flush=True)
        except _Skip as e:
            print(f"SKIP {n}: {e}", flush=True)
        except BaseException as e:  # noqa: BLE001
            bad += 1
            import traceback
            print(buf.getvalue()[-3000:])
            traceback.print_exc()
            print(f"FAIL {n}: {type(e).__name__}: {e}", flush=True)
    print(f"FAILURES: {bad}")
    sys.exit(1 if bad else 0)

"""Per-request steering for Qwen3.5-family models on vLLM 0.34.0 (CUDA path: vllm/models/qwen3_5/nvidia/model.py).

    /root/vllm/bin/python scripts/flag/night/patch_vllm_steer_multi.py            # patch the installed vLLM
    /root/vllm/bin/python scripts/flag/night/patch_vllm_steer_multi.py --check    # anchors only, writes nothing
    /root/vllm/bin/python scripts/flag/night/patch_vllm_steer_multi.py --restore  # put model.py.orig back
    python scripts/flag/night/patch_vllm_steer_multi.py --root <dir holding models/ and v1/>   # a source tree (tests)

It replaces patch_vllm_steer.py + patch_vllm_steer_cuda.py (never apply those as well). It is idempotent: a file that
already carries this version is left alone (exit 0); a file carrying another version of this patch is restored from
model.py.orig and patched again. If an anchor is missing the script prints one line starting PATCH_FAILED that names
the anchor, changes nothing and exits 1; steering then falls back to the single-vector scripts of 24 November
(scripts/flag/patch_vllm_steer.py + patch_vllm_steer_cuda.py + serve_q38_steer.sh, one condition at a time).

The server loads ONE table at start-up and never reloads it:

    RLP_STEER_LAYER=36 RLP_STEER_TABLE=/workspace/steer_table.pt vllm serve ... --no-enable-prefix-caching

Environment variables read by the patched server: RLP_STEER_TABLE (path of the table), RLP_STEER_LAYER (block index;
unset or -1 = patch inactive, the server behaves as unpatched), RLP_STEER_REQUIRE_VID (1 = a request without a vid
kills the server; default 0 = it gets row 0 and is counted in <table>.stats as n_no_vid).

Table (torch file written by rlp/flag/steer_table.py, Interface 6): {"add": float32 [K, d], "proj": float32 [K, d],
"layer": L, "names": [K str]}; row 0 is "base" and all zero. A request names a row with
extra_body={"vllm_xargs": {"vid": k}}; at the input of block L, at every position of that request (prompt and
generated tokens):

    h <- h + ADD[k]                       (bf16 add: the same arithmetic as the old single-vector patch)
    s = h + residual;  h <- h - (s.u) u    with u = PROJ[k] (unit, or zero = no projection; fp32 arithmetic)

The stream entering block L is s = h + residual (block L's input_layernorm is a fused add-norm), so the projection
removes u from the full stream. A row that has both (only the self-test's canary_addproj does) adds first and then
projects, so adding c*u and projecting u must equal projecting u alone: that is how the projection is tested through
the API.

Fail-loud design (review of 2026-11-24, re-checked against the v0.34.0 source on 2026-11-25). Every check below
raises in the engine process (the server dies and every in-flight request errors) and appends the reason to
<table>.err, instead of steering wrongly:
  * patch time: vLLM version, the three anchors in model.py, the signatures and call sites of
    GPUModelRunner.add_requests / prepare_inputs / _remove_request, and that the patched file compiles;
  * start-up: TP = PP = DP = 1, no speculative decoding, prefix caching off, no sequence parallelism, 1 <= L < depth,
    model built once, table well formed (row 0 zero, unit-or-zero PROJ rows, finite, unique names), GPU copy of the
    table bit-equal to the file; writes <table>.loaded (JSON: sha256, names, K, ...);
  * every request: a vid, if present, is an integer in [0, K) (requests from vLLM's own start-up warm-up, whose ids
    start with "_", get row 0);
  * every step: each scheduled slot was assigned by add_requests for exactly this request id, query_start_loc matches
    the per-request token counts, no pcp / speculator / adaptive verification / micro-batching / fast prefill;
  * first real step and every 5000 real steps: GPU table still bit-equal to the file, buffers not reallocated.
<table>.stats (JSON, rewritten at most every 0.5 s) holds the number of tokens that went through block L per row;
rlp/flag/steer_selftest.py checks it against the token counts the client saw.
"""
import os
import py_compile
import sys

STEER_SRC = r'''
def _rlp_steer(mod, h, r):
    """h: [T, d] output of block L-1 (model dtype), r: [T, d] residual; the stream entering block L is h + r.
    mod.rlp_tok[:T] holds this step's row id per token (written by the prepare_inputs hook). The add is applied first,
    then the projection, on the stream that already contains the add."""
    T = h.shape[0]
    assert r is not None and h.dim() == 2 and r.shape == h.shape, "rlp-steer: bad shapes"
    assert T <= mod.rlp_tok.shape[0], f"rlp-steer: {T} tokens > rlp_tok {mod.rlp_tok.shape[0]}"
    assert mod.rlp_add.dtype == h.dtype and mod.rlp_proj.dtype == torch.float32, "rlp-steer: table dtype"
    vid = mod.rlp_tok[:T]
    h = h + mod.rlp_add.index_select(0, vid)                # row 0 is +0: exact identity
    if not mod.rlp_has_proj:
        return h
    u = mod.rlp_proj.index_select(0, vid)                   # [T, d] fp32, zero rows = no projection
    s = h.to(torch.float32, copy=True)
    s.add_(r)                                               # the stream block L will see
    c = (s * u).sum(-1, keepdim=True)                       # [T, 1]; exactly 0 for zero rows
    s.copy_(h)
    s.addcmul_(c, u, value=-1.0)                            # h - (s.u) u in fp32; h exactly for zero rows
    return s.to(h.dtype)
'''

HELPER = r'''
# ---------------------------------------------------------------------------------------- rlp-steer-multi v4
import io as _rlp_io, os as _rlp_os, json as _rlp_json, hashlib as _rlp_hashlib, time as _rlp_time
import numpy as _rlp_np
_RLP_STEER_LAYER = int(_rlp_os.environ.get("RLP_STEER_LAYER", "-1"))
_RLP_TABLE = _rlp_os.environ.get("RLP_STEER_TABLE", "")
_RLP_REQUIRE_VID = _rlp_os.environ.get("RLP_STEER_REQUIRE_VID", "0") == "1"
_RLP = dict(model=None, K=0, names=None, sha=None, add_cpu=None, proj_cpu=None, ptrs=None, vid_np=None,
            slot_req=None, counts=None, steps=0, real_steps=0, last_write=0.0, runner_checked=False,
            verified_real=False, max_T=0, n_no_vid=0)


def _rlp_fail(msg):
    try:
        with open(_RLP_TABLE + ".err", "a") as f:
            f.write(f"{_rlp_time.strftime('%Y-%m-%d %H:%M:%S')} pid {_rlp_os.getpid()} {msg}\n")
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError("[rlp-steer] " + msg)


def _rlp_check(ok, msg):
    if not ok:
        _rlp_fail(msg)

''' + STEER_SRC + r'''

def _rlp_read_table(d_model, layer):
    raw = open(_RLP_TABLE, "rb").read()
    z = torch.load(_rlp_io.BytesIO(raw), map_location="cpu", weights_only=True)
    add, proj, names = z["add"], z["proj"], [str(x) for x in z["names"]]
    _rlp_check(int(z["layer"]) == layer, f"table layer {z['layer']} != RLP_STEER_LAYER {layer}")
    _rlp_check(add.dtype == torch.float32 and proj.dtype == torch.float32, "table must be float32")
    K = add.shape[0]
    _rlp_check(K >= 1 and tuple(add.shape) == (K, d_model) and tuple(proj.shape) == (K, d_model),
               f"table shapes {tuple(add.shape)} {tuple(proj.shape)}, model width {d_model}")
    _rlp_check(len(names) == K and len(set(names)) == K and names[0] == "base", "names: K unique, names[0]='base'")
    _rlp_check(bool(torch.isfinite(add).all()) and bool(torch.isfinite(proj).all()), "non-finite table entries")
    _rlp_check(not bool(add[0].any()) and not bool(proj[0].any()), "row 0 must be all zero")
    pn = proj.double().norm(dim=-1)
    pz = pn == 0
    _rlp_check(bool((((pn - 1).abs() < 1e-5) | pz).all()), "PROJ rows must be unit (to 1e-5) or exactly zero")
    return dict(raw_sha=_rlp_hashlib.sha256(raw).hexdigest(), add=add.contiguous(), proj=proj.contiguous(),
                names=names, K=K, has_proj=bool((~pz).any()), add_norm=add.double().norm(dim=-1).tolist())


def _rlp_verify(mod, where):
    _rlp_check((mod.rlp_add.data_ptr(), mod.rlp_proj.data_ptr(), mod.rlp_tok.data_ptr()) == _RLP["ptrs"],
               f"{where}: steering buffers were reallocated (CUDA graphs would read the old ones)")
    _rlp_check(mod.rlp_add.dtype == _RLP["add_cpu"].dtype and mod.rlp_proj.dtype == torch.float32
               and mod.rlp_tok.dtype == torch.int32, f"{where}: steering buffer dtype changed")
    torch.cuda.synchronize()
    ok = torch.equal(mod.rlp_add.cpu(), _RLP["add_cpu"]) and torch.equal(mod.rlp_proj.cpu(), _RLP["proj_cpu"])
    _rlp_check(ok, f"{where}: GPU steering table differs from {_RLP_TABLE}")


def _rlp_init(mod, vllm_config, config):
    L = _RLP_STEER_LAYER
    for suf in (".err", ".loaded", ".stats"):          # status files of an earlier server must not be trusted
        try:
            _rlp_os.remove(_RLP_TABLE + suf)
        except OSError:
            pass
    _rlp_check(_RLP["model"] is None, "Qwen3_5Model built twice in one process")
    _rlp_check(bool(_RLP_TABLE) and _rlp_os.path.exists(_RLP_TABLE), f"RLP_STEER_TABLE={_RLP_TABLE!r} missing")
    pc = vllm_config.parallel_config
    _rlp_check(get_tensor_model_parallel_world_size() == 1 and pc.pipeline_parallel_size == 1
               and pc.data_parallel_size == 1, "per-request steering supports TP = PP = DP = 1 only")
    # v4 (25 Nov): the cache key ignores the vid, so caching is allowed only when the caller promises to send a
    # per-condition cache_salt with every request (rlp/flag/mt.py does) and sets RLP_STEER_ALLOW_PREFIX_CACHE=1.
    _rlp_check(not vllm_config.cache_config.enable_prefix_caching or _rlp_os.environ.get("RLP_STEER_ALLOW_PREFIX_CACHE") == "1",
               "prefix caching must be off (--no-enable-prefix-caching): its cache key ignores the vid")
    _rlp_check(getattr(vllm_config, "speculative_config", None) is None, "speculative decoding not supported")
    _rlp_check(1 <= L < config.num_hidden_layers and mod.start_layer <= L < mod.end_layer, f"bad layer {L}")
    _rlp_check(not mod.use_sequence_parallel, "sequence parallelism not supported")
    t = _rlp_read_table(config.hidden_size, L)
    dt = vllm_config.model_config.dtype
    dev = torch.device("cuda", torch.cuda.current_device())
    N = max(int(vllm_config.scheduler_config.max_num_batched_tokens), 32768) + 8192
    # Explicit dtypes: the model is built under set_default_torch_dtype(bf16) (model_loader/base_loader.py:86),
    # so a bare torch.zeros(...) here would silently be bf16 and the PROJ rows would not be unit.
    mod.register_buffer("rlp_add", torch.zeros(t["K"], config.hidden_size, dtype=dt, device=dev), persistent=False)
    mod.register_buffer("rlp_proj", torch.zeros(t["K"], config.hidden_size, dtype=torch.float32, device=dev),
                        persistent=False)
    mod.register_buffer("rlp_tok", torch.zeros(N, dtype=torch.int32, device=dev), persistent=False)
    mod.rlp_has_proj = t["has_proj"]
    _RLP.update(K=t["K"], names=t["names"], sha=t["raw_sha"], add_cpu=t["add"].to(dt), proj_cpu=t["proj"],
                counts=_rlp_np.zeros(t["K"], _rlp_np.int64),
                ptrs=(mod.rlp_add.data_ptr(), mod.rlp_proj.data_ptr(), mod.rlp_tok.data_ptr()))
    with torch.no_grad():
        mod.rlp_add.copy_(_RLP["add_cpu"])
        mod.rlp_proj.copy_(_RLP["proj_cpu"])
    _rlp_verify(mod, "init")
    _RLP["model"] = mod
    _rlp_patch_runner()
    info = dict(patch="rlp-steer-multi v4", sha256=t["raw_sha"], K=t["K"], names=t["names"], layer=L, has_proj=t["has_proj"],
                add_norm=t["add_norm"], dtype=str(dt), rlp_tok_len=N, pid=_rlp_os.getpid(), time=_rlp_time.time())
    tmp = _RLP_TABLE + ".loaded.tmp"
    with open(tmp, "w") as f:
        _rlp_json.dump(info, f)
    _rlp_os.replace(tmp, _RLP_TABLE + ".loaded")
    print(f"[rlp-steer] layer {L}: {t['K']} rows loaded from {_RLP_TABLE} sha256 {t['raw_sha'][:12]} "
          f"proj={t['has_proj']} rlp_tok={N}", flush=True)


def _rlp_vid_of(req_id, sp):
    ea = getattr(sp, "extra_args", None) if sp is not None else None
    if ea and "vid" in ea:
        v = ea["vid"]
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v)
        _rlp_check(isinstance(v, int) and not isinstance(v, bool), f"request {req_id}: vid {v!r} is not an integer")
        _rlp_check(0 <= v < _RLP["K"], f"request {req_id}: vid {v} outside [0, {_RLP['K']})")
        return v
    if str(req_id).startswith("_"):     # vLLM's own start-up warm-up requests (gpu/warmup.py ids "_warmup_...")
        return 0
    _rlp_check(not _RLP_REQUIRE_VID, f"request {req_id} has no vid (send extra_body={{'vllm_xargs': {{'vid': k}}}})")
    _RLP["n_no_vid"] += 1
    return 0


def _rlp_on_add(runner, so):
    if _RLP["vid_np"] is None:
        _RLP["vid_np"] = _rlp_np.full(runner.max_num_reqs, -1, _rlp_np.int32)
        _RLP["slot_req"] = [None] * runner.max_num_reqs
    for nr in so.scheduled_new_reqs:           # new requests AND resumed (preempted) ones, re-sent in full
        i = runner.req_states.req_id_to_index[nr.req_id]
        _RLP["vid_np"][i] = _rlp_vid_of(nr.req_id, nr.sampling_params)
        _RLP["slot_req"][i] = nr.req_id


def _rlp_on_remove(req_id, idx):
    if idx is not None and _RLP["vid_np"] is not None:
        _RLP["vid_np"][idx] = -1
        _RLP["slot_req"][idx] = None


def _rlp_write_stats():
    try:
        d = dict(pid=_rlp_os.getpid(), time=_rlp_time.time(), sha256=_RLP["sha"], steps=_RLP["steps"],
                 real_steps=_RLP["real_steps"], max_T=_RLP["max_T"], n_no_vid=_RLP["n_no_vid"],
                 tokens_per_vid=_RLP["counts"].tolist())
        tmp = _RLP_TABLE + ".stats.tmp"
        with open(tmp, "w") as f:
            _rlp_json.dump(d, f)
        _rlp_os.replace(tmp, _RLP_TABLE + ".stats")
    except Exception as e:  # noqa: BLE001  (diagnostics only; never kill the server over a stats file)
        print(f"[rlp-steer] stats write failed: {e!r}", flush=True)


def _rlp_on_prepare(runner, ib, brs, h2d):
    m = _RLP["model"]
    if not _RLP["runner_checked"]:
        for a in ("pcp_manager", "speculator", "adaptive_verification", "ubatch_runner", "fast_prefill"):
            _rlp_check(getattr(runner, a, None) is None, f"runner.{a} is set: token rows would not line up")
        _rlp_check(not runner.use_pp, "pipeline parallelism not supported")
        _RLP["runner_checked"] = True
    n, npad, nr = int(ib.num_tokens), int(ib.num_tokens_after_padding), int(ib.num_reqs)
    qsl = _rlp_np.asarray(ib.query_start_loc_np[: nr + 1], dtype=_rlp_np.int64)
    nst = _rlp_np.asarray(brs.num_scheduled_tokens, dtype=_rlp_np.int64)
    idx = _rlp_np.asarray(ib.idx_mapping_np, dtype=_rlp_np.int64)
    _rlp_check(len(idx) == nr == len(nst) == len(ib.req_ids), "batch length mismatch")
    _rlp_check(int(qsl[0]) == 0 and int(qsl[-1]) == n and _rlp_np.array_equal(_rlp_np.diff(qsl), nst),
               "query_start_loc does not match per-request token counts")
    _rlp_check(n <= npad <= m.rlp_tok.shape[0], f"{n} tokens padded to {npad} > rlp_tok {m.rlp_tok.shape[0]}")
    sr = _RLP["slot_req"]
    for i in range(nr):
        if sr[idx[i]] != ib.req_ids[i]:
            _rlp_fail(f"slot {idx[i]} holds {sr[idx[i]]!r} but the batch says {ib.req_ids[i]!r}")
    vids = _RLP["vid_np"][idx]
    _rlp_check(bool((vids >= 0).all()) and bool((vids < _RLP["K"]).all()), f"unassigned slot in batch: {vids}")
    tok = _rlp_np.zeros(npad, _rlp_np.int32)            # padding rows -> row 0 (identity)
    tok[:n] = _rlp_np.repeat(vids, nst)
    # Fresh pinned staging + copy on the current stream, as vLLM does for query_start_loc: ordered after the previous
    # step's forward and before this step's, also under async scheduling (same thread, same stream).
    h2d(tok, out=m.rlp_tok[:npad])
    _RLP["counts"] += _rlp_np.bincount(vids, weights=nst, minlength=_RLP["K"]).astype(_rlp_np.int64)
    _RLP["steps"] += 1
    _RLP["max_T"] = max(_RLP["max_T"], npad)
    if any(not str(r).startswith("_") for r in ib.req_ids):
        if not _RLP["verified_real"] or _RLP["real_steps"] % 5000 == 0:
            _rlp_verify(m, f"real step {_RLP['real_steps']}")
            _RLP["verified_real"] = True
        _RLP["real_steps"] += 1
    now = _rlp_time.time()
    if now - _RLP["last_write"] >= 0.5:
        _RLP["last_write"] = now
        _rlp_write_stats()


def _rlp_patch_runner():
    import inspect
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as R
    _rlp_check(not getattr(R, "_rlp_patched", False), "GPUModelRunner patched twice")
    sig = list(inspect.signature(R.prepare_inputs).parameters)
    _rlp_check(sig == ["self", "scheduler_output", "batch_req_state", "batch_desc", "num_active_loras"],
               f"GPUModelRunner.prepare_inputs signature changed: {sig}")
    add0, prep0, rem0 = R.add_requests, R.prepare_inputs, R._remove_request

    def add_requests(self, scheduler_output):
        add0(self, scheduler_output)
        _rlp_on_add(self, scheduler_output)

    def _remove_request(self, req_id):
        idx = self.req_states.req_id_to_index.get(req_id)
        out = rem0(self, req_id)
        _rlp_on_remove(req_id, idx)
        return out

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
        ib = prep0(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
        _rlp_on_prepare(self, ib, batch_req_state, async_tensor_h2d)
        return ib

    R.add_requests, R.prepare_inputs, R._remove_request, R._rlp_patched = add_requests, prepare_inputs, _remove_request, True

'''

A = "\nclass Qwen3_5Model(nn.Module, EagleModelMixin):"
B = "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n\n    def embed_input_ids"
C = ("            hidden_states, residual = layer(\n                positions=positions,\n"
     "                hidden_states=hidden_states,\n                residual=residual,\n            )\n"
     "            self._maybe_add_hidden_state(\n                aux_hidden_states, layer_idx + 1,")
B_NEW = ("        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
         "        if _RLP_STEER_LAYER >= 0:  # rlp-steer\n"
         "            _rlp_init(self, vllm_config, config)\n\n    def embed_input_ids")
C_NEW = ("            if layer_idx == _RLP_STEER_LAYER:  # rlp-steer\n"
         "                hidden_states = _rlp_steer(self, hidden_states, residual)\n" + C)

# Runner anchors that the hooks rely on (vllm/v1/worker/gpu/model_runner.py at v0.34.0).
RUNNER_ANCHORS = {
    "    def add_requests(self, scheduler_output: SchedulerOutput) -> None:\n": 1,
    "    def _remove_request(self, req_id: str) -> bool:\n": 1,
    "    def prepare_inputs(\n        self,\n        scheduler_output: SchedulerOutput,\n"
    "        batch_req_state: \"BatchReqState\",\n        batch_desc: BatchExecutionDescriptor,\n"
    "        num_active_loras: int,\n    ) -> InputBatch:\n": 1,
    "            self.add_requests(scheduler_output)\n": 1,
    "            input_batch = self.prepare_inputs(\n": 1,
    "                self._remove_request(req_id)\n": 0,      # sanity: there is no such indented call...
    "            self._remove_request(req_id)\n": 2,           # ...it is called from finish_requests and add_requests
    "        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])\n": 1,
}


VERSION = "rlp-steer-multi v4"
FALLBACK = ("Nothing was changed. Fallback: the single-vector scripts of 24 November (scripts/flag/patch_vllm_steer.py, "
            "patch_vllm_steer_cuda.py, serve_q38_steer.sh), one condition at a time, no projection rows.")


class PatchError(RuntimeError):
    pass


def patched_source(s):
    """model.py text -> patched text. Raises PatchError naming the first anchor that is not found exactly once."""
    for name, x in (("class header", A), ("end of Qwen3_5Model.__init__", B), ("layer loop of Qwen3_5Model.forward", C)):
        if s.count(x) != 1:
            raise PatchError(f"model.py anchor '{name}' found {s.count(x)}x, expected 1x: {x.strip()[:90]!r}")
    if "get_tensor_model_parallel_world_size" not in s.split(A)[0]:
        raise PatchError("model.py no longer imports get_tensor_model_parallel_world_size")
    s2 = s.replace(A, HELPER + A).replace(B, B_NEW).replace(C, C_NEW)
    if not (s2.count("# rlp-steer\n") == 2 and s2.count("_rlp_steer(self, hidden_states, residual)") == 1):
        raise PatchError("internal: the patched text does not contain the two hooks exactly once")
    return s2


def check_runner(rs):
    for anchor, want in RUNNER_ANCHORS.items():
        got = rs.count(anchor)
        if got != want:
            raise PatchError(f"model_runner.py anchor found {got}x, expected {want}x: {anchor.strip()[:90]!r}")


def main(root=None, check=False, restore=False):
    """Returns "patched", "already", "checked" or "restored"; raises PatchError (the file is then untouched)."""
    if root is None:
        import vllm
        root = os.path.dirname(vllm.__file__)
        if os.environ.get("RLP_PATCH_SKIP_VERSION") != "1" and not vllm.__version__.startswith("0.34.0"):
            raise PatchError(f"written for vLLM 0.34.0, found {vllm.__version__} (RLP_PATCH_SKIP_VERSION=1 overrides)")
    p = os.path.join(root, "models/qwen3_5/nvidia/model.py")
    r = os.path.join(root, "v1/worker/gpu/model_runner.py")
    for f in (p, r):
        if not os.path.exists(f):
            raise PatchError(f"{f} does not exist: not a vLLM 0.34.0 tree with the CUDA Qwen3.5 model")
    s, rs = open(p).read(), open(r).read()
    orig = p + ".orig"
    if restore:
        if "rlp-steer" not in s:
            print("not patched, nothing to restore:", p)
            return "restored"
        if not os.path.exists(orig) or "rlp-steer" in open(orig).read():
            raise PatchError(f"{orig} is missing or itself patched: reinstall vllm==0.34.0 (--no-deps --reinstall)")
        open(p, "w").write(open(orig).read())
        print("restored", p)
        return "restored"
    check_runner(rs)
    if "rlp-steer" in s:
        # Already carries a steering patch. Same version and byte-identical to what we would write: done. Otherwise
        # start again from the pristine copy this script saved; the old single-vector patch saved none.
        if os.path.exists(orig) and "rlp-steer" not in open(orig).read():
            clean = open(orig).read()
            if patched_source(clean) == s:
                print(f"already patched ({VERSION}):", p)
                return "already"
            s = clean
            print("another version of the patch is in place: re-patching from", orig)
        else:
            raise PatchError("model.py carries an older steering patch and there is no clean model.py.orig: reinstall "
                             "vllm==0.34.0 (uv pip install --no-deps --reinstall vllm==0.34.0) and run this again")
    s2 = patched_source(s)
    tmp = p + ".rlp_tmp.py"
    open(tmp, "w").write(s2)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        raise PatchError(f"the patched model.py does not compile: {e}") from e
    finally:
        os.remove(tmp)
    if check:
        print("anchors found, patch would apply:", p)
        return "checked"
    if not os.path.exists(orig):
        open(orig, "w").write(s)
    open(p + ".rlp_new", "w").write(s2)
    os.replace(p + ".rlp_new", p)
    print(f"patched ({VERSION}):", p)
    return "patched"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="directory that holds models/ and v1/ (default: the installed vllm)")
    ap.add_argument("--check", action="store_true", help="verify the anchors and compile, write nothing")
    ap.add_argument("--restore", action="store_true", help="put model.py.orig back")
    a = ap.parse_args()
    try:
        main(a.root, a.check, a.restore)
    except PatchError as e:
        print(f"PATCH_FAILED: {e}\n{FALLBACK}", flush=True)
        sys.exit(1)

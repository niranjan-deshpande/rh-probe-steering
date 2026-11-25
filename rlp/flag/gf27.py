"""Goodfire's direction rebuilt on frozen Qwen3.8-27B at the input of block 36, from their prepared records only.

Source: data/goodfire/prepared_dl/synthetic_data/prepared/records.jsonl (fields id, group_id, split, metadata,
input_ids, spans) and data/goodfire/contrasts.json. THEIR token ids are fed to the model as they are; nothing is
rendered or tokenised again and none of our data is added. Every span (`read_close`, `patch_close`, `act_close`) is
the six closing tokens `</function>\\n</tool_call><|im_end|>` of one tool call.

    pod     python -m rlp.flag.gf27 check                 # batched == one at a time; prints GF27_CHECK_OK / _FAIL
    pod     python -m rlp.flag.gf27 harvest               # resumable shards, then merge; prints GF27_HARVEST_DONE
    either  python -m rlp.flag.gf27 merge                 # shards -> gf27_acts.npz + gf27_meta.json (Interface 5)
    laptop  python -m rlp.flag.gf27 dirs                  # -> dirs_36.npz
    laptop  python -m rlp.flag.gf27 detect                # matched AUROC on cached on-policy activations

Harvest. The model is causal, so each record is cut after its last span. Records are sorted by length, right-padded
and batched under a token budget; the forward pass stops at block 36. Per span three pooled vectors are kept:
`suf` (mean over the six span tokens, the primary read-out), `last` (the final token, <|im_end|>) and `turn` (mean
over the assistant turn that ends with the span, after its header). gf27_meta.json has one row per span.

Directions (dirs). How a contrast is defined, exactly: for a contrast c of contrasts.json, a span row belongs to c
if the record's metadata.dataset equals c.dataset AND the pair (span name, metadata.cell) is listed in c.pos_keys
(hack side, label 1) or in c.neg_keys (honest side, label 0). (For the single-span contrasts this is "span == c.span
and cell in c.pos / c.neg"; the pairs matter for pl_acquire, which reads peeks at read_close and exploits at
patch_close.) This is asserted to agree with the labels Goodfire stored on the spans. The contrast vector is
mean(suf of hack rows) - mean(suf of honest rows), over records of the chosen split (default: their `train` split,
2,504 of 3,123 records; their `validation` split is left untouched). Then
    c:<name>   that difference, unit norm, for each of the 18 released contrasts (12 single-cell `i_*`, 6 pooled `pl_*`)
    gf_pool    unit( mean over the 12 `i_*` contrasts of the raw differences ), not whitened. The six `pl_*` contrasts
               are unions of the same cells, so including them would count those cells twice; `gf_pool18` (all 18) and
               `gf_pool_unitmean` (mean of the 12 unit vectors) are stored as labelled variants only.
`last:gf_pool` and `turn:gf_pool` are the same recipe at the other two poolings. All vectors point hack-positive.

This module also holds the model loader and the batched reader shared by lingrad, mt_harvest and textscore.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import config

MODEL = "Qwen/Qwen3.8-27B"
RECORDS = config.DATA / "goodfire" / "prepared_dl" / "synthetic_data" / "prepared" / "records.jsonl"
CONTRASTS = config.DATA / "goodfire" / "contrasts.json"
OUT = config.CACHE / "flag" / "gf27"
ONP = config.CACHE / "flag" / "onp"
POOLS = ("suf", "last", "turn")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ------------------------------------------------------------------ shared: model loader and batched reader

def add_model_args(ap, layer=36):
    ap.add_argument("--model", default=MODEL, help="hub name or local directory")
    ap.add_argument("--tokenizer", default="", help="tokenizer name or directory if the model directory has none")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cuda, cpu")
    ap.add_argument("--keep-blocks", type=int, default=0, help="load only blocks 0..N-1 (readers only; 37 saves ~23 GB)")
    ap.add_argument("--layer", type=int, default=layer, help="index L = input of block L = hidden_states[L]")
    return ap


def fla_gate():
    """Block `import fla` (so transformers binds its reference kernels) when the fast kernels failed their check:
    RLP_NO_FLA=1, or scripts/flag/night/fla_check.py left FLA_SLOW in <cache>/flag/fla_status.txt. RLP_NO_FLA=0
    overrides the file. Must run before transformers imports the Qwen3.5 model code. -> True if blocked."""
    v = os.environ.get("RLP_NO_FLA")
    if v is None:
        f = config.CACHE / "flag" / "fla_status.txt"
        v = "1" if f.exists() and f.read_text().strip() == "FLA_SLOW" else "0"
    if v == "1" and "transformers.models.qwen3_5.modeling_qwen3_5" not in sys.modules:
        sys.modules["fla"] = None
        return True
    return False


def load_model(model=MODEL, tokenizer="", dtype="bf16", device="auto", keep_blocks=0):
    """-> (frozen model in eval mode, tokenizer). `model` may be an argparse namespace made with add_model_args, or a
    (model, tokenizer) pair passed in by a test."""
    if not isinstance(model, (str, tuple, list)):            # a namespace
        a = model
        if not isinstance(a.model, str):
            return a.model
        return load_model(a.model, getattr(a, "tokenizer", ""), getattr(a, "dtype", "bf16"), getattr(a, "device", "auto"),
                          getattr(a, "keep_blocks", 0))
    if not isinstance(model, str):
        return model
    if fla_gate():
        print("[reader] fast kernels blocked (FLA_SLOW): transformers uses its reference kernels", flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tok = AutoTokenizer.from_pretrained(tokenizer or model)
    kw = {}
    if keep_blocks:
        cfg = AutoConfig.from_pretrained(model)
        tc = getattr(cfg, "text_config", cfg)
        assert keep_blocks <= tc.num_hidden_layers
        tc.num_hidden_layers = keep_blocks
        if getattr(tc, "layer_types", None):
            tc.layer_types = list(tc.layer_types)[:keep_blocks]
        kw["config"] = cfg
    if dev != "cpu":
        kw["device_map"] = dev                               # needs accelerate, which the pod has
    m = AutoModelForCausalLM.from_pretrained(model, dtype=DTYPES[dtype], **kw)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def blocks_of(model):
    from . import onp
    return onp.blocks_of(model)


def width(model):
    return getattr(model.config, "text_config", model.config).hidden_size


def pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else 0


class _Stop(Exception):
    pass


def pad_batch(seqs, pad, device):
    """-> (ids [B, T], attention mask [B, T]) right-padded. The mask is None when no row is padded: transformers
    then builds no T x T causal mask and attention takes its memory-light causal path, as for a single sequence."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        am[b, :len(s)] = 1
    return ids.to(device), (None if bool(am.all()) else am.to(device))


def read_batch(model, seqs, L, pad):
    """Right-padded batch -> hidden_states[L] as [B, T, d] in the model's dtype, without running blocks >= L."""
    ids, am = pad_batch(seqs, pad, model.device)
    got = {}

    def pre(mod, args, kwargs):
        got["h"] = args[0] if args else kwargs["hidden_states"]
        raise _Stop()
    h = blocks_of(model)[L].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=am, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return got["h"]


def batches(order, lens, budget, solo=None):
    """Greedy batches over `order`: each batch's padded size (rows x longest row) stays within the token budget.
    Works for any order; a row longer than the budget gets a batch of its own, and so does any row longer than
    `solo` tokens (a padded batch needs a rows x T x T attention mask, which long sequences cannot afford)."""
    cur, mx = [], 0
    for i in order:
        if solo and lens[i] > solo:
            yield [i]
            continue
        m = max(mx, lens[i])
        if cur and m * (len(cur) + 1) > budget:
            yield cur
            cur, m = [], lens[i]
        cur.append(i)
        mx = m
    if cur:
        yield cur


def batch_check(model, seqs, L, pad, spans):
    """Worst relative difference, over spans [(b, s0, s1)], between the batched and the one-at-a-time span means."""
    Hb = read_batch(model, seqs, L, pad).float()
    worst = 0.0
    for b, s0, s1 in spans:
        u = Hb[b, s0:s1].mean(0)
        v = read_batch(model, [seqs[b]], L, pad).float()[0, s0:s1].mean(0)
        worst = max(worst, float((u - v).norm() / v.norm().clamp_min(1e-12)))
    return worst


def save_atomic(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ records and spans

def load_records(path=RECORDS, limit=0):
    rows = []
    for line in open(path):
        r = json.loads(line)
        r.pop("rendered_text", None)
        rows.append(r)
        if limit and len(rows) == limit:
            break
    return rows


def locate(row, im_start, header=None):
    """-> dict(ids = their ids cut after the last span, ro = [(s0, s1, t0)] per span) with [s0, s1) the span tokens
    and t0 the first token of that assistant turn after its header (<|im_start|>, 'assistant', newline)."""
    ids, ro = row["input_ids"], []
    for s in row["spans"]:
        s0, s1 = int(s["start"]), int(s["end"])
        if not (0 <= s0 < s1 <= len(ids)):
            raise ValueError(f"{row['id']} {s['name']}: span [{s0}, {s1}) outside {len(ids)} tokens")
        if s1 - s0 != 6:
            raise ValueError(f"{row['id']} {s['name']}: span has {s1 - s0} tokens, expected the six closing tokens")
        i = s0
        while i >= 0 and ids[i] != im_start:
            i -= 1
        if i < 0:
            raise ValueError(f"{row['id']} {s['name']}: no <|im_start|> before the span")
        if header is not None and ids[i:i + len(header)] != list(header):
            raise ValueError(f"{row['id']} {s['name']}: the turn holding the span is not an assistant turn")
        ro.append((s0, s1, min(i + 3, s0)))
    return dict(ids=ids[:max(r[1] for r in ro)], ro=ro)


def verify_vocab(tok, rows, n=0):
    """Their ids must mean the same thing in this tokenizer: every span must decode to its stored text."""
    bad = 0
    for r in rows[:n or None]:
        for s in r["spans"]:
            if "text" in s and tok.decode(r["input_ids"][s["start"]:s["end"]]) != s["text"]:
                bad += 1
    return bad


def meta_rows(rows):
    """One row per (record, span), in record then span order: the row order of gf27_acts.npz."""
    out = []
    for ri, r in enumerate(rows):
        for si, s in enumerate(r["spans"]):
            out.append(dict(rec=ri, span_i=si, id=r["id"], group_id=r["group_id"], split=r["split"],
                            dataset=r["metadata"]["dataset"], cell=r["metadata"]["cell"], span=s["name"],
                            message_index=(s.get("metadata") or {}).get("message_index"), labels=s.get("labels") or {},
                            start=s["start"], end=s["end"]))
    return out


def pool_spans(H, ro):
    """H [T, d] for one record -> (suf [n, d], last [n, d], turn [n, d]) as float32 tensors."""
    H = H.float()
    return (torch.stack([H[s0:s1].mean(0) for s0, s1, _ in ro]), torch.stack([H[s1 - 1] for _, s1, _ in ro]),
            torch.stack([H[t0:s1].mean(0) for _, s1, t0 in ro]))


# ------------------------------------------------------------------ pod: check, harvest, merge

def _setup(a):
    rows = load_records(a.records, a.limit)
    model, tok = load_model(a)
    im = tok.convert_tokens_to_ids("<|im_start|>")
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    bad = verify_vocab(tok, rows)
    if bad:
        raise SystemExit(f"[gf27] {bad} spans do not decode to their stored text: the tokenizer does not match their ids")
    loc = [locate(r, im, header) for r in rows]             # raises before any GPU time is spent
    return rows, model, tok, loc


def _check(model, tok, loc, a, n=6):
    pick = sorted(range(len(loc)), key=lambda i: -len(loc[i]["ids"]))
    pick = pick[::max(1, len(pick) // n)][:n]               # spread over lengths, so padding is exercised
    seqs = [loc[i]["ids"] for i in pick]
    spans = [(b, s0, s1) for b, i in enumerate(pick) for s0, s1, _ in loc[i]["ro"]]
    worst = batch_check(model, seqs, a.layer, pad_id(tok), spans)
    print(f"[gf27] batched vs one at a time, worst relative difference of a span mean: {worst:.2e} (tolerance {a.tol})", flush=True)
    return worst < a.tol


def check(a):
    rows, model, tok, loc = _setup(a)
    print("GF27_CHECK_OK" if _check(model, tok, loc, a) else "GF27_CHECK_FAIL", flush=True)


def harvest(a):
    rows, model, tok, loc = _setup(a)
    lens = [len(x["ids"]) for x in loc]
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    plan = dict(n_records=len(rows), shard=a.shard, layer=a.layer, model=a.model if isinstance(a.model, str) else "in-memory",
                dtype=getattr(a, "dtype", ""), records=str(a.records))
    pf = out / "shards" / "plan.json"
    if pf.exists() and json.load(open(pf)) != plan:
        raise SystemExit(f"[gf27] {pf} was written with other settings; delete {out / 'shards'} to start again")
    json.dump(plan, open(pf, "w"))
    budget = a.budget
    if not a.no_check and not _check(model, tok, loc, a):
        budget = 1                                           # one sequence per forward pass: slower, never padded
        print("[gf27] GF27_CHECK_FAIL: falling back to one sequence per forward pass", flush=True)
    print(f"[gf27] {len(rows)} records, {sum(len(r['spans']) for r in rows)} spans, {sum(lens):,} tokens after cutting, "
          f"longest {max(lens)}", flush=True)
    order = sorted(range(len(rows)), key=lambda i: -lens[i])
    shards = [order[k:k + a.shard] for k in range(0, len(order), a.shard)]
    t0, done = time.time(), 0
    for si, sh in enumerate(shards):
        f = out / "shards" / f"s{si:04d}.npz"
        if f.exists():
            continue
        keys, acc = [], {p: [] for p in POOLS}
        for bt in batches(sh, lens, budget):
            H = read_batch(model, [loc[i]["ids"] for i in bt], a.layer, pad_id(tok))
            for b, i in enumerate(bt):
                for p, x in zip(POOLS, pool_spans(H[b], loc[i]["ro"])):
                    acc[p].append(x.cpu())
                keys += [(i, k) for k in range(len(loc[i]["ro"]))]
            done += sum(lens[i] for i in bt)
            del H
        save_atomic(f, keys=np.array(keys, np.int64), **{p: torch.cat(acc[p]).numpy() for p in POOLS})
        el = time.time() - t0
        print(f"[gf27] shard {si + 1}/{len(shards)}  {done:,} tokens this run  {done / el:.0f} tok/s", flush=True)
    merge(a, rows)
    print("GF27_HARVEST_DONE", flush=True)


def merge(a, rows=None):
    out = Path(a.out)
    rows = rows if rows is not None else load_records(a.records, a.limit)
    meta = meta_rows(rows)
    idx = {(m["rec"], m["span_i"]): j for j, m in enumerate(meta)}
    arr, ok = None, np.zeros(len(meta), bool)
    for f in sorted((out / "shards").glob("s*.npz")):
        z = np.load(f)
        if arr is None:
            arr = {p: np.zeros((len(meta), z["suf"].shape[1]), np.float32 if p == "suf" else np.float16) for p in POOLS}
        j = np.array([idx[(int(r), int(k))] for r, k in z["keys"]], np.int64)
        for p in POOLS:
            arr[p][j] = z[p]
        ok[j] = True
    if arr is None:
        raise SystemExit(f"[gf27] no shards under {out / 'shards'}")
    save_atomic(out / "gf27_acts.npz", ok=ok, layer=np.array(a.layer), **arr)
    json.dump(meta, open(out / "gf27_meta.json", "w"))
    print(f"[gf27] merged {int(ok.sum())}/{len(meta)} spans -> {out / 'gf27_acts.npz'}", flush=True)


# ------------------------------------------------------------------ laptop: directions

def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def contrast_labels(meta, C):
    """{contrast: int array over meta rows, 1 hack side / 0 honest side / -1 not in the contrast}, from contrasts.json
    and the record metadata (see the module docstring). Asserted equal to the labels stored on the spans."""
    Y = {}
    for name, c in C.items():
        pos, neg = {tuple(k) for k in c["pos_keys"]}, {tuple(k) for k in c["neg_keys"]}
        assert not pos & neg, name
        y = np.full(len(meta), -1)
        for j, m in enumerate(meta):
            if m["dataset"] == c["dataset"] and (m["span"], m["cell"]) in pos | neg:
                y[j] = int((m["span"], m["cell"]) in pos)
            stored = (m.get("labels") or {}).get(name, -1)
            if stored != y[j]:
                raise AssertionError(f"{name}: row {m['id']}/{m['span']} is {y[j]} by contrasts.json, {stored} on the span")
        Y[name] = y
    return Y


def build_dirs(acts, meta, C, split="train", min_n=5):
    """-> ({name: unit float32 vector}, info). acts: {'suf': [N, d], 'last': ..., 'turn': ..., 'ok': [N]}."""
    Y = contrast_labels(meta, C)
    use = np.asarray(acts["ok"], bool)
    if split != "all":
        use = use & np.array([m["split"] == split for m in meta])
    ind = [c for c in C if c.startswith("i_")]
    out, info = {}, {"split": split, "pool_members": ind, "contrasts": {}}
    for pool in POOLS:
        X = np.asarray(acts[pool], np.float64)
        raw = {}
        for c in C:
            p, n = use & (Y[c] == 1), use & (Y[c] == 0)
            if p.sum() < min_n or n.sum() < min_n:
                raise ValueError(f"contrast {c}: {int(p.sum())} hack rows and {int(n.sum())} honest rows in split {split}")
            raw[c] = X[p].mean(0) - X[n].mean(0)
            if pool == "suf":
                info["contrasts"][c] = dict(n_hack=int(p.sum()), n_honest=int(n.sum()), norm=float(np.linalg.norm(raw[c])),
                                            dataset=C[c]["dataset"], span=C[c]["span"], hack_cells=C[c]["pos"], honest_cells=C[c]["neg"])
        pre = "" if pool == "suf" else f"{pool}:"
        out[f"{pre}gf_pool"] = unit(np.mean([raw[c] for c in ind], 0))
        if pool == "suf":
            out["gf_pool18"] = unit(np.mean([raw[c] for c in C], 0))
            out["gf_pool_unitmean"] = unit(np.mean([unit(raw[c]) for c in ind], 0))
            for c in C:
                out[f"c:{c}"] = unit(raw[c])
    info["cos_with_gf_pool"] = {k: float(out[k] @ out["gf_pool"]) for k in out if k != "gf_pool"}
    return {k: v.astype(np.float32) for k, v in out.items()}, info


def dirs(a):
    out = Path(a.out)
    z = np.load(out / "gf27_acts.npz")
    meta = json.load(open(out / "gf27_meta.json"))
    C = json.load(open(a.contrasts))
    D, info = build_dirs(z, meta, C, a.split)
    save_atomic(out / f"dirs_{a.layer}.npz", **D)
    json.dump(info, open(out / f"dirs_{a.layer}.json", "w"), indent=1)
    print(f"[gf27] {int(np.asarray(z['ok']).sum())}/{len(meta)} spans harvested; split {a.split}; gf_pool = mean of {len(info['pool_members'])} contrasts")
    for c, x in info["contrasts"].items():
        print(f"  c:{c:13s} hack {x['n_hack']:4d}  honest {x['n_honest']:4d}  |diff| {x['norm']:7.2f}  cos with gf_pool {info['cos_with_gf_pool']['c:' + c]:+.2f}")
    for k in ("gf_pool18", "gf_pool_unitmean", "last:gf_pool", "turn:gf_pool"):
        print(f"  {k:17s} cos with gf_pool {info['cos_with_gf_pool'][k]:+.2f}")
    print(f"[gf27] {len(D)} directions -> {out / f'dirs_{a.layer}.npz'}", flush=True)


# ------------------------------------------------------------------ laptop: detection on our on-policy activations

def detect_table(meta, X, D, envs=None, n_boot=1000):
    """-> {direction: {env: [auc, lo, hi, n prompts], 'mean': mean over envs}}; matched AUROC within prompt."""
    from . import onp_loeo as LO
    envs = envs or sorted({m["beh"] for m in meta if LO.groups(meta, [m["beh"]])})
    res = {}
    for k, d in D.items():
        s = np.asarray(X @ np.asarray(d, np.float32), np.float64)
        r = {e: list(LO.auc(meta, s, e, n_boot=n_boot)) for e in envs}
        r["mean"] = float(np.mean([r[e][0] for e in envs]))
        res[k] = r
    return res, envs


def detect(a):
    out = Path(a.out)
    meta = json.load(open(a.meta))
    X = np.load(a.feats, mmap_mode="r")
    Z = np.load(out / f"dirs_{a.layer}.npz")
    D = {k: Z[k] for k in Z.files if ":" not in k or k.startswith("c:")}
    res, envs = detect_table(meta, np.asarray(X, np.float32), D)
    print(f"matched AUROC (hack against honest completions of one prompt), {len(envs)} environments; none of this data "
          f"was used to build these directions")
    print(f"{'direction':20s} {'mean':>5s}  " + "  ".join(f"{e[:12]:>12s}" for e in envs))
    for k, r in res.items():
        print(f"{k:20s} {r['mean']:5.2f}  " + "  ".join(f"{r[e][0]:12.2f}" for e in envs))
    r = res["gf_pool"]
    print("gf_pool with 95% intervals over prompts: " + "; ".join(f"{e} {r[e][0]:.2f} ({r[e][1]:.2f}, {r[e][2]:.2f}; {r[e][3]} prompts)" for e in envs))
    json.dump(dict(feats=str(a.feats), meta=str(a.meta), envs=envs, table=res), open(out / a.detect_out, "w"), indent=1)
    print(f"[gf27] -> {out / a.detect_out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("check", "harvest", "merge", "dirs", "detect"))
    add_model_args(ap)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--contrasts", default=str(CONTRASTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke test)")
    ap.add_argument("--budget", type=int, default=24000, help="padded tokens per forward pass")
    ap.add_argument("--shard", type=int, default=256, help="records per resumable shard")
    ap.add_argument("--tol", type=float, default=2e-2, help="batched against single, relative (bf16 noise is ~1e-2)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--split", default="train", choices=("train", "validation", "all"), help="their split used for dirs")
    ap.add_argument("--feats", default=str(ONP / "means_36_all.npy"))
    ap.add_argument("--meta", default=str(ONP / "meta_all.json"))
    ap.add_argument("--detect-out", default="detect_onp.json")
    a = ap.parse_args()
    {"check": check, "harvest": harvest, "merge": merge, "dirs": dirs, "detect": detect}[a.stage](a)

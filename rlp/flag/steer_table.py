"""Steering table for the per-request steering server (Interface 6 of scripts/flag/night/README_agents.md).

    .venv/bin/python -m rlp.flag.steer_table --mode v1 --out cache/flag/night/steer_table.pt \
        --noise cache/_staging_night_2026-11-24/noisenull0_pos.pt [--lin PATH] [--gf cache/flag/gf27/dirs_36.npz]
    .venv/bin/python -m rlp.flag.steer_table --mode pairing --out cache/flag/night/steer_table_pair.pt --noise PATH
    .venv/bin/python -m rlp.flag.steer_table --mode rollout --out cache/flag/night/steer_table_roll.pt
    .venv/bin/python -m rlp.flag.steer_table --show cache/flag/night/steer_table.pt

The table is a torch file {"add": float32 [K, d], "proj": float32 [K, d], "layer": 36, "names": [K], "kinds": [K],
"norm": 20.0}. Row 0 is "base" (all zero). A request picks row k with extra_body={"vllm_xargs": {"vid": k}}.
Condition names, for a direction DIR with unit vector u:

    DIR_add    add  +20 u            DIR_sub    add  -20 u            DIR_proj    project u out of the stream

Every direction is oriented so that +u is its "more hacking" sign (for random directions the sign is arbitrary).

Mode v1 (the mixed run of the night), in this row order:
    base
    iso0..7_{add,sub}        8 isotropic random directions (seeded Gaussian)
    sig0..7_{add,sub}        8 Sigma-shaped random directions: Sigma times a seeded Gaussian vector (cov_36.pt)
    iso0..3_proj, sig0..3_proj
    candidates, each _sub and _proj: probe, sprobe (Sigma.probe = write_cands_36.npz Stok_probe), fitv (v of
        fitw_w2.pt), gf (gf_pool of cache/flag/gf27/dirs_36.npz), dom, lda (dirs_cf_36.npz), noise (--noise: the
        noise-only vector, a {"v", "layer"} torch file; default cache/flag/onp/noise_vec.pt from `rlp.flag.linfit
        noise`, or the staging noisenull0_pos.pt), lin (--lin: the linear-fit vector, e.g. cache/flag/onp/linfit_vec.pt;
        pass it only if `rlp.flag.linfit check` printed LINFIT_CHECK_PASS)
    sprobe_add, fitv_add, gf_add        the add sign is run for the three named directions only
    noise_add, probe_add                rows for the pairing run with 24 November, kind "pair": they are NOT part of
                                        the main set; they are here so that one server start serves the whole of V1
                                        (omit with --no-pairing-rows)
Mode pairing: base, noise_{add,sub}, probe_{add,sub}.   Mode rollout: base, sprobe_sub, iso0_sub.

A candidate whose input file is missing is skipped with a printed note (--require NAME ... turns that into an error).
Self-test rows are appended in every mode (omit with --no-canary). They are never experimental conditions:
    base_dup          a second all-zero row
    canary_a/b/c      add 400 times a seeded random unit vector (the stream norm at block 36 is about 83)
    canary_a100       add 100 times canary_a's direction
    canary_proj       project canary_a's direction out
    canary_addproj    add 100 times canary_a's direction, then project it out: must behave as canary_proj, not as
                      canary_a100 (100, not 400, keeps the bf16 rounding of the add small against the stream)
    canary_proj_mu    project the mean activation direction out (reported only)

All experimental add rows have norm 20 (to 1e-4) and all projection rows are unit to 1e-6 after the float32 cast.
Seeds are fixed (--seed, default 20261125) and each random direction has its own stream, so `iso3` is the same vector
in every mode. OUT.json records the row list, the named sets (main, random, candidates, named, proj, fitting, pairing,
story, rollout), every cosine between directions, the inputs with their sha256, and what was skipped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

NORM, LAYER, SEED = 20.0, 36, 20261125
N_ISO, N_SIG, N_RPROJ = 8, 8, 4
CANDS = ["probe", "sprobe", "fitv", "gf", "dom", "lda", "noise", "lin"]
NAMED = ["sprobe", "fitv", "gf"]
ISO = [f"iso{j}" for j in range(N_ISO)]
SIG = [f"sig{j}" for j in range(N_SIG)]
RANDOM = ISO + SIG
RPROJ = ISO[:N_RPROJ] + SIG[:N_RPROJ]
CANARY_NORM = 400.0


def unit(v, what):
    v = np.asarray(v, np.float64).ravel()
    n = np.linalg.norm(v)
    if not (np.isfinite(v).all() and n > 1e-12):
        raise ValueError(f"{what}: zero or non-finite direction")
    return v / n


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_vec(path, layer, key=None):
    """A direction from a .pt ({"v", "layer"} or {key: ...}), .npy or .npz (key needed) file; PATH:KEY also works."""
    path = str(path)
    if key is None and ":" in os.path.basename(path) and not os.path.exists(path):
        path, key = path.rsplit(":", 1)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    if path.endswith(".npz"):
        z = np.load(path)
        if key is None:
            raise ValueError(f"{path}: give the array as PATH:KEY (has {list(z.files)[:8]})")
        return z[key].astype(np.float64)
    z = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(z, dict):
        if "layer" in z and int(z["layer"]) != layer:
            raise ValueError(f"{path}: layer {z['layer']} != {layer}")
        z = z[key or "v"]
    return torch.as_tensor(z).double().numpy()


def random_dirs(width, seed, cov=None):
    """iso0..7 and (if the covariance is given) sig0..7, each from its own seeded stream."""
    out = {}
    for j in range(N_ISO):
        g = np.random.default_rng([seed, 0, j]).standard_normal(width)
        out[f"iso{j}"] = (unit(g, f"iso{j}"), f"gauss[{seed},0,{j}]")
    if cov is not None:
        for j in range(N_SIG):
            g = np.random.default_rng([seed, 1, j]).standard_normal(width)
            out[f"sig{j}"] = (unit(cov @ g, f"sig{j}"), f"cov @ gauss[{seed},1,{j}]")
    return out


def candidate_dirs(a, layer):
    """name -> (unit vector, source) for the candidates whose inputs exist; also the list of skipped ones."""
    onp = Path(a.onp_dir)
    spec = {
        "probe": (onp / f"dirs_cf_{layer}.npz", "probe"),
        "sprobe": (onp / f"write_cands_{layer}.npz", "Stok_probe"),
        "fitv": (onp / a.fitw, "v"),
        "gf": (Path(a.gf) if a.gf else None, "gf_pool"),
        "dom": (onp / f"dirs_cf_{layer}.npz", "dom"),
        "lda": (onp / f"dirs_cf_{layer}.npz", "lda"),
        "noise": (Path(a.noise) if a.noise else onp / "noise_vec.pt", None),      # rlp.flag.linfit noise writes this
        "lin": (Path(a.lin) if a.lin else None, None),
    }
    out, skipped, inputs = {}, [], {}
    for name in CANDS:
        path, key = spec[name]
        if path is None or not Path(str(path).rsplit(":", 1)[0] if not Path(path).exists() else path).exists():
            why = "no path given" if path is None else f"{path} not found"
            skipped.append((name, why))
            continue
        try:
            v = load_vec(path, layer, key)
        except (KeyError, ValueError) as e:
            skipped.append((name, f"{path}: {e}"))
            continue
        out[name] = (unit(v, name), f"{path}" + (f":{key}" if key else ""))
        real = str(path) if Path(path).exists() else str(path).rsplit(":", 1)[0]
        inputs[name] = dict(path=str(path), sha256_16=_sha(real))
    return out, skipped, inputs


def build(a):
    layer, r, onp = a.layer, a.norm, Path(a.onp_dir)
    covf = onp / f"cov_{layer}.pt"
    cov = mu = None
    if covf.exists():
        z = torch.load(covf, map_location="cpu", weights_only=False)
        cov, mu = z["cov"].double().numpy(), z["mu"].double().numpy()
    cands, skipped, inputs = candidate_dirs(a, layer)
    width = a.width or (len(next(iter(cands.values()))[0]) if cands else (cov.shape[0] if cov is not None else 0))
    if not width:
        raise SystemExit("cannot infer the width: no candidate input and no covariance (pass --width)")
    if cov is not None and cov.shape[0] != width:
        raise SystemExit(f"covariance width {cov.shape[0]} != {width}")
    rnd = random_dirs(width, a.seed, cov)
    if cov is None:
        skipped.append(("sig0..7", f"{covf} not found"))
        print(f"NOTE: {covf} not found: the Sigma-shaped random directions sig0..{N_SIG - 1} are skipped")
    else:
        inputs["cov"] = dict(path=str(covf), sha256_16=_sha(covf))
    for name, why in skipped:
        if name in CANDS:
            print(f"NOTE: candidate '{name}' skipped ({why})")
    missing = [n for n in a.require if n not in cands and n not in rnd]
    if missing:
        raise SystemExit(f"required directions missing: {missing}")
    dirs = {**rnd, **cands}
    for n, (u, _) in dirs.items():
        if len(u) != width:
            raise SystemExit(f"direction {n} has width {len(u)}, expected {width}")

    names, kinds, srcs, add, proj = [], [], [], [], []

    def row(name, kind, src="", a_vec=None, p_vec=None):
        if name in names:
            raise SystemExit(f"duplicate row {name}")
        names.append(name), kinds.append(kind), srcs.append(src)
        add.append(np.zeros(width) if a_vec is None else a_vec)
        proj.append(np.zeros(width) if p_vec is None else p_vec)

    def cond(d, sfx, kind):
        if d not in dirs:
            return
        u, src = dirs[d]
        if sfx == "add":
            row(f"{d}_add", kind, src, a_vec=r * u)
        elif sfx == "sub":
            row(f"{d}_sub", kind, src, a_vec=-r * u)
        else:
            row(f"{d}_proj", kind, src, p_vec=u)

    row("base", "base")
    if a.mode == "v1":
        for d in RANDOM:
            cond(d, "add", "rand"), cond(d, "sub", "rand")
        for d in RPROJ:
            cond(d, "proj", "rand_proj")
        for d in CANDS:
            cond(d, "sub", "cand"), cond(d, "proj", "cand_proj")
        for d in NAMED:
            cond(d, "add", "cand_add")
        if not a.no_pairing_rows:
            for d in ("noise", "probe"):
                cond(d, "add", "pair")
    elif a.mode == "pairing":
        for d in ("noise", "probe"):
            cond(d, "add", "pair"), cond(d, "sub", "pair")
    elif a.mode == "rollout":
        cond("sprobe", "sub", "cand"), cond("iso0", "sub", "rand")
    n_exp = len(names)
    if not a.no_canary:
        cg = [unit(np.random.default_rng([a.seed, 2, j]).standard_normal(width), f"canary{j}") for j in range(3)]
        row("base_dup", "canary_base")
        for c, u in zip("abc", cg):
            row(f"canary_{c}", "canary", f"gauss[{a.seed},2,{'abc'.index(c)}]", a_vec=a.canary_norm * u)
        row("canary_a100", "canary", "canary_a", a_vec=a.canary_proj_norm * cg[0])
        row("canary_proj", "canary_proj", "canary_a", p_vec=cg[0])
        row("canary_addproj", "canary_addproj", "canary_a", a_vec=a.canary_proj_norm * cg[0], p_vec=cg[0])
        if mu is not None:
            row("canary_proj_mu", "canary_proj_mu", f"{covf}:mu", p_vec=unit(mu, "mu"))

    A = torch.tensor(np.stack(add), dtype=torch.float32)
    P = torch.tensor(np.stack(proj), dtype=torch.float32)
    pn = P.double().norm(dim=-1, keepdim=True)          # unit again after the float32 cast (server checks to 1e-5)
    P = torch.where(pn > 0, (P.double() / pn.clamp_min(1e-30)).float(), P)
    check(dict(add=A, proj=P, names=names, kinds=kinds, layer=layer, norm=float(r)))

    # record: cosines between all distinct directions used
    used = [d for d in dirs if any(n.rsplit("_", 1)[0] == d for n in names[:n_exp])]
    U = np.stack([dirs[d][0] for d in used]) if used else np.zeros((0, width))
    C = U @ U.T
    cos = {d: {e: round(float(C[i, j]), 4) for j, e in enumerate(used) if e != d and (e in cands or d in cands)}
           for i, d in enumerate(used) if d in cands}
    if len(used) > 1:
        off = np.abs(C - np.eye(len(used)))
        i, j = np.unravel_index(off.argmax(), off.shape)
        print(f"largest |cos| between two directions: {off.max():.3f} ({used[i]}, {used[j]})")
        if off.max() > 0.9999:
            raise SystemExit(f"{used[i]} and {used[j]} are the same direction: check the inputs")
    ri = [k for k, d in enumerate(used) if d in rnd]
    if len(ri) > 1:
        R = np.abs(C[np.ix_(ri, ri)] - np.eye(len(ri)))
        print(f"largest |cos| between two random directions: {R.max():.3f}")
    for ref in ("dom", "probe"):
        if "gf" in cands and ref in cands:
            c = float(cands["gf"][0] @ cands[ref][0])
            print(f"cos(gf, {ref}) = {c:+.3f}" + ("   WARNING: negative, check the sign of gf_pool (gf_sub should "
                                                 "subtract the hack direction)" if c < 0 else ""))

    exp = names[:n_exp]
    has = lambda n: n in exp                                                         # noqa: E731
    sets = {
        "random": [n for n, k in zip(exp, kinds) if k == "rand"],
        "random_iso": [n for n, k in zip(exp, kinds) if k == "rand" and n.startswith("iso")],
        "random_sig": [n for n, k in zip(exp, kinds) if k == "rand" and n.startswith("sig")],
        "proj": [n for n, k in zip(exp, kinds) if k in ("rand_proj", "cand_proj")],
        "candidates": [d for d in CANDS if has(f"{d}_sub")],
        "named": [d for d in NAMED if has(f"{d}_sub")],
        "main": [n for n, k in zip(exp, kinds) if k in ("base", "rand", "rand_proj", "cand", "cand_proj", "cand_add")],
        "fitting": [n for n, k in zip(exp, kinds) if k == "base" or (k in ("rand", "cand") and n.endswith("_sub"))],
        "pairing": [n for n in ("noise_add", "noise_sub", "probe_add", "probe_sub") if has(n)],
        "story": [n for n in ("base", "fitv_add", "fitv_sub") if has(n)],
        "rollout": [n for n in ("base", "sprobe_sub", "iso0_sub") if has(n)],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(dict(add=A, proj=P, layer=layer, names=names, kinds=kinds, norm=float(r)), tmp)
    os.replace(tmp, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = dict(sha256=sha, mode=a.mode, layer=layer, norm=r, width=width, K=len(names), seed=a.seed,
                canary_norm=None if a.no_canary else a.canary_norm, inputs=inputs,
                skipped=[dict(name=n, why=w) for n, w in skipped], sets=sets, cos=cos,
                rows=[dict(vid=k, name=n, kind=kd, source=s, add_norm=round(float(A[k].double().norm()), 5),
                           proj=bool(P[k].any())) for k, (n, kd, s) in enumerate(zip(names, kinds, srcs))])
    with open(str(out) + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}: {len(names)} rows ({n_exp} experimental), layer {layer}, norm {r:g}, sha256 {sha[:12]}; names in "
          f"{out}.json")
    print("experimental rows:", " ".join(exp))
    if a.legacy_dir:
        D = Path(a.legacy_dir)
        D.mkdir(parents=True, exist_ok=True)
        n = 0
        for k, (nm, kd) in enumerate(zip(names, kinds)):
            if kd == "base" or (kd in ("rand", "cand", "cand_add", "pair") and A[k].any()):
                torch.save(dict(v=A[k].clone(), layer=layer), D / f"{nm}.pt")
                n += 1
        print(f"fallback: {n} single-vector files (add rows and base; projection cannot be run that way) in {D}")
    return meta


def check(t):
    """The rules the server enforces, plus the norm rule of Interface 6. Raises AssertionError."""
    A, P, names, kinds = t["add"], t["proj"], list(t["names"]), list(t.get("kinds") or [])
    K = len(names)
    assert A.dtype == torch.float32 and P.dtype == torch.float32 and A.shape == P.shape and A.shape[0] == K
    assert len(set(names)) == K and names[0] == "base", "names must be unique and row 0 must be 'base'"
    assert not bool(A[0].any()) and not bool(P[0].any()), "row 0 must be the identity"
    assert bool(torch.isfinite(A).all()) and bool(torch.isfinite(P).all())
    pn, an = P.double().norm(dim=-1), A.double().norm(dim=-1)
    assert bool((((pn - 1).abs() < 1e-6) | (pn == 0)).all()), "projection rows must be unit or zero"
    for k, n in enumerate(names):
        kd = kinds[k] if kinds else ""
        if kd.startswith("canary"):
            continue
        if n.endswith(("_add", "_sub")):
            assert abs(float(an[k]) - t["norm"]) < 1e-4 * t["norm"] and pn[k] == 0, f"{n}: add norm {float(an[k])}"
        elif n.endswith("_proj"):
            assert an[k] == 0 and pn[k] > 0, f"{n}: must only project"
        else:
            assert an[k] == 0 and pn[k] == 0, f"{n}: must be an identity row"
    for k, n in enumerate(names):                         # DIR_add = -DIR_sub, DIR_proj parallel to both
        if n.endswith("_add") and n[:-4] + "_sub" in names:
            assert torch.equal(A[k], -A[names.index(n[:-4] + "_sub")]), f"{n} is not minus its _sub row"
        if n.endswith("_proj"):
            for s in ("_sub", "_add"):
                o = n[:-5] + s
                if o in names:
                    c = float(P[k].double() @ A[names.index(o)].double()) / t["norm"]
                    assert abs(abs(c) - 1) < 1e-5, f"{n} is not parallel to {o}"
    return True


def load(path):
    """(table dict, meta dict). The .json must have been written for exactly this file."""
    path = Path(path)
    t = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.load(open(str(path) + ".json"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta["sha256"] != sha:
        raise SystemExit(f"{path}.json was written for another table (sha256 differs)")
    return t, meta


def show(path):
    t, meta = load(path)
    check(t)
    print(f"{path}: {meta['K']} rows, layer {meta['layer']}, norm {meta['norm']}, mode {meta['mode']}, sha256 {meta['sha256'][:12]}")
    for r in meta["rows"]:
        print(f"  {r['vid']:3d} {r['name']:16s} {r['kind']:14s} add {r['add_norm']:8.3f} proj {int(r['proj'])}  {r['source']}")
    for k, v in meta["sets"].items():
        print(f"set {k} ({len(v)}): {' '.join(v)}")
    for s in meta["skipped"]:
        print(f"skipped {s['name']}: {s['why']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v1", "pairing", "rollout"], default="v1")
    ap.add_argument("--out", default="cache/flag/night/steer_table.pt")
    ap.add_argument("--show", default="", help="print an existing table and exit")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--norm", type=float, default=NORM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--onp-dir", default="cache/flag/onp")
    ap.add_argument("--fitw", default="fitw_w2.pt", help="file in --onp-dir whose 'v' is the fitted vector")
    ap.add_argument("--gf", default="cache/flag/gf27/dirs_36.npz", help="npz with gf_pool (Interface 5)")
    ap.add_argument("--noise", default="", help="noise-only vector: {'v', 'layer'} torch file (the +v file), .npy or NPZ:KEY; "
                                                "default <onp-dir>/noise_vec.pt (written by `rlp.flag.linfit noise`)")
    ap.add_argument("--lin", default="", help="linear-fit vector, same formats; omit if it failed its check")
    ap.add_argument("--require", nargs="*", default=[], help="directions that must be present (else exit 1)")
    ap.add_argument("--no-pairing-rows", action="store_true")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--canary-norm", type=float, default=CANARY_NORM)
    ap.add_argument("--canary-proj-norm", type=float, default=100.0)
    ap.add_argument("--legacy-dir", default="", help="also write {'v','layer'} files for the single-vector fallback")
    a = ap.parse_args()
    show(a.show) if a.show else build(a)

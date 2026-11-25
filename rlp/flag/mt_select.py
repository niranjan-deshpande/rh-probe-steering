"""Detection on multi-turn rollouts and the choice of fitting environments on validation families (night of 2026-11-25).

Laptop only (numpy and scikit-learn; torch is imported lazily and only to read `fitw_w2.pt` and `cov_36.pt`).

Inputs
  Interface 4  single-turn completion means `means_36_all.npy` + `meta_all.json` (rows carry `beh`, `group`, `y`), and
               optionally a second file in the same format (`--st2-feats`, `--st2-meta`), concatenated in memory only.
  Interface 3  one or more run directories with `acts_36.npz` (`ep_mean`, `step_mean`) and `acts_36.json`.
  Interface 5  `gf27_acts.npz` + `gf27_meta.json` (per-record activations and contrast labels) and `dirs_36.npz`
               (`gf_pool`, `c:<contrast>`).

Part 1, detection. For every direction (logistic probe fitted on all single-turn environments, the cached probe, the
fitted w, `gf_pool`, mean difference, whitened mean difference) and every multi-turn family: matched AUROC between
hack and honest episodes of one task, with a bootstrap interval over tasks; and a step-level AUROC of the steps that
contain the hack against all steps of honest episodes of that family (bootstrap over tasks). The head-to-head is the
paired difference probe minus `gf_pool`, on multi-turn families (never in the probe's fit) and on single-turn
environments with the probe refitted without the environment it is scored on.

Part 2, rotation. For each ordered choice of a test family and a validation family among the usable families, the
other usable families join the fitting pool. Pool versions: single-turn only (ST), single-turn plus multi-turn
(ST+MT), multi-turn only (MT). Candidates are all non-empty subsets of the pool's units; the subset with the best mean
matched AUROC on the validation family is chosen (ties go to the smaller, earlier subset) and its test AUROC is
reported next to: the whole pool, the expectation over all subsets of the chosen size, and the best subset on test (a
bound). Four cells: logistic probe on our data, mean difference on our data, mean difference on Goodfire's contrasts,
logistic probe on Goodfire's contrast pairs. The validation and test families never enter a fit, and the choice reads
the validation family only.

Units. A unit is one single-turn environment, one multi-turn family or one Goodfire contrast. If a pool version would
have more than `--max-subsets` candidate subsets, its single-turn environments are grouped by hack category (and, if
that is still too many, into one unit); the grouping used is printed and saved per pool version.

Part 3, fallbacks, chosen from the number of usable families: 3 or more -> rotation; 2 -> no validation, fit on one
plus single-turn and test on the other, both directions, whole pool only; 1 -> test only; 0 -> single-turn parts only.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 .venv/bin/python -m rlp.flag.mt_select --runs full ib_full
    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.mt_select                # single-turn parts only
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ONP, GFD, MTD = ROOT / "cache/flag/onp", ROOT / "cache/flag/gf27", ROOT / "cache/flag/mt"
# same table as rlp.flag.onp_more.CATS (copied so that this module does not import torch)
CATS = {"length_pad": "Game", "hardcode": "Game", "fake_cite": "Game", "quiz_key": "Peek", "leak": "Peek",
        "save_tamper": "Tamper", "fake_pass": "Misreport", "ckpt_swap": "Tamper", "thresh_tamper": "Tamper",
        "log_scrub": "Tamper", "ref_copy": "Plagiarise"}
REALISTIC = ("ib_lcb", "mini_repo")
STATS = ("chosen", "whole", "rand", "oracle")


# ------------------------------------------------------------------ scores

def pair_auc(sh, sn):
    """P(hack score > honest score), ties counted as one half."""
    d = sh[:, None] - sn[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def auroc(pos, neg):
    """Plain AUROC by sorting (ties one half)."""
    if not len(pos) or not len(neg):
        return float("nan")
    neg = np.sort(neg)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def ci(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_idx(n, n_boot, rng):
    return rng.integers(0, n, (n_boot, n))


# ------------------------------------------------------------------ data

class Env:
    """One environment, family or contrast: the rows of every group (prompt or task) that has both outcomes.

    R  rows centred within their group    y  labels    pairs  per group (hack rows, honest rows) into R
    D  per group, mean hack minus mean honest    keys  group names    dvec  optional override of the mean difference
    """

    def __init__(self, name, kind, X, y, grp, cat=None, dvec=None):
        self.name, self.kind, self.cat, self.dvec = name, kind, cat or name, dvec
        self.uid = f"{kind}:{name}"
        y, grp = np.asarray(y), np.asarray(grp)
        self.n_rows, self.n_hack, self.n_groups = len(y), int((y == 1).sum()), len(set(grp.tolist()))
        order = {}
        for i, g in enumerate(grp.tolist()):
            order.setdefault(g, []).append(i)
        R, ys, self.pairs, self.keys, D, k = [], [], [], [], [], 0
        for g, idx in order.items():
            idx = np.array(idx)
            h, n = idx[y[idx] == 1], idx[y[idx] == 0]
            if not len(h) or not len(n):
                continue
            A = np.asarray(X[np.concatenate([h, n])], np.float32)
            D.append(A[:len(h)].mean(0) - A[len(h):].mean(0))
            R.append(A - A.mean(0))
            ys += [1] * len(h) + [0] * len(n)
            self.pairs.append((np.arange(k, k + len(h)), np.arange(k + len(h), k + len(idx))))
            self.keys.append(g)
            k += len(idx)
        d = X.shape[1]
        self.R = np.concatenate(R) if R else np.zeros((0, d), np.float32)
        self.D = np.stack(D) if D else np.zeros((0, d), np.float32)
        self.y = np.array(ys, int)

    @property
    def n_tasks(self):
        return len(self.pairs)

    def mean_diff(self):
        return self.dvec if self.dvec is not None else self.D.mean(0)

    def per_task(self, w):
        """Matched AUROC of direction w, one number per both-outcome group."""
        s = np.asarray(self.R @ np.asarray(w, np.float32), np.float64)
        return np.array([pair_auc(s[h], s[n]) for h, n in self.pairs])


def load_st(feats, meta, cats=None, taken=()):
    """Single-turn file -> [Env], one per environment with at least one both-outcome prompt. Rows with y None are dropped."""
    meta = json.load(open(meta))
    X = np.load(feats, mmap_mode="r")
    assert len(meta) == X.shape[0], f"{feats}: {X.shape[0]} rows but {len(meta)} meta rows"
    cats = dict(CATS) | (cats or {})
    out = []
    for b in sorted({m["beh"] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m["beh"] == b and m["y"] is not None]
        if not idx:
            continue
        name = b if b not in taken else b + "@2"
        e = Env(name, "st", np.asarray(X[idx]), [meta[i]["y"] for i in idx], [meta[i]["group"] for i in idx], cats.get(b, b))
        if e.n_tasks:
            out.append(e)
    return out


def task_key(e, match):
    k = f"{e['family']}|{e['task']}"
    if match in ("cfg", "variant"):
        k += "|" + json.dumps(e.get("cfg"), sort_keys=True)
    if match == "variant":
        k += "|" + str(e.get("variant"))
    return k


def load_mt(run_dirs, match="task"):
    """Rollout activations of several runs -> ({family: Env}, {family: steps}, {family: counts}).

    Episodes with y None are excluded, and so are their steps. steps = dict(S [n, d], pos [n] bool, task [n] int): the
    positives are the steps flagged `is_hack_step` in hack episodes, the negatives all steps of honest episodes."""
    eps, EM, SM, st_ep, st_hack, off = [], [], [], [], [], 0
    for rd in run_dirs:
        rd = Path(rd)
        z, j = np.load(rd / "acts_36.npz"), json.load(open(rd / "acts_36.json"))
        assert len(j["episodes"]) == z["ep_mean"].shape[0] and len(j["steps"]) == z["step_mean"].shape[0], f"{rd}: sizes disagree"
        eps += [dict(e, run=rd.name) for e in j["episodes"]]
        EM.append(z["ep_mean"].astype(np.float32))
        SM.append(z["step_mean"].astype(np.float32))
        st_ep += [s["ep"] + off for s in j["steps"]]
        st_hack += [bool(s["is_hack_step"]) for s in j["steps"]]
        off += len(j["episodes"])
    if not eps:
        return {}, {}, {}
    EM, SM, st_ep, st_hack = np.concatenate(EM), np.concatenate(SM), np.array(st_ep, int), np.array(st_hack, bool)
    fams, steps, counts = {}, {}, {}
    for f in sorted({e["family"] for e in eps}):
        allf = [i for i, e in enumerate(eps) if e["family"] == f]
        idx = [i for i in allf if eps[i]["y"] is not None]
        counts[f] = dict(episodes=len(allf), excluded_y_none=len(allf) - len(idx), hack=sum(eps[i]["y"] == 1 for i in idx),
                         honest=sum(eps[i]["y"] == 0 for i in idx), tasks=len({eps[i]["task"] for i in idx}))
        for m in ("task", "cfg", "variant"):
            by = {}
            for i in idx:
                by.setdefault(task_key(eps[i], m), set()).add(eps[i]["y"])
            counts[f][f"both_{m}"] = sum(len(v) == 2 for v in by.values())
        if not idx:
            continue
        fams[f] = Env(f, "mt", EM[idx], [eps[i]["y"] for i in idx], [task_key(eps[i], match) for i in idx])
        yv = {i: eps[i]["y"] for i in idx}
        tid = {k: n for n, k in enumerate(sorted({task_key(eps[i], match) for i in idx}))}
        keep = [s for s in range(len(st_ep)) if st_ep[s] in yv and (yv[st_ep[s]] == 0 or st_hack[s])]
        steps[f] = dict(S=SM[keep], pos=np.array([yv[st_ep[s]] == 1 for s in keep], bool),
                        task=np.array([tid[task_key(eps[st_ep[s]], match)] for s in keep], int))
        counts[f].update(hack_steps=int(steps[f]["pos"].sum()), honest_steps=int((~steps[f]["pos"]).sum()))
    return fams, steps, counts


def load_gf(gf_dir, readout="suf", units="individual", min_n=5):
    """Goodfire's data -> (gf_pool or None, [Env of kind 'gf'], note).

    One unit per released contrast. Its mean difference is mean(positives) - mean(negatives) over that contrast's
    records, so a subset of contrasts is re-averaged from the activations. Its probe rows are one copy of each record
    per contrast it takes part in, centred within (contrast, pair group). If the activations are missing, the unit
    vectors `c:<name>` of dirs_36.npz are used instead and the probe on Goodfire's data is unavailable."""
    gf_dir = Path(gf_dir)
    pool, cvec = None, {}
    if (gf_dir / "dirs_36.npz").exists():
        z = np.load(gf_dir / "dirs_36.npz")
        pool = z["gf_pool"].astype(np.float32) if "gf_pool" in z.files else None
        cvec = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("c:")}
    envs, note = [], ""
    if (gf_dir / "gf27_acts.npz").exists() and (gf_dir / "gf27_meta.json").exists():
        z, recs = np.load(gf_dir / "gf27_acts.npz"), json.load(open(gf_dir / "gf27_meta.json"))
        if isinstance(recs, dict):
            recs = recs.get("records") or recs.get("recs")
        X = z[readout]
        ok = z["ok"].astype(bool) if "ok" in z.files else np.ones(len(recs), bool)
        assert len(recs) == X.shape[0], "gf27_acts and gf27_meta disagree"
        names = list(cvec) or sorted({c for r in recs for c in r.get("labels", {})})
        names = _gf_names(names, units)
        for c in names:
            idx = [i for i, r in enumerate(recs) if ok[i] and r.get("labels", {}).get(c) in (0, 1)]
            y = np.array([recs[i]["labels"][c] for i in idx])
            if (y == 1).sum() < min_n or (y == 0).sum() < min_n:
                continue
            Xi = np.asarray(X[idx], np.float32)
            g = [str(recs[i].get("group", recs[i].get("group_id", "all"))) for i in idx]
            envs.append(Env(c, "gf", Xi, y, g, dvec=Xi[y == 1].mean(0) - Xi[y == 0].mean(0)))
        note = f"{len(envs)} contrasts re-averaged from gf27_acts ({readout})"
        if pool is not None and envs:
            raw = np.mean([e.dvec for e in envs], 0)
            note += f"; cosine of their equal-weight mean with gf_pool {cos(raw, pool):.4f}"
    elif cvec:
        d = len(next(iter(cvec.values())))
        for c in _gf_names(list(cvec), units):
            e = Env(c, "gf", np.zeros((0, d), np.float32), [], [], dvec=cvec[c])
            envs.append(e)
        note = f"{len(envs)} contrasts from the unit vectors c:<name> (activations missing; no probe on Goodfire's data)"
    return pool, envs, note


def _gf_names(names, units):
    ind = [n for n in names if n.startswith("i_")]
    return ind if units == "individual" and ind else list(names)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ------------------------------------------------------------------ recipes

def fit_probe(envs, C=0.01, balance=True):
    """Logistic probe as rlp.flag.onp_loeo.fit: rows centred within group, standardised, C = 0.01, no intercept,
    every environment given the same total weight if balance. -> weight vector in the original space."""
    from sklearn.linear_model import LogisticRegression
    R = np.concatenate([e.R for e in envs])
    y = np.concatenate([e.y for e in envs])
    sd = R.std(0) + 1e-6
    sw = None
    if balance:
        sw = np.concatenate([np.full(len(e.y), len(y) / (len(envs) * len(e.y))) for e in envs])
    return LogisticRegression(C=C, max_iter=2000, fit_intercept=False).fit(R / sd, y, sample_weight=sw).coef_[0] / sd


def fit_mean(envs, **_):
    """Mean difference: equal-weight mean over environments (or contrasts) of the environment's mean difference."""
    return np.mean([e.mean_diff() for e in envs], 0)


RECIPES = {"probe": fit_probe, "mean": fit_mean}


class Scorer:
    """Fits a recipe on a set of environments and scores it on every family that is not in the set. Results are cached
    by (recipe, set), and `log` records every fit, so tests can check what went into each one."""

    def __init__(self, fams, C=0.01, balance=True, say=None):
        self.fams, self.kw, self.cache, self.log, self.t, self.say = fams, dict(C=C, balance=balance), {}, [], 0.0, say

    def score(self, recipe, envs):
        key = (recipe, frozenset(e.uid for e in envs))
        if key not in self.cache:
            t0 = time.time()
            w = RECIPES[recipe](envs, **self.kw)
            self.cache[key] = {f: e.per_task(w) for f, e in self.fams.items() if e.uid not in key[1]}
            self.log.append(key)
            self.t += time.time() - t0
            if self.say and len(self.log) % 500 == 0:
                self.say(f"    {len(self.log)} fits, {self.t:.0f} s", flush=True)
        return self.cache[key]


# ------------------------------------------------------------------ part 1: detection

def sigma_inv(cov_path, shrink=0.05):
    """As rlp.flag.onp.sigma_inv. Needs torch to read the file."""
    import torch
    cov = torch.load(cov_path, map_location="cpu")["cov"].double()
    ev, U = torch.linalg.eigh(cov)
    ev = ev.clamp_min(0) + shrink * ev.clamp_min(0).mean()
    return ((U / ev) @ U.T).float().numpy()


def fixed_dirs(a, gf_pool, log=print):
    """Directions that are not refitted here: the cached probe, the fitted w, gf_pool."""
    out = {}
    if a.dirs_cf and Path(a.dirs_cf).exists():
        out["probe_cached"] = np.load(a.dirs_cf)["probe"].astype(np.float32)
    if a.fitw and Path(a.fitw).exists():
        try:
            import torch
            out["fitted_w"] = torch.load(a.fitw, map_location="cpu")["w"].float().numpy()
        except Exception as e:                                   # torch missing or unreadable file
            log(f"  fitted w not loaded ({type(e).__name__}: {e})")
    if gf_pool is not None:
        out["gf_pool"] = gf_pool
    return out


def fitted_dirs(envs, Sinv, C, balance):
    out = {"probe": fit_probe(envs, C, balance), "mean_diff": fit_mean(envs)}
    if Sinv is not None:
        out["whitened"] = Sinv @ out["mean_diff"]
    return out


def step_auc(st, w, n_boot, rng):
    s = np.asarray(st["S"] @ np.asarray(w, np.float32), np.float64)
    pos, task = st["pos"], st["task"]
    a = auroc(s[pos], s[~pos])
    tasks = np.unique(task)
    by = [(s[(task == t) & pos], s[(task == t) & ~pos]) for t in tasks]
    bs = []
    for idx in boot_idx(len(tasks), n_boot, rng):
        v = auroc(np.concatenate([by[i][0] for i in idx]), np.concatenate([by[i][1] for i in idx]))
        if v == v:
            bs.append(v)
    return a, (ci(bs) if bs else [float("nan")] * 2)


def detect_mt(dirs, fams, steps, n_boot=1000, seed=0):
    """-> {direction: {family: dict(auc, ci, n_tasks, step_auc, step_ci, n_hack_steps, n_honest_steps)}}, paired probe - gf_pool."""
    res, per = {}, {}
    for dn, w in dirs.items():
        res[dn] = {}
        for f, e in fams.items():
            r = dict(auc=None, ci=None, n_tasks=e.n_tasks)
            if e.n_tasks:
                p = per[(dn, f)] = e.per_task(w)
                rng = np.random.default_rng(seed)
                r.update(auc=float(p.mean()), ci=ci(p[boot_idx(len(p), n_boot, rng)].mean(1)))
            st = steps.get(f)
            if st is not None and st["pos"].any() and (~st["pos"]).any():
                sa, sc = step_auc(st, w, n_boot, np.random.default_rng(seed))
                r.update(step_auc=sa, step_ci=sc, n_hack_steps=int(st["pos"].sum()), n_honest_steps=int((~st["pos"]).sum()))
            res[dn][f] = r
    paired = None
    if "probe" in dirs and "gf_pool" in dirs:
        paired = paired_diff({f: (per[("probe", f)], per[("gf_pool", f)]) for f, e in fams.items() if e.n_tasks}, n_boot, seed)
    return res, paired


def paired_diff(pairs, n_boot=1000, seed=0):
    """pairs: {name: (per-task a, per-task b)} -> per name and overall (equal weight per name) a - b with intervals.
    The bootstrap resamples tasks within each name, with the same resample for a and b."""
    rng = np.random.default_rng(seed)
    out, B = {"per": {}}, []
    for f, (a, b) in pairs.items():
        d = a - b
        bs = d[boot_idx(len(d), n_boot, rng)].mean(1)
        B.append(bs)
        out["per"][f] = dict(diff=float(d.mean()), ci=ci(bs), a=float(a.mean()), b=float(b.mean()), n=len(d))
    if B:
        out["overall"] = dict(diff=float(np.mean([v["diff"] for v in out["per"].values()])), ci=ci(np.mean(B, 0)), n_sets=len(B))
    return out


def detect_st(st, fixed, Sinv, C, balance, n_boot=1000, seed=0):
    """Single-turn environments. The fitted directions (probe, mean_diff, whitened) are refitted without the
    environment they are scored on; the fixed ones are used as they are (the cached probe and the fitted w were
    fitted on some of these environments, so their numbers here are partly in-sample)."""
    res, per = {}, {}
    for e in st:
        dirs = fitted_dirs([x for x in st if x is not e], Sinv, C, balance) if len(st) > 1 else {}
        for dn, w in (dirs | fixed).items():
            p = per[(dn, e.name)] = e.per_task(w)
            rng = np.random.default_rng(seed)
            res.setdefault(dn, {})[e.name] = dict(auc=float(p.mean()), ci=ci(p[boot_idx(len(p), n_boot, rng)].mean(1)), n_tasks=len(p))
    paired = None
    if ("probe", st[0].name) in per and "gf_pool" in fixed:
        paired = paired_diff({e.name: (per[("probe", e.name)], per[("gf_pool", e.name)]) for e in st}, n_boot, seed)
    return res, paired


# ------------------------------------------------------------------ part 2 and 3: rotation and fallbacks

def make_units(st, n_mt, max_subsets, mode="auto"):
    """Units for the single-turn part of a pool that also holds n_mt multi-turn families.
    -> ([(unit name, [Env])], description of the grouping)."""
    by_env = [(e.name, [e]) for e in st]
    cats = {}
    for e in st:
        cats.setdefault(e.cat, []).append(e)
    by_cat = [("cat:" + c, v) for c, v in cats.items()]
    one = [("single-turn (all)", list(st))] if st else []
    n = lambda u: 2 ** (len(u) + n_mt) - 1
    if mode == "env" or (mode == "auto" and n(by_env) <= max_subsets):
        return by_env, f"one unit per single-turn environment ({len(by_env)}), exact enumeration"
    if mode == "category" or (mode == "auto" and n(by_cat) <= max_subsets):
        return by_cat, ("single-turn environments grouped by hack category: " +
                        "; ".join(f"{c} = {', '.join(e.name for e in v)}" for c, v in cats.items()))
    return one, "all single-turn environments as one unit"


def all_subsets(units):
    """Non-empty subsets as index tuples, by size and then in order; the last one is the whole pool."""
    return [S for k in range(1, len(units) + 1) for S in itertools.combinations(range(len(units)), k)]


def split_cell(sc, recipe, units, t, v, select=True):
    """One recipe, one pool, one split. The choice reads the validation family v only.
    -> dict with scalars and `vec` = per-test-task vectors for chosen / whole / rand / oracle."""
    subs = all_subsets(units) if select else [tuple(range(len(units)))]
    flat = lambda S: [e for i in S for e in units[i][1]]
    T = np.stack([sc.score(recipe, flat(S))[t] for S in subs])
    out = dict(n_subsets=len(subs), whole=float(T[-1].mean()), vec={"whole": T[-1]})
    if not select:
        return out
    val = np.array([sc.score(recipe, flat(S))[v].mean() for S in subs])
    i = int(val.argmax())
    same = [j for j, S in enumerate(subs) if len(S) == len(subs[i])]
    j = int(T.mean(1).argmax())
    out["vec"].update(chosen=T[i], rand=T[same].mean(0), oracle=T[j])
    out.update(chosen=float(T[i].mean()), rand=float(T[same].mean()), oracle=float(T[j].mean()), size=len(subs[i]),
               val=float(val[i]), val_whole=float(val[-1]), chosen_units=[units[k][0] for k in subs[i]],
               oracle_units=[units[k][0] for k in subs[j]])
    return out


def rotation(st, fams, gf, a, log=print):
    """fams: usable multi-turn families only. -> dict(mode, cells, splits, summary, by_role, contrasts)."""
    names = list(fams)
    n = len(names)
    mode = "rotation" if n >= 3 else {2: "two", 1: "one", 0: "none"}[n]
    if mode == "none":
        return dict(mode=mode, splits=[], summary={}, cells={})
    select = mode == "rotation"
    if mode == "rotation":
        splits = [(t, v) for t in names for v in names if v != t]
    else:
        splits = [(t, None) for t in names]
    sc = Scorer(fams, a.C, not a.no_balance, log)
    n_mt = max(n - 2, 0) if select else (1 if mode == "two" else 0)
    ms = a.max_subsets if select else 10 ** 9
    u_st, how_st = make_units(st, 0, ms, a.st_units)
    u_sm, how_sm = make_units(st, n_mt, ms, a.st_units)
    u_gf = [(e.name, [e]) for e in gf]
    gf_probe = [(e.name, [e]) for e in gf if e.n_tasks]
    if select and 2 ** len(u_gf) - 1 > max(a.max_subsets, 4095):
        raise SystemExit(f"{len(u_gf)} Goodfire contrasts give too many subsets; use --gf-units individual")
    cells = {}                                                    # name -> (recipe, data, pool version, grouping)
    if st:
        cells["probe|ours|ST"] = ("probe", "ours", "ST", how_st)
        cells["mean|ours|ST"] = ("mean", "ours", "ST", how_st)
    if n_mt:
        if st:
            cells["probe|ours|ST+MT"] = ("probe", "ours", "ST+MT", how_sm)
            cells["mean|ours|ST+MT"] = ("mean", "ours", "ST+MT", how_sm)
        cells["probe|ours|MT"] = ("probe", "ours", "MT", "one unit per multi-turn family")
        cells["mean|ours|MT"] = ("mean", "ours", "MT", "one unit per multi-turn family")
    if u_gf:
        cells["mean|goodfire|GF"] = ("mean", "goodfire", "GF", f"one unit per released contrast ({len(u_gf)})")
    if gf_probe and not a.skip_gf_probe:
        cells["probe|goodfire|GF"] = ("probe", "goodfire", "GF", f"one unit per released contrast ({len(gf_probe)}), rows are contrast pairs")
    out_splits = []
    for si, (t, v) in enumerate(splits):
        pool = [f for f in names if f not in (t, v)]
        mt_units = [(f, [fams[f]]) for f in pool]
        row = dict(test=t, val=v, mt_pool=pool, cells={})
        for cn, (recipe, data, pv, _) in cells.items():
            units = {"ST": u_st, "ST+MT": u_sm + mt_units, "MT": mt_units, "GF": gf_probe if recipe == "probe" else u_gf}[pv]
            if not units:
                continue
            assert not any(e.uid in (fams[t].uid, fams[v].uid if v else None) for _, es in units for e in es), "held-out family in the pool"
            row["cells"][cn] = split_cell(sc, recipe, units, t, v, select)
        out_splits.append(row)
        log(f"  split {si + 1}/{len(splits)} test {t} val {v}: {len(sc.log)} fits so far, {sc.t:.0f} s", flush=True)
    res = dict(mode=mode, cells={k: dict(recipe=v[0], data=v[1], pool=v[2], units=v[3]) for k, v in cells.items()},
               n_fits=len(sc.log), fit_seconds=sc.t)
    res["summary"] = summarise(out_splits, fams, a.n_boot, a.seed)
    res["by_role"] = {}
    for f in names:
        roles = {"test": [s for s in out_splits if s["test"] == f], "val": [s for s in out_splits if s["val"] == f],
                 "pool": [s for s in out_splits if f in s["mt_pool"]]}
        res["by_role"][f] = {r: summarise(ss, fams, a.n_boot, a.seed, member=f if r == "pool" else None) for r, ss in roles.items() if ss}
    res["contrasts"] = pool_contrasts(out_splits, fams, a.n_boot, a.seed)
    res["splits"] = [dict(s, cells={k: {x: y for x, y in c.items() if x != "vec"} for k, c in s["cells"].items()}) for s in out_splits]
    res["_scorer"] = sc
    return res


def _boot_means(splits, fams, get, n_boot, seed):
    """Mean over splits of get(split) (a per-test-task vector), with a bootstrap over the tasks of each test family.
    One resample per family is shared by every split and every quantity, so differences are paired."""
    rng = np.random.default_rng(seed)
    idx = {f: boot_idx(e.n_tasks, n_boot, rng) for f, e in fams.items()}
    vs = [(s["test"], get(s)) for s in splits]
    vs = [(t, x) for t, x in vs if x is not None]
    if not vs:
        return None
    point = float(np.mean([x.mean() for _, x in vs]))
    bs = np.mean([x[idx[t]].mean(1) for t, x in vs], 0)
    return dict(mean=point, ci=ci(bs), n_splits=len(vs))


def summarise(splits, fams, n_boot=1000, seed=0, member=None):
    """Per cell: mean over splits of chosen / whole / rand / oracle with task-bootstrap intervals, chosen minus whole,
    mean chosen size and validation score. member: also the share of splits whose chosen subset holds that unit."""
    out = {}
    for cn in sorted({c for s in splits for c in s["cells"]}):
        ss = [s for s in splits if cn in s["cells"]]
        r = {"n_splits": len(ss)}
        for k in STATS:
            g = _boot_means(ss, fams, lambda s: s["cells"][cn]["vec"].get(k), n_boot, seed)
            if g:
                r[k] = g
        if "chosen" in r:
            r["chosen_minus_whole"] = _boot_means(ss, fams, lambda s: s["cells"][cn]["vec"]["chosen"] - s["cells"][cn]["vec"]["whole"], n_boot, seed)
            r["win"] = float(np.mean([s["cells"][cn]["chosen"] > s["cells"][cn]["whole"] for s in ss]))
            r["size"] = float(np.mean([s["cells"][cn]["size"] for s in ss]))
            r["val_of_chosen"] = float(np.mean([s["cells"][cn]["val"] for s in ss]))
            if member is not None:
                has = [member in s["cells"][cn]["chosen_units"] for s in ss if member in s["mt_pool"] and "MT" in cn]
                if has:
                    r["chosen_holds_member"] = float(np.mean(has))
        r["per_test"] = {t: float(np.mean([s["cells"][cn]["whole"] for s in ss if s["test"] == t])) for t in sorted({s["test"] for s in ss})}
        out[cn] = r
    return out


def pool_contrasts(splits, fams, n_boot=1000, seed=0):
    """Paired differences on the same test families: does multi-turn data in the fit help, and ours against Goodfire's."""
    want = [("probe: ST+MT minus ST, whole pool", "probe|ours|ST+MT", "probe|ours|ST", "whole"),
            ("probe: MT minus ST, whole pool", "probe|ours|MT", "probe|ours|ST", "whole"),
            ("mean difference: ST+MT minus ST, whole pool", "mean|ours|ST+MT", "mean|ours|ST", "whole"),
            ("probe: ST+MT minus ST, chosen", "probe|ours|ST+MT", "probe|ours|ST", "chosen"),
            ("probe on ours (ST) minus mean difference on Goodfire, whole pool", "probe|ours|ST", "mean|goodfire|GF", "whole"),
            ("probe on ours (ST+MT) minus mean difference on Goodfire, whole pool", "probe|ours|ST+MT", "mean|goodfire|GF", "whole"),
            ("recipe on our data: probe minus mean difference (ST), whole pool", "probe|ours|ST", "mean|ours|ST", "whole"),
            ("recipe on their data: probe minus mean difference, whole pool", "probe|goodfire|GF", "mean|goodfire|GF", "whole"),
            ("data under the probe: ours (ST) minus Goodfire, whole pool", "probe|ours|ST", "probe|goodfire|GF", "whole"),
            ("data under the mean difference: ours (ST) minus Goodfire, whole pool", "mean|ours|ST", "mean|goodfire|GF", "whole")]
    out = {}
    for name, ca, cb, k in want:
        ss = [s for s in splits if ca in s["cells"] and cb in s["cells"] and k in s["cells"][ca]["vec"] and k in s["cells"][cb]["vec"]]
        if ss:
            out[name] = _boot_means(ss, fams, lambda s: s["cells"][ca]["vec"][k] - s["cells"][cb]["vec"][k], n_boot, seed)
    return out


# ------------------------------------------------------------------ printing

def f3(x, c=None):
    if x is None or x != x:
        return "  -  "
    return f"{x:.3f}" + (f" [{c[0]:.3f}, {c[1]:.3f}]" if c else "")


def print_detection(title, res, paired, log=print):
    log(f"\n{title}")
    sets = list(next(iter(res.values()))) if res else []
    for f in sets:
        n = next(iter(res.values()))[f].get("n_tasks")
        log(f"  {f}  (both-outcome tasks or prompts: {n})")
        for dn, r in res.items():
            if f not in r:
                continue
            x = r[f]
            line = f"    {dn:13s} matched AUROC {f3(x['auc'], x['ci'])}"
            if "step_auc" in x:
                line += f"   step AUROC {f3(x['step_auc'], x['step_ci'])} ({x['n_hack_steps']} hack steps, {x['n_honest_steps']} honest steps)"
            log(line)
    if res:
        log("  mean over sets: " + "  ".join(
            f"{dn} {np.mean([x['auc'] for x in r.values() if x['auc'] is not None]):.3f}" for dn, r in res.items()
            if any(x["auc"] is not None for x in r.values())))
    if paired:
        log("  head-to-head, probe minus gf_pool (paired, bootstrap over tasks or prompts):")
        for f, x in paired["per"].items():
            log(f"    {f:16s} {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]   probe {x['a']:.3f}  gf_pool {x['b']:.3f}  n {x['n']}")
        o = paired.get("overall")
        if o:
            log(f"    {'overall':16s} {o['diff']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]   (equal weight per set, {o['n_sets']} sets)")


LABEL = {"probe|ours": "logistic probe on our data", "mean|ours": "mean difference on our data",
         "mean|goodfire": "mean difference on Goodfire's contrasts", "probe|goodfire": "logistic probe on Goodfire's contrast pairs"}


def print_summary(summ, cells, select, log=print, indent="  "):
    iv = lambda g, sign="": f"{g['mean']:{sign}.3f} [{g['ci'][0]:{sign}.3f}, {g['ci'][1]:{sign}.3f}]"
    head = f"{indent}{'recipe and data':44s} {'pool':5s} {'splits':>6s}  {'whole pool':22s}"
    if select:
        head += f"{'chosen on validation':22s} {'random':>6s} {'bound':>6s} {'size':>5s} {'val':>6s}  {'chosen minus whole':25s} {'wins':>5s}"
    log(head)
    for cn, r in summ.items():
        c = cells[cn]
        line = f"{indent}{LABEL[c['recipe'] + '|' + c['data']]:44s} {c['pool']:5s} {r['n_splits']:6d}  {iv(r['whole']):22s}"
        if select and "chosen" in r:
            line += (f"{iv(r['chosen']):22s} {r['rand']['mean']:6.3f} {r['oracle']['mean']:6.3f} {r['size']:5.1f} {r['val_of_chosen']:6.3f}  "
                     f"{iv(r['chosen_minus_whole'], '+'):25s} {100 * r['win']:4.0f}%"
                     + (f"  chosen subset holds this family in {100 * r['chosen_holds_member']:.0f}% of splits" if "chosen_holds_member" in r else ""))
        log(line)
    if select:
        log(f"{indent}(random = expectation over all subsets of the chosen size; bound = best subset on test, which no procedure can use; "
            f"size = mean number of units chosen; val = validation AUROC of the chosen subset; wins = share of splits where chosen beats whole)")


def print_rotation(rot, realistic, log=print):
    mode = rot["mode"]
    if mode == "none":
        log("\nRotation: no usable multi-turn family, so only the single-turn parts were run.")
        return
    what = {"rotation": "rotation over ordered (test, validation) pairs; the other usable families join the pool",
            "two": "fallback for two usable families: no validation, fit on one plus single-turn, test on the other, whole pool only",
            "one": "fallback for one usable family: test set only, whole pool only"}[mode]
    log(f"\nChoosing fitting environments ({what}). Test matched AUROC, mean over splits, intervals by bootstrap over test tasks.")
    for pv in sorted({c["pool"] for c in rot["cells"].values()}):
        log(f"  units, pool {pv}: " + "; ".join(sorted({c['units'] for c in rot['cells'].values() if c['pool'] == pv})))
    print_summary(rot["summary"], rot["cells"], mode == "rotation", log)
    log("  whole pool by test family: ")
    for cn, r in rot["summary"].items():
        log(f"    {cn:20s} " + "  ".join(f"{t} {x:.3f}" for t, x in r["per_test"].items()))
    if rot.get("contrasts"):
        log("  paired differences on the same test tasks:")
        for k, x in rot["contrasts"].items():
            log(f"    {k:70s} {x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  ({x['n_splits']} splits)")
    if realistic and realistic in rot.get("by_role", {}):
        for role, summ in rot["by_role"][realistic].items():
            log(f"\n  {realistic} as {role} family ({ {'test': 'the test set', 'val': 'the family the choice is made on', 'pool': 'a member of the fitting pool'}[role] }):")
            print_summary(summ, rot["cells"], mode == "rotation", log, "    ")
    elif realistic:
        log(f"\n  {realistic} is not among the usable families, so it has no role in the rotation (see the detection table).")
    log(f"  ({rot['n_fits']} distinct fits, {rot['fit_seconds']:.0f} s)")


# ------------------------------------------------------------------ main

def run(a, log=print):
    t0 = time.time()
    cats = json.loads(a.cats) if a.cats else None
    st = load_st(a.st_feats, a.st_meta, cats)
    if a.st2_feats:
        st += load_st(a.st2_feats, a.st2_meta, cats, taken={e.name for e in st})
    log("Single-turn environments (prompts with both outcomes, rows in them): " +
        ", ".join(f"{e.name} {e.n_tasks}/{len(e.y)}" for e in st))
    runs = [Path(r) if Path(r).exists() else MTD / r for r in a.runs]
    fams, steps, counts = load_mt(runs, a.match)
    usable = {f: e for f, e in fams.items() if e.n_tasks >= a.min_tasks}
    if counts:
        log(f"\nMulti-turn families (matching within {a.match}; usable = at least {a.min_tasks} tasks with both outcomes):")
        for f, c in counts.items():
            log(f"  {f:14s} episodes {c['episodes']:4d} (y None, excluded: {c['excluded_y_none']})  hack {c['hack']:4d}  honest {c['honest']:4d}  "
                f"tasks {c['tasks']:3d}  with both outcomes: by task {c['both_task']}, by task and cfg {c['both_cfg']}, "
                f"by task, cfg and variant {c['both_variant']}  -> {'usable' if f in usable else 'not usable'}")
    else:
        log("\nNo multi-turn run given (--runs): single-turn parts only.")
    gf_pool, gf, note = load_gf(a.gf_dir, a.gf_readout, a.gf_units)
    log(f"\nGoodfire: gf_pool {'loaded' if gf_pool is not None else 'missing'}; {note or 'no contrast data'}")
    Sinv = None
    if not a.no_whiten and a.cov and Path(a.cov).exists():
        try:
            Sinv = sigma_inv(a.cov)
        except Exception as e:
            log(f"  whitening skipped ({type(e).__name__}: {e})")
    fixed = fixed_dirs(a, gf_pool, log)
    res = dict(args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, families=counts,
               usable=list(usable), single_turn={e.name: dict(prompts=e.n_tasks, rows=len(e.y), category=e.cat) for e in st},
               goodfire=dict(note=note, contrasts=[e.name for e in gf]))
    res["detection_single_turn"], res["paired_single_turn"] = detect_st(st, fixed, Sinv, a.C, not a.no_balance, a.n_boot, a.seed)
    print_detection("Detection on single-turn environments (probe, mean_diff and whitened are refitted without the environment scored; "
                    "probe_cached and fitted_w saw some of these environments)", res["detection_single_turn"], res["paired_single_turn"], log)
    if fams:
        dirs = fitted_dirs(st, Sinv, a.C, not a.no_balance) | fixed
        res["detection_rollouts"], res["paired_rollouts"] = detect_mt(dirs, fams, steps, a.n_boot, a.seed)
        print_detection("Detection on multi-turn rollouts (probe, mean_diff and whitened fitted on all single-turn environments; "
                        "no direction saw a rollout)", res["detection_rollouts"], res["paired_rollouts"], log)
    realistic = a.realistic if a.realistic != "auto" else next((f for f in REALISTIC if f in fams), None)
    res["realistic"] = realistic
    if not a.skip_rotation:
        rot = rotation(st, usable, gf, a, log)
        rot.pop("_scorer", None)
        res["rotation"] = rot
        print_rotation(rot, realistic, log)
    res["seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}  ({res['seconds']:.0f} s)")
    return res


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--st-feats", default=str(ONP / "means_36_all.npy"))
    ap.add_argument("--st-meta", default=str(ONP / "meta_all.json"))
    ap.add_argument("--st2-feats", help="second single-turn file, same format (new environments)")
    ap.add_argument("--st2-meta")
    ap.add_argument("--cats", help="JSON {environment: hack category} to add to the built-in table (used only when grouping)")
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (paths, or names under cache/flag/mt)")
    ap.add_argument("--gf-dir", default=str(GFD))
    ap.add_argument("--gf-readout", default="suf")
    ap.add_argument("--gf-units", choices=("individual", "all"), default="individual",
                    help="individual = the one-variant contrasts (i_*) when present; the pooled ones are unions of them")
    ap.add_argument("--dirs-cf", default=str(ONP / "dirs_cf_36.npz"))
    ap.add_argument("--fitw", default=str(ONP / "fitw_w2.pt"))
    ap.add_argument("--cov", default=str(ONP / "cov_36.pt"))
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--min-tasks", type=int, default=2, help="a family is usable with at least this many both-outcome tasks")
    ap.add_argument("--match", choices=("task", "cfg", "variant"), default="task",
                    help="episodes are matched within task; cfg adds the pilot configuration, variant also the hack/benign variant")
    ap.add_argument("--max-subsets", type=int, default=4095, help="largest number of candidate subsets per pool before grouping")
    ap.add_argument("--st-units", choices=("auto", "env", "category", "one"), default="auto")
    ap.add_argument("--realistic", default="auto", help="family reported on its own in each role (auto: ib_lcb, else mini_repo)")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--no-balance", action="store_true", help="do not give every environment the same total weight in the probe")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-rotation", action="store_true")
    ap.add_argument("--skip-gf-probe", action="store_true", help="skip the logistic probe on Goodfire's contrast pairs (the slowest cell)")
    ap.add_argument("--out", default=str(ONP / "night_select.json"))
    return ap


if __name__ == "__main__":
    run(parser().parse_args())

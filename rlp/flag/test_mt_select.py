"""Tests for rlp.flag.mt_select on synthetic activations: a planted direction shared by every environment and family,
plus a nuisance direction that is specific to each one.

    HF_HUB_OFFLINE=1 .venv/bin/python -m pytest rlp/flag/test_mt_select.py -q
    HF_HUB_OFFLINE=1 .venv/bin/python -m rlp.flag.test_mt_select           # the same tests without pytest
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from rlp.flag import mt_select as M

D = 48
U = np.zeros(D, np.float32); U[0] = 1.0                      # the planted, shared direction
ST = ["length_pad", "hardcode", "quiz_key", "leak", "save_tamper", "poison"]
FAMS = ["answer_key", "leak_outputs", "scorer_copy", "save_game", "release_gate", "ib_lcb"]


def nuis(i):
    v = np.zeros(D, np.float32); v[1 + i] = 1.0
    return v


def rows(rng, n_groups, per, shared, nu, strength=3.0, noise=1.0):
    """-> X, y, group index. x = group offset + y * (shared * U + strength * nuisance) + noise."""
    X, y, g = [], [], []
    for k in range(n_groups):
        off = rng.normal(0, 5, D)
        for j in range(per):
            yy = j % 2
            X.append(off + yy * (shared * U + strength * nu) + rng.normal(0, noise, D))
            y.append(yy); g.append(k)
    return np.array(X, np.float32), y, g


def write_st(tmp, rng, envs=ST, name="st"):
    X, meta = [], []
    for i, e in enumerate(envs):
        shared = -4.0 if e == "poison" else 4.0                # 'poison' carries the planted direction with the wrong sign
        x, y, g = rows(rng, 8, 6, shared, nuis(i), strength=4.0 if e == "poison" else 3.0)
        X.append(x)
        meta += [dict(beh=e, group=f"{e}|tempt|{e}/{gg}", y=yy) for yy, gg in zip(y, g)]
    meta[0]["y"] = None                                        # an invalid completion
    meta.append(dict(beh="only_honest", group="only_honest|tempt|0", y=0)); X.append(rng.normal(0, 1, (1, D)).astype(np.float32))
    np.save(tmp / f"{name}.npy", np.concatenate(X)); json.dump(meta, open(tmp / f"{name}.json", "w"))
    return str(tmp / f"{name}.npy"), str(tmp / f"{name}.json")


def write_run(tmp, rng, fams, name="run", shared=4.0, scramble=None, n_tasks=6, one_sided=()):
    """Interface 3. scramble: a family whose activations are replaced by noise. one_sided: families with no honest episode
    in any hack task (not usable)."""
    eps, EM, steps, SM = [], [], [], []
    for fi, f in enumerate(fams):
        x, y, g = rows(rng, n_tasks, 6, shared, nuis(10 + FAMS.index(f)))
        if f == scramble:
            x = rng.normal(0, 1, x.shape).astype(np.float32)
        for xi, yy, gg in zip(x, y, g):
            if f in one_sided:
                yy = 1 if gg % 2 else 0
            ep = len(eps)
            eps.append(dict(key=f"{f}/{gg}/hack", family=f, task=f"{f}-{gg}", variant="hack" if yy else "benign", k=ep, cfg="full", y=yy))
            EM.append(xi)
            for m in range(3):
                hs = bool(yy and m == 2)
                steps.append(dict(ep=ep, msg=2 * m + 2, is_hack_step=hs))
                SM.append(rng.normal(0, 1, D) + (3 * U if hs else 0))
        ep = len(eps)                                          # one truncated episode per family, far out along U
        eps.append(dict(key=f"{f}/x", family=f, task=f"{f}-0", variant="hack", k=ep, cfg="full", y=None))
        EM.append(100 * U)
        steps.append(dict(ep=ep, msg=2, is_hack_step=True)); SM.append(-100 * U)
    d = tmp / name
    d.mkdir()
    np.savez(d / "acts_36.npz", ep_mean=np.array(EM, np.float32), step_mean=np.array(SM, np.float32))
    json.dump(dict(episodes=eps, steps=steps), open(d / "acts_36.json", "w"))
    return str(d)


def write_gf(tmp, rng, acts=True):
    names = ["i_a", "i_b", "i_c", "i_bad", "pl_ab"]
    recs, X = [], []
    for ci, c in enumerate(names[:4]):
        for g in range(10):
            for lab in (1, 0):
                labels = {c: lab}
                if c in ("i_a", "i_b"):
                    labels["pl_ab"] = lab
                recs.append(dict(rid=f"{c}/{g}/{lab}", family="ds", group=f"ds{ci}/{g}", labels=labels))
                sh = (-1.0 if c == "i_bad" else 1.0) * lab
                X.append(rng.normal(0, 3, D) * 0 + g + lab * 2 * nuis(30 + ci) + sh * 4 * U + rng.normal(0, 0.5, D))
    X = np.array(X, np.float32)
    d = tmp / "gf27"
    d.mkdir()
    dirs = {}
    for c in names:
        y = np.array([r["labels"].get(c, -1) for r in recs])
        v = X[y == 1].mean(0) - X[y == 0].mean(0)
        dirs["c:" + c] = v / np.linalg.norm(v)
    pool = np.mean([X[np.array([r["labels"].get(c, -1) for r in recs]) == 1].mean(0) -
                    X[np.array([r["labels"].get(c, -1) for r in recs]) == 0].mean(0) for c in names[:4]], 0)
    np.savez(d / "dirs_36.npz", gf_pool=pool / np.linalg.norm(pool), **dirs)
    if acts:
        np.savez(d / "gf27_acts.npz", suf=X, last=X, turn=X, ok=np.ones(len(recs), bool))
        json.dump(recs, open(d / "gf27_meta.json", "w"))
    return str(d)


def args(tmp, st, runs=(), gf=None, **kw):
    a = M.parser().parse_args([])
    a.st_feats, a.st_meta = st
    a.runs, a.gf_dir = list(runs), gf or str(tmp / "nogf")
    a.dirs_cf = a.fitw = a.cov = str(tmp / "missing")
    a.n_boot, a.out = 200, str(tmp / "out.json")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def world():
    tmp = Path(tempfile.mkdtemp(prefix="mt_select_"))
    rng = np.random.default_rng(0)
    return tmp, write_st(tmp, rng), rng


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol


class Lines(list):
    def __call__(self, *x, **k):
        self.append(" ".join(str(i) for i in x))


# ------------------------------------------------------------------ small pieces

def test_auc_helpers():
    assert near(M.pair_auc(np.array([2.0, 1.0]), np.array([1.0, 0.0])), 0.875)
    assert near(M.auroc(np.array([2.0, 1.0]), np.array([1.0, 0.0])), 0.875)
    rng = np.random.default_rng(1)
    p, n = rng.normal(0.5, 1, 40).round(1), rng.normal(0, 1, 70).round(1)
    assert near(M.auroc(p, n), M.pair_auc(p, n))
    assert np.isnan(M.auroc(np.array([]), n))


def test_env_keeps_only_both_outcome_groups():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    e = M.Env("e", "st", X, [1, 0, 0, 1, 1, 0], ["a", "a", "b", "c", "d", "d"])
    assert e.n_tasks == 2 and e.keys == ["a", "d"] and len(e.y) == 4
    assert np.allclose(e.R[:2].sum(0), 0) and np.allclose(e.D[0], X[0] - X[1])
    assert list(e.per_task(np.array([1.0, 0.0]))) == [0.0, 0.0] and list(e.per_task(np.array([-1.0, 0.0]))) == [1.0, 1.0]


def test_cats_match_repo_table():
    src = (Path(M.__file__).parent / "onp_more.py").read_text() + (Path(M.__file__).parent / "onp_loeo.py").read_text()
    for e, c in M.CATS.items():
        assert f'"{e}": "{c}"' in src


def test_subsets_and_units():
    st = [M.Env(n, "st", np.zeros((2, 2), np.float32), [1, 0], [0, 0], cat=c) for n, c in
          [("a", "G"), ("b", "G"), ("c", "P"), ("d", "T")]]
    S = M.all_subsets([1, 2, 3])
    assert len(S) == 7 and S[-1] == (0, 1, 2) and S[0] == (0,)
    u, how = M.make_units(st, 2, 63)
    assert [n for n, _ in u] == ["a", "b", "c", "d"] and "exact" in how
    u, how = M.make_units(st, 2, 62)                          # 2^(4+2)-1 = 63 is too many, 2^(3+2)-1 = 31 is not
    assert [n for n, _ in u] == ["cat:G", "cat:P", "cat:T"] and [e.name for e in u[0][1]] == ["a", "b"] and "G = a, b" in how
    u, how = M.make_units(st, 2, 10)
    assert len(u) == 1 and len(u[0][1]) == 4
    assert len(M.make_units(st, 2, 10, "env")[0]) == 4


def test_probe_matches_existing_recipe_on_real_cache():
    feats, meta = M.ONP / "means_36_all.npy", M.ONP / "meta_all.json"
    if not feats.exists():
        print("  (skipped: cached single-turn activations are not here)")
        return
    from rlp.flag import onp_loeo as LO
    st = {e.name: e for e in M.load_st(feats, meta)}
    X, mt = np.load(feats), json.load(open(meta))
    envs = ["leak", "fake_pass", "ref_copy"]
    ref = LO.fit(mt, X, envs, balance=True)
    w = M.fit_probe([st[e] for e in envs])
    assert M.cos(w, ref["probe"]) > 0.9999
    assert M.cos(M.fit_mean([st[e] for e in envs]), ref["dom"]) > 0.9999
    s = np.asarray(X @ w, np.float64)
    assert near(st["quiz_key"].per_task(w).mean(), LO.auc(mt, s, "quiz_key", n_boot=1)[0], 1e-6)
    assert st["quiz_key"].n_tasks == len(LO.groups(mt, ["quiz_key"]))


# ------------------------------------------------------------------ loaders

def test_loaders_exclude_y_none():
    tmp, st, rng = world()
    envs = M.load_st(*st)
    assert [e.name for e in envs] == sorted(ST)               # the environment with one outcome is dropped
    assert len(next(e for e in envs if e.name == "length_pad").y) == 8 * 6 - 1   # the row with y None is dropped; its prompt keeps both outcomes
    r1, r2 = write_run(tmp, rng, FAMS[:2], "r1"), write_run(tmp, rng, FAMS[2:4], "r2")
    fams, steps, counts = M.load_mt([r1, r2])
    assert list(fams) == sorted(FAMS[:4])
    for f, e in fams.items():
        assert counts[f]["excluded_y_none"] == 1 and counts[f]["episodes"] == 37 and e.n_rows == 36 and e.n_tasks == 6
        assert np.abs(e.R).max() < 50                          # the truncated episode (100 * U) is not in the data
        assert np.abs(steps[f]["S"]).max() < 50 and steps[f]["pos"].sum() == 18 and (~steps[f]["pos"]).sum() == 54
        assert counts[f]["both_task"] == 6 and counts[f]["both_variant"] == 0
    assert M.load_mt([r1], "variant")[0]["answer_key"].n_tasks == 0


def test_gf_units_are_reaveraged():
    tmp, st, rng = world()
    pool, gf, note = M.load_gf(write_gf(tmp, rng))
    assert [e.name for e in gf] == ["i_a", "i_b", "i_c", "i_bad"]          # pooled contrasts are left out by default
    assert M.cos(M.fit_mean(gf), pool) > 0.9999 and "1.0000" in note
    z, recs = np.load(Path(tmp) / "gf27/gf27_acts.npz")["suf"], json.load(open(Path(tmp) / "gf27/gf27_meta.json"))
    y = np.array([r["labels"].get("i_b", -1) for r in recs])
    assert np.allclose(gf[1].mean_diff(), z[y == 1].mean(0) - z[y == 0].mean(0), atol=1e-5)
    assert gf[0].n_tasks == 10 and len(gf[0].y) == 20
    assert [e.name for e in M.load_gf(str(Path(tmp) / "gf27"), units="all")[1]][-1] == "pl_ab"


def test_gf_without_activations_uses_unit_vectors():
    tmp_path, rng = Path(tempfile.mkdtemp(prefix="mt_select_")), np.random.default_rng(0)
    pool, gf, note = M.load_gf(write_gf(tmp_path, rng, acts=False))
    assert len(gf) == 4 and all(e.n_tasks == 0 for e in gf) and "unit vectors" in note
    assert M.load_gf(str(tmp_path / "absent")) == (None, [], "")


# ------------------------------------------------------------------ detection

def test_detection_and_head_to_head():
    tmp, st, rng = world()
    fams, steps, _ = M.load_mt([write_run(tmp, rng, FAMS[:3])])
    dirs = {"probe": U, "gf_pool": nuis(40), "nuisance0": nuis(10)}
    res, paired = M.detect_mt(dirs, fams, steps, 200)
    for f in fams:
        assert res["probe"][f]["auc"] > 0.85 and res["probe"][f]["ci"][0] <= res["probe"][f]["auc"] <= res["probe"][f]["ci"][1]
        assert res["probe"][f]["step_auc"] > 0.9 and res["probe"][f]["n_hack_steps"] == 18
        assert 0.3 < res["gf_pool"][f]["auc"] < 0.7 and 0.3 < res["gf_pool"][f]["step_auc"] < 0.7
    assert res["nuisance0"]["answer_key"]["auc"] > 0.9 and res["nuisance0"]["leak_outputs"]["auc"] < 0.7   # family-specific
    assert paired["overall"]["diff"] > 0.25 and paired["overall"]["ci"][0] > 0 and len(paired["per"]) == 3
    assert M.detect_mt({"probe": U}, fams, steps, 50)[1] is None


def test_single_turn_detection_is_leave_one_out():
    tmp, st, rng = world()
    envs = [e for e in M.load_st(*st) if e.name != "poison"]
    res, paired = M.detect_st(envs, {"gf_pool": nuis(40)}, None, 0.01, True, 100)
    assert set(res) == {"probe", "mean_diff", "gf_pool"}
    # a probe fitted on the environment itself would use its nuisance and score near 1; left out, it relies on U
    assert all(0.6 < r["auc"] <= 1.0 for r in res["probe"].values())
    w = M.fit_probe([e for e in envs if e.name != "leak"])
    assert abs(w[1 + ST.index("leak")]) < 0.2 * abs(w[0])
    assert paired["overall"]["diff"] > 0.2


# ------------------------------------------------------------------ rotation and fallbacks

def rot(tmp, st, rng, fams, gf=True, name="run", **kw):
    a = args(tmp, st, [write_run(tmp, rng, fams, name, **{k: kw.pop(k) for k in ("scramble", "one_sided") if k in kw})],
             write_gf(tmp, rng) if gf and not (tmp / "gf27").exists() else str(tmp / "gf27"), **kw)
    stE = M.load_st(a.st_feats, a.st_meta)
    F, _, _ = M.load_mt(a.runs)
    usable = {f: e for f, e in F.items() if e.n_tasks >= a.min_tasks}
    return M.rotation(stE, usable, M.load_gf(a.gf_dir)[1], a, log=lambda *x, **k: None), usable


def test_rotation_four_families_no_leak():
    tmp, st, rng = world()
    r, fams = rot(tmp, st, rng, FAMS[:4])
    assert r["mode"] == "rotation" and len(r["splits"]) == 12
    assert set(r["cells"]) == {"probe|ours|ST", "mean|ours|ST", "probe|ours|ST+MT", "mean|ours|ST+MT", "probe|ours|MT",
                               "mean|ours|MT", "mean|goodfire|GF", "probe|goodfire|GF"}
    for s in r["splits"]:
        held = {s["test"], s["val"]}
        assert len(s["mt_pool"]) == 2 and not held & set(s["mt_pool"])
        for cn, c in s["cells"].items():
            assert not held & set(c["chosen_units"]) and not held & set(c["oracle_units"])
            assert c["oracle"] >= c["chosen"] - 1e-12 and c["oracle"] >= c["whole"] - 1e-12 and c["oracle"] >= c["rand"] - 1e-12
            assert c["val"] >= c["val_whole"] - 1e-12
        assert s["cells"]["probe|ours|ST"]["n_subsets"] == 63 and s["cells"]["probe|ours|ST+MT"]["n_subsets"] == 255
        assert s["cells"]["probe|ours|MT"]["n_subsets"] == 3 and s["cells"]["mean|goodfire|GF"]["n_subsets"] == 15
    sc = r["_scorer"]
    for (recipe, key), scores in sc.cache.items():            # a fit is never scored on a family it was fitted on
        assert not {f"mt:{f}" for f in scores} & key
    # the planted direction is shared, so fits transfer; the wrong-signed units hurt and selection drops them
    S = r["summary"]
    assert S["probe|ours|ST"]["chosen"]["mean"] > 0.8 and S["mean|goodfire|GF"]["chosen"]["mean"] > 0.8
    assert S["mean|ours|ST"]["chosen"]["mean"] >= S["mean|ours|ST"]["whole"]["mean"]
    assert all("poison" not in s["cells"]["mean|ours|ST"]["chosen_units"] for s in r["splits"])
    assert all("i_bad" not in s["cells"]["mean|goodfire|GF"]["chosen_units"] for s in r["splits"])
    assert S["probe|ours|ST"]["chosen"]["ci"][0] <= S["probe|ours|ST"]["chosen"]["mean"] <= S["probe|ours|ST"]["chosen"]["ci"][1]
    assert set(r["by_role"]["answer_key"]) == {"test", "val", "pool"}
    assert r["by_role"]["answer_key"]["test"]["probe|ours|ST"]["n_splits"] == 3
    assert r["by_role"]["answer_key"]["pool"]["probe|ours|MT"]["n_splits"] == 6
    assert "chosen_holds_member" in r["by_role"]["answer_key"]["pool"]["probe|ours|ST+MT"]
    assert "probe: ST+MT minus ST, whole pool" in r["contrasts"]
    json.dumps({k: v for k, v in r.items() if k != "_scorer"})


def test_choice_does_not_read_the_test_family():
    """Replacing one family's activations by noise must leave every choice and validation score unchanged in the splits
    where that family is the test set, and must change the test numbers."""
    tmp, st, _ = world()
    r1, _ = rot(tmp, st, np.random.default_rng(5), FAMS[:4], name="a")
    r2, _ = rot(tmp, st, np.random.default_rng(5), FAMS[:4], name="b", scramble="save_game")
    n = 0
    for s1, s2 in zip(r1["splits"], r2["splits"]):
        assert (s1["test"], s1["val"]) == (s2["test"], s2["val"])
        if s1["test"] != "save_game":
            continue
        n += 1
        for cn in s1["cells"]:
            assert s1["cells"][cn]["chosen_units"] == s2["cells"][cn]["chosen_units"]
            assert near(s1["cells"][cn]["val"], s2["cells"][cn]["val"], 1e-12)
        assert abs(s2["cells"]["probe|ours|ST"]["whole"] - 0.5) < 0.2 < s1["cells"]["probe|ours|ST"]["whole"] - 0.5
    assert n == 3


def test_fit_inputs_exclude_held_out_families():
    tmp, st, rng = world()
    seen, cur = [], {}
    real, real_cell = M.fit_probe, M.split_cell

    def cell(sc, recipe, units, t, v, select=True):
        before = len(seen)
        out = real_cell(sc, recipe, units, t, v, select)
        for used in seen[before:]:
            assert f"mt:{t}" not in used and f"mt:{v}" not in used
        cur[(t, v)] = True
        return out
    M.RECIPES["probe"] = lambda envs, **kw: (seen.append({e.uid for e in envs}), real(envs, **kw))[1]
    M.split_cell = cell
    try:
        rot(tmp, st, rng, FAMS[:4])
    finally:
        M.RECIPES["probe"], M.split_cell = real, real_cell
    assert len(cur) == 12 and len(seen) > 100


def test_grouping_keeps_whole_pool_and_is_reported():
    tmp, st, rng = world()
    r1, _ = rot(tmp, st, np.random.default_rng(3), FAMS[:4], name="a")
    r2, _ = rot(tmp, st, np.random.default_rng(3), FAMS[:4], name="b", max_subsets=100)
    assert "exact" in r1["cells"]["probe|ours|ST+MT"]["units"] and "grouped by hack category" in r2["cells"]["probe|ours|ST+MT"]["units"]
    assert "exact" in r2["cells"]["probe|ours|ST"]["units"]                    # 63 subsets still fit
    for s1, s2 in zip(r1["splits"], r2["splits"]):
        assert near(s1["cells"]["probe|ours|ST+MT"]["whole"], s2["cells"]["probe|ours|ST+MT"]["whole"], 1e-9)
        assert s2["cells"]["probe|ours|ST+MT"]["n_subsets"] == 2 ** (4 + 2) - 1  # Game, Peek, Tamper, poison: 4 categories, plus 2 families
        assert all(u.startswith("cat:") or u in FAMS for u in s2["cells"]["probe|ours|ST+MT"]["chosen_units"])


def test_fallback_three_families():
    tmp, st, rng = world()
    r, _ = rot(tmp, st, rng, FAMS[:3])
    assert r["mode"] == "rotation" and len(r["splits"]) == 6
    assert all(len(s["mt_pool"]) == 1 and s["cells"]["probe|ours|MT"]["n_subsets"] == 1 for s in r["splits"])


def test_fallback_two_families():
    tmp, st, rng = world()
    r, _ = rot(tmp, st, rng, FAMS[:2] + ["ib_lcb"], one_sided=("ib_lcb",))     # the third family has no task with both outcomes
    assert r["mode"] == "two" and [(s["test"], s["val"], s["mt_pool"]) for s in r["splits"]] == [
        ("answer_key", None, ["leak_outputs"]), ("leak_outputs", None, ["answer_key"])]
    for s in r["splits"]:
        assert set(s["cells"]) == set(r["cells"]) and "probe|ours|ST+MT" in s["cells"]
        assert all(set(c) == {"n_subsets", "whole"} and c["n_subsets"] == 1 for c in s["cells"].values())
    assert "chosen" not in r["summary"]["probe|ours|ST"] and r["summary"]["probe|ours|ST"]["whole"]["mean"] > 0.8
    for (recipe, key), scores in r["_scorer"].cache.items():
        assert "mt:ib_lcb" not in key and "ib_lcb" not in scores


def test_fallback_one_and_zero_families():
    tmp, st, rng = world()
    r, _ = rot(tmp, st, rng, FAMS[:1])
    assert r["mode"] == "one" and len(r["splits"]) == 1 and r["splits"][0]["mt_pool"] == []
    assert set(r["cells"]) == {"probe|ours|ST", "mean|ours|ST", "mean|goodfire|GF", "probe|goodfire|GF"}
    assert all(not any(k.startswith("mt:") for k in key) for _, key in r["_scorer"].cache)
    r0, _ = rot(tmp, st, rng, FAMS[:2], name="none", one_sided=tuple(FAMS[:2]))
    assert r0["mode"] == "none" and r0["splits"] == []


def test_min_tasks_threshold():
    tmp, st, rng = world()
    r, _ = rot(tmp, st, rng, FAMS[:3], min_tasks=7)                            # each family has 6 both-outcome tasks
    assert r["mode"] == "none"


# ------------------------------------------------------------------ end to end

def test_cli_end_to_end():
    tmp, st, rng = world()
    st2 = write_st(tmp, rng, ["length_pad", "brand_new"], "st2")
    a = args(tmp, st, [write_run(tmp, rng, FAMS[:2], "r1"), write_run(tmp, rng, FAMS[2:], "r2")], write_gf(tmp, rng),
             st2_feats=st2[0], st2_meta=st2[1], max_subsets=600)
    L = Lines()
    res = M.run(a, L)
    out = "\n".join(L)
    saved = json.load(open(a.out))
    assert saved["rotation"]["mode"] == "rotation" and len(saved["rotation"]["splits"]) == 30
    assert saved["realistic"] == "ib_lcb" and set(saved["rotation"]["by_role"]["ib_lcb"]) == {"test", "val", "pool"}
    assert "length_pad@2" in saved["single_turn"] and "brand_new" in saved["single_turn"]
    assert "grouped by hack category" in saved["rotation"]["cells"]["probe|ours|ST+MT"]["units"]
    assert saved["paired_rollouts"]["overall"]["n_sets"] == 6 and "gf_pool" in saved["detection_rollouts"]
    assert saved["paired_single_turn"] is not None and "step_auc" in saved["detection_rollouts"]["probe"]["ib_lcb"]
    for needle in ("Detection on multi-turn rollouts", "head-to-head, probe minus gf_pool", "chosen on validation",
                   "ib_lcb as test family", "ib_lcb as val family", "ib_lcb as pool family", "bound = best subset on test"):
        assert needle in out, needle


def test_cli_single_turn_only():
    tmp, st, rng = world()
    L = Lines()
    res = M.run(args(tmp, st), L)
    out = "\n".join(L)
    assert res["rotation"]["mode"] == "none" and "single-turn parts" in out and "detection_rollouts" not in res
    assert set(res["detection_single_turn"]) == {"probe", "mean_diff"}


if __name__ == "__main__":
    import time
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for k, v in tests:
        t0 = time.time()
        v()
        print(f"ok  {k}  ({time.time() - t0:.1f} s)", flush=True)
    print(f"{len(tests)} passed")

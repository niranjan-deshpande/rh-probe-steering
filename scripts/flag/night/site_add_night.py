"""Insert (or replace) the overnight-run part of site/flag-direction.html. Chart data are read from the caches; the
prose is in scripts/flag/night/site_night_body.html and is typed by hand from results/07_overnight.md. Laptop only.

    .venv/bin/python scripts/flag/night/site_add_night.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
onp = ROOT / "cache/flag/onp"
st = json.load(open(onp / "night_steer_summary.json"))
sel = json.load(open(onp / "night_select.json"))
r1 = lambda x: round(100 * x, 1)
r3 = lambda x: round(x, 3)

m, c = st["main"], st["main"]["conditions"]
N = {"base": r1(m["base"]["rate"]),
     "rnd_sub": {k: [r1(c[f"{k}{i}_sub"]["S"]) for i in range(8)] for k in ("iso", "sig")},
     "rnd_add": {k: [r1(c[f"{k}{i}_add"]["S"]) for i in range(8)] for k in ("iso", "sig")},
     "sub": {k: r1(v["S"]) for k, v in m["subtract"].items()},
     "add": {k: r1(c[f"{k}_add"]["S"]) for k in ("sprobe", "fitv", "gf")},
     "fit": {"base": r1(st["fitting"]["base"]["rate"]), "rnd": [r1(x) for x in st["fitting"]["null"]["values"]],
             "sub": {k: r1(v["S"]) for k, v in st["fitting"]["subtract"].items()}}}
ds = sel["detection_single_turn"]
N["det"] = {k: {e: [r3(v["auc"]), r3(v["ci"][0]), r3(v["ci"][1])] for e, v in ds[k].items()} for k in ("probe", "gf_pool")}
dr = sel.get("detection_rollouts") or {}
N["roll"] = {k: {f: [r3(v["auc"]), r3(v["ci"][0]), r3(v["ci"][1]), v["n_tasks"], r3(v["step_auc"]), r3(v["step_ci"][0]), r3(v["step_ci"][1])]
                 for f, v in dr[k].items() if v.get("auc") is not None and f in sel.get("usable", [])} for k in ("probe", "gf_pool") if k in dr}

body = (ROOT / "scripts/flag/night/site_night_body.html").read_text()
js = (ROOT / "scripts/flag/night/site_night_charts.js").read_text().replace("__NIGHT_DATA__", json.dumps(N))

p = ROOT / "site/flag-direction.html"
h = p.read_text()
B0, B1, J0, J1 = "<!-- night:start -->", "<!-- night:end -->", "// night:start", "// night:end"
if B0 in h:
    h = h[:h.index(B0)] + B0 + "\n" + body + "\n" + h[h.index(B1):]
else:
    a = h.index('<p class="part" id="part-a">')
    h = h[:a] + B0 + "\n" + body + "\n" + B1 + "\n\n" + h[a:]
if J0 in h:
    h = h[:h.index(J0)] + J0 + "\n" + js + "\n" + h[h.index(J1):]
else:
    a = h.index("// steering: base rate, random condition changes")
    h = h[:a] + J0 + "\n" + js + "\n" + J1 + "\n\n" + h[a:]
p.write_text(h)
print("ok", len(h), "chars;", "rollout families:", list(N["roll"].get("probe", {})))

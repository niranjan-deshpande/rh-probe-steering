"""Regenerate the 24 November data on site/flag-direction.html from the corrected caches (thresh_tamper relabelled):
DATA (set curve, effects, detection), the best-set table and the one-environment heat table. Prints the numbers the
prose quotes so that they can be typed by hand. Laptop only."""
import collections, json, re
onp = "cache/flag/onp/"
S = json.load(open(onp + "mix_all_k_summary.json")); A = json.load(open(onp + "mix_all_k.json")); F = json.load(open(onp + "fitted_detect_all.json"))
p = "site/flag-direction.html"; h = open(p).read()
old = json.loads(re.search(r"const DATA=(\{.*?\});\n", h, re.S).group(1))
D = dict(old)
D["sets"] = {str(k): S[str(k)]["sets"] for k in range(1, 10)}
D["tmean"] = [S[str(k)]["tmean"] for k in range(1, 10)]; D["worst"] = [S[str(k)]["worst"] for k in range(1, 10)]
D["eff"] = {e: S["effect"][e] for e in old["eff"]}; D["det"] = F
h = re.sub(r"const DATA=\{.*?\};\n", lambda m: "const DATA=" + json.dumps(D) + ";\n", h, count=1, flags=re.S)
print("tmean", [round(x, 2) for x in D["tmean"]], "worst", [round(x, 2) for x in D["worst"]])
print("eff", {e: [round(x, 3) for x in v] for e, v in D["eff"].items()})
for k in ("probe", "fitted w", "fitted v"):
    print(k, {e: round(F[k][e][0], 2) for e in ["fake_cite", "save_tamper", "fake_pass", "ref_copy", "ckpt_swap", "thresh_tamper", "log_scrub"]})
# best and worst sets per size
by = collections.defaultdict(dict)
for r in A: by[(r["k"], tuple(r["S"]))][r["t"]] = r["auc"]
rows = []
for k in range(2, 7):
    c = [(sum(v.values()) / len(v), min(v.values()), s) for (kk, s), v in by.items() if kk == k]
    m, w, s = max(c); rows.append(f"<tr><td>{k}</td><td>{', '.join(s)}</td><td>{m:.2f}</td><td>{w:.2f}</td></tr>")
    mw, ww, sw = min(c); print("worst set", k, sw, round(mw, 2), {t: round(a, 2) for t, a in by[(k, sw)].items() if a < 0.45})
print("\n".join(rows))
h2 = re.sub(r"(<thead><tr><th>Size</th><th>Best set</th><th>Mean</th><th>Worst target</th></tr></thead>\n<tbody>\n).*?(</tbody>)", lambda m: m.group(1) + "\n".join(rows) + "\n" + m.group(2), h, count=1, flags=re.S)
assert h2 != h or True; h = h2
# heat table (one fitting environment)
E = ["hardcode", "length_pad", "fake_cite", "leak", "quiz_key", "save_tamper", "ckpt_swap", "thresh_tamper", "fake_pass", "ref_copy"]
def cell(a):
    if a >= 0.5: al = (a - 0.42) * 1.2; return f'<td style="background:rgba(35,95,166,{al:.2f});color:{"#fff" if al > 0.545 else "#2a2318"}">{a:.2f}</td>'
    al = (0.58 - a) * 1.2; return f'<td style="background:rgba(176,112,42,{al:.2f});color:{"#fff" if al > 0.545 else "#2a2318"}">{a:.2f}</td>'
body = ""
for r in E:
    v = by[(1, (r,))]; body += f'<tr><td>{r.replace("_", " ")} <span style="color:#8a7d6d">({D["cat"][r]})</span></td>'
    body += "".join('<td style="color:#8a7d6d">–</td>' if t == r else cell(v[t]) for t in E) + f"<td><b>{sum(v.values()) / len(v):.2f}</b></td></tr>"
    print("one env", r, {t: round(a, 2) for t, a in v.items() if a < 0.35}, round(sum(v.values()) / len(v), 2))
h3 = re.sub(r"(<th>ref copy</th><th>mean</th></tr></thead><tbody>).*?(</tbody></table></div>\n<p class=\"fig-note\">Matched AUROC of a probe fitted on the row)", lambda m: m.group(1) + body + m.group(2), h, count=1, flags=re.S)
assert h3 != h; h = h3
open(p, "w").write(h)

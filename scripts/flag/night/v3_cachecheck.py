"""Check, on the live steering server with prefix caching ON, that (1) rows 43 (sprobe_sub) and 47 (gf_sub) change the
next-token distribution against row 0, (2) a canary row wrecks it, (3) a completion served from a cached prefix under
the same row and cache salt equals the cold one, and (4) the cache was in fact hit. Prints V3_CACHECHECK_PASS or
V3_CACHECHECK_FAIL as the last line.   python scripts/flag/night/v3_cachecheck.py
"""
import json, sys, urllib.request, uuid
import torch

B = "http://127.0.0.1:8000"
M = "Qwen/Qwen3.8-27B"
names = torch.load(__import__("os").environ.get("STEER_TABLE", "cache/flag/night/steer_table.pt"), map_location="cpu")["names"]
ROW = {n: names.index(n) for n in ("base", "sprobe_sub", "gf_sub", "canary_a")}
assert "STEER_TABLE" in __import__("os").environ or (ROW["sprobe_sub"] == 43 and ROW["gf_sub"] == 47), ROW
P = "".join(f"Step {i}: the operator reads gauge {i * 7 % 13} and writes the value in the log book. " for i in range(160))
Q = "Summary of the procedure in one sentence: The operator"


def post(prompt, vid, salt, max_tokens=24):
    body = {"model": M, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "logprobs": 8, "seed": 1,
            "vllm_xargs": {"vid": vid}, "cache_salt": salt}
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
    c = json.loads(r.read())["choices"][0]
    return c["text"], c["logprobs"]["top_logprobs"][0]


def dist(a, b):
    ks = set(a) & set(b)
    return max((abs(a[k] - b[k]) for k in ks), default=9.9) if len(ks) >= 4 else 9.9


def metric(name):
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    return sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))


u = uuid.uuid4().hex[:6]
ok = True
res = {}
for n in ("base", "sprobe_sub", "gf_sub", "canary_a"):
    res[n] = post(P + Q, ROW[n], f"{u}-cold-{n}")
noise = dist(res["sprobe_sub"][1], post(P + Q, ROW["sprobe_sub"], f"{u}-cold2")[1])
print("noise floor (two cold runs, same row):", round(noise, 4))
for n in ("sprobe_sub", "gf_sub", "canary_a"):
    d = dist(res["base"][1], res[n][1])
    print(f"row {n} against base: max |dlogp| {d:.3f}; text {res[n][0][:60]!r}")
    ok &= d > max(0.02, 3 * noise)
print("base text:", repr(res["base"][0][:60]))
ok &= res["canary_a"][0] != res["base"][0]
h0, q0 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
for n in ("sprobe_sub", "gf_sub"):
    s = f"{u}-warm-{n}"
    post(P, ROW[n], s, max_tokens=1)
    w = post(P + Q, ROW[n], s)
    d = dist(res[n][1], w[1])
    print(f"row {n}: cached prefix against cold: max |dlogp| {d:.4f}; same text {w[0] == res[n][0]}")
    ok &= d <= max(0.3, 3 * noise)
h1, q1 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
print(f"prefix cache: {h1 - h0:.0f} hit tokens of {q1 - q0:.0f} queried")
ok &= (h1 - h0) > 1000
# a cached prefix of one row must not be served to another row (different salt): row 0 after the warm rows is unchanged
d = dist(res["base"][1], post(P + Q, ROW["base"], f"{u}-cold3")[1])
print("base again:", round(d, 4)); ok &= d <= max(0.3, 3 * noise)
print("V3_CACHECHECK_PASS" if ok else "V3_CACHECHECK_FAIL")
sys.exit(0 if ok else 1)

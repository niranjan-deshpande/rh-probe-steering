"""Diagnostic (lead, 25 Nov): does a base request fail the counting task under concurrency even when NO other row is
in the batch? usage: leaktest.py MODE  with MODE in {zero, canary, mild}."""
import sys, re, random, json, torch
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from rlp.flag.steer_selftest import COUNT_HEAD
mode = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 160
names = torch.load("cache/flag/night/steer_table.pt", map_location="cpu")["names"]
other = {"zero": [0], "canary": [names.index("canary_a"), names.index("canary_b")], "mild": [names.index("sprobe_add"), names.index("sig0_sub"), names.index("iso0_proj")]}[mode]
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=600)
rng = random.Random(1)
def call(i):
    M = rng.choice([8, 60, 106, 250, 600, 1200]); k = 16
    v = 0 if i % 2 == 0 else rng.choice(other)
    p = COUNT_HEAD + ", ".join(map(str, range(1, M + 1))) + ","
    r = c.completions.create(model="Qwen/Qwen3.8-27B", prompt=p, max_tokens=k * (len(str(M + k)) + 3) + 8, temperature=0.0, extra_body={"vllm_xargs": {"vid": v}})
    t = r.choices[0].text
    return dict(i=i, M=M, vid=v, ok=[int(x) for x in re.findall(r"\d+", t)[:k]] == list(range(M + 1, M + 1 + k)), text=t[-70:])
jobs = list(range(N)); 
with ThreadPoolExecutor(96) as ex: R = list(ex.map(call, jobs))
b = [r for r in R if r["i"] % 2 == 0]
print(mode, "base ok", sum(r["ok"] for r in b), "/", len(b), "| other-half ok", sum(r["ok"] for r in R if r["i"] % 2), "/", N - len(b))
for r in b:
    if not r["ok"]: print("  FAIL M=%d %r" % (r["M"], r["text"]))

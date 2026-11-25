"""Diagnostic (lead): does the server reuse a prefix when the next prompt extends prompt + completion exactly?"""
import time, sys
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=600)
M = "Qwen/Qwen3.8-27B"
base = "Notes on rivers.\n" + " ".join(f"River {i} flows past town {i*7%91}." for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 400)) + "\nSummary:"
import urllib.request
def hits():
    for l in urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode().splitlines():
        if l.startswith("vllm:prefix_cache_hits_total"): return float(l.split()[-1])
def go(p, n=40):
    h0 = hits(); t0 = time.time()
    r = c.completions.create(model=M, prompt=p, max_tokens=n, temperature=0.0, extra_body={"skip_special_tokens": False})
    d = getattr(r.usage, "prompt_tokens_details", None)
    return r.choices[0].text, r.usage.prompt_tokens, (hits() - h0, round(time.time() - t0, 1))
t, n, k = go(base); print("first", n, k)
t2, n2, k2 = go(base); print("same again", n2, k2)
for wait in (0, 20, 60):
    time.sleep(wait)
    p = base + t + "\nMore notes follow.\n" + "x y z " * 50
    t3, n3, k3 = go(p); print(f"extended after {wait}s", n3, k3)
    base, t = p, t3

import json, glob, os, sys
from transformers import AutoTokenizer
from rlp.flag import mt, q38 as Q
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
fs = sorted(glob.glob("cache/flag/mt/m1/ck/**/*.json", recursive=True), key=os.path.getmtime)[-6:]
for f in fs:
    d = json.load(open(f)); ms = d.get("messages") or d.get("msgs") or []
    a = [m for m in ms if m.get("role") == "assistant"]
    print(os.path.basename(f), "keys", list(d)[:8], "n_assistant", len(a), "with _raw", sum(1 for m in a if m.get("_raw")))
    if len(a) >= 2 and a[-1].get("_raw"):
        llm = mt.RawLLM.__new__(mt.RawLLM); llm.tok = tok
        task = None
        i = max(j for j, m in enumerate(ms) if m.get("role") == "assistant")
        tools = d.get("tools")
        try:
            full = Q.prompt(tok, ms, tools); inc = llm._prompt(ms, tools)
            prev = llm._prompt(ms[:i], tools)
            print("   len full", len(full), "inc", len(inc), "same", full == inc, "inc extends prev", inc.startswith(prev))
            k = next((j for j in range(min(len(full), len(inc))) if full[j] != inc[j]), None)
            if k is not None: print("   first diff at", k, repr(full[k-60:k+60]), "|||", repr(inc[k-60:k+60]))
        except Exception as e: print("   err", type(e).__name__, e)

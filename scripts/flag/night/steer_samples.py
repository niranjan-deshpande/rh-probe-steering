"""Write matched completions (same prompt, same seed index) under several steering conditions to a markdown file, for
reading by eye. Laptop only, reads cache/flag/base_rate/n1_*.

    .venv/bin/python scripts/flag/night/steer_samples.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "cache/flag/base_rate"
ENVS = ["ckpt_swap", "ref_copy", "thresh_tamper", "save_tamper", "puzzle_leak"]
CONDS = [("base", "Unsteered"), ("sprobe_sub", "Σ·probe subtracted"), ("fitv_sub", "Fitted v subtracted"),
         ("noise_sub", "Noise-only vector subtracted"), ("gf_sub", "Goodfire direction subtracted")]


def load(c, e):
    r = {(x["uid"], x["k"]): x for x in map(json.loads, open(B / f"n1_{c}/{e}__tempt.jsonl"))}
    l = {(x["uid"], x["k"]): x for x in map(json.loads, open(B / f"n1_{c}/{e}__tempt.labels.jsonl"))}
    return r, l


def clip(s, head=1500, tail=2500):
    s = s.strip()
    if len(s) <= head + tail + 200:
        return s
    return s[:head] + f"\n\n[... {len(s) - head - tail:,} characters omitted ...]\n\n" + s[-tail:]


def rep(s):  # share of 4-word sequences that are repeats, a crude looping measure
    w = s.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / max(1, len(g))


out = ["# Steered completions to read by eye (overnight run, 25 November)\n",
       "Each block is one prompt from a held-out environment, with the model's completion under five conditions. "
       "The steering norm is 20 at the input of block 36. `hack` and `valid` are the judge's labels; a completion is "
       "invalid when it hit the 5,000-token limit before giving an answer. Reasoning longer than about 4,000 "
       "characters is clipped in the middle. Two prompts per environment: the first where the unsteered model hacks "
       "and `Σ·probe` subtracted does not, and the first where `Σ·probe` subtracted is invalid or still hacks (or, if "
       "none, the second prompt of the first kind).\n"]
stats = {c: [] for c, _ in CONDS}
for e in ENVS:
    D = {c: load(c, e) for c, _ in CONDS}
    keys = sorted(D["base"][0])
    for c, _ in CONDS:
        stats[c] += [(rep(D[c][0][k]["reasoning"] or ""), D[c][1][k]["valid"]) for k in keys if k in D[c][0]]
    lab = lambda c, k: D[c][1][k]
    a = [k for k in keys if lab("base", k)["hit"] and lab("sprobe_sub", k)["valid"] and not lab("sprobe_sub", k)["hit"]]
    b = [k for k in keys if not lab("sprobe_sub", k)["valid"] or lab("sprobe_sub", k)["hit"]]
    pick = (a[:1] + [k for k in b if k not in a[:1]][:1]) or keys[:1]
    if len(pick) < 2:
        pick += [k for k in (a + keys) if k not in pick][:1]
    for k in pick:
        out.append(f"\n---\n\n## {e}, prompt `{k[0]}`\n")
        out.append("**Prompt (user turn):**\n\n```text\n" + D["base"][0][k]["messages"][-1]["content"].strip() + "\n```\n")
        for c, name in CONDS:
            r, l = D[c][0][k], D[c][1][k]
            out.append(f"\n### {name}: hack = {bool(l['hit'])}, valid = {l['valid']}, {r['completion_tokens']:,} tokens, "
                       f"finish = {r['finish_reason']}\n")
            out.append("<details><summary>Reasoning</summary>\n\n```text\n" + clip(r["reasoning"] or "(none)") + "\n```\n\n</details>\n")
            out.append("**Answer:**\n\n````text\n" + (clip(r["response"]) if r["response"].strip() else "(no answer: token limit reached during reasoning)") + "\n````\n")
p = ROOT / "logs/flag/night/steer_samples.md"
p.write_text("\n".join(out))
print("wrote", p, len("\n".join(out)), "chars")
for c, name in CONDS:
    v = [x for x, _ in stats[c]]
    print(f"{name:32s} n={len(v)} repeated 4-grams in reasoning: median {sorted(v)[len(v)//2]:.2f}, max {max(v):.2f}, >0.5: {sum(x>0.5 for x in v)}")

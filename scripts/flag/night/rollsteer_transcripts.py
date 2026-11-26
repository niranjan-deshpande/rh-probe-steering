"""Matched transcripts for the rollout-steering runs: the same task and seed under each arm, trimmed for reading.

    .venv/bin/python scripts/flag/night/rollsteer_transcripts.py   ->  logs/flag/night/rollsteer_transcripts.md
"""
import json
from pathlib import Path

MT = Path("cache/flag/mt")
ARMS = [("v4_base", "Unsteered"), ("v3_sprobe_sub", "Σ·probe subtracted"), ("v3_gf_sub", "Goodfire's direction subtracted"),
        ("v4_sig5_sub", "Random Σ-shaped vector subtracted"), ("v5_spec_prompt", "Unsteered, with the instruction in the system prompt")]


def load(run, fam):
    p = MT / run / f"{fam}__hack.jsonl"
    return {(e["task"], e["k"]): e for e in map(json.loads, open(p))} if p.exists() else {}


def cut(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n * 2 // 3] + f"\n[... {len(s) - n} characters cut ...]\n" + s[-n // 3:]


def render(e, max_turns=14):
    out, msgs = [], e["messages"][2:]
    a = [m for m in msgs if m["role"] == "assistant"]
    keep = set(range(4)) | set(range(len(a) - (max_turns - 4), len(a)))
    t = -1
    for m in msgs:
        if m["role"] == "assistant":
            t += 1
            if t not in keep:
                if t == 4:
                    out.append(f"*[turns 5 to {len(a) - (max_turns - 4)} omitted]*\n")
                continue
            out.append(f"**Turn {t + 1}, model.**")
            if m.get("reasoning_content"):
                out.append("Thinking:\n```\n" + cut(m["reasoning_content"], 900) + "\n```")
            if m.get("content"):
                out.append("Says:\n```\n" + cut(m["content"], 900) + "\n```")
            for c in m.get("tool_calls") or []:
                out.append(f"Calls `{c['function']['name']}`:\n```\n" + cut(json.dumps(c["function"]["arguments"], ensure_ascii=False), 700) + "\n```")
        elif t in keep:
            out.append("Tool output:\n```\n" + cut(m["content"], 350) + "\n```")
    return "\n".join(out)


def main():
    L = ["# Matched transcripts: steering inside rollouts (25 November)", "",
         "Each section is one task with the same seed under each condition. Long thinking, commands and tool outputs "
         "are cut, and the middle turns of long episodes are omitted. `hack` is the verdict of the programmatic detector.", ""]
    for fam in ("release_gate", "save_game"):
        R = {run: load(run, fam) for run, _ in ARMS}
        key = next(k for k in sorted(R["v4_base"]) if R["v4_base"][k]["info"]["hit"] and all(k in R[r] for r, _ in ARMS[:3]))
        e0 = R["v4_base"][key]
        L += [f"## {fam}, task `{key[0]}`, seed {key[1]}", "", "Task given to the model:", "", "> " + e0["messages"][1]["content"].replace("\n", "\n> "), ""]
        for run, label in ARMS:
            if key not in R[run]:
                continue
            i = R[run][key]["info"]
            L += [f"### {label}", "",
                  f"Outcome: hack = {bool(i['hit'])}" + (f" ({i['kind']})" if i["hit"] else "") +
                  f"; stopped by `{i['stop']}`; {i['n_turns']} turns; {i['n_gen_tokens']:,} generated tokens.", "", render(R[run][key]), ""]
    Path("logs/flag/night/rollsteer_transcripts.md").write_text("\n".join(L))
    print("ok", sum(len(x) for x in L))


main()

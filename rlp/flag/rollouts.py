"""Goodfire's released agent rollouts (Goodfire/reward-hack-data) -> conversations the reader model can be run on,
with the token span of every labelled passage.

A trace is a tree of nodes linked by `parent`. Each generated assistant step appears twice: a `sampled` node (with
reasoning_content / content / tool_calls separated; this is what the judge labelled, by `node_index`) and an unsampled
copy with the reasoning inlined, which is the one later nodes descend from. We walk the path from the last node to the
root and, for every assistant message on it, use the sampled sibling's fields.

A passage is one non-empty channel (reasoning / content / tool_calls) of one sampled assistant step.

Rollouts are longer than the reader's context, so `windows()` cuts each one into overlapping windows of whole
messages. Every window starts with the system and first user message; a passage is scored in the window that gives it
the most preceding context.
"""
import json
import re

from . import render as R

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def load(dirpath):
    """-> list of dict(trace_id, task, split, verdict, verdict_enacted, messages, passages) for every rollout.
    passages: {(message index in `messages`, channel): dict(consensus, statuses, categories, passage_n)}"""
    traces = [json.loads(l) for l in open(f"{dirpath}/traces.jsonl")]
    roll = [json.loads(l) for l in open(f"{dirpath}/judge_rollouts.jsonl")]
    plab = {}
    for l in open(f"{dirpath}/judge_passages.jsonl"):
        p = json.loads(l)
        plab[(p["trace_id"], p["node_index"], p["channel"])] = p
    out = []
    for row, rr in zip(traces, roll):
        t = row["traces"][0]
        assert t["id"] == rr["trace_id"]
        nodes = t["nodes"]
        # path from the last node to the root
        path, i = [], len(nodes) - 1
        while i is not None:
            path.append(i)
            i = nodes[i].get("parent")
        path = path[::-1]
        sampled_by_parent = {}
        for j, n in enumerate(nodes):
            if n.get("sampled") and n["message"].get("role") == "assistant":
                sampled_by_parent.setdefault(n.get("parent"), j)
        messages, passages = [], {}
        for j in path:
            n = nodes[j]
            m = n["message"]
            role = m.get("role")
            if role == "assistant":
                sj = j if n.get("sampled") else sampled_by_parent.get(n.get("parent"))
                src = nodes[sj]["message"] if sj is not None else m
                content, reasoning = src.get("content") or "", src.get("reasoning_content") or ""
                if sj is None and "<think>" in content:           # no sampled sibling: split the inlined reasoning
                    reasoning = "\n".join(x.strip() for x in THINK.findall(content))
                    content = THINK.sub("", content)
                msg = {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning.strip(),
                       "tool_calls": [{"type": "function", "function": {"name": tc.get("name"), "arguments": tc.get("arguments")}}
                                      for tc in (src.get("tool_calls") or [])]}
                if sj is not None:
                    for ch in ("reasoning", "content", "tool_calls"):
                        p = plab.get((t["id"], sj, ch))
                        if p is not None:
                            passages[(len(messages), ch)] = dict(consensus=p["consensus"], statuses=p["carrier_statuses"],
                                                                 categories=p["carrier_categories"], passage_n=p["passage_n"],
                                                                 chars=p["chars"])
                messages.append(msg)
            elif role == "tool":
                messages.append({"role": "tool", "content": _text(m.get("content"))})
            else:
                messages.append({"role": role, "content": _text(m.get("content"))})
        out.append(dict(trace_id=t["id"], task=rr["task"], split=rr["split"], verdict=rr["verdict"],
                        verdict_enacted=rr["verdict_enacted"], messages=R.norm_messages(messages), passages=passages))
    return out


def _text(c):
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c or ""


def tokenize(tok, messages):
    """-> (ids, per-message token range [t0, t1), assistant blocks with token spans per channel) or None."""
    text = R.render(tok, messages)
    blks = R.blocks(text, messages)
    if blks is None:
        return None
    ids, off = R.encode(tok, text)
    for b in blks:
        for k, s in zip(("turn", "reasoning", "content", "tool_calls"),
                        R.tok_spans(off, [b["turn"], b["reasoning"], b["content"], b["tool_calls"]])):
            b[k + "_tok"] = s
    return ids, blks


def windows(tok, messages, max_tokens, head=2, keep=1 / 3):
    """Cut a conversation into windows of whole messages that each fit max_tokens when rendered.
    Every window = messages[:head] + messages[a:b]. Returns [(a, b)] with a >= head; the last `keep` of a window's
    tokens are repeated at the start of the next, and a window never starts on a tool message (its call would be
    missing). A single message larger than the budget is left out (its passages are not scored)."""
    n = len(messages)
    cost = [len(tok(R.render(tok, [m] if m["role"] != "tool" else [{"role": "user", "content": m["content"]}]),
                    add_special_tokens=False)["input_ids"]) + 4 for m in messages]
    budget = max_tokens - sum(cost[:head]) - 16
    out, a = [], head
    while True:
        b, used = a, 0
        while b < n and used + cost[b] <= budget:
            used += cost[b]
            b += 1
        if b == a:                       # a single message larger than the budget: skip it
            b = a + 1
            if b >= n:
                break
            a = b
            continue
        out.append((a, b))
        if b >= n:
            break
        # next window starts where about `keep` of this window's tokens remain as context
        half, c = used * (1 - keep), a
        acc = 0
        while c < b - 1 and acc + cost[c] <= half:
            acc += cost[c]
            c += 1
        c = max(c, a + 1)
        while c < b and messages[c]["role"] == "tool":
            c += 1
        if c >= b:
            c = b
        a = c
    return out


def assign(wins, n_messages, head=2):
    """For each message index, the window in which it has the most preceding context: the earliest window that
    contains it in its second half, else the last window that contains it. -> {message index: window index}"""
    best = {}
    for mi in range(head, n_messages):
        cands = [(wi, mi - a) for wi, (a, b) in enumerate(wins) if a <= mi < b]
        if cands:
            best[mi] = max(cands, key=lambda x: x[1])[0]
    return best

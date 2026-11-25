"""Qwen3.8-27B prompt rendering and completion parsing for generation (added 2026-11-23 for the base-rate pilot).

Qwen3.8 differs from Qwen3-14B (rlp.flag.render) in three ways that matter for the agent harness:

  - Its chat template keeps reasoning in every assistant turn (`preserve_thinking`, on by default), so the sentinel
    workaround in render.py is not needed.
  - Tool calls are written as XML,  <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>,
    not as a JSON object.
  - The generation prompt ends inside an open think block (`<|im_start|>assistant\n<think>\n`), so a completion starts
    with reasoning text and has no opening tag.

Only generation is ported here. Token spans for reading activations (render.blocks and friends) are NOT ported yet.
"""
import json
import re

MODEL = "Qwen/Qwen3.8-27B"
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)     # the model card's thinking-mode settings
EFFORT = "medium"                                          # the template adds no instruction text at this level
STOP = "<|im_end|>"
TH_CLOSE = "</think>"

CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
ANY_CALL = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.S)


def is_q38(model):
    return "Qwen3.8" in model


def norm(messages):
    out = []
    for m in messages:
        m = dict(m)
        m["content"] = m.get("content") or ""
        tcs = []
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            a = f.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {"raw": a}
            tcs.append({"type": "function", "function": {"name": f.get("name"), "arguments": a if isinstance(a, dict) else {}}})
        if tcs:
            m["tool_calls"] = tcs
        else:
            m.pop("tool_calls", None)
        out.append(m)
    return out


def prompt(tok, messages, tools=None):
    """Text the model generates from: the conversation, then the assistant header and an open think block."""
    return tok.apply_chat_template(norm(messages), tools=tools, tokenize=False, add_generation_prompt=True,
                                   reasoning_effort=EFFORT)


def _types(tools):
    t = {}
    for tool in tools or []:
        f = tool["function"]
        t[f["name"]] = {k: v.get("type") for k, v in f["parameters"].get("properties", {}).items()}
    return t


def parse(text, tools=None):
    """Raw completion (starting inside the think block) -> assistant message. A completion with no </think> ran out
    of tokens while reasoning, so all of it is reasoning."""
    if TH_CLOSE in text:
        reasoning, body = text.split(TH_CLOSE, 1)
    else:
        reasoning, body = text, ""
    reasoning = reasoning.replace("<think>", "").strip()
    types, calls = _types(tools), []
    for raw in ANY_CALL.findall(body):
        m = CALL.search(raw)
        if not m:
            calls.append({"type": "function", "function": {"name": "invalid", "arguments": {"raw": raw[:2000]}}})
            continue
        name, args = m.group(1).strip(), {}
        for k, v in PARAM.findall(m.group(2)):
            k = k.strip()
            if types.get(name, {}).get(k) in ("integer", "number", "boolean", "object", "array"):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            args[k] = v
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    content = ANY_CALL.sub("", body).strip()
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}

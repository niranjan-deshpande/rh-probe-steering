"""Render conversations with the reader model's chat template and locate token spans.

A conversation is a list of OpenAI-style messages (system / user / assistant with optional reasoning_content and
tool_calls / tool). The Qwen3 template writes an assistant turn as

    <|im_start|>assistant\n[<think>\n{reasoning}\n</think>\n\n]{content}[\n<tool_call>\n{json}\n</tool_call>]*<|im_end|>

`blocks()` finds each assistant turn in the rendered text and splits it into the three channels Goodfire labels
(reasoning / content / tool_calls), as character spans; `tok_spans()` converts character spans to token indices.
"""
import json

A_OPEN, T_END = "<|im_start|>assistant\n", "<|im_end|>"
TH_OPEN, TH_CLOSE, TC_OPEN = "<think>\n", "\n</think>\n\n", "<tool_call>\n"

S_OPEN, S_CLOSE = "\u2402THINK\u2402", "\u2402/THINK\u2402"

TOOLS = [  # the three tools used in Goodfire's synthetic conversations (their release does not ship schemas)
    {"type": "function", "function": {"name": "run", "description": "Run a shell command in the repository.",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "str_replace", "description": "Replace one occurrence of old_str with new_str in a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"},
                                                                                      "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the test suite under a path.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def norm_messages(messages):
    """Tool-call arguments as dicts, None content as '', so the template renders every source the same way."""
    out = []
    for m in messages:
        m = dict(m)
        m["content"] = m.get("content") or ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {"raw": a}
                tcs.append({"type": "function", "function": {"name": f.get("name"), "arguments": a}})
            m["tool_calls"] = tcs
        else:
            m.pop("tool_calls", None)
        # The Qwen3 template drops reasoning from assistant turns that precede the last user message. Agent rollouts
        # contain several user messages, so reasoning is carried inside content behind sentinels and restored to
        # <think> tags after templating (render), which keeps it in every turn.
        r = (m.pop("reasoning_content", None) or "").strip()
        if m["role"] == "assistant" and r:
            m["content"] = S_OPEN + r + S_CLOSE + m["content"].strip()
        out.append(m)
    return out


def render(tok, messages, tools=None, gen=False):
    """Full text. gen=True appends the assistant header exactly as generation with thinking off would see it."""
    t = tok.apply_chat_template(norm_messages(messages), tools=tools, tokenize=False, add_generation_prompt=gen,
                                enable_thinking=False)
    t = t.replace(S_OPEN, TH_OPEN).replace(S_CLOSE, TH_CLOSE)
    return t.replace("<think>\n\n</think>\n\n" + TH_OPEN, TH_OPEN)     # the template's empty block on the last turn


def _pieces(m):
    """The reasoning and content strings of an assistant message, whether or not norm_messages has merged them."""
    c = m.get("content") or ""
    r = m.get("reasoning_content") or ""
    if c.startswith(S_OPEN):
        r, c = c[len(S_OPEN):].split(S_CLOSE, 1)
    return [x.strip() for x in (r, c) if x.strip()]


def blocks(text, messages):
    """One dict per assistant message, in order: {"mi": message index, "turn": (c0, c1), "reasoning": span | None,
    "content": span | None, "tool_calls": span | None}. turn covers the header through <|im_end|>. Returns None if
    the assistant turns in the text cannot be matched one-to-one with the messages."""
    ai = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    out, cur = [], 0
    for mi in ai:
        m = messages[mi]
        a = text.find(A_OPEN, cur)
        if a < 0:
            return None
        b0 = a + len(A_OPEN)
        # the turn ends at the first <|im_end|> after everything this message is known to contain
        probe = b0
        for piece in _pieces(m):
            j = text.find(piece, probe)
            if j < 0:
                return None
            probe = j + len(piece)
        e = text.find(T_END, probe)
        if e < 0:
            return None
        blk = {"mi": mi, "turn": (a, e + len(T_END)), "reasoning": None, "content": None, "tool_calls": None}
        body0 = b0
        if text.startswith(TH_OPEN, b0):
            j = text.find(TH_CLOSE, b0)
            if j >= 0 and j < e:
                if j > b0 + len(TH_OPEN):
                    blk["reasoning"] = (b0 + len(TH_OPEN), j)
                body0 = j + len(TH_CLOSE)
        n_tc = len(m.get("tool_calls") or [])
        tc0 = e
        if n_tc:
            j = e
            for _ in range(n_tc):
                j = text.rfind(TC_OPEN, body0, j)
                if j < 0:
                    return None
            tc0 = j
            blk["tool_calls"] = (tc0, e + len(T_END))          # includes </tool_call><|im_end|>
        if text[body0:tc0].strip():
            blk["content"] = (body0, tc0 if n_tc else e + len(T_END))
        out.append(blk)
        cur = e + len(T_END)
    return out


def encode(tok, text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]


def tok_spans(offsets, spans):
    """Token index ranges [t0, t1) covering each character span (None stays None). offsets are sorted, so one pass."""
    import bisect
    starts = [o[0] for o in offsets]
    ends = [o[1] for o in offsets]
    out = []
    for s in spans:
        if s is None:
            out.append(None)
            continue
        t0 = bisect.bisect_right(ends, s[0])          # first token ending after the span start
        t1 = bisect.bisect_left(starts, s[1])         # first token starting at or after the span end
        out.append((t0, t1) if t1 > t0 else None)
    return out

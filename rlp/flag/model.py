"""Reader model loading and the two hooks everything else uses: capture hidden_states[L] with an early exit, and
overwrite the component along a direction at chosen positions.

Convention (same as the rest of the repo): hidden_states[L] is the input to decoder block L (0 = embeddings), so a
forward pre-hook on block L reads and writes exactly what output_hidden_states would return at index L.
"""
import contextlib

import torch


def load(name=None, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from . import MODEL
    name = name or MODEL
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def blocks_of(model):
    return model.model.layers


class _Stop(Exception):
    pass


@contextlib.contextmanager
def capture(model, layers, stop=True):
    """Collect hidden_states[L] for L in layers into the yielded dict; with stop=True the forward pass is abandoned
    after the highest layer (call through `run_capture`)."""
    got, hooks = {}, []
    top = max(layers)

    def mk(L):
        def pre(mod, args, kwargs):
            got[L] = args[0] if args else kwargs["hidden_states"]
            if stop and L == top:
                raise _Stop()
        return pre
    blks = blocks_of(model)
    for L in layers:
        hooks.append(blks[L].register_forward_pre_hook(mk(L), with_kwargs=True))
    try:
        yield got
    finally:
        for h in hooks:
            h.remove()


def run_capture(model, layers, **inputs):
    """-> {L: [B, T, d]} without computing blocks above max(layers)."""
    with torch.no_grad(), capture(model, layers) as got:
        try:
            model(**inputs, use_cache=False)
        except _Stop:
            pass
    return got


@contextlib.contextmanager
def set_along(model, L, fn):
    """Pre-hook on block L that replaces its input h with fn(h). fn sees [B, T, d] for the tokens being processed."""
    def pre(mod, args, kwargs):
        if args:
            return (fn(args[0]),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = fn(kwargs["hidden_states"])
        return args, kwargs
    h = blocks_of(model)[L].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def overwrite(h, d, value, mask):
    """h - (h.d) d + value d at masked positions. d: [dm] unit vector; value: scalar or [B] or [B, T]; mask: [B, T]
    bool. Everything orthogonal to d is left as it is."""
    d = d.to(h.dtype)
    proj = h @ d                                              # [B, T]
    value = torch.as_tensor(value, device=h.device, dtype=h.dtype)
    while value.dim() < proj.dim():
        value = value.unsqueeze(-1)
    delta = (value - proj) * mask.to(h.dtype)
    return h + delta.unsqueeze(-1) * d

"""Activations for the synthetic conversations (GPU).   python -m rlp.flag.harvest [--limit N]

For every (row, readout) record, teacher-force the whole conversation through the reader model and store the mean
residual-stream activation over two position sets of the readout's assistant turn, at each layer in flag.LAYERS:

    close   the last 3 tokens of the turn (`}}\\n`, `</tool_call>`, `<|im_end|>` for a tool call). This is the analogue
            of Goodfire's readout, the closing tokens of the tool call.
    turn    every token of the turn after the `<|im_start|>assistant\\n` header (written reasoning included).
    reason  the tokens of the turn's written reasoning only (zeros if the turn has none).

Output: cache/flag/synth_acts.npz (close, turn: [n records, n layers, d] float16) + synth_meta.json (the records).
"""
import argparse
import json

import numpy as np
import torch

from .. import config
from . import LAYERS, model as M, render as R, synth as S

OUT = config.CACHE / "flag"
HEADER = 3            # <|im_start|> assistant \n
CLOSE = 3


def locate(tok, row):
    """-> (ids, {message index: (t0, t1)}) with [t0, t1) the tokens of that assistant turn after its header."""
    msgs = R.norm_messages(row["messages"])
    text = R.render(tok, row["messages"], tools=S.tools_for(row))
    blks = R.blocks(text, msgs)
    if blks is None:
        return None
    ids, off = R.encode(tok, text)
    spans = R.tok_spans(off, [b["turn"] for b in blks])
    rs = R.tok_spans(off, [b["reasoning"] for b in blks])
    return ids, {b["mi"]: (s[0] + HEADER, s[1], r) for b, s, r in zip(blks, spans, rs) if s is not None}


def run(model, tok, rows, recs, layers=LAYERS, token_budget=24000, log=print):
    dev = next(model.parameters()).device
    by_row = {}
    for i, r in enumerate(recs):
        by_row.setdefault(r["row"], []).append(i)
    loc = {ri: locate(tok, rows[ri]) for ri in by_row}
    bad = [ri for ri, v in loc.items() if v is None]
    order = sorted((ri for ri in by_row if loc[ri] is not None), key=lambda ri: len(loc[ri][0]))
    d = model.config.hidden_size
    close = np.zeros((len(recs), len(layers), d), np.float16)
    turn = np.zeros((len(recs), len(layers), d), np.float16)
    reason = np.zeros((len(recs), len(layers), d), np.float16)
    ok = np.zeros(len(recs), bool)
    i = 0
    while i < len(order):
        T = len(loc[order[i]][0])
        n = max(1, token_budget // T)
        batch = order[i:i + n]
        T = max(len(loc[ri][0]) for ri in batch)
        ids = torch.full((len(batch), T), tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros((len(batch), T), dtype=torch.long)
        for b, ri in enumerate(batch):
            x = loc[ri][0]
            ids[b, :len(x)] = torch.tensor(x)
            am[b, :len(x)] = 1
        got = M.run_capture(model, layers, input_ids=ids.to(dev), attention_mask=am.to(dev))
        H = torch.stack([got[L] for L in layers], 1).float()          # [B, nL, T, d]
        for b, ri in enumerate(batch):
            for ci in by_row[ri]:
                sp = loc[ri][1].get(recs[ci]["mi"])
                if sp is None:
                    continue
                t0, t1, rsp = sp
                if rsp is not None:
                    reason[ci] = H[b, :, rsp[0]:rsp[1]].mean(1).cpu().numpy()
                turn[ci] = H[b, :, t0:t1].mean(1).cpu().numpy()
                close[ci] = H[b, :, max(t0, t1 - CLOSE):t1].mean(1).cpu().numpy()
                ok[ci] = True
        i += len(batch)
        if (i // max(1, len(batch))) % 20 == 0:
            log(f"[harvest] {i}/{len(order)} rows", flush=True)
    return close, turn, reason, ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = S.load_rows()
    recs = S.records(rows)
    if a.limit:
        keep = set(range(a.limit))
        recs = [r for r in recs if r["row"] in keep]
    model, tok = M.load(a.model)
    close, turn, reason, ok, bad = run(model, tok, rows, recs)
    print(f"[harvest] records {len(recs)}, located {int(ok.sum())}, rows that failed to parse {len(bad)}", flush=True)
    np.savez(OUT / "synth_acts.npz", close=close, turn=turn, reason=reason, ok=ok, layers=np.array(LAYERS))
    json.dump(recs, open(OUT / "synth_meta.json", "w"))


if __name__ == "__main__":
    main()

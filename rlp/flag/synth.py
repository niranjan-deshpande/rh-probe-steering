"""Synthetic minimal-pair agent conversations: Goodfire's released set plus our own families, in one schema.

Row schema (Goodfire's `synthetic_data/dataset.jsonl`): id, group_id, split (train / validation, by group),
metadata.dataset (family), metadata.cell (honest or a hack variant), messages, readouts. A readout names one assistant
message (`selector.message_index`) and gives 0/1 labels for the contrasts it takes part in. A contrast compares the
same message position across cells of one group, so its positives and negatives share everything before that message.

Families and roles:
    train      tz_tamper_suite_pos_v1, tz_tamper_validator_pos_v1, tz_acquire_pos_v1 (Goodfire) + ours_support
    held out   tz_tamper_scorefile_pos_v1 (Goodfire) + ours_report
"Goodfire's direction" is built from all four tz_ families, as they do; ours never sees the held-out families.
"""
import collections
import json

from .. import config

GF = config.DATA / "goodfire"
OURS = config.DATA / "flag"

GOODFIRE = ("tz_tamper_suite_pos_v1", "tz_tamper_validator_pos_v1", "tz_tamper_scorefile_pos_v1", "tz_acquire_pos_v1")
TRAIN = ("tz_tamper_suite_pos_v1", "tz_tamper_validator_pos_v1", "tz_acquire_pos_v1", "ours_support")
HELD = ("tz_tamper_scorefile_pos_v1", "ours_report")
SWE_TRAIN = TRAIN[:3]


def load_rows(ours=True, reasoning=None):
    """reasoning: add the written thought (data/flag/reasoning.json, see gen_reasoning.py) as reasoning_content to
    every assistant message a contrast reads. None = use it if the file exists."""
    rows = [json.loads(l) for l in open(GF / "dataset.jsonl")]
    p = OURS / "ours.jsonl"
    if ours and p.exists():
        rows += [json.loads(l) for l in open(p)]
    rp = OURS / "reasoning.json"
    if reasoning or (reasoning is None and rp.exists()):
        from .gen_reasoning import key_of
        th = json.load(open(rp))
        for r in rows:                       # every assistant message with a written thought, in whichever cell it
            for mi, m in enumerate(r["messages"]):      # appears, so cells that share a message stay identical there
                if m["role"] != "assistant":
                    continue
                t = th.get(key_of(r["group_id"], mi, m))
                if t:
                    r["messages"][mi] = dict(m, reasoning_content=t)
    return rows


def load_contrasts(ours=True):
    C = json.load(open(GF / "contrasts.json"))
    p = OURS / "ours_contrasts.json"
    if ours and p.exists():
        C.update(json.load(open(p)))
    return C


def individual(C):
    """The one-hack-variant contrasts (names starting with i_); pooled ones (pl_) are unions of these."""
    return {k: v for k, v in C.items() if k.startswith("i_")}


def records(rows):
    """One record per (row, readout): the unit activations are harvested for.
    -> list of dict(rid, row, family, group, split, cell, mi, labels)"""
    out = []
    for ri, r in enumerate(rows):
        for ro in r["readouts"]:
            out.append(dict(rid=f"{r['id']}|{ro['name']}", row=ri, family=r["metadata"]["dataset"], group=r["group_id"],
                            split=r["split"], cell=r["metadata"]["cell"], mi=ro["selector"]["message_index"],
                            labels=ro["labels"]))
    return out


def pairs(rows, C=None, families=None, split=None):
    """Minimal pairs for the causal loss: for every individual contrast and group, (positive row, negative row, k)
    where k is the message both readouts point at and messages[:k] are identical.
    -> list of dict(contrast, family, group, split, pos, neg, k) with pos / neg as row indices"""
    C = individual(C or load_contrasts())
    by = collections.defaultdict(dict)
    for ri, r in enumerate(rows):
        by[r["group_id"]][r["metadata"]["cell"]] = ri
    out = []
    for name, c in C.items():
        if families is not None and c["dataset"] not in families:
            continue
        for g, cells in by.items():
            if not g.startswith(c["dataset"] + "/"):
                continue
            for pc in c["pos"]:
                for nc in c["neg"]:
                    if pc not in cells or nc not in cells:
                        continue
                    rp, rn = rows[cells[pc]], rows[cells[nc]]
                    if split is not None and rp["split"] != split:
                        continue
                    k = next((ro["selector"]["message_index"] for ro in rp["readouts"] if ro["labels"].get(name) == 1), None)
                    if k is None or rp["messages"][:k] != rn["messages"][:k] or rp["messages"][k] == rn["messages"][k]:
                        continue
                    out.append(dict(contrast=name, family=c["dataset"], group=g, split=rp["split"], pos=cells[pc],
                                    neg=cells[nc], k=k))
    return out


def tools_for(row):
    """Tool schemas shown in the system prompt. Our families ship their own; Goodfire's use the three SWE tools."""
    from .render import TOOLS
    return row.get("tools") or TOOLS

"""Counts for the write-up: per run, family and variant: episodes, valid, hits by kind, stops, cut-off turns, solved,
tasks with both outcomes. Laptop only. Usage: .venv/bin/python scripts/flag/night/mt_counts.py [run ...]"""
import collections, glob, json, os, sys
root = "cache/flag/mt"
runs = sys.argv[1:] or sorted(r for r in os.listdir(root) if "smoke" not in r)
for r in runs:
    for f in sorted(glob.glob(f"{root}/{r}/*__*.jsonl")):
        eps = [json.loads(l) for l in open(f)]
        if not eps: continue
        fam, var = eps[0]["family"], eps[0]["variant"]
        valid = [e for e in eps if e["info"]["stop"] != "truncated"]
        hits = [e for e in valid if e["info"]["hit"]]
        kinds = collections.Counter(e["info"]["kind"] for e in hits)
        stops = collections.Counter(e["info"]["stop"] for e in eps)
        turns = sum(e["info"]["n_turns"] for e in eps); trunc = sum(e["info"]["n_trunc"] for e in eps)
        solved = [e["info"]["solved"] for e in valid if e["info"]["solved"] is not None]
        by = collections.defaultdict(set)
        for e in valid: by[e["task"]].add(bool(e["info"]["hit"]))
        both = sum(1 for v in by.values() if len(v) == 2)
        gen = sum(e["info"]["n_gen_tokens"] for e in eps)
        cfg = eps[0].get("cfg"); cfg = cfg if isinstance(cfg, str) else json.dumps(cfg, sort_keys=True)
        print(f"{r:22s} {fam:13s} {var:6s} eps {len(eps):3d} valid {len(valid):3d} hits {len(hits):3d} {dict(kinds)} "
              f"tasks {len(by)} both {both} trunc_turns {trunc}/{turns} solved {sum(map(bool, solved))}/{len(solved)} "
              f"stops {dict(stops)} gen {gen/1e6:.2f}M")
    f0 = glob.glob(f"{root}/{r}/*__*.jsonl")
    if f0:
        e = json.loads(open(f0[0]).readline()); print(f"{'':22s} cfg {json.dumps(e.get('cfg'))[:200]}")

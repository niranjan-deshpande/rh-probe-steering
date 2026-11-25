#!/bin/bash
# Relabel after the 2026-11-24 fix to envs2._bash_hits (commented-out commands no longer count as actions), then rerun
# every laptop table that used those labels. Old labels and tables are kept in cache/_backup_labels_2026-11-24/.
set -e
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
B=cache/_backup_labels_2026-11-24; O=cache/flag/onp
if [ ! -d $B ]; then
  mkdir -p $B/onp
  for d in cache/flag/base_rate/env2 cache/flag/base_rate/q38_env2 cache/flag/base_rate/s3_*; do
    mkdir -p $B/$(basename $d); cp $d/*.labels.jsonl $d/summary.json $B/$(basename $d)/
  done
  cp $O/more_envs.json $O/loeo_all.json $O/mix.json $O/mix_all_k.json $O/mix_all_k_summary.json $O/fitted_detect_all.json \
     $O/steer3_summary.json $O/h36_env2.json $O/meta_all.json $B/onp/
fi
for d in env2 q38_env2 $(cd cache/flag/base_rate && ls -d s3_*); do
  .venv/bin/python -m rlp.flag.base_rate label --name $d --workers 6 > /dev/null
done
echo RELABEL_DONE
.venv/bin/python - <<'PY'
import json
m = json.load(open("cache/flag/onp/h36_env2.json")); lab = {}
for f in __import__("glob").glob("cache/flag/base_rate/env2/*.labels.jsonl"):
    for l in open(f):
        j = json.loads(l); lab[(j["uid"], j["k"])] = j
ch = 0
for r in m:
    j = lab[(r["uid"], r["k"])]
    y = None if (j.get("trunc") or j.get("empty") or not j.get("valid", True)) and r["y"] is None else int(bool(j["hit"]))
    if r["y"] is not None and y != r["y"]:
        ch += 1; r["y"] = y
json.dump(m, open("cache/flag/onp/h36_env2.json", "w"))
print("h36_env2 labels changed:", ch)
PY
.venv/bin/python -m rlp.flag.onp_more
.venv/bin/python -m rlp.flag.onp_loeo --feats $O/means_36_all.npy --meta $O/meta_all.json --no-lda --out loeo_all.json \
  --cats '{"ckpt_swap": "Tamper", "thresh_tamper": "Tamper", "log_scrub": "Tamper", "ref_copy": "Plagiarise"}'
.venv/bin/python -m rlp.flag.onp_steer2 --prefix s3_ --cands fitww2 sprobe20 lda20 --rand r220 r320 r420 r520 r620 r720 \
  --base base --groups '{"new": ["ckpt_swap", "thresh_tamper", "ref_copy"]}' --out $O/steer3_summary.json
.venv/bin/python -m rlp.flag.onp_mix
echo RELABEL_CHAIN_DONE

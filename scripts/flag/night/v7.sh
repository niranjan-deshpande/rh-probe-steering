#!/bin/bash
# Pod, 25 November evening: finisher after the owner moved the shutdown to 9 pm ET. Replaces the supervisor of v6.sh
# (its stages keep running). Stops everything at CUTOFF (8:45 pm ET). The second random vector stops after 16 episodes.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
CUTOFF=1795657500
C="--api raw --gen-budget 30000"
SG="--families save_game,release_gate --tasks 8"
alive() { [ -s /workspace/running/$1 ] && kill -0 -- "-$(cat /workspace/running/$1)" 2>/dev/null; }
( while alive v5_cap; do sleep 20; done
  bash $S/stage.sh v5_cap 120 python scripts/flag/night/v5_single.py cap --conc 8 ) &
( while alive v4_benign; do sleep 20; done
  bash $S/stage.sh v4_benign 120 python -m rlp.flag.mt run --run v4_sprobe_benign --vid 43 $SG --hack-eps 0 --benign-eps 4 $C --conc 3 --active 3
  bash $S/stage.sh v4_leak 120 python -m rlp.flag.mt run --run v4_sprobe_other --vid 43 --families leak_outputs --tasks 8 --hack-eps 4 --benign-eps 0 $C --conc 4 --active 4 ) &
( while alive v4_sig5; do sleep 20; done
  alive v5_spec || bash $S/stage.sh v5_spec 150 python -m rlp.flag.mt run --run v5_spec_prompt --vid 0 --cfg @scripts/flag/night/spec_cfg.json $SG --hack-eps 2 --benign-eps 0 $C --conc 5 --active 5 ) &
( while alive v4_sig7 && [ "$(cat cache/flag/mt/v4_sig7_sub/*__hack.jsonl 2>/dev/null | wc -l)" -lt 16 ]; do sleep 30; done
  alive v4_sig7 && kill -- "-$(cat /workspace/running/v4_sig7)" ) &
sleep 90
while [ "$(date +%s)" -lt "$CUTOFF" ] && pgrep -f "flag.m[t] run|v5_singl[e]" > /dev/null; do sleep 30; done
for f in /workspace/running/v*; do [ -s "$f" ] && kill -- "-$(cat "$f")" 2>/dev/null; done
sleep 10
for r in v3_gf_sub v4_base v4_sig5_sub v4_sig7_sub v4_sprobe_benign v4_sprobe_other v5_spec_prompt; do python -m rlp.flag.mt stats --run $r; done > /workspace/logs/v4_stats.txt 2>&1
echo V4_END >> /workspace/logs/v4.log
echo V5_END >> /workspace/logs/v5.log

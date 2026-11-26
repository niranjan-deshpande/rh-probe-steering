#!/bin/bash
# Pod, 25 November 6:15 pm ET: triage of the queue after throughput fell to 80 tok/s with 20 open conversations.
# At most 13 rollout conversations; everything stops at CUTOFF (7:45 pm ET) so the last sync beats the 8 pm shutdown.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
CUTOFF=1795653900
C="--api raw --gen-budget 30000"
SG="--families save_game,release_gate --tasks 8"
bash $S/stage.sh v5_cap 85 python scripts/flag/night/v5_single.py cap --conc 8 &
( bash $S/stage.sh v4_benign 85 python -m rlp.flag.mt run --run v4_sprobe_benign --vid 43 $SG --hack-eps 0 --benign-eps 4 $C --conc 3 --active 3
  bash $S/stage.sh v4_leak 60 python -m rlp.flag.mt run --run v4_sprobe_other --vid 43 --families leak_outputs --tasks 8 --hack-eps 4 --benign-eps 0 $C --conc 3 --active 3 ) &
( while [ -e /workspace/running/v4_sig5 ] && kill -0 -- "-$(cat /workspace/running/v4_sig5)" 2>/dev/null; do sleep 20; done
  bash $S/stage.sh v5_spec 85 python -m rlp.flag.mt run --run v5_spec_prompt --vid 0 --cfg @scripts/flag/night/spec_cfg.json $SG --hack-eps 2 --benign-eps 0 $C --conc 5 --active 5 ) &
sleep 60
while [ "$(date +%s)" -lt "$CUTOFF" ] && pgrep -f "flag.m[t] run|v5_singl[e]" > /dev/null; do sleep 30; done
for f in /workspace/running/v*; do [ -s "$f" ] && kill -- "-$(cat "$f")" 2>/dev/null; done
sleep 10
for r in v3_gf_sub v4_base v4_sig5_sub v4_sig7_sub v4_iso0_sub v4_sprobe_benign v4_sprobe_other v5_spec_prompt; do python -m rlp.flag.mt stats --run $r; done > /workspace/logs/v4_stats.txt 2>&1
echo V4_END >> /workspace/logs/v4.log
echo V5_END >> /workspace/logs/v5.log

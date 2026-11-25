#!/bin/bash
# Pod, 25 November afternoon: rigour checks for steering inside multi-turn rollouts. Every run uses the same
# per-episode budget of 30,000 generated tokens (stop "budget"). Three runs at a time, 5 requests each, in priority order.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
C="--api raw --conc 5 --active 5 --gen-budget 30000"
SG="--families save_game,release_gate --tasks 8"
cat > /workspace/logs/v4_jobs.txt <<J
v4_gf     python -m rlp.flag.mt run --run v3_gf_sub --vid 47 $SG --hack-eps 2 --benign-eps 0 $C
v4_base   python -m rlp.flag.mt run --run v4_base --vid 0 $SG --hack-eps 2 --benign-eps 0 $C
v4_sig5   python -m rlp.flag.mt run --run v4_sig5_sub --vid 28 $SG --hack-eps 2 --benign-eps 0 $C
v4_sig7   python -m rlp.flag.mt run --run v4_sig7_sub --vid 32 $SG --hack-eps 2 --benign-eps 0 $C
v4_iso0   python -m rlp.flag.mt run --run v4_iso0_sub --vid 2 $SG --hack-eps 2 --benign-eps 0 $C
v4_benign python -m rlp.flag.mt run --run v4_sprobe_benign --vid 43 $SG --hack-eps 0 --benign-eps 4 $C
v4_leak   python -m rlp.flag.mt run --run v4_sprobe_other --vid 43 --families leak_outputs --tasks 8 --hack-eps 8 --benign-eps 0 $C
v4_akey   python -m rlp.flag.mt run --run v4_sprobe_other --vid 43 --families answer_key --tasks 8 --hack-eps 4 --benign-eps 0 $C
v4_scorer python -m rlp.flag.mt run --run v4_sprobe_other --vid 43 --families scorer_copy --tasks 8 --hack-eps 2 --benign-eps 0 $C
J
xargs -P 3 -L 1 bash -c 'n=$0; bash scripts/flag/night/stage.sh $n 200 "$@"' < /workspace/logs/v4_jobs.txt
for r in v3_gf_sub v4_base v4_sig5_sub v4_sig7_sub v4_iso0_sub v4_sprobe_benign v4_sprobe_other; do python -m rlp.flag.mt stats --run $r; done > /workspace/logs/v4_stats.txt 2>&1
echo V4_END

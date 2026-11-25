#!/bin/bash
# Pod B, stage M1: the five scripted families, first batch (8 tasks x 4 hack + 6 benign episodes).
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh m1 240 python -m rlp.flag.mt run --run m1 --families answer_key,leak_outputs,scorer_copy,save_game,release_gate --tasks 8 --hack-eps 4 --benign-eps 6 --conc 7 --active 7
python -m rlp.flag.mt stats --run m1 > /workspace/logs/m1_stats.txt 2>&1
echo M1_END

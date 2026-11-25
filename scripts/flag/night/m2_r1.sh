#!/bin/bash
# Pod B, stage M2 round 1: three ImpossibleBench variants in parallel with M1 (10 tasks x 3 impossible + 1 solvable).
cd /workspace/rlp; S=scripts/flag/night
for v in published budget minimal; do
  bash $S/stage.sh ib_r1_$v 200 python -m rlp.flag.mt run --run ib_r1_$v --families ib_lcb --tasks 10 --hack-eps 3 --benign-eps 1 --conc 2 --active 2 --cfg "{\"name\":\"$v\"}" &
done
wait
for v in published budget minimal; do python -m rlp.flag.mt stats --run ib_r1_$v > /workspace/logs/ib_r1_${v}_stats.txt 2>&1; done
echo M2R1_END

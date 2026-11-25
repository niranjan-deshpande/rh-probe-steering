#!/bin/bash
# Pod B, ImpossibleBench round 3. Round 2: plain wording 5/5 conform, strict wording + note 0/4. Round 3 tries two
# wordings in between (mild: "say so in your final answer"; mild2: "the specification is the source of truth").
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh ib_r3_mild 240 python -m rlp.flag.mt run --run ib_r3_mild --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg '{"name":"published","instruction":"mild","max_submits":5,"max_turns":15}' &
bash $S/stage.sh ib_r3_mild2 240 python -m rlp.flag.mt run --run ib_r3_mild2 --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg '{"name":"published","instruction":"mild2","max_submits":5,"max_turns":15}' &
wait
for v in mild mild2; do python -m rlp.flag.mt stats --run ib_r3_$v > /workspace/logs/ib_r3_${v}_stats.txt 2>&1; done
echo M2R3_END

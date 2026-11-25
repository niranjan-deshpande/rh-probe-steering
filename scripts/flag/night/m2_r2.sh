#!/bin/bash
# Pod B, ImpossibleBench round 2. Round 1 (small n): strict wording 0/5, plain wording + budget note 6/7, minimal
# scaffold 6/6. Round 2 takes the two settings in between: plain wording without the note, strict wording with it.
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh ib_r2_plain 240 python -m rlp.flag.mt run --run ib_r2_plain --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg '{"name":"published","instruction":"plain","max_submits":5,"max_turns":15}' &
bash $S/stage.sh ib_r2_strictnote 240 python -m rlp.flag.mt run --run ib_r2_strictnote --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg '{"name":"budget","instruction":"strict","max_turns":15}' &
wait
for v in plain strictnote; do python -m rlp.flag.mt stats --run ib_r2_$v > /workspace/logs/ib_r2_${v}_stats.txt 2>&1; done
echo M2R2_END

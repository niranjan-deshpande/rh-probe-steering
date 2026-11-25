#!/bin/bash
# Pod B, 5:50 am: the scripted batch is nearly done, so the freed slots extend the ImpossibleBench full run ("mild"
# wording, same cfg as pod A) to tasks 15-29, and add 4 more hack-variant episodes per task to leak_outputs (the
# scripted family with both outcomes), as the spec's expansion rule allows.
cd /workspace/rlp; S=scripts/flag/night
C='"name":"published","max_submits":5,"max_turns":15,"instruction":"mild"'
IDS="15 16 17 18 19 20 21 22 23 24 25 26 27 28 29"
bash $S/stage.sh ib_full2_oneoff 110 python -m rlp.flag.mt run --run ib_full2_oneoff --families ib_lcb --task-ids $IDS --hack-eps 3 --benign-eps 1 --conc 4 --active 4 --cfg "{$C,\"impossible\":\"oneoff\"}" &
bash $S/stage.sh ib_full2_conflicting 110 python -m rlp.flag.mt run --run ib_full2_conflicting --families ib_lcb --task-ids $IDS --hack-eps 3 --benign-eps 0 --conc 4 --active 4 --cfg "{$C,\"impossible\":\"conflicting\"}" &
while pgrep -f "m1b.sh" > /dev/null; do sleep 20; done
bash $S/stage.sh m1_expand 110 python -m rlp.flag.mt run --run m1 --families leak_outputs,answer_key --tasks 8 --hack-eps 8 --benign-eps 6 --conc 5 --active 5 &
wait
echo B_IB2_END

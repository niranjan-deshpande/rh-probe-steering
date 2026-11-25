#!/bin/bash
# Pod A, 5:00 am: ImpossibleBench full run on the "mild" wording (closest to the range: 4/5 conform with one honest
# outcome; plain 5/5, mild2 5/5, strict 0/9), 15 tasks x 2 impossible versions x 3 episodes + solvable once; plus a
# round-4 pilot of one more wording between mild and strict (mild3 = strict without the last sentence and "STOP").
cd /workspace/rlp; S=scripts/flag/night
C='"name":"published","max_submits":5,"max_turns":15'
bash $S/stage.sh ib_full_oneoff 170 python -m rlp.flag.mt run --run ib_full_oneoff --families ib_lcb --tasks 15 --hack-eps 3 --benign-eps 1 --conc 5 --active 5 --cfg "{$C,\"instruction\":\"mild\",\"impossible\":\"oneoff\"}" &
bash $S/stage.sh ib_full_conflicting 170 python -m rlp.flag.mt run --run ib_full_conflicting --families ib_lcb --tasks 15 --hack-eps 3 --benign-eps 0 --conc 5 --active 5 --cfg "{$C,\"instruction\":\"mild\",\"impossible\":\"conflicting\"}" &
bash $S/stage.sh ib_r4_mild3 170 python -m rlp.flag.mt run --run ib_r4_mild3 --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg "{$C,\"instruction\":\"mild3\"}" &
wait
echo A_IB_END

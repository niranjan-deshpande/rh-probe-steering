#!/bin/bash
# Usage: harvest.sh START_EPOCH [st]   Waits until START_EPOCH, stops generation and the server, then harvests the
# activations of every rollout run on this pod (and, with "st", the new single-turn completions).
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
while [ $(date +%s) -lt $1 ]; do sleep 20; done
bash $S/killall_mt.sh; bash $S/serve.sh stop
for r in $(ls cache/flag/mt | grep -v smoke); do python -m rlp.flag.mt stats --run $r > /workspace/logs/stats_$r.txt 2>&1; done
RUNS=$(ls cache/flag/mt | grep -v smoke | tr "\n" " ")
bash $S/stage.sh h2_mt 100 python -m rlp.flag.mt_harvest mt --run $RUNS
[ "$2" = st ] && bash $S/stage.sh h2_st 20 python -m rlp.flag.mt_harvest st --names np_base n1_base --tag night
echo HARVEST_END

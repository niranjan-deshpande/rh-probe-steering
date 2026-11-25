#!/bin/bash
# Pod A, 7:30 am: the first harvest took nine minutes, so the planned ImpossibleBench full run (unfinished at 7:00)
# is resumed from its checkpoints until the second harvest at 8:10 am ET. Same runs, same cfg as a_ib.sh.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
rm -f /workspace/running/server.pid
MAX_MODEL_LEN=65536 bash $S/serve.sh start plain || exit 1
C='"name":"published","max_submits":5,"max_turns":15'
bash $S/stage.sh ib_full_oneoff_b 60 python -m rlp.flag.mt run --run ib_full_oneoff --families ib_lcb --tasks 15 --hack-eps 3 --benign-eps 1 --conc 6 --active 6 --cfg "{$C,\"instruction\":\"mild\",\"impossible\":\"oneoff\"}" &
bash $S/stage.sh ib_full_conflicting_b 60 python -m rlp.flag.mt run --run ib_full_conflicting --families ib_lcb --tasks 15 --hack-eps 3 --benign-eps 0 --conc 7 --active 7 --cfg "{$C,\"instruction\":\"mild\",\"impossible\":\"conflicting\"}" &
wait
echo A_IB2_END

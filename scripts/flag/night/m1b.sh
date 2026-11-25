#!/bin/bash
# Pod B, 3:20 am replan (throughput is about 270 tok/s): the two families with mixed outcomes keep 4 hack + 6 benign
# episodes per task; save_game, release_gate (near 100% hack) and scorer_copy (0%) are cut to 2 hack + 4 benign.
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh m1_mixed 300 python -m rlp.flag.mt run --run m1 --families answer_key,leak_outputs --tasks 8 --hack-eps 4 --benign-eps 6 --conc 7 --active 7 &
bash $S/stage.sh m1_rest 300 python -m rlp.flag.mt run --run m1 --families save_game,release_gate,scorer_copy --tasks 8 --hack-eps 2 --benign-eps 4 --conc 3 --active 3 &
wait
python -m rlp.flag.mt stats --run m1 > /workspace/logs/m1_stats.txt 2>&1
echo M1_END

#!/bin/bash
# Pod B: keep only the "mild" round-3 pilot (mild2 stopped at 5/5 conform).
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh ib_r3_mild 240 python -m rlp.flag.mt run --run ib_r3_mild --families ib_lcb --tasks 8 --hack-eps 3 --benign-eps 1 --conc 3 --active 3 --cfg '{"name":"published","instruction":"mild","max_submits":5,"max_turns":15}'
echo M2R3B_END

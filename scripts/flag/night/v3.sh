#!/bin/bash
# Steering inside rollouts (25 November, afternoon): the two tampering families, the same 16 hack-variant episodes per
# family as run m1 (tasks 0-7, k 0-1, same seeds), with Sigma.probe subtracted (row 43) and Goodfire's direction
# subtracted (row 47) at every request. One server, prefix caching on, cache keyed by condition (cache_salt in mt.py).
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
T=cache/flag/night/steer_table.pt
RLP_STEER_ALLOW_PREFIX_CACHE=1 MAX_MODEL_LEN=65536 EXTRA_ARGS="--enable-prefix-caching" bash $S/serve.sh start steer $T || { echo V3_SERVER_FAILED; exit 1; }
bash $S/stage.sh v3_cachecheck 15 python scripts/flag/night/v3_cachecheck.py || { echo V3_CACHECHECK_FAILED; exit 1; }
A="--families save_game,release_gate --tasks 8 --hack-eps 2 --benign-eps 0 --api raw --conc 6 --active 6"
bash $S/stage.sh v3_sprobe 240 python -m rlp.flag.mt run --run v3_sprobe_sub --vid 43 $A &
bash $S/stage.sh v3_gf 240 python -m rlp.flag.mt run --run v3_gf_sub --vid 47 $A &
wait
for r in v3_sprobe_sub v3_gf_sub; do python -m rlp.flag.mt stats --run $r; done > /workspace/logs/v3_stats.txt 2>&1
echo V3_END

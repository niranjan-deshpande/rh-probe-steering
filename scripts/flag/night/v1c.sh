#!/bin/bash
# Pod A: after v1b.sh ends, rerun the fitting group (its first try lacked data/manif), then free the GPU and serve plain.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
until grep -qx V1B_END /workspace/logs/v1b.log; do sleep 20; done
rm -f /workspace/failed/v1_fit
bash $S/stage.sh v1_fit 60 python -m rlp.flag.gen_multi run --table cache/flag/night/steer_table.pt --prefix n1_ --conds @fitting --preset fitting --label
bash $S/serve.sh stop
MAX_MODEL_LEN=65536 bash $S/serve.sh start plain
echo V1C_END

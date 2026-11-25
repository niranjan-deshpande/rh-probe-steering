#!/bin/bash
# Step 4 (GPU part): harvest synthetic activations (with written reasoning), rebuild read-out directions, train the
# gradient-descent direction, and read our own rollouts. Aborts on the first failed step. Logs to logs/flag/.
# Needs cache/flag/rollouts/<NAME>.jsonl (from rlp.flag.agent) and the vLLM server stopped so the GPU is free.
# usage: bash scripts/flag/run_step4.sh [NAME=pilot] [STEPS=120]
set -euo pipefail
source /workspace/rlp/scripts/flag/env.sh
NAME=${1:-pilot}; STEPS=${2:-120}
mkdir -p logs/flag
python -m rlp.flag.harvest              2>&1 | tee logs/flag/8_harvest.log
python -m rlp.flag.dirs                 2>&1 | tee logs/flag/9_dirs.log
python -m rlp.flag.train --steps $STEPS 2>&1 | tee logs/flag/10_train.log
python -m rlp.flag.read --rollouts $NAME --own --max-tokens 40000 2>&1 | tee logs/flag/11_read.log
echo STEP4_DONE

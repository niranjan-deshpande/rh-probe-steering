#!/bin/bash
# Step 3: harvest synthetic activations, build read-out directions, train the gradient-descent direction, read
# Goodfire's labelled rollouts, score. Aborts on the first failed step. Logs to logs/flag/.
# usage: bash scripts/flag/run_step3.sh [N_ROLLOUTS=150] [STEPS=120]
set -euo pipefail
source /workspace/rlp/scripts/flag/env.sh
N=${1:-150}; STEPS=${2:-120}
mkdir -p logs/flag
python -m rlp.flag.test_flag            2>&1 | tee logs/flag/0_test.log | tail -3
python -m rlp.flag.harvest              2>&1 | tee logs/flag/1_harvest.log
python -m rlp.flag.dirs                 2>&1 | tee logs/flag/2_dirs.log
python -m rlp.flag.train --steps $STEPS 2>&1 | tee logs/flag/3_train.log
python -m rlp.flag.read --n $N          2>&1 | tee logs/flag/4_read.log
python -m rlp.flag.score                2>&1 | tee logs/flag/5_score.log
echo STEP3_DONE

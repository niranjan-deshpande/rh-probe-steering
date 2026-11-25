#!/bin/bash
# Stage 1 of the on-policy run (2026-11-24): the frozen Qwen3.8-27B sampled 8 times per prompt on the environments
# where it hacks unprompted at an intermediate rate. Fitting environments feed the vector's loss; held-out ones are
# only scored. Assumes serve_q38.sh is up. Output: cache/flag/base_rate/onp/.
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache
cd /workspace/rlp
G="python -m rlp.flag.base_rate gen --name onp --model Qwen/Qwen3.8-27B --samples 8 --workers ${WORKERS:-96}"
# fitting environments (40 items: 30 fit, 10 validation)
$G --n 40 --behs length_pad quiz_key --conds neutral tempt --max-tokens 6000
$G --n 40 --behs leak --conds tempt --max-tokens 8000
$G --n 40 --behs hardcode --conds neutral tempt --max-tokens 9000
# held-out environments (30 items)
$G --n 30 --behs save_tamper fake_pass chess_key --conds tempt --max-tokens 8000
$G --n 30 --behs fake_cite --conds neutral --max-tokens 9000
python -m rlp.flag.base_rate label --name onp --workers 16
echo ONPOLICY_GEN_DONE

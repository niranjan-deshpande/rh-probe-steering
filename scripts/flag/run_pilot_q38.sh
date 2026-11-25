#!/bin/bash
# Base-rate pilot on Qwen3.8-27B: single-turn hack environments first (cheap), then the 90 ImpossibleBench episodes
# that the Qwen3-14B pilot ran. Assumes serve_q38.sh is up. Stops at the first failure.
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache
cd /workspace/rlp
M=Qwen/Qwen3.8-27B
python -m rlp.flag.base_rate gen --name q38 --model $M --n 30 --conds neutral tempt --workers 64
python -m rlp.flag.base_rate label --name q38 --workers 16
python -m rlp.flag.agent --out pilot_q38 --model $M --n-tasks 15 --workers 48
echo PILOT_Q38_DONE

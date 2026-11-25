#!/bin/bash
# Steering block of 2026-11-24 (transformed detector, random-direction null). One condition per vector, on prompts
# no direction or earlier steering run has seen: items 60-89 of two held-out environments and items 70-99 of three
# fitting environments. fake_cite and hardcode are left out (6,000 and 4,000 tokens per completion). The cap is 5,000
# tokens for every condition. Labels are computed after all generation, so the GPU does not idle.
# Usage: bash run_onp_steer2.sh <vec> [<vec> ...]     Output: cache/flag/base_rate/s2_<vec>/
set -uo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache HF_HUB_OFFLINE=1
cd /workspace/rlp
for V in "$@"; do
  python -m rlp.flag.onp setvec --vec $V || { echo "SETVEC_FAILED $V"; continue; }
  G="python -m rlp.flag.base_rate gen --name s2_$V --model Qwen/Qwen3.8-27B --n ${N:-30} --workers 60 --max-tokens 5000"
  $G --start 60 --behs save_tamper fake_pass --conds tempt &
  $G --start 70 --behs length_pad leak --conds tempt &
  $G --start 70 --behs quiz_key --conds neutral &
  wait
  (python -m rlp.flag.base_rate label --name s2_$V --workers 8 > /workspace/label_s2_$V.log 2>&1 &)
  echo "STEER_COND_DONE $V $(date +%H:%M:%S)"
done
echo ONP_STEER2_DONE

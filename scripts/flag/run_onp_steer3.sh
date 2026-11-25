#!/bin/bash
# Second steering test of 2026-11-24: the same vectors on three environments built that day, which no detector fit,
# no steering fit and no earlier steering run has seen (items 40-69; items 0-39 were sampled for detection only).
# Usage: bash run_onp_steer3.sh <vec> [<vec> ...]     Output: cache/flag/base_rate/s3_<vec>/
set -uo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache HF_HUB_OFFLINE=1
cd /workspace/rlp
for V in "$@"; do
  python -m rlp.flag.onp setvec --vec $V || { echo "SETVEC_FAILED $V"; continue; }
  G="python -m rlp.flag.base_rate gen --name s3_$V --model Qwen/Qwen3.8-27B --n 30 --start 40 --workers 50 --max-tokens 5000 --conds tempt"
  $G --behs ckpt_swap thresh_tamper &
  $G --behs ref_copy &
  wait
  (python -m rlp.flag.base_rate label --name s3_$V --workers 8 > /workspace/label_s3_$V.log 2>&1 &)
  echo "STEER_COND_DONE $V $(date +%H:%M:%S)"
done
sleep 45
echo ONP_STEER3_DONE

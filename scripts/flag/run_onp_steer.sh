#!/bin/bash
# Stage 4 of the on-policy run: free generation with a vector added, hacks counted by the program detectors.
# Usage: bash run_onp_steer.sh <vec> [<vec> ...]   with vectors in cache/flag/onp/steer/<vec>.pt ("base" is zero).
# Assumes serve_q38_steer.sh is up. Prompts are ones no vector was fitted on: items 40-69 of the fitting
# environments, items 30-59 of the held-out ones, and the first 30 of four environments with a base rate near zero.
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache HF_HUB_OFFLINE=1
cd /workspace/rlp
for V in "$@"; do
  python -m rlp.flag.onp setvec --vec $V
  G="python -m rlp.flag.base_rate gen --name st_$V --model Qwen/Qwen3.8-27B --n 30 --workers ${WORKERS:-120}"
  $G --start 40 --behs length_pad leak --conds tempt --max-tokens 6000 &
  $G --start 40 --behs quiz_key --conds neutral --max-tokens 6000 &
  $G --start 40 --behs hardcode --conds neutral --max-tokens 8000 &
  $G --start 30 --behs save_tamper fake_pass --conds tempt --max-tokens 8000 &
  $G --start 30 --behs fake_cite --conds neutral --max-tokens 8000 &
  $G --start 0 --behs false_report ticket_close reward_tamper --conds tempt --max-tokens 6000 &
  $G --start 0 --behs runtests --conds neutral --max-tokens 6000 &
  wait
  python -m rlp.flag.base_rate label --name st_$V --workers 16
  echo "STEER_COND_DONE $V"
done
echo ONP_STEER_DONE

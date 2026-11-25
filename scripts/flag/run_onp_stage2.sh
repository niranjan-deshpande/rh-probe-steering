#!/bin/bash
# Stage 2 of the on-policy run: read activations of the frozen model on the Stage 1 completions, then closed-form
# directions and detection scores. The generation server must be stopped first (it holds the GPU).
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false RLP_CACHE=/workspace/rlp/cache HF_HUB_OFFLINE=1
cd /workspace/rlp
python -m rlp.flag.onp harvest
python -m rlp.flag.onp dirs
echo ONP_STAGE2_DONE

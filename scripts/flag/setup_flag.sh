#!/bin/bash
# Pod setup for rlp.flag: venv on container disk (reusing the image's torch), transformers pinned to the version the
# CPU tests ran on, Qwen3-14B download. No secrets are needed: every model and dataset used is public.
set -euo pipefail
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pipcache HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/rlp
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q "transformers==5.22.2" accelerate hf_transfer huggingface_hub numpy
python - <<'P'
import torch, transformers
print("torch", torch.__version__, "tf", transformers.__version__, torch.cuda.get_device_name())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().abs().mean()) > 0)
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors", "*.txt"]))
P
echo SETUP_DONE

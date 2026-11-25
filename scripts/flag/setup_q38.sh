#!/bin/bash
# Pod setup for the Qwen3.8-27B base-rate pilot (generation only): a client venv for rlp.flag / rlp.manif and a
# separate vLLM venv, both on the container disk, then the model download (56 GB). No secrets are needed.
set -euo pipefail
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pipcache HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/rlp
python3 -c "
import torch
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
print('gpu', torch.cuda.get_device_name(), 'matmul ok', float((x @ x).float().abs().mean()) > 0)"
python3 -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q --upgrade pip
/root/venv/bin/pip install -q "transformers>=5.8" hf_transfer huggingface_hub numpy openai pandas pyarrow chess accelerate scikit-learn flash-linear-attention
python3 -m venv /root/vllm
/root/vllm/bin/pip install -q --upgrade pip uv
/root/vllm/bin/uv pip install -q --python /root/vllm/bin/python "vllm>=0.17" "transformers>=5.8" hf_transfer
/root/vllm/bin/python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'tf', transformers.__version__)"
/root/venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.8-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja']))"
echo Q38_SETUP_DONE

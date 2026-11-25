#!/bin/bash
# Second venv for the vLLM server (it pins its own torch / transformers); the rlp.flag venv gets the client packages.
set -euo pipefail
export HF_HOME=/root/hf PIP_CACHE_DIR=/root/pipcache
/root/venv/bin/pip install -q openai pandas pyarrow
python3 -m venv /root/vllm
/root/vllm/bin/pip install -q --upgrade pip uv
/root/vllm/bin/uv pip install -q --python /root/vllm/bin/python "vllm==0.11.0" "transformers>=4.56,<5"
/root/vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)"
echo VLLM_SETUP_DONE

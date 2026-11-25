#!/bin/bash
# Raw-completions server for Qwen3.8-27B, text only (prompts are rendered by rlp.flag.q38, so no chat template or
# tool parser is used). Needs vLLM 0.17 or later.
export HF_HOME=/root/hf
exec /root/vllm/bin/vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --language-model-only --max-model-len 40000 \
  --gpu-memory-utilization 0.92 --port 8000 --max-num-seqs ${MAX_SEQS:-64}

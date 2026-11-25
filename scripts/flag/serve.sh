#!/bin/bash
# Raw-completions server for the agent harness (no chat template or tool parser: prompts are rendered by rlp.flag).
export HF_HOME=/root/hf VLLM_ENABLE_V1_MULTIPROCESSING=0
exec /root/vllm/bin/vllm serve Qwen/Qwen3-14B --dtype bfloat16 --max-model-len 40000 --gpu-memory-utilization 0.92 --port 8000 --max-num-seqs 64

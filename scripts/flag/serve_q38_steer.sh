#!/bin/bash
# Steered raw-completions server for Qwen3.8-27B. Needs scripts/flag/patch_vllm_steer.py applied to the vLLM venv.
# Usage: LAYER=<block index> bash serve_q38_steer.sh   (vector file: /workspace/steer_current.pt, swapped at run time)
# 2026-11-24: memory utilisation raised from 0.92 (vLLM 0.34 warned it kept 4.8 GiB out of the KV cache) and
# max-num-seqs lowered to what the cache holds at these lengths (about 64-80 sequences on an A100 80 GB).
export HF_HOME=/root/hf RLP_STEER_LAYER=${LAYER:?} RLP_STEER_FILE=/workspace/steer_current.pt
exec /root/vllm/bin/vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --language-model-only --max-model-len 20000 \
  --gpu-memory-utilization ${GPU_UTIL:-0.97} --port 8000 --max-num-seqs ${MAX_SEQS:-80} --no-enable-prefix-caching

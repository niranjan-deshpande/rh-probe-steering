# Shared environment for the night scripts ON THE POD. Source it; it does not change directory or shell options.
# Every path can be overridden (the local tests point NIGHT_ROOT at a temporary directory).
export NIGHT_ROOT="${NIGHT_ROOT:-/workspace}"
export NIGHT_REPO="${NIGHT_REPO:-$NIGHT_ROOT/rlp}"
export NIGHT_VENV="${NIGHT_VENV:-/root/vllm}"
export NIGHT_LOGS="$NIGHT_ROOT/logs" NIGHT_DONE="$NIGHT_ROOT/done" NIGHT_FAILED="$NIGHT_ROOT/failed"
export NIGHT_RUN="$NIGHT_ROOT/running" NIGHT_WD="$NIGHT_ROOT/watchdog"
export NIGHT_MODEL="${NIGHT_MODEL:-Qwen/Qwen3.8-27B}"
export HF_HOME="${HF_HOME:-/root/hf}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false
export RLP_CACHE="${RLP_CACHE:-$NIGHT_REPO/cache}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
case ":$PATH:" in *":$NIGHT_VENV/bin:"*) ;; *) export PATH="$NIGHT_VENV/bin:$PATH" ;; esac

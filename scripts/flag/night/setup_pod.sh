#!/bin/bash
# Idempotent pod setup for the night of 2026-11-24/25. No secrets, no .env: the model and every package are public.
#
# Usage (on the pod, after push.sh):
#   setsid -f bash /workspace/rlp/scripts/flag/night/setup_pod.sh < /dev/null > /dev/null 2>&1
#   # poll:  tail -n 1 /workspace/logs/setup.log | grep -x SETUP_OK
# Everything goes to /workspace/logs/setup.log (appended). The last line is exactly SETUP_OK on success, or starts
# with SETUP_FAILED. Re-running after success takes a few seconds and prints SETUP_OK again.
#
# What it does:
#   1. starts the Qwen3.8-27B download (56 GB) in the background, into HF_HOME=/root/hf (log: logs/download.log);
#   2. builds ONE venv at /root/vllm: vLLM 0.34.0 (brings its own torch and the matching triton), transformers
#      5.22.2 for the reader, flash-linear-attention 0.6.0 and the client packages. All Python on the pod runs here;
#   3. tests the GPU with a matrix multiply inside that venv and reports whether the fast kernels import;
#   4. waits for the download and checks that the model resolves offline.
# Knobs: VLLM_VERSION, TF_VERSION, FLA_VERSION, DOWNLOAD_TIMEOUT_MIN (default 45), SETUP_NO_REDIRECT=1 (log to stdout).
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HUB_OFFLINE=0
source "$HERE/env.sh"
export HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1 PIP_CACHE_DIR=/root/pipcache UV_CACHE_DIR=/root/uvcache
VLLM_VERSION="${VLLM_VERSION:-0.34.0}"; TF_VERSION="${TF_VERSION:-5.22.2}"; FLA_VERSION="${FLA_VERSION:-0.6.0}"
V="$NIGHT_VENV"; DL=/root/dl; M=/root/.night_setup
mkdir -p "$NIGHT_LOGS" "$NIGHT_DONE" "$NIGHT_FAILED" "$NIGHT_RUN" "$NIGHT_WD" "$M" "$HF_HOME"
if [ "${SETUP_NO_REDIRECT:-0}" != 1 ]; then exec >> "$NIGHT_LOGS/setup.log" 2>&1; fi
say() { echo "[setup $(date -u +%H:%M:%SZ)] $*"; }
trap 'rc=$?; trap - ERR; echo "SETUP_FAILED rc=$rc line=$LINENO"; exit $rc' ERR
exec 9> "$M/lock"
if command -v flock > /dev/null; then
  flock -n 9 || { trap - ERR; echo "SETUP_FAILED another setup_pod.sh is running"; exit 1; }
fi
say "start (vllm $VLLM_VERSION, transformers $TF_VERSION, fla $FLA_VERSION)"
df -h /root "$NIGHT_ROOT" | sed 's/^/  /' || true
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  gpu: /' || true

# --- 1. model download, in the background, from a tiny venv that is ready in seconds ------------------------------
alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2> /dev/null; }
start_download() {
  if [ ! -x "$DL/bin/python" ]; then
    python3 -m venv "$DL"
    "$DL/bin/pip" install -q --upgrade pip
    "$DL/bin/pip" install -q huggingface_hub hf_transfer
  fi
  cat > "$M/download.py" <<'P'
import sys, time
from huggingface_hub import snapshot_download
for attempt in range(4):
    try:
        print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.md", "LICENSE", ".gitattributes"]), flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"download attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(20)
else:
    sys.exit(1)
P
  setsid bash -c '"$1/bin/python" -u "$2/download.py" "$3" && touch "$2/download_ok" && echo DOWNLOAD_OK || echo DOWNLOAD_FAILED' \
    _ "$DL" "$M" "$NIGHT_MODEL" > "$NIGHT_LOGS/download.log" 2>&1 < /dev/null 9>&- &
  echo $! > "$M/download.pid"
  say "download of $NIGHT_MODEL started (pid $(cat "$M/download.pid"), log $NIGHT_LOGS/download.log)"
}
if [ -f "$M/download_ok" ]; then say "download: already complete"
elif alive "$M/download.pid"; then say "download: already running (pid $(cat "$M/download.pid"))"
else start_download; fi

# --- 2. the one venv -------------------------------------------------------------------------------------------------
EXTRAS="hf_transfer huggingface_hub accelerate scikit-learn scipy numpy pandas pyarrow openai safetensors einops chess python-dotenv datasets"
if [ -f "$M/venv_ok" ] && [ "$(cat "$M/venv_ok")" = "$VLLM_VERSION $TF_VERSION $FLA_VERSION" ]; then
  say "venv: already built"
else
  [ -x "$V/bin/python" ] || python3 -m venv "$V"
  "$V/bin/pip" install -q --upgrade pip uv
  UVI=("$V/bin/uv" pip install -q --python "$V/bin/python")
  # One resolve with every pin, so nothing can move vLLM's torch / triton afterwards.
  if "${UVI[@]}" "vllm==$VLLM_VERSION" "transformers==$TF_VERSION" "flash-linear-attention==$FLA_VERSION" $EXTRAS; then
    say "venv: pinned resolve ok"
  else
    say "SETUP_WARN pinned resolve failed; installing vllm==$VLLM_VERSION with its own transformers, then FLA without dependencies"
    "${UVI[@]}" "vllm==$VLLM_VERSION" "transformers>=5.8" $EXTRAS
    "${UVI[@]}" --no-deps "flash-linear-attention==$FLA_VERSION" || say "SETUP_WARN flash-linear-attention did not install"
  fi
  echo "$VLLM_VERSION $TF_VERSION $FLA_VERSION" > "$M/venv_ok"
fi

# --- 3. checks inside the venv ---------------------------------------------------------------------------------------
"$V/bin/python" - "$VLLM_VERSION" <<'P'
import sys
from importlib.metadata import version
import torch
names = ["vllm", "torch", "triton", "transformers", "flash-linear-attention", "numpy", "accelerate", "scikit-learn"]
vs = {}
for n in names:
    try:
        vs[n] = version(n)
    except Exception:  # noqa: BLE001
        vs[n] = "MISSING"
print("versions:", " ".join(f"{k}={v}" for k, v in vs.items()))
assert vs["vllm"].startswith(sys.argv[1]), f"vllm {vs['vllm']}, wanted {sys.argv[1]}"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
assert float((x @ x).float().abs().mean()) > 0
print("gpu:", torch.cuda.get_device_name(), "matmul ok")
P
# The fast kernels are optional (the reader falls back to the slow reference ones); fla_check.py is the real test.
if "$V/bin/python" -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2> "$M/fla_import.err"; then
  say "FLA_IMPORT_OK"
else
  say "FLA_IMPORT_FAILED: $(tail -n 3 "$M/fla_import.err" | tr '\n' ' ')"
fi

# --- 4. wait for the model -------------------------------------------------------------------------------------------
t0=$(date +%s); lim=$(( ${DOWNLOAD_TIMEOUT_MIN:-45} * 60 )); restarted=0
while [ ! -f "$M/download_ok" ]; do
  if ! alive "$M/download.pid"; then
    sleep 2
    [ -f "$M/download_ok" ] && break
    if [ "$restarted" = 0 ]; then
      restarted=1; say "download process ended without success; restarting once"; start_download
    else
      say "download failed twice: $(tail -n 3 "$NIGHT_LOGS/download.log" | tr '\n' ' ')"; false
    fi
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$lim" ]; then say "download not finished after ${DOWNLOAD_TIMEOUT_MIN:-45} min"; false; fi
  say "waiting for the download: $(du -sh "$HF_HOME" 2> /dev/null | cut -f1) so far"
  sleep 30
done
HF_HUB_OFFLINE=1 "$V/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
print('model at', snapshot_download(sys.argv[1], local_files_only=True))" "$NIGHT_MODEL"
date -u +%FT%TZ > "$NIGHT_DONE/setup"
trap - ERR
echo SETUP_OK

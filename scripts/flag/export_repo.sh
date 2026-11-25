#!/bin/bash
# Copy the rlp/flag work (code, results docs, small result files, the directions) into the checkout of the GitHub
# repository rh-probe-steering. Never copies .env, .venv, caches of activations or transcripts of unfinished runs.
# Usage: bash scripts/flag/export_repo.sh /path/to/rh-probe-steering     (then commit and push there)
set -euo pipefail
D="${1:?target checkout}"; cd "$(dirname "$0")/../.."
R="rsync -rt --prune-empty-dirs --exclude __pycache__ --exclude .DS_Store --exclude .env*"
mkdir -p "$D/rlp" "$D/scripts" "$D/docs" "$D/site" "$D/results/night" "$D/results/onp" "$D/results/mt" "$D/directions"
cp rlp/__init__.py rlp/config.py "$D/rlp/"
$R rlp/flag "$D/rlp/"
$R scripts/flag "$D/scripts/"
cp docs/INFRA.md "$D/docs/"; cp site/flag-direction.html "$D/site/"
$R --include '*.json' --exclude '*' cache/flag/ "$D/results/"
$R --include '*.json' --exclude '*' cache/flag/onp/ "$D/results/onp/"
$R --include '*.json' --include 'steer_table.pt*' --exclude '*' cache/flag/night/ "$D/results/night/"
for r in v3_sprobe_sub; do mkdir -p "$D/results/mt/$r"; $R --include '*.jsonl' --exclude '*' cache/flag/mt/$r/ "$D/results/mt/$r/"; done
cp logs/flag/night/steer_samples.md "$D/results/night/" 2>/dev/null || true
cp cache/flag/onp/dirs_cf_36.npz cache/flag/onp/write_cands_36.npz "$D/directions/"
cp cache/flag/gf27/dirs_36.npz "$D/directions/goodfire_dirs_36.npz"
echo EXPORT_OK

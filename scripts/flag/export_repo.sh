#!/bin/bash
# Copy the rlp/flag work (code, results docs, small result files, the directions) into the checkout of the GitHub
# repository rh-probe-steering. Never copies .env, .venv, caches of activations or transcripts of unfinished runs.
# Usage: bash scripts/flag/export_repo.sh /path/to/rh-probe-steering     (then commit and push there)
set -euo pipefail
D="${1:?target checkout}"; cd "$(dirname "$0")/../.."
R="rsync -rt --prune-empty-dirs --exclude __pycache__ --exclude .DS_Store --exclude .env*"
mkdir -p "$D/rlp" "$D/scripts" "$D/docs" "$D/site" "$D/results/night" "$D/results/onp" "$D/results/mt" "$D/directions"
cp rlp/__init__.py rlp/config.py "$D/rlp/"
mkdir -p "$D/rlp/manif" "$D/data/flag" "$D/results/night/v5" "$D/results/transcripts"
# The single-turn base environments live in rlp/manif/envs.py; nine flag modules import them. The rest of rlp/manif is not exported.
cp rlp/manif/__init__.py rlp/manif/envs.py "$D/rlp/manif/"
cp data/flag/*.json data/flag/*.jsonl "$D/data/flag/"
$R rlp/flag "$D/rlp/"
$R scripts/flag "$D/scripts/"
cp docs/INFRA.md "$D/docs/"; cp site/flag-direction.html "$D/site/"
$R --include '*.json' --exclude '*' cache/flag/ "$D/results/"
$R --include '*.json' --exclude '*' cache/flag/onp/ "$D/results/onp/"
$R --include '*.json' --include 'steer_table.pt*' --exclude '*' cache/flag/night/ "$D/results/night/"
# Finished rollout runs only (25 November). Runs still in progress are added by name when they end.
for r in m1 v3_sprobe_sub v3_gf_sub v4_base v4_sig5_sub v4_sig7_sub v4_iso0_sub v4_sprobe_benign v4_sprobe_other v5_spec_prompt; do mkdir -p "$D/results/mt/$r"; $R --include '*.jsonl' --exclude '*' cache/flag/mt/$r/ "$D/results/mt/$r/"; done
cp logs/flag/night/steer_samples.md "$D/results/night/" 2>/dev/null || true
cp logs/flag/night/rollsteer_transcripts.md "$D/results/transcripts/" 2>/dev/null || true
# Single-turn checks of 25 November: summaries and the stories. The benchmark questions and completions (items.jsonl,
# cap_*.jsonl) hold third-party test items and are not exported.
cp cache/flag/night/v5/cap_summary.json cache/flag/night/v5/story_*.json* "$D/results/night/v5/"
cp cache/flag/night/steer_table_v8.pt cache/flag/night/steer_table_v8.pt.json cache/flag/night/v8_cands.npz cache/flag/night/v8_cands.json "$D/results/night/" 2>/dev/null || true
cp cache/flag/onp/dirs_cf_36.npz cache/flag/onp/write_cands_36.npz "$D/directions/"
cp cache/flag/gf27/dirs_36.npz "$D/directions/goodfire_dirs_36.npz"
echo EXPORT_OK

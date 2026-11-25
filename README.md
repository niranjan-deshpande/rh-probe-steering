# rh-probe-steering

A linear direction for reward hacking in Qwen3.8-27B (input of block 36) that both detects hacking and, when
subtracted, removes it. The direction is a logistic probe `w` trained on on-policy rollouts, and the steering vector is
`Σ·w` (the probe multiplied by the activation covariance). It is compared against Goodfire's published direction.

This is a snapshot of work in progress. Numbers in `rlp/flag/results/` are the record.

## Where to look

| What | Where |
|---|---|
| Question, status, how to run | `rlp/flag/README.md` |
| Results by topic (start with the latest) | `rlp/flag/results/`, index in `rlp/flag/RESULTS.md` |
| Overnight run of 25 Nov 2026 (detection and steering against Goodfire's) | `rlp/flag/results/07_overnight.md` |
| Charts page | `site/flag-direction.html` |
| Method | `rlp/flag/METHOD.md` |
| The directions (probe, `Σ·probe` candidates, Goodfire's at block 36) | `directions/` |
| Steering table used by the patched vLLM server (68 rows, norm 20) | `results/night/steer_table.pt` and `.pt.json` |
| Small result files (JSON) | `results/` (copied from `cache/flag/` in the working tree) |
| Steered multi-turn rollouts, `Σ·probe` subtracted (32 episodes) | `results/mt/v3_sprobe_sub/` |
| Matched single-turn completions under steering | `results/night/steer_samples.md` |
| Pod scripts, vLLM steering patch | `scripts/flag/night/` |
| Pods, models, judge | `docs/INFRA.md` |

## Not included

Activation caches, the covariance matrix (105 MB), full transcripts of the unsteered rollouts and API keys are not in
the repository. Code paths refer to `cache/flag/...`; the files under `results/` and `directions/` are copies from there.
Runs still in progress on 25 Nov (random-vector controls, benign variants, capability benchmarks, prompt baseline) will
be added when they finish.

## Running

From the repo root, with a virtual environment at `.venv`: `.venv/bin/python -m rlp.flag.<module>`.

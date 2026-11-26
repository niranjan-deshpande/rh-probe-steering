# rh-probe-steering

A linear direction for reward hacking in Qwen3.8-27B (input of block 36) that both detects hacking and, when
subtracted, removes it. The direction is a logistic probe `w` trained on on-policy rollouts, and the steering vector is
`Σ·w` (the probe multiplied by the activation covariance). It is compared against Goodfire's published direction.

This is a snapshot of work in progress. Numbers in `rlp/flag/results/` are the record.

**New here? Read [`rlp/flag/HANDOFF.md`](rlp/flag/HANDOFF.md) first.** It says what the goal is now (steering, not
detection), what has and has not been tested, where every environment came from, and how this repository is laid
out. Much of the older documentation still describes earlier goals, and the handoff file says which parts.

## Where to look

| What | Where |
|---|---|
| Orientation for a new collaborator | `rlp/flag/HANDOFF.md` |
| Question, status, how to run | `rlp/flag/README.md` |
| Steering inside multi-turn rollouts, capability costs (25 Nov 2026, daytime) | `rlp/flag/results/08_rollout_steering.md` |
| The same task under each steering condition, trimmed for reading | `results/transcripts/rollsteer_transcripts.md` |
| Results by topic (start with the latest) | `rlp/flag/results/`, index in `rlp/flag/RESULTS.md` |
| Overnight run of 25 Nov 2026 (detection and steering against Goodfire's) | `rlp/flag/results/07_overnight.md` |
| Charts page | `site/flag-direction.html` |
| Method | `rlp/flag/METHOD.md` |
| The directions (probe, `Σ·probe` candidates, Goodfire's at block 36) | `directions/` |
| Steering table used by the patched vLLM server (68 rows, norm 20) | `results/night/steer_table.pt` and `.pt.json` |
| Small result files (JSON) | `results/` (copied from `cache/flag/` in the working tree) |
| Rollout transcripts of every finished run (unsteered `m1` and `v4_base`, steered, controls, benign, prompt baseline) | `results/mt/<run>/` |
| Candidate vectors and steering table of the selection run started on the night of 25 Nov | `results/night/v8_cands.*`, `steer_table_v8.pt` |
| Matched single-turn completions under steering | `results/night/steer_samples.md` |
| Pod scripts, vLLM steering patch | `scripts/flag/night/` |
| Pods, models, judge | `docs/INFRA.md` |

## Not included

Activation caches, the covariance matrix (105 MB), the source data of the environments and benchmarks, completions on
benchmark test items and API keys are not in the repository. Code paths refer to `cache/flag/...`; the files under
`results/` and `directions/` are copies from there (`rlp/flag/HANDOFF.md`, section 9, gives the mapping). The run
started on the night of 25 Nov (vector selection by steering, strength curve, difficulty curve) is in progress and
its results are not in this snapshot.

## Running

From the repo root, with a virtual environment at `.venv`: `.venv/bin/python -m rlp.flag.<module>`.

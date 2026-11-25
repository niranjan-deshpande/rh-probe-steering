# 03. Step 4: our own Qwen3-14B rollouts on ImpossibleBench (2026-11-15)

**Question.** When the rollouts come from the reader model itself and contain attempted and enacted hacks, does our
direction flag hack passages better than Goodfire's recipe?
**Short answer.** It roughly ties. In matched pairs our gradient-descent direction is ahead by +0.05 (−0.05, +0.13),
which is not distinguishable from zero with 36 rollouts. The logistic probe is the most consistent candidate.
**Code.** `rlp/flag/agent.py`, `own.py`, `read.py`, `matched.py`, `train.py`. Rerun: `scripts/flag/serve.sh` and
`python -m rlp.flag.agent` on a pod, `scripts/flag/run_step4.sh`, then `own --judge` and `matched --own` locally.
**Files.** `cache/flag/rollouts/pilot.jsonl`, `cache/flag/judge/own_pilot.json`, `cache/flag/read/pilot/`,
`cache/flag/matched_pilot.json`, `cache/flag/causal_eval.json`, `cache/flag/dirs_*.npz`, `logs/flag/7_pilot.log` to
`11_read.log`.

## What changed from step 3

1. **Own rollouts.** `agent.py` runs Qwen3-14B with thinking on, through a vLLM raw-completions server, on
   Impossible-LiveCodeBench. It re-implements ImpossibleBench's two scaffolds with their prompts and limits
   (safety-research/impossiblebench at commit 061dc3d). In the minimal scaffold the model answers with a code block
   and is shown the test error, for up to 10 attempts. In the tools scaffold it has bash, python, text-editor, think
   and submit tools, with up to 10 submissions and 50 messages. Each task has three versions: `original` (solvable),
   and `oneoff` and `conflicting` (the tests cannot be passed honestly).
2. **Written reasoning** was added to the training transcripts ([01_data_and_judge.md](01_data_and_judge.md)), and a
   `reason` harvest was added.
3. **Whole-turn activations** became the primary harvest. The chosen layer is 24.

## The pilot

15 tasks × 3 versions × 2 scaffolds = 90 episodes. `gpt-6.1-terra` judged every passage once.

| Scaffold | Version | Episodes | Tests passed | Rollouts flagged | Hack passages (considered / attempted / enacted) |
|---|---|---|---|---|---|
| minimal | conflicting | 15 | 0 | 7 | 67 (26 / 10 / 31) |
| minimal | oneoff | 15 | 2 | 12 | 52 (15 / 10 / 27) |
| minimal | original | 15 | 4 | 6 | 47 (9 / 9 / 29) |
| tools | conflicting | 15 | 0 | 6 | 29 (14 / 4 / 11) |
| tools | oneoff | 15 | 1 | 7 | 46 (20 / 6 / 20) |
| tools | original | 15 | 2 | 6 | 34 (6 / 4 / 24) |

In total there are 275 hack passages in 44 of 90 rollouts, 185 of them attempted or enacted, against 2,014 clean
passages in the 87 rollouts that could be read (three exceeded the context). By channel: reasoning 118, content 105,
tool calls 52. By category: special-casing the tests 185, misrepresented compliance 82, test modification 1, other 7. This is far more
attempted-or-enacted material than step 3 had (25 passages).

Two things to keep in mind about these labels:

- The model is a weak solver. It passed the tests in 9 of 90 episodes, and in only 3 of 60 impossible-version
  episodes. So "enacted" mostly means it wrote special-cased code that still failed.
- The judge flagged hacks on the solvable `original` tasks about as often as on the impossible ones (12 of 30
  rollouts against 32 of 60). That is plausible for a model that special-cases when it is stuck, but nobody has read
  those passages to confirm the labels.

## Matched-pair detection (`matched.py`)

Mean-pooling, layer 24. 132 of the 275 hack passages have a matched clean passage, in 36 rollouts.

| Subset | Hack passages | Goodfire recipe (`turn:gf_pool`) | `turn:dom_ours` | `turn:probe_ours` | `gd_mean` | gd vs Goodfire | Random p95 |
|---|---|---|---|---|---|---|---|
| Matched | 132 | 0.67 (0.55, 0.81) | 0.69 (0.56, 0.81) | 0.75 (0.65, 0.85) | 0.72 (0.59, 0.85) | +0.05 (−0.05, +0.13) | 0.71 |
| Matched + position | 67 | 0.62 (0.45, 0.77) | 0.64 (0.47, 0.79) | 0.69 (0.51, 0.84) | 0.65 (0.47, 0.81) | +0.03 (−0.06, +0.11) | 0.65 |
| Attempted or enacted only | 79 | 0.73 (0.58, 0.84) | 0.74 (0.60, 0.85) | 0.77 (0.64, 0.88) | 0.80 (0.66, 0.89) | +0.06 (−0.08, +0.19) | 0.70 |
| Reasoning channel | 59 | 0.60 (0.40, 0.81) | 0.61 (0.40, 0.82) | 0.72 (0.57, 0.88) | 0.61 (0.41, 0.81) | +0.01 (−0.04, +0.07) | 0.76 |
| Content + tool calls | 73 | 0.73 (0.56, 0.85) | 0.75 (0.59, 0.86) | 0.77 (0.65, 0.87) | 0.82 (0.69, 0.91) | +0.09 (−0.06, +0.24) | 0.70 |
| Minimal scaffold | 56 | 0.79 (0.65, 0.88) | 0.80 (0.65, 0.89) | 0.75 (0.62, 0.85) | 0.80 (0.65, 0.90) | +0.02 (−0.14, +0.16) | 0.66 |
| Tools scaffold | 76 | 0.58 (0.42, 0.79) | 0.60 (0.43, 0.80) | 0.74 (0.60, 0.89) | 0.67 (0.50, 0.85) | +0.08 (+0.02, +0.16) | 0.77 |

What we take from this:

- **Against step 3, the sign changed.** The trained direction was behind Goodfire's recipe by 0.06 there and is ahead
  by 0.05 here. The interval includes zero, so this is not a confirmed win.
- **The scaffolds differ.** On the minimal scaffold every direction does well and they are tied. On the tools scaffold
  Goodfire's recipe falls to 0.58 and the trained direction is ahead by +0.08 (+0.02, +0.16), though no direction
  clears random p95 there.
- **Pooling matters.** With max-pooling the trained direction drops to 0.63 and is behind Goodfire's recipe at 0.71,
  a difference of −0.08 (−0.20, +0.04). Its edge exists only with mean-pooling.
- **The probe is the most consistent.** It is at or near the top in most rows. With max-pooling and the position
  control it scores 0.87 (0.78, 0.96) against 0.60 for Goodfire's recipe, a difference of +0.28 (+0.13, +0.44). That
  is the best of many variants looked at after the fact, so it is a lead and not a result.
- **`dom_ours` and `gf_pool` are nearly the same direction** and score within 0.02 of each other everywhere. Adding
  our families to a difference of means changes little.

The full table, with max-pooling and the `reason:` and `close:` harvests, is in `cache/flag/matched_pilot.json` and is
reprinted by `python -m rlp.flag.matched --rollouts pilot --own`.

## Causal comparison (`causal_eval.json`)

Setting the value along a direction at dose 1; A_H and A_N in nats, with the fraction of pairs above zero.

| Direction | Validation A_H | Validation A_N | Held-out A_H | Held-out A_N |
|---|---|---|---|---|
| `gd_mean` | +17.4 (100%) | +30.7 (100%) | +7.1 (65%) | +27.8 (99%) |
| `gf_pool` | −13.6 (7%) | +26.4 (98%) | −11.3 (9%) | +17.3 (89%) |
| `dom_ours` | −16.5 (3%) | +30.1 (100%) | −12.2 (6%) | +18.2 (94%) |
| `probe_ours` | −3.3 (4%) | +5.0 (98%) | −4.1 (2%) | +3.5 (96%) |
| random | about 0 (≈ 50%) | about 0 (≈ 50%) | about 0 | about 0 |

The effects are several times larger than in step 3, because the deciding turns now contain reasoning text and the
direction acts on all of it. The read-out directions now push strongly toward honest, but setting their "hack" value
makes the hack turn less likely, not more. Only the trained direction moves the model in both directions, and its
toward-hack effect weakens on held-out families (65% of pairs). The three seeds have pairwise cosine 0.97, and
`gd_mean` has cosine 0.58 with `dom_ours` (it was 0.12 in step 3).

## Cost

- The pilot took about 95 minutes on one RTX PRO 6000 (96 GB, $2.69 per hour), about $4.30. Generation ran at roughly
  1,000 to 1,200 tokens per second in total at 48 concurrent episodes. The model writes several thousand reasoning
  tokens per attempt, and most tools-scaffold episodes ran to the 50-message limit (about 48 messages per episode on average, against
  about 19 for minimal).
- Harvest, directions, training (3 seeds, 120 steps) and reading the 90 rollouts took about 50 minutes.
- Judging the 90 rollouts used about 2.1 million input tokens.
- At the pilot's rate, the full benchmark (103 tasks × 3 × 2 = 618 episodes) would cost about $30 and 11 hours. It
  was not run.

## What is left

The open decision, as put to the owner on 2026-11-15 and not yet answered:

1. Fix the comparison in advance: mean-pooling, matched pairs, `gd_mean` and `turn:probe_ours` against
   `turn:gf_pool`, with the tools scaffold reported separately.
2. Run about 150 tools and 150 minimal episodes (roughly $15) to bring intervals to about ±0.06. Minimal-only is
   cheaper (about a third of the cost per episode) but is where the directions are tied.

Smaller items that do not need a GPU:

- Read a sample of the judge's flags on `original` tasks to check the labels.
- Only 132 of 275 hack passages find a match. Check whether relaxing the length tolerance changes the picture.

Not started: whether the signal persists while a model is trained with RL (the owner's secondary goal).

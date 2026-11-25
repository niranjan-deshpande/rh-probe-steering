# 02. Step 3: detection on Goodfire's labelled rollouts (2026-11-15)

**Question.** On the rollouts Goodfire released with passage labels, does our direction flag hack passages better
than Goodfire's recipe?
**Short answer.** No. Our gradient-descent direction was slightly worse than Goodfire's recipe, and no direction beat
random directions once passage length was controlled. The test was also weak, for reasons given at the end.
**Code.** `rlp/flag/rollouts.py`, `read.py`, `score.py`, `matched.py`, `train.py`. Rerun on a pod with
`scripts/flag/run_step3.sh`, then `python -m rlp.flag.matched --rollouts ib_qwen38` locally.
**Files.** `cache/flag/step3/` (directions, `causal_eval.json`, `score_ib_qwen38.json`, `matched_ib_qwen38.json`),
`cache/flag/read/ib_qwen38/`, `data/goodfire/ib_qwen38/`, `logs/flag/0_test.log` to `6_causal.log`.

**Note on versions.** These directions were trained on synthetic transcripts without written reasoning, and the layer
chosen was 32. Step 4 (file 03) retrained on transcripts with reasoning, at layer 24. Do not mix the two sets.

## The rollouts

Goodfire released agent rollouts with judge labels for three models (Kimi K3, GLM 5.2 and Qwen 3.8 Max). None is a
model we can run, so we teacher-force Qwen3-14B on another model's rollouts. We used the Qwen 3.8 Max rollouts on
ImpossibleBench (`data/goodfire/ib_qwen38/`), which use the tools scaffold. Labels are a three-vote consensus per
passage: positive if all votes flagged it, negative if none did, ambiguous otherwise (left out).

We read 150 rollouts. They contain 596 hack passages and 6,004 clean passages. Only 25 of the hack passages are
attempted or enacted hacks. The rest are hacks the model considered in its reasoning. In the full release the ratio is
the same: 1,130 of 1,201 hack passages are considered only. Rollouts have a median of about 42,000 tokens, so they are
read in overlapping 32,000-token windows, which drops under 1% of passages.

## Plain passage AUROC (`score.py`)

Max-pooling over the passage's tokens, layer 32, hack against clean passages:

| Score | AUROC |
|---|---|
| Passage length alone | 0.82 |
| Random directions, median / 95th percentile | 0.77 / 0.87 |
| Goodfire's recipe, closing tokens (`close:gf_pool`) | 0.53 (0.51, 0.56) |
| Difference of means on our data (`close:dom_ours`) | 0.66 (0.63, 0.69) |
| Logistic probe on our data (`close:probe_ours`) | 0.81 (0.79, 0.83) |
| Gradient descent (`gd_mean`) | 0.64 (0.62, 0.67) |

The first two rows are the reason these numbers cannot be used. Hack passages are much longer than clean ones, and a
maximum over more tokens is larger for almost any direction. After the log token count is regressed out, every
direction is at or below 0.51. This is why `matched.py` was written.

## Matched pairs (`matched.py`)

Each hack passage is compared with clean passages in the same rollout and channel and within a factor of two in
length. Mean-pooling, layer 32. "vs Goodfire" is the paired difference from `turn:gf_pool`.

| Subset | Hack passages | Goodfire recipe (`turn:gf_pool`) | `turn:dom_ours` | `gd_mean` | gd vs Goodfire | Random p95 | Position alone |
|---|---|---|---|---|---|---|---|
| Matched | 466 | 0.81 (0.77, 0.85) | 0.76 (0.72, 0.81) | 0.75 (0.71, 0.79) | −0.06 (−0.07, −0.04) | 0.82 | 0.87 |
| Matched + position | 224 | 0.75 (0.69, 0.80) | 0.68 (0.61, 0.74) | 0.71 (0.65, 0.77) | −0.04 (−0.06, −0.01) | 0.84 | 0.63 |
| Reasoning channel | 309 | 0.77 (0.73, 0.81) | 0.74 (0.69, 0.79) | 0.71 (0.66, 0.76) | −0.06 (−0.08, −0.04) | 0.81 | 0.87 |
| Content + tool calls | 157 | 0.89 (0.80, 0.95) | 0.82 (0.73, 0.89) | 0.83 (0.76, 0.90) | −0.05 (−0.07, −0.03) | 0.91 | 0.88 |
| Attempted or enacted only | 15 | 0.30 (0.18, 0.69) | 0.33 (0.20, 0.65) | 0.29 (0.18, 0.68) | −0.01 (−0.03, 0.00) | 0.79 | 0.85 |

What we take from this:

- Goodfire's recipe built on whole-turn activations is ahead of our trained direction by a small, consistent margin.
- No direction clears the 95th percentile of random directions. A random direction does this well because, even after
  matching on length, hack passages come later in the rollout (position alone scores 0.87), and many directions drift
  with position.
- Goodfire's recipe on closing tokens, which is the closest analogue of what they describe, is at 0.54 here. The
  whole-turn harvest is what makes their recipe competitive on reasoning-heavy rollouts.
- The logistic probe is below chance in this table (0.49 matched) although it led the plain table. Its plain score
  came from length.
- There are too few attempted-or-enacted passages (15 with a match, in 5 rollouts) to say anything about them.

## Causal comparison (`train.py`, `causal_eval.json`)

Setting the value along a direction at dose 1 (see [../METHOD.md](../METHOD.md)); A_H and A_N in nats, with the
fraction of pairs above zero.

| Direction | Validation A_H | Validation A_N | Held-out A_H | Held-out A_N |
|---|---|---|---|---|
| `gd_mean` | +3.5 (89%) | +3.6 (81%) | +3.3 (65%) | −1.5 (29%) |
| `gf_pool` | −0.2 (48%) | +0.2 (56%) | −0.2 (41%) | +0.2 (59%) |
| `dom_ours` | −0.3 (30%) | +0.3 (71%) | −0.2 (41%) | +0.1 (55%) |
| `probe_ours` | −0.1 (43%) | +0.1 (66%) | −0.1 (47%) | 0.0 (55%) |

So the trained direction steers the model's choice on families it was trained on, and the read-out directions do
not. On held-out families its toward-honest effect has the wrong sign. The three seeds agree with each other (pairwise
cosine 0.83 to 0.91) and are nearly orthogonal to the read-out directions (cosine about 0.12 with `dom_ours`).

## Why this test was weak, and what changed for step 4

1. The rollouts come from a different model, so we read Qwen3-14B's activations on text it did not write.
2. Almost all hack passages are considered-only reasoning, and the training transcripts had no reasoning text at all.
3. The tools scaffold rarely produces enacted hacks. Goodfire's own source manifest shows the minimal scaffold passing
   impossible tests in 23–30% of episodes against about 2% for tools.
4. Position in the rollout is a strong confound that matching on length does not remove.

Step 4 addresses the first three: we generate rollouts from Qwen3-14B itself, on both scaffolds, and add written
reasoning to the training transcripts. See [03_own_rollouts.md](03_own_rollouts.md).

## Background: why the 4B Goodfire-style direction looked weak earlier

An earlier 4B reproduction in this repo (`v_hack`, see `rlp/hackdir/`) scored poorly, while a class-project
reproduction on a larger model reached 0.93. Both used the same pairs file and recipe. The difference was the test.
The class project scored verbalised, considered hacks in agent reasoning on a frozen model. The 4B test scored
silent, enacted hacks by RL-trained hackers.

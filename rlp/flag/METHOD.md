# Method: directions and scoring in `rlp/flag/`

This file explains how the directions are built and how detection is scored. Results are in
[RESULTS.md](RESULTS.md). The module docstrings repeat the essentials next to the code.

**Conventions.** The reader model is Qwen3-14B (40 blocks, width 5120), frozen. `hidden_states[L]` is the input to
block L, so index 0 is the embeddings. Activations are harvested at layers 8, 12, 16, 20, 24, 28 and 32. A direction is
a unit vector at one layer, oriented so that hacking projects higher.

## Minimal pairs

Every direction is built from the same synthetic data (described in
[results/01_data_and_judge.md](results/01_data_and_judge.md)). A minimal pair is one conversation prefix followed by
one assistant turn, the "deciding turn", that is either a hack or its matched non-hack. The two versions share
everything before that turn. We call the hack turn y+ and the non-hack turn y−.

Activations of the deciding turn are averaged over one of three position sets (`harvest.py`):

- `close`: the last three tokens of the turn. This is the analogue of Goodfire's read-out position.
- `turn`: every token of the turn after the assistant header, written reasoning included. This is the primary set
  since step 4 (`read.PRIMARY`).
- `reason`: the tokens of the turn's written reasoning only.

## Baselines (`dirs.py`)

- `gf_pool`: Goodfire's recipe on Goodfire's four families. For each of their contrasts we take the mean activation of
  positives minus negatives, and we average the contrasts with equal weight. `gf_<contrast>` are the single-contrast
  versions.
- `dom_swe`: the same recipe on the three coding families we train on.
- `dom_ours`: the same recipe on our training set (three coding families plus one non-coding family).
- `probe_ours`: L2-regularised logistic regression on our training set, with each (family, label) cell weighted
  equally and the regularisation strength chosen by cross-validation grouped by conversation group.
- `rnd0`–`rnd99`: random unit directions. Their 95th percentile is the bar a direction has to clear.

The layer used downstream is chosen in `dirs.py` from training families only: it is the layer where `dom_ours`
separates validation-split conversations best, by d′. That was layer 32 in step 3 and layer 24 in step 4.

## The gradient-descent direction (`train.py`)

The idea is to find a direction whose value the model actually uses when it decides to hack, not only one that
correlates with hacking. We train a unit vector d by gradient descent so that setting the value along d changes what
the model writes next. Everything uses the frozen base model on teacher-forced transcripts. There is no fine-tuned
model.

**Intervention.** At layer L, on the tokens of the deciding turn, we overwrite the component of the residual stream
along d with a target value and leave everything orthogonal to d alone (`model.overwrite`). The targets come from the
data. Let m_H and m_N be the mean activations on hack turns and on non-hack turns of the training pairs, and let
g = (m_H − m_N)·d be the natural gap along d. For a dose t,

- "toward hack" sets the value to m_N·d + t·g, so t = 1 is the average hack value;
- "toward honest" sets the value to m_H·d − t·g, so t = 1 is the average non-hack value.

If d does not separate hack turns from non-hack turns, then g = 0, the two interventions are the same, and the loss
cannot fall. So a direction that does well must also be one whose projection reads the hack. This is what ties the
causal objective to detection.

**Loss.** Let lp(y) be the summed log-probability of a turn's tokens, and lp₀ the same without intervention.

- A_H = [lp_H(y+) − lp₀(y+)] − [lp_H(y−) − lp₀(y−)]. This is positive when setting the hack value makes the hack turn
  more likely relative to the non-hack turn.
- A_N = [lp_N(y−) − lp₀(y−)] − [lp_N(y+) − lp₀(y+)]. This is positive when setting the honest value makes the hack
  turn less likely, including while the hack is being written.
- L = softplus(−β·A_H) + softplus(−β·A_N) + μ·[max(0, KL_H − τ) + max(0, KL_N − τ)].

KL is the per-token divergence from the unmodified model on generic text (`data/flag/generic.json`, 64 Alpaca
prompts) with the value set on every token, so the direction is penalised for damaging ordinary behaviour. The
settings are β = 0.1, μ = 10, τ = 0.1, and the dose is sampled from {1, 2, 4} each step.

**Optimisation.** d starts as a random unit vector and is renormalised after each step. Defaults: 3 seeds, batch 4
with 4 accumulation steps, deciding turns capped at 192 tokens (`--max-turn`), 120 steps in the runs so far
(`--steps`; the CLI default is 200). The prefix is never intervened on, so it is run once without gradients and its
key/value cache is shared by the six rows per pair (y+ and y− under no intervention, toward hack, toward honest).
The loss does not care about the sign of d, so directions are flipped to be hack-positive after training, and
`gd_mean` is the normalised mean of the oriented seeds.

**Causal comparison.** After training, `causal_eval` reports A_H and A_N in nats at dose 1 for every direction, on
validation pairs of the training families and on pairs from held-out families, with the fraction of pairs above zero.
`python -m rlp.flag.train --eval-only` reruns it without training. Results are in
[results/02_goodfire_rollouts.md](results/02_goodfire_rollouts.md) and
[results/03_own_rollouts.md](results/03_own_rollouts.md).

**Implementation notes that cost time.**

- Log-probabilities are computed only at the deciding turn's tokens. A full-vocabulary log-softmax over the whole
  sequence runs out of memory.
- When a turn has written reasoning, the turn starts at the assistant header. When it has none, the header is followed
  by an empty think block, as the Qwen3 template would write it.
- The Qwen3 chat template drops reasoning from all but the last assistant turn. `render.py` works around this with
  sentinel strings so that the text the model is read on is the text it generated from.
- Float32 matters in the tests (rotary embeddings under left padding, finite-difference checks). `test_flag.py`
  encodes the tolerances that work.

This recipe is a lean version of the draft in [`rlp/manif/RECIPE_SPEC.md`](../manif/RECIPE_SPEC.md). It keeps the
overwrite intervention and the two-sided causal loss. It drops the fine-tuned hacker model, the multi-term loss and
the hack taxonomy grid.

## Scoring (`read.py`, `matched.py`)

`read.py` teacher-forces each rollout through the reader. Rollouts are often longer than the context, so they are cut
into overlapping windows of whole messages, and each passage is scored in the window that gives it the most preceding
context. For each passage and direction we store the projection pooled over the passage's tokens by maximum and by
mean.

**Why not plain AUROC.** Hack passages differ from clean ones in ways that have nothing to do with hacking. On
Goodfire's rollouts they are longer and come later in the rollout, so that passage length alone scores 0.82 and
the 95th percentile of random directions is 0.87 by max-pooling. `score.py` reports these plain numbers and is kept only for
reference.

**Matched pairs (`matched.py`, the scorer to use).** Each hack passage is compared only with clean passages that are
in the same rollout, in the same channel (reasoning, content or tool calls), and within a factor of two in token
length. A stricter variant also requires relative position within 0.15. The score is the probability that the hack
passage projects higher than a matched clean passage, with each hack passage weighted equally, so 0.5 is chance.
Intervals come from resampling rollouts 500 times. The table also reports the same score for length and position
alone, the random-direction percentiles, and each direction's paired difference from `turn:gf_pool`.

Variants printed by `matched.py`: matched; matched + position; attempted-or-enacted hacks only; reasoning channel
only; content and tool calls only; and each scaffold separately. Direction names carry their harvest as a prefix
(`turn:`, `reason:`, `close:`); `gd_s0`–`gd_s2` and `gd_mean` are the trained directions.

**A caution for anyone extending this.** The variants and pooling choices multiply quickly, and with about 36
rollouts the intervals are wide. Pick the comparison before looking at new data. The proposed primary is mean-pooling,
matched pairs, `gd_mean` and `turn:probe_ours` against `turn:gf_pool`, with the tools scaffold reported separately.

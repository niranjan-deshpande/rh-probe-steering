# 01. Training data and the judge

**Question.** What are the directions trained on, and how are rollout passages labelled?
**Short answer.** Directions are trained on synthetic minimal-pair agent conversations: Goodfire's released set plus
two families we wrote. Rollout passages are labelled by one pass of `gpt-6.1-terra` with our own rubric.
**Code.** `rlp/flag/synth.py`, `gen_families.py`, `gen_reasoning.py`, `judge.py`, `own.py`.
**Files.** `data/goodfire/dataset.jsonl`, `data/goodfire/contrasts.json`, `data/flag/`, `cache/flag/judge/`.

## Goodfire's synthetic set

Goodfire released the synthetic conversations their direction is built from (`Goodfire/reward-hack-data`, folder
`synthetic_data`). There are 3,123 multi-turn agent conversations in four families:

| Family | Conversations | Cells (one honest cell plus these hack variants) |
|---|---|---|
| `tz_acquire_pos_v1` | 1,323 | reading hidden tests, fixtures or a reference solution (`peek_*`), and then using what was read (`exploit_*`) |
| `tz_tamper_suite_pos_v1` | 1,000 | `rm_test`, `edit_test`, `conftest_ignore`, `env_var`: removing, editing or disabling tests |
| `tz_tamper_validator_pos_v1` | 400 | `print_line`: printing the line the validator looks for |
| `tz_tamper_scorefile_pos_v1` | 400 | `write_score`: writing the score file directly |

Each conversation belongs to a group. Conversations in a group share a prefix and differ in one assistant message,
which is either honest or one of several hack variants (the conversation's "cell"). A "readout" names that message and
gives 0/1 labels for the contrasts it takes part in. There are 18 contrasts; the 12 named `i_*` compare one hack cell
with its matched cell and the 6 named `pl_*` are pooled. In the acquire family an `exploit_*` cell is contrasted with
the matching `peek_*` cell, and a `peek_*` cell with `honest`. Groups are split into train and validation.

## Our two families (`gen_families.py`)

Goodfire's families are all coding tasks. The owner asked for one or two families of a different kind, so we wrote
two, 600 conversations each (200 per cell), in the same schema (`data/flag/ours.jsonl`, 1,200 conversations).

- `ours_support` (training; overreach). A support agent is scored on five-star surveys. The customer asks for
  something the stated policy does not allow. The cells are `honest` (decline and offer what policy allows), `grant`
  (give it anyway) and `extra` (give it plus an unrequested credit). The tool is `resolve_ticket`.
- `ours_report` (held out; misreport). An operations assistant is scored on its report saying the task is complete.
  The check it runs comes back partial or failed. The cells are `honest`, `claim` (report it complete) and
  `fabricate` (report it complete with invented specifics). The tool is `file_report`.

`gpt-6.1-terra` wrote only the scenario details. The conversations are assembled in code, so the cells of a group
differ in exactly one message. The contrasts are `i_sup_grant`, `i_sup_extra`, `i_rep_claim` and `i_rep_fab`.

## Roles

| Role | Families |
|---|---|
| Train | `tz_tamper_suite_pos_v1`, `tz_tamper_validator_pos_v1`, `tz_acquire_pos_v1`, `ours_support` |
| Held out | `tz_tamper_scorefile_pos_v1`, `ours_report` |
| "Goodfire's direction" (`gf_pool`) | all four `tz_` families, as Goodfire does |

Our directions never see the held-out families. Goodfire's recipe is given all of Goodfire's data, including the
family that is held out for us, so the comparison does not handicap it.

On the synthetic held-out families every direction, including most random ones, separates hack from honest turns
almost perfectly (random 95th percentile 0.89 to 1.00). The synthetic set therefore cannot rank directions. Only the
rollout tests in files 02 and 03 can.

## Written reasoning (`gen_reasoning.py`, added for step 4)

The synthetic conversations contain only tool calls, but in real rollouts most hack passages are reasoning text. After
step 3 we had the judge model write a short first-person thought (two to four sentences) before each deciding action.
For a hack action the thought makes plain that the agent knows this is not the intended solution. For an honest action
it is ordinary problem-solving reasoning. One thought is written per distinct (group, message position, message), and
it is applied to every assistant message that matches, so conversations that share a message share its thought and
remain minimal pairs. The result is `data/flag/reasoning.json` with 4,511 thoughts (median 34 words, none missing).
`synth.load_rows()` applies them by default when the file exists.

The step-3 directions in `cache/flag/step3/` were trained without these thoughts. Everything in `cache/flag/` itself
was trained with them.

## The judge (`judge.py`)

Goodfire labelled their rollouts with three votes of `gpt-5.6-sol` and did not release the prompt. We wrote our own
rubric to match their label definitions: the same hack categories, and a status of considered, attempted or enacted.
A hack that is only considered in the reasoning counts as a hack passage. The judge sees the whole transcript with
each assistant step numbered and returns the (step, channel) passages that qualify. The owner chose one judge pass,
not a three-vote consensus, and `gpt-6.1-terra` over `gpt-6-luna`.

We checked both models on 24 of Goodfire's rollouts (12 judged hacking and 12 clean by their consensus):

| Judge | Rollout agreement | Passage precision | Passage recall | Kappa | Attempted/enacted found |
|---|---|---|---|---|---|
| `gpt-6.1-terra` | 22 / 24 | 0.95 | 0.69 | 0.78 | 7 / 9 |
| `gpt-6-luna` | 23 / 24 | 0.94 | 0.53 | 0.66 | 3 / 9 |

Terra misses about a third of the passages Goodfire's consensus flagged, mostly ones where a hack is only considered.
It rarely flags a passage they did not. Each rollout costs about 42,000 input tokens to judge. The account's rate
limit is one million tokens per minute, so `own.judge_all` uses three workers; with twelve, a third of the calls
failed and had to be retried.

For our own rollouts there is also a label that needs no judge. On the `conflicting` and `oneoff` splits of
ImpossibleBench the tests cannot be passed honestly, so an episode that passes them enacted a hack
(`enacted_by_tests` in `own.load`).

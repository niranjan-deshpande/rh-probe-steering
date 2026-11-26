# 08. Steering inside multi-turn rollouts (2026-11-25, daytime)

**Question.** The overnight run ([07](07_overnight.md)) showed that subtracting `Σ·probe` removes hacking on
single-turn environments. Does it also remove hacking when the model works as an agent over many turns, does it do
so better than Goodfire's direction and than random vectors, and what does it cost in ordinary capability?

**Short answer.** Yes on the first two, with a measurable cost. Subtracting `Σ·probe` took hacking from 26 of 32
episodes to 0 of 32, and every steered episode ended by itself with an honest submission. Goodfire's direction and
two random vectors also lowered hacking, but they did it by derailing the episode. A one-paragraph instruction in the
system prompt is a strong baseline (3 hacks in 30). The cost of our direction is a small drop on HumanEval and a
lower solve rate on one benign task family.

## Setup

- Model Qwen3.8-27B, frozen, steered at the input of block 36 at every token, norm 20, through the per-request
  steering patch in vLLM (`patch_vllm_steer_multi.py` v4, rows of `cache/flag/night/steer_table.pt`).
- Two multi-turn families from 07, `save_game` (a trading game whose save file can be edited) and `release_gate`
  (a performance gate whose threshold, benchmark or auth check can be bypassed). 16 hack-variant episodes each
  (8 tasks, 2 seeds). The hack variants cannot be solved honestly, so the honest outcome is to submit a failing
  result.
- Hacks are detected programmatically by the runner (`mt.py`), not by a judge.
- Every run of today used a budget of 30,000 generated tokens per episode (`--gen-budget`), added so that rambling
  conditions could not consume the pod. The evaluation applies the same rule to runs made without it (last night's
  unsteered run `m1` and the two `v3` runs): an episode of more than 46,000 generated tokens (budget plus one turn)
  is counted as derailed. **This changes one `m1` episode and one Goodfire episode from hack to derailed and is a
  post hoc rule for those runs.**
- Outcomes: **hack** (detector fired); **finished** (no hack, the episode ended by itself: submitted, finished or
  used all its submissions); **derailed** (no hack, ran out of budget, context or turn length).
- Intervals are 95% bootstrap intervals over tasks, paired on task and seed with the unsteered run on the same server.

## 1. Hack variants, `save_game` + `release_gate`

| Condition | n | hack | finished | derailed | median generated tokens | hack rate against unsteered (paired) |
|---|---|---|---|---|---|---|
| Unsteered, last night (`m1`) | 32 | 28 | 2 | 2 | 14,255 | |
| Unsteered, same server (`v4_base`) | 32 | 26 | 3 | 3 | 14,355 | |
| `Σ·probe` subtracted | 32 | 0 | 32 | 0 | 6,781 | −81 points (−94, −66) |
| Goodfire's direction subtracted | 32 | 2 | 0 | 30 | 39,377 | −75 points (−88, −59) |
| Random `Σ`-shaped vector 1 subtracted | 27 | 11 | 1 | 15 | 30,168 | −37 points (−62, −12) |
| Random `Σ`-shaped vector 2 subtracted | 16 | 1 | 0 | 15 | 39,514 | −69 points (−88, −44) |
| Unsteered + instruction in the system prompt | 30 | 3 | 18 | 9 | 24,839 | −70 points (−87, −52) |

- **`Σ·probe`.** 0 of 32 hacks; the one-sided 95% upper bound on the hack rate is about 9%. All 32 episodes finished
  by themselves, with about half the tokens of the unsteered run. Honest finishing rose from 9% to 100% (+91 points
  (+75, +100)).
- **Transcripts (post hoc reading of all 32).** In `save_game` the model only reads the save file. In `release_gate`
  it never edits the threshold, the benchmark or the simulator and always keeps `auth.verify`; several episodes add
  an in-process cache, which is the honest optimisation the family was designed around. Matched transcripts:
  `logs/flag/night/rollsteer_transcripts.md`.
- **Goodfire's direction** lowers hacking nearly as much, but 30 of 32 episodes derail (context or budget exhausted,
  or a turn cut at 16,000 tokens) and none finishes honestly. So it stops the hack by stopping the agent. This is
  our construction of their direction on this model at one norm, not their published pipeline.
- **Random vectors** drawn with the activation covariance also lower hacking, again by derailing (15 of 27 and 15 of
  16 episodes). So "hack rate fell" is not evidence of specificity on its own in rollouts; the specific result is
  that `Σ·probe` lowers hacking **and** the episode finishes. Vector 1 lacks 5 episodes and vector 2 was stopped at
  16 by design; an isotropic random vector was dropped after one episode (which hacked) to save time.
- **Instruction baseline.** Appending three sentences to the system prompt ("Complete every task only by legitimate
  means ...", `scripts/flag/night/spec_cfg.json`) gives 3 hacks in 30, 18 honest finishes and 9 derailed, with more
  tokens than the unsteered run. It is close to steering on hack rate and worse on finishing (60% against 100%).
  Two episodes were not reached before the cutoff.
- The two unsteered runs differ in `save_game` (14 of 16 last night, 10 of 16 today, where 3 hit the budget), so the
  paired comparison uses today's.

## 2. Benign variants under steering

On the benign variants the target can be reached honestly. Unsteered (`m1`, last night) the model solved 64 of 64.
With `Σ·probe` subtracted it solved 52 of 64: 32 of 32 in `release_gate` and 20 of 32 in `save_game` (−19 points
(−33, −6) pooled). No steered benign episode hacked or derailed.

In the failed `save_game` episodes I read, the model trades one good on one route, falls short of the target and
states that the target is unattainable. It does not touch the save file. So steering costs effort or planning on the
harder honest task. Caveat: the unsteered comparison is last night's server and has no token budget, although benign
episodes are short (median under 4,000 tokens) so the budget does not bind.

## 3. A third family: `leak_outputs`

`leak_outputs` (expected outputs are readable on disk) was the only other family with an informative unsteered hack
rate. Steered, 10 of 31 episodes hacked against 13 of the same 31 unsteered last night (−10 points (−28, +6)); solve
rate 21 of 31. **Steering does not measurably reduce this hack.** Reading a file that happens to hold the answers is
a different behaviour from editing a grader, and the direction was fitted on environments of the second kind. One
episode was not reached before the cutoff.

## 4. Capability benchmarks (single turn, same questions and seeds, thinking on)

| Benchmark | n | unsteered | `Σ·probe` subtracted | difference (paired, over questions) |
|---|---|---|---|---|
| MATH level 5 | 200 | 82.5% | 82.5% | 0.0 (−5.0, +5.0) |
| MMLU | 300 | 87.0% | 86.7% | −0.3 (−3.0, +2.3) |
| HumanEval | 164 | 98.8% | 95.7% | −3.0 (−6.1, −0.6) |

HumanEval loses five problems and gains none; the five are ordinary logic errors, not formatting failures. MATH is
scored by normalised string match of the boxed answer, and a completion cut at 8,000 tokens counts as wrong (20
unsteered, 13 steered), so its absolute level is a lower bound.

## 5. Lying in short stories (after arXiv 2610.29129)

Six prompts ask for a 400-word story in which the protagonist is asked a direct question where honesty is costly;
20 stories per prompt per condition; a blind judge (`gpt-6.1-terra`) labels the protagonist's final answer.

| Condition | truth | lie | evasion | none |
|---|---|---|---|---|
| Unsteered | 118 | 1 | 1 | 0 |
| `Σ·probe` subtracted | 119 | 1 | 0 | 0 |
| `Σ·probe` added | 112 | 3 | 4 | 1 |

Adding the direction lowers truthful answers by 5.0 points (−10.0, −0.8); subtracting changes nothing measurable
(+0.8 (−1.7, +3.3)). The unsteered model is at 98%, so the subtract arm has no room and the add effect rests on
eight stories. Weak evidence that the direction overlaps with dishonesty in general.

## Caveats

- 16 episodes per family and condition, two families, one set of seeds, one steering norm.
- Hack detection is programmatic and family-specific; a hack of a kind the detector does not know would be missed.
  The transcript reading in section 1 is the check on that, and it was done post hoc by the assistant, not blind.
- The 30,000-token rule was introduced during the day (see Setup).
- The direction was chosen on single-turn environments last night; nothing was tuned on these rollouts.

## What is left (nothing is running; each needs a go)

- **Probe-gated steering** (the capping idea of arXiv 2610.29129): subtract along `Σ·probe` only when the probe reads
  above a threshold set on benign activations, and test whether the HumanEval and benign `save_game` costs go away
  while the hack rate stays near zero.
- A lower steering norm as the cheaper alternative to gating.
- An unsteered benign run and `leak_outputs` run on the same server, to remove the cross-night comparison.
- Detection on rollouts is still tied with Goodfire's (07); this run was about steering only.

## Files and cost

- Runner: `rlp/flag/mt.py` (`--gen-budget`, `cfg.system_add` added today). Evaluation: `rlp/flag/rollsteer_eval.py`
  → `cache/flag/onp/v5_rollsteer.json`.
- Runs: `cache/flag/mt/{v3_sprobe_sub,v3_gf_sub,v4_base,v4_sig5_sub,v4_sig7_sub,v4_iso0_sub,v4_sprobe_benign,v4_sprobe_other,v5_spec_prompt}`;
  baseline `cache/flag/mt/m1`.
- Single turn: `scripts/flag/night/v5_single.py`, `v5_story_judge.py` → `cache/flag/night/v5/`.
- Pod scripts: `scripts/flag/night/{v3,v4,v6,v7}.sh`; transcripts: `scripts/flag/night/rollsteer_transcripts.py`.
- One A100 pod at $2.19 per hour from 11:51 am to 8:47 pm ET, about $19.60, plus 360 judge calls (not metered, well
  under $1). About 2 hours of that was lost to my running too many conversations at once, which collapsed the prefix
  cache. Pod deleted.

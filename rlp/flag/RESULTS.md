# `rlp/flag/` results: index and summary

This page gives the findings in brief and points to the file that holds each one. The method is in
[METHOD.md](METHOD.md). All numbers come from files under `cache/flag/`; intervals in parentheses are 95% bootstrap
intervals over rollouts.

## Summary (items 1 to 4 as of 2026-11-15; items 5 to 8 added as they were run)

**The goal** is a direction that flags reward-hacking passages in agent rollouts better than Goodfire's direction.
**It has not been met yet.** As of 2026-11-25 (item 8): on Qwen3.8-27B our steering vector beats Goodfire's direction in
steering and our probe beats it in detection on single-turn environments, but on multi-turn agent rollouts, which is
what the goal is about, the two are not distinguishable.

1. **Data and judge.** We train on Goodfire's 3,123 synthetic minimal-pair conversations plus 1,200 of our own in two
   non-coding families, with one coding family and one of ours held out. A single `gpt-6.1-terra` pass agrees with
   Goodfire's three-vote labels on 22 of 24 rollouts, with passage precision 0.95 and recall 0.69.
   → [results/01_data_and_judge.md](results/01_data_and_judge.md)

2. **Step 3: Goodfire's own Qwen 3.8 Max rollouts (negative).** On 150 rollouts, no direction beat the 95th
   percentile of random directions once passages were matched on rollout, channel and length. Our gradient-descent
   direction scored 0.75 against 0.81 for Goodfire's recipe, a difference of −0.06 (−0.07, −0.04). The test itself was
   weak: only 25 of 596 hack passages were attempted or enacted, the rest were hacks the model only considered in its
   reasoning, and position in the rollout alone scored 0.87.
   → [results/02_goodfire_rollouts.md](results/02_goodfire_rollouts.md)

3. **Step 4: our own Qwen3-14B rollouts (inconclusive, slightly positive).** We added written reasoning to the
   training transcripts, retrained, and ran the 14B model on ImpossibleBench ourselves (90-episode pilot; 275 hack
   passages, 185 attempted or enacted). In matched pairs with mean-pooling, our gradient-descent direction scored
   0.72 (0.59, 0.85) against 0.67 (0.55, 0.81) for Goodfire's recipe, a difference of +0.05 (−0.05, +0.13). The
   logistic probe scored 0.75 (0.65, 0.85). On the tools scaffold alone the trained direction is ahead by
   +0.08 (+0.02, +0.16); with max-pooling it is behind by −0.08 (−0.20, +0.04).
   → [results/03_own_rollouts.md](results/03_own_rollouts.md)

4. **Causal check.** The gradient-descent direction is the only one that moves the model both ways. In step 4,
   setting the hack value along it makes the hack turn more likely by 17 nats on validation pairs (100% of pairs) and
   setting the honest value makes it less likely by 31 nats (100%). Goodfire's recipe and the other read-out
   directions push toward honest but go the wrong way toward hacking (about −14 nats). On held-out families the
   toward-hack effect of the trained direction falls to 7 nats (65% of pairs).
   → [results/03_own_rollouts.md](results/03_own_rollouts.md), "Causal comparison"

5. **Qwen3.8-27B base-rate pilot (2026-11-23, for the new recipe).** No direction was trained. Qwen3.8-27B solved
   10 of 15 solvable ImpossibleBench tasks in the minimal scaffold (14B passed 9 of 90 episodes overall) and hacked in
   1 of 30 impossible ones. On 25 single-turn environments it hacks unprompted at usable rates in about 8, mostly
   "game the metric" and "use a leaked answer" hacks; grader tampering and misreporting are near zero. The run was cut
   off with 65 of 90 episodes saved, and a 40,000-token context ended many episodes early.
   → [results/04_q38_base_rates.md](results/04_q38_base_rates.md)

6. **On-policy directions on Qwen3.8-27B (2026-11-24, first run of the new recipe).** On the frozen model's own hack
   and honest samples from the same prompt, a logistic probe and a covariance-whitened mean difference at layer 36
   reach matched AUROC 0.92 and 0.89 on three held-out environments; the plain mean difference gets 0.64. In free
   generation, subtracting a hack direction at norm 20 cuts the hack rate on fresh prompts of the fitting environments
   from 41.7% by 22 to 33 points (random directions: −11 to +10) while honest solves rise. Steering on held-out
   environments is not established, because random directions also move those rates by up to +26 points. The
   recipe's fitted vector was not obtained: the teacher-forced steering term had no signal.
   Text-only control (post hoc): a TF-IDF classifier matches the probe on the fitting environments (0.97) but
   reaches 0.67 on held-out ones, 0.25 below the probe (0.20 to 0.30). A small unrelated reader model (Qwen3-0.6B
   activations over the same text) reaches 0.84, so the probe's lead over text is about 0.08 (0.04 to 0.12).
   → [results/05_onpolicy_directions.md](results/05_onpolicy_directions.md)

7. **One detector transformed for steering, prefix resampling, more environments (2026-11-24, second run).** A
   detector `w` was fitted with an on-policy steering term from prefix resampling, steering along `Σw`. Subtracting
   the fitted vector in free generation cuts hacking from 61% to 3% on three environments nothing had seen (two
   tampering, one plagiarism; 12 random conditions: −33 to +14 points) and from 43% to 0% on the earlier held-out
   pair, with honest solve rates rising. Held-out steering is now established against eight random directions for the
   probe, `Σ·probe` and the whitened direction (signed contrasts +43 to +52 points, random RMS 20, p about 0.02 to
   0.03). The transform `Σw` did not steer better than `w` itself. One fit and one seed; "more cautious in general"
   is not ruled out. Detection: the morning's probe reaches 0.92, 0.86 and 0.81 on three new environments; more
   fitting environments help on average (0.72 with one, 0.85 with four) but category variety does not, and single
   environments can lower transfer by 0.1 to 0.3.
   → [results/06_transformed_detector.md](results/06_transformed_detector.md)

8. **Overnight run: noise control, Goodfire's direction on this model, multi-turn rollouts (2026-11-25).** In
   steering, ours beats Goodfire's: subtracting `Σ·probe` on five environments outside every fit lowers hacking from
   48% by 44 points [−58, −32] (Holm p 0.024, none of 32 random conditions as low) and Goodfire's direction does
   nothing (+6); paired difference −50 [−64, −36]. In detection on single-turn environments ours also wins (0.887
   against 0.471). **On multi-turn rollouts it does not:** over six rollout families the probe minus Goodfire's
   direction is −0.010 [−0.115, +0.085], and our probe is inverted on two families. The fitted vector's advantage of
   24 November is mostly reproduced by a noise control. The model hacks unprompted in rollouts (two tampering
   families about 90%, leaked outputs 39%, ImpossibleBench 48% under one instruction wording and 0% under the
   published one). Choosing fitting environments on a validation family helps the probe on rollouts (0.759 against
   0.586 for the whole pool). Step-level detection favours the probe, but that comparison is post hoc and position
   explains it on ImpossibleBench.
   → [results/07_overnight.md](results/07_overnight.md)

## Main caveats

- The step-4 sample is small: 36 rollouts contribute matched pairs, so intervals are about ±0.12.
- Many scoring variants were looked at in step 4 (two poolings, three harvests, six subsets). The comparison was not
  fixed in advance, so the probe's best numbers in particular should be treated as leads.
- Qwen3-14B is a weak solver on this benchmark. It passed the tests in 9 of 90 episodes, so "enacted" mostly means it
  wrote special-cased code that still failed.
- The judge's labels on our own rollouts have not been spot-checked by hand.

## Files

| File | Holds |
|---|---|
| [results/01_data_and_judge.md](results/01_data_and_judge.md) | Training data, family roles, written reasoning, judge rubric and its check against Goodfire's labels |
| [results/02_goodfire_rollouts.md](results/02_goodfire_rollouts.md) | Step 3 in full: plain and matched scores, causal comparison, why the test was weak |
| [results/03_own_rollouts.md](results/03_own_rollouts.md) | Step 4 in full: the pilot, matched scores by subset, causal comparison, costs, what is left |
| [results/04_q38_base_rates.md](results/04_q38_base_rates.md) | Qwen3.8-27B pilot: unprompted hack rates on 25 single-turn environments, ImpossibleBench solve and hack rates, cost |
| [results/05_onpolicy_directions.md](results/05_onpolicy_directions.md) | Qwen3.8-27B on-policy matched pairs: closed-form directions as detectors, the failed fit of the new recipe, steering in free generation with random controls, cost |
| [results/06_transformed_detector.md](results/06_transformed_detector.md) | Second run of 2026-11-24: ten new environments and their pilot, detection with more environments and mixes, prefix resampling, the fit of `w` steering along `Σw`, steering against a random-direction null on two prompt sets, cost |
| [results/07_overnight.md](results/07_overnight.md) | Overnight run of 2026-11-25: Goodfire's direction built on Qwen3.8-27B, steering against 32 random conditions of two kinds, the noise control for the fitted vector, six multi-turn families including every ImpossibleBench variant tried, detection on rollouts, choosing fitting environments on validation, deviations and cost |

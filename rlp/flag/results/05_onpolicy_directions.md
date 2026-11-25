# On-policy hack directions on Qwen3.8-27B: detection and steering (2026-11-24)

**Why this was run.** The owner's 2026-11-23 recipe asks for a vector `v` on a frozen model that steers toward reward
hacking, separates hack from honest rollouts, leaves unrelated text alone and stays on the activation manifold
(`L = L_steer + λ1·L_AUC + λ2·L_KL + λ3·vᵀΣ⁻¹v`), with everything judged on-policy. This run built the on-policy data
for it, scored simple directions as baselines, tried to fit `v` with the recipe, and tested the directions by steering
the model in free generation.

The model's weights are never changed. Only directions are fitted. "Fitting environments" below are the environments
whose samples were used to fit directions; "held-out environments" were never used for that.

**Summary.**

1. On matched pairs of the model's own hack and honest samples from the same prompt, a logistic probe and a
   covariance-whitened mean difference separate the two with matched AUROC about 0.9 on three held-out environments.
   The plain mean difference does not transfer (0.64).
2. The recipe's steering term, as we implemented it (teacher-forced on the model's own sampled completions), has
   almost no signal, so the fit reduced to an overfitted detection term. We stopped it. This part of the recipe is
   not yet tested fairly.
3. In free generation, subtracting any of the three hack directions lowers the hack rate on fresh prompts of the
   fitting environments by 22 to 33 points, well outside what two random directions do (−11 to +10). Honest solve
   rates go up at the same time.
4. On held-out environments the steering result is not established. The effects are large (up to +33 and −19 points)
   but random directions of the same norm also move held-out hack rates by up to +26 points.
5. The teacher-forced steering measure and the on-policy one disagree: the first says these directions do nothing,
   the second says they change hack rates by tens of points.

## Setup

- **Model.** `Qwen/Qwen3.8-27B`, frozen, thinking on, served with vLLM 0.34 as in the base-rate pilot
  ([04_q38_base_rates.md](04_q38_base_rates.md)).
- **Data (Stage 1).** Single-turn environments from `rlp/manif/envs.py`, 8 samples per prompt, labelled by the
  program detectors. Fitting environments: `length_pad`, `quiz_key`, `hardcode` (neutral and pressure prompts) and
  `leak` (pressure), 40 items each; items 0–29 fit directions and items 30–39 are validation. Held-out environments:
  `save_tamper`, `fake_pass`, `chess_key` (pressure) and `fake_cite` (neutral), 30 items each. 3,200 completions, of
  which 2,960 had a usable label. Script: `scripts/flag/run_onpolicy_q38.sh`. Output: `cache/flag/base_rate/onp/`.
- **Activations (Stage 2).** The frozen model reads each of its own completions once. We keep the mean residual
  activation over completion tokens at layers 16, 24, 28, 32, 36, 40 and 48 (layer L means the input of block L), and
  a token covariance Σ per layer from the fitting completions plus generic text. Code: `rlp/flag/onp.py harvest`.
  Output: `cache/flag/onp/` (`means.npy`, `cov_36.pt`, `meta.json`).
- **Matched AUROC.** For each prompt that has both hack and honest samples, the probability that a hack sample scores
  above an honest sample of the same prompt, averaged over prompts and then over environments. Intervals are 95%
  bootstrap intervals over prompts.

## Stage 1: the on-policy data

Hacks and honest samples, and the number of prompts that have both:

| Role | Environment | Hack | Honest | Prompts with both |
|---|---|---|---|---|
| fit | `length_pad` | 241 | 238 | 59 |
| fit | `quiz_key` | 106 | 373 | 14 |
| fit | `leak` | 180 | 56 | 19 |
| fit | `hardcode` | 288 | 98 | 21 |
| validation | the same four | 272 | 255 | 33 |
| held out | `save_tamper` | 137 | 102 | 28 |
| held out | `fake_pass` | 39 | 182 | 17 |
| held out | `fake_cite` | 133 | 92 | 29 |
| held out | `chess_key` | 168 | 0 | 0 |

`chess_key` is unusable: every valid sample was a hack (the model solved none of the puzzles), so it has no pairs and
was dropped from everything below. `hardcode` under pressure is nearly saturated (242 hacks of 250 valid samples).

## Stage 2: closed-form directions as detectors

Four directions were computed per layer from the 113 fitting prompts that have both classes:

- **Mean difference**: the within-prompt difference between mean hack and mean honest activation, averaged with equal
  weight per environment.
- **Whitened mean difference**: Σ⁻¹ times the mean difference (Σ shrunk by 5% of its mean eigenvalue). This is the
  linear-discriminant direction.
- **Σ-weighted mean difference**: Σ times the mean difference.
- **Logistic probe** on prompt-centred activations (`C = 0.01`).

The layer was chosen on the validation prompts among layers up to 40, which gave **layer 36**. Results at layer 36,
pooling all completion tokens:

| Direction | Validation (fitting envs) | Held-out mean | `save_tamper` | `fake_pass` | `fake_cite` |
|---|---|---|---|---|---|
| Logistic probe | 0.95 | 0.92 | 0.94 (0.90, 0.97) | 0.85 (0.74, 0.93) | 0.98 (0.96, 0.99) |
| Whitened mean difference | 0.96 | 0.89 | 0.86 (0.80, 0.91) | 0.82 (0.71, 0.91) | 1.00 (1.00, 1.00) |
| Mean difference | 0.90 | 0.64 | 0.55 (0.43, 0.67) | 0.62 (0.48, 0.74) | 0.76 (0.70, 0.82) |
| Σ-weighted mean difference | 0.75 | 0.54 | 0.52 (0.41, 0.64) | 0.40 (0.30, 0.52) | 0.70 (0.59, 0.80) |

- **Random directions are a wide null here.** For 20 random directions, the 95th percentile of |AUROC − 0.5| is 0.27
  on validation and 0.22 on held-out. Mean activations differ between hack and honest samples along many directions,
  for example through answer length. The probe and the whitened direction are far outside this; the plain mean
  difference on held-out environments is not.
- **Whitening is what transfers.** The mean difference detects within the fitting environments and fails outside
  them. The whitened version transfers. The probe and the whitened direction have cosine only 0.32 with each other.
- **Whitened directions are off-manifold by the recipe's own measure.** At equal norm, `vᵀΣ⁻¹v` is 394 for the
  whitened direction, 385 for a random direction and 11 for the mean difference. So the recipe's manifold term pushes
  away from the direction that detects best on held-out environments. This tension is worth a decision before the
  recipe is tried again.
- Other layers and pools are in `logs/flag/q38_onp/dirs.log`. Reasoning-only tokens give nearly the same numbers.
  Answer-only tokens are worse on held-out environments (probe 0.86 at layer 36).
- The layer was chosen on validation prompts, and the held-out numbers were visible in the same log, but the rule
  (validation mean of three directions) was written before the run.

## Stage 3: fitting the vector with the recipe (did not work as implemented)

**Implementation.** `rlp/flag/onp.py train`. The steering term was the change in mean log-probability per token of a
hack completion minus the same change for an honest completion of the same prompt, when `v` is added at layer 36 at
every position. The completions are the model's own samples, but this is still teacher forcing. The detection term
is a pairwise softplus on the 113 fitting pairs. The KL term uses 64 generic passages. Initial `v` was the mean
difference at 4% of the mean residual norm (83.4).

**What happened.** In the one real run (λ1 = 0.3, λ2 = 1, λ3 = 0.001), stopped near step 100 of 200:

- The steering gain stayed at noise level (between −0.001 and +0.007 nats per token at the logged steps up to 75).
- The detection term fitted the 113 pairs perfectly by step 20. Held-out matched AUROC rose from 0.64 to 0.82 at step
  25 and then fell to 0.69 at step 50 and 0.73 at step 75, so the fit was overfitting.
- The norm stayed near 4, because nothing in the loss rewarded growing it.

**Diagnostic.** To see whether a larger norm would give the steering term signal, we measured the same teacher-forced
gain for fixed directions at norms 10, 20 and 40 on 18 fitting pairs (`scripts/flag/onp_steer_diag.py`, output
`cache/flag/onp/diag_36_cf.json`):

| Direction, norm | Gain when added (nats/token) | Gain when subtracted | KL on generic text |
|---|---|---|---|
| Probe, 20 | +0.004 (SE 0.005) | −0.006 (0.004) | 0.024 / 0.021 |
| Whitened, 20 | 0.000 (0.004) | −0.008 (0.003) | 0.037 / 0.027 |
| Mean difference, 40 | −0.005 (0.026) | −0.111 (0.030) | 0.201 / 0.180 |
| Random, 20 | +0.001 (0.003) | −0.005 (0.004) | 0.019 / 0.018 |

Adding any direction lowers the log-probability of hack and honest completions by almost the same amount. I think
the reason is that a completion is thousands of tokens long and the hack and honest versions differ at few of them,
so a per-token average over a fixed completion barely registers a change in which way the model would go.

**Reading.** The recipe has not been tested fairly. What failed is one way of writing `L_steer`. Three fits planned
for different λ3 values were not run, to avoid spending three hours on a loss with no steering signal.

## Stage 4: steering in free generation

**Method.** vLLM was patched so that a vector is added to the residual stream entering block 36 at every position of
every request (`scripts/flag/patch_vllm_steer.py`, then `patch_vllm_steer_cuda.py`; server
`scripts/flag/serve_q38_steer.sh`). With a zero vector the greedy output is identical to the unpatched server. Each
direction was scaled to **norm 20** (24% of the mean residual norm) and both added and subtracted. Each condition
generated one completion for 330 prompts that no direction was fitted on: items 40–69 of the four fitting
environments, items 30–59 of `save_tamper`, `fake_pass` and `fake_cite`, and 30 items each of four environments
where the base rate is zero (`false_report`, `ticket_close`, `reward_tamper`, `runtests`). Script:
`scripts/flag/run_onp_steer.sh`. Table: `rlp/flag/onp_steer.py`, output `cache/flag/onp/steer_summary.json`.

The hack rate is detector hits over all completions. Differences from the unsteered run are paired by prompt, with
95% bootstrap intervals over prompts. "Fit envs" pools 120 prompts and "held-out envs" pools 90.

| Condition | Fit envs, hack rate | Change | Held-out envs, hack rate | Change | Honest solves | Valid (fit / held) |
|---|---|---|---|---|---|---|
| Unsteered | 41.7% | | 37.8% | | 52% | 92 / 88% |
| Probe, subtracted | 12.5% | −29.2 (−37.5, −20.0) | 28.9% | −8.9 (−18.9, +1.1) | 67% | 98 / 98% |
| Probe, added | 55.8% | +14.2 (+5.0, +23.3) | 48.9% | +11.1 (−1.1, +23.3) | 42% | 88 / 81% |
| Whitened, subtracted | 19.2% | −22.5 (−31.7, −13.3) | 18.9% | −18.9 (−28.9, −8.9) | 62% | 93 / 92% |
| Whitened, added | 55.0% | +13.3 (+3.3, +23.3) | 71.1% | +33.3 (+22.2, +44.4) | 54% | 92 / 94% |
| Mean difference, subtracted | 8.3% | −33.3 (−41.7, −25.8) | 21.1% | −16.7 (−27.8, −6.7) | 66% | 98 / 84% |
| Mean difference, added | 60.0% | +18.3 (+9.2, +27.5) | 38.9% | +1.1 (−11.1, +13.3) | 29% | 80 / 83% |
| Random A, subtracted | 51.7% | +10.0 (0.0, +20.0) | 54.4% | +16.7 (+4.4, +28.9) | 49% | 92 / 81% |
| Random A, added | 30.8% | −10.8 (−19.2, −1.7) | 34.4% | −3.3 (−14.4, +7.8) | 62% | 93 / 96% |
| Random B, subtracted | 35.8% | −5.8 (−15.0, +3.3) | 52.2% | +14.4 (+2.2, +26.7) | 55% | 92 / 93% |
| Random B, added | 38.3% | −3.3 (−12.5, +5.8) | 63.3% | +25.6 (+14.4, +36.7) | 61% | 92 / 96% |

"Honest solves" is the share of `hardcode`, `leak`, `fake_pass` and `runtests` answers that pass the hidden tests.

**Is the effect on the fitting environments specific to the hack directions?** For subtraction, yes on this evidence.
The four random conditions change the rate by between −10.8 and +10.0 points, and subtracting a hack direction
changes it by −22.5 to −33.3. Subtraction also raised honest solves (52% to 62–67%) and valid answers, so the model
was not simply degraded. Per environment, subtracting the probe took `leak` from 21 to 5 hacks of 30, `hardcode` from
11 to 2 (with solves from 10 to 24), `length_pad` from 10 to 2 and `quiz_key` from 8 to 6. For addition the margin is
thin: +13.3 to +18.3 against a random maximum of +10.0, and adding the mean difference also cut solves to 29% and
valid answers to 80%.

**Is there a specific effect on held-out environments?** Not shown. Random directions changed the held-out rate by
+16.7, −3.3, +14.4 and +25.6 points, so the whitened direction's +33.3 is not clearly outside the null. The
reductions (−18.9 whitened, −16.7 mean difference) are below every random condition, but with four random conditions
that is weak, and one random condition lowered `save_tamper` alone by 27 points. Per environment the largest effects
of the hack directions are on `save_tamper` (17 of 30 unsteered; 1 to 5 when subtracting; 28 when adding the whitened
direction) and `fake_pass` (2 unsteered; 15 when adding the whitened direction, but 19 when subtracting random A).
The single-environment intervals are about ±20 points.

**No new hack types appeared.** All four zero-rate environments stayed at 0 of 30 in every condition.

**Teacher-forced and on-policy measures disagree.** At norm 20 the teacher-forced gain of the probe and whitened
directions was zero within error (table in Stage 3), while in free generation the same vectors changed hack rates by
13 to 33 points. A direction chosen or rejected on the teacher-forced number would have been judged wrongly.

## Text-only control for Stage 2 (post hoc, 2026-11-24)

**Why.** The probe reads activations averaged over the completion, and the completion contains the hack. A detector
that reads only the completion text might do as well, in which case the activations add nothing. This control was
added after seeing the Stage 2 result.

**What was run.** A TF-IDF logistic regression on the completion text (reasoning and answer), with the same rows,
the same fit / validation / held-out split, prompt-centred features and the same matched AUROC. Two versions: word
1-2-grams and character 2-5-grams, vocabulary from the fitting rows, regularisation chosen on the validation prompts
(which the 27B probe did not get; it used a fixed C). Code: `rlp/flag/onp_text.py score`. Output:
`cache/flag/onp/text_control.json`. The 27B rows below are recomputed from the saved directions and reproduce Stage 2.

| Detector | Validation | Held-out mean | `save_tamper` | `fake_pass` | `fake_cite` | Probe minus this, held-out mean |
|---|---|---|---|---|---|---|
| 27B logistic probe | 0.95 | 0.92 | 0.94 [0.90, 0.97] | 0.85 [0.74, 0.93] | 0.98 [0.96, 0.99] | |
| 27B whitened mean difference | 0.96 | 0.89 | 0.86 [0.80, 0.91] | 0.82 [0.71, 0.91] | 1.00 [1.00, 1.00] | +0.03 [-0.01, +0.05] |
| TF-IDF words | 0.97 | 0.67 | 0.87 [0.82, 0.91] | 0.78 [0.67, 0.88] | 0.36 [0.26, 0.45] | +0.25 [+0.20, +0.30] |
| TF-IDF characters | 0.93 | 0.68 | 0.89 [0.85, 0.94] | 0.74 [0.60, 0.87] | 0.40 [0.31, 0.50] | +0.24 [+0.19, +0.29] |

- **On fresh prompts of the fitting environments the text alone is enough** (0.97 against 0.95). Within an
  environment, surface text separates hack from honest samples as well as activations do.
- **On held-out environments the probe is 0.25 ahead** (paired bootstrap over prompts, interval excludes zero). Most
  of the gap is `fake_cite`, where the text classifier is below chance (0.36), so the words that marked a hack in the
  fitting environments mark the honest answer there.
- **The text classifier still transfers to `save_tamper` (0.87) and `fake_pass` (0.78).** On those two the probe's
  lead is 0.07, with overlapping intervals. So part of the held-out transfer in Stage 2 is available from text.
- Completion length is not the signal: its matched AUROC is 0.52, 0.35 and 0.47 on the three held-out environments.

**A stronger text reader closes most of the gap.** `onp_text embed` reads the same completion text with an unrelated
small model (Qwen3-0.6B, mean activations over the completion, same probe recipe, layer chosen on validation
prompts). Its best layer by validation (21 of 28) reaches 0.95 on validation and **0.84 held-out** (`save_tamper`
0.91 [0.87, 0.95], `fake_pass` 0.65 [0.52, 0.78], `fake_cite` 0.97 [0.94, 0.99]). The 27B probe is ahead by 0.08
[0.04, 0.12] on the held-out mean, and nearly all of that is `fake_pass` (0.85 against 0.65). Layer 14 gives 0.85 and
the last layer 0.51, so the small reader's transfer depends on the layer. An LLM judge was not tried.

**Reading.** Most of the held-out detection in Stage 2 can be had from the text of the completion by a model 45
times smaller. What the 27B activations add is a lead of about 0.08, concentrated in misreporting. The claim that
survives is "a linear probe on the model's own activations is the best of the detectors tried and transfers to new
hack classes", not "activations see something the text does not show".

**Prior work (literature check by a subagent, 2026-11-24; not exhaustive).** The closest work is Goodfire's
reward-hacking monitor (arXiv 2609.19101), which transfers a mean-difference direction from synthetic pairs to real
benchmarks but does not hold out hack types and has no trained probe or random-direction null; PRIME (arXiv
2606.09711), which holds out exploit families that are all pytest gaming, on RL-trained models; and hacktrace (arXiv
2610.03055), which uses a within-problem AUC and finds TF-IDF close to the probe (0.953 against 0.987) in
distribution. None was found that fits on gaming and peeking and tests on tampering, misreporting and fabricated
citations with on-policy program-labelled samples. Earlier truth-probe work found whitened directions transfer worse
than the plain mean difference, the opposite of Stage 2.

## Caveats

- **Thin null for steering.** Two random directions (four conditions) are not enough to set a null for effects of
  this size. Random perturbations at norm 20 are clearly not neutral, especially on `fake_pass` and `save_tamper`.
- **One norm, one layer, one sample per prompt.** No dose-response. 30 prompts per environment.
- **Fresh prompts, same environments.** The "fit envs" steering result is on new items of environments the
  directions were fitted on. It shows the directions are causal there, not that they generalise.
- **Detectors, not a judge.** A hack is a program-detector hit. Truncated and invalid answers count as no hack;
  `fake_cite` and `hardcode` lose 20–40% of answers that way, and steering changed how many (`fake_cite` valid 13 to
  28 of 30 across conditions), which confounds that environment.
- **Hack types.** The fitting environments are "game the metric" and "use a leaked answer" hacks. The held-out set
  adds one tamper and one misreport environment. Agent rollouts (ImpossibleBench) were not used.
- **Detection confound.** Mean activations over a completion can separate hack from honest samples through length or
  format. The matched design removes prompt effects, not these. No length-matched control was run.
- **No reading of samples.** Steered completions were not read by hand.
- **The Stage 3 fit** was implemented and tuned during the run (learning rate, sequence cap, gradient checkpointing),
  and only one configuration ran for more than 30 steps.

## What is left

- A steering objective with signal. Options are the log-probability of only the tokens where hack and honest
  completions diverge, or a search over a few candidate directions scored by on-policy hack rate (each condition
  costs about $0.45).
- A real null for steering: 10 or more random directions on the held-out environments (about $9).
- Whether the manifold term should be `vᵀΣ⁻¹v`, given that it penalises the whitened direction as much as a random one.

## Cost and speed

One A100 80 GB at $2.19 per hour from 12:59 am to 8:54 am ET, about **$17.40**. Stage 1 took 3 hours at roughly
1,000 generated tokens per second with 120 concurrent requests. The activation pass took 46 minutes. Each steering
condition took about 12 minutes at roughly 750 tokens per second. Logs: `logs/flag/q38_onp/`.

Two engineering notes. On a 128-core pod the closed-form step hangs unless `OMP_NUM_THREADS` is set (8 works). And
gradients through blocks 36–63 need about 13 MB per token, so the fit uses gradient checkpointing (`--ckpt 1`) to
take 3,000-token sequences in 59 GB.

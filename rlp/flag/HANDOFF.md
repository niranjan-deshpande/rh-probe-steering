# Handoff: what this repository is doing, as of the night of 25 November 2026

This file is for a collaborator (or their agent) arriving cold. It is long on purpose. It records what we are trying
to do right now, what has and has not been tested, where every environment came from, and the practical knowledge
that is not written anywhere else.

**If you are an agent and the repository looks confusing, that is expected and it is not your fault.** The project
changed direction three times in two weeks, and the older files still describe the older goals. You do not need to
understand everything. Read sections 1 to 4 of this file, then open only what your task needs. Nothing here is
urgent, nothing breaks if you read slowly, and it is always fine to stop and ask the owner what they want.

Contents: 1. The goal now · 2. Vocabulary · 3. How we got here · 4. What is established · 5. What has and has not been
tested · 6. Where the environments came from · 7. The directions · 8. The run in progress · 9. Layout of this
repository · 10. Practical knowledge · 11. Working rules

---

## 1. The goal now

We have one frozen model, `Qwen/Qwen3.8-27B`, and we work at one place in it: the residual stream at the input of
decoder block 36. No weights are ever trained. We only fit **directions** (vectors of width 5,120) and either read
them (detection) or add a multiple of them to the stream at every token (steering).

**We are now optimising for steering, not detection.** The project began as a detection project, and the folder name
(`flag`), the README question and most of the older results files still say so. The owner changed the priority on
the evening of 25 November, after the steering result in `results/08_rollout_steering.md` turned out much better than
expected.

The claim we are trying to establish is:

> Subtracting one direction from the residual stream stops the model from reward hacking in multi-turn agent tasks,
> while the model keeps working: it still finishes the task honestly and does not lose general capability.

The open worry, which is the owner's and which the data partly supports, is that the direction does not remove
hacking as such. It may make the model **give up sooner**, which looks the same on tasks that cannot be solved
honestly. That would make it useless during reinforcement learning, where persistence on hard tasks is what you want.
Most of what is planned next is about telling these two apart.

Detection is still of interest, but it is not what we are selecting for.

## 2. Vocabulary

- **Reward hack / hack.** The model gets a task to pass by a route the task did not intend: editing a grader, a
  threshold or a save file, reading a leaked answer key, padding an answer to meet a length score, and so on. The
  prompt never asks for it. Every hack label in this repository on the 27B model comes from a **program detector**
  (code that inspects the answer or the working directory), not from a judge model.
- **Layer 36.** Index L means the input to decoder block L (`hidden_states[L]`), everywhere in the code.
- **Probe, `w`.** A logistic-regression direction that separates the model's own hack completions from its own
  honest completions of the same prompt. Fitted on mean activations over completion tokens.
- **`Σ`.** The covariance of layer-36 activations over tokens (`cov_36.pt`, 105 MB, not in the repository).
- **`Σ·probe`** (also `Σw`, `sprobe`, `Stok_probe`). The probe multiplied by `Σ`, normalised. This is the steering
  vector. The idea is that a detector reads along `w`, and the direction that moves that read-out most cheaply, for
  a model whose activations have covariance `Σ`, is `Σw`.
- **Subtracted / added.** We add `−α·u` or `+α·u` to the stream at every token, where `u` is the unit vector.
- **Norm.** `α`. It has been 20 in every steering run up to 25 November. The mean norm of the stream at this layer is
  about 83, so 20 is about 24% of it. **It was never tuned.** See section 5.
- **Matched AUROC.** For each prompt (or task) that has both hack and honest samples, the probability that a hack
  sample scores above an honest sample of the same prompt, averaged over prompts and then over environments. It
  removes differences between prompts. 0.5 is chance.
- **Single-turn environment.** One prompt, one answer. The loophole is visible in the prompt.
- **Multi-turn family / rollout.** The model acts as an agent with tools (`bash`, `read_file`, `write_file`,
  `submit`) in a sandboxed working directory for up to 30 turns.
- **Hack variant / benign variant** (multi-turn). Two versions of the same task with identical tools and limits. In
  the hack variant the honest route cannot succeed, so the only way to pass is to cheat. In the benign variant the
  honest route succeeds. The honest outcome on a hack variant is therefore to submit a failing result.
- **Hack / finished / derailed** (outcome of a hack-variant episode, defined in `rollsteer_eval.py`). Hack: the
  detector fired. Finished: no hack and the episode ended by itself. Derailed: no hack and the episode ran out of
  token budget, context or turn length. A steering vector that only makes the model ramble shows up as derailed.
- **Goodfire's direction** (`gf_pool`, `gf_sub`). The reward-hacking direction of Goodfire's paper (arXiv
  2609.19101), rebuilt by us on this model. See the caveat in section 7.
- **Post hoc.** An analysis chosen after seeing the data. We label these every time.

## 3. How we got here

1. **15 November, Qwen3-14B.** The original plan: train a direction to flag hacking passages in agent rollouts, with
   a judge model for labels (`results/01` to `03`). It roughly tied Goodfire's direction. This phase is finished and
   its code (`train.py`, `judge.py`, `score.py`, `agent.py`, `matched.py`, ...) is not used by anything current.
2. **23 November, new model and recipe.** Moved to Qwen3.8-27B because 14B was too weak an agent, and to on-policy
   data: the frozen model's own hack and honest completions of the same prompt, labelled by program detectors
   (`results/04`).
3. **24 November, two runs.** A logistic probe detects hacks on environments it never saw. Steering with it works in
   free generation. A vector fitted with a gradient-based steering term looked better (`results/05`, `06`).
4. **25 November, overnight.** That fitted vector's advantage was mostly reproduced by a noise control, so it was
   retired and the closed-form `Σ·probe` became the steering vector. Goodfire's direction was rebuilt on this model.
   Multi-turn rollout families were built. On rollouts, detection tied Goodfire's (`results/07`).
5. **25 November, daytime.** Steering inside rollouts: 0 hacks in 32 episodes, all finishing honestly, where
   Goodfire's direction and random vectors only derailed the agent. Small capability costs appeared (`results/08`).
6. **25 November, evening.** The owner made steering the priority. A selection run was started (section 8).

## 4. What is established

All of this is on one model, mostly one seed set, and should be read as a strong pilot.

**Steering inside rollouts** (`results/08_rollout_steering.md`; two families, 16 hack-variant episodes each):

| Condition | Episodes | Hack | Finished | Derailed |
|---|---|---|---|---|
| Unsteered | 32 | 26 | 3 | 3 |
| `Σ·probe` subtracted, norm 20 | 32 | 0 | 32 | 0 |
| Goodfire's direction subtracted, norm 20 | 32 | 2 | 0 | 30 |
| Random `Σ`-shaped vector 1 | 27 | 11 | 1 | 15 |
| Random `Σ`-shaped vector 2 | 16 | 1 | 0 | 15 |
| Unsteered, with an instruction in the system prompt | 30 | 3 | 18 | 9 |

- Our vector lowers hacking by 81 points (95% interval −94 to −66) and every episode finishes honestly.
- **Hack rate alone is not evidence of anything in rollouts.** Any large push lowers it by breaking the agent. The
  claim always has to be "fewer hacks and the episode finishes".
- A three-sentence instruction in the system prompt is a strong baseline. Any future claim should include it.

**Costs of our vector at norm 20:**

- HumanEval 98.8% → 95.7% (−3.0 points, interval −6.1 to −0.6). MATH level 5 and MMLU unchanged.
- On benign `save_game` (solvable honestly), solve rate 32 of 32 unsteered → 20 of 32 steered. The steered model
  plays a simple route, falls short, and declares the target unattainable. This is the direct evidence for the
  "gives up sooner" worry.
- Steered outputs are shorter everywhere (10 to 15% fewer tokens on benchmarks, about half on hack variants).

**Where it did nothing:** the `leak_outputs` family (reading a file that holds the expected outputs). 10 of 31
hacks steered against 13 of 31 unsteered.

**Steering on single-turn environments** (`results/07`, section 2): on five environments outside every fit, 50
fresh prompts, against 32 random-vector conditions, `Σ·probe` subtracted beats Goodfire's by 50 points (−64 to −36)
and is the only vector that moves hacking in both directions.

**Detection:** on single-turn environments our probe is far ahead of Goodfire's (matched AUROC 0.887 against 0.47,
leave one environment out). **On multi-turn rollouts the two are tied** (difference −0.010, interval −0.115 to
+0.085, over six families). Choosing which single-turn environments to fit on, using a validation rollout family,
lifts our probe from 0.586 to 0.759 on a test family. Adding rollout data to the fitting pool did not help.

**Other things worth knowing:**

- Random vectors drawn with the activation covariance lower single-turn hacking by about 13 points with either
  sign. A steering null needs these, not only isotropic random vectors.
- Adding the direction makes protagonists of short stories lie slightly more (truthful 118 of 120 → 112 of 120).
  Weak evidence that it overlaps with dishonesty in general.
- The gradient-fitted vector `v` (`fitw.py`, `onp.py train`) is retired. Do not build on it.
- Teacher-forced steering losses had no signal. Only free generation tells you whether a vector steers.

## 5. What has and has not been tested

This is the section to read twice. The pattern is that **detection was validated broadly and steering narrowly.**

| Question | Detection | Steering |
|---|---|---|
| Single-turn environments | 11 environments, leave-one-out and leave-category-out | 11 environments in total: 5 held out of every fit (the clean test), 3 from 24 November, 3 fitting environments |
| Multi-turn families | 6 families, including ImpossibleBench | **2 families with a positive result** (`save_game`, `release_gate`), 1 with none (`leak_outputs`). `answer_key`, `scorer_copy`, `ib_lcb`, `mini_repo`: **never steered** |
| Which fitting environments are best | Selected on validation rollout families, 20 splits | **Never selected, until the run in section 8.** The vector in use was fitted on four environments chosen by hand on 24 November for a detection experiment |
| Random-vector controls | 20 random directions | Single-turn: 32 conditions. Rollouts: 2 `Σ`-shaped vectors at norm 20, incomplete; no isotropic control |
| Strength | not applicable | **One norm (20), never tuned, for every vector including Goodfire's** |
| Layer | 7 layers compared on 24 November, 36 chosen | **Only layer 36** |
| Against Goodfire's | Yes, both settings | At equal norm 20 only. Their vector is much further off-manifold at that norm (below), so the comparison is not yet fair |

**Not tested at all:**

- Persistence on hard but solvable tasks at graded difficulty (the "gives up sooner" question). One data point
  exists: benign `save_game` at one difficulty.
- Whether a lower norm keeps the effect and drops the cost.
- Steering only when the probe fires (the capping idea of arXiv 2610.29129). A free diagnostic on stored
  activations says a turn-level gate would open too late in two of three families, so it would have to act per token.
- Any use during training. Nothing here has been run inside an RL loop.
- A second model, or a second set of seeds for the rollout result.
- Concealed hacks. In our families the model usually hacks openly.
- Whether the programmatic detectors miss hacks of an unanticipated kind under steering. The only check is that one
  agent read all 32 steered transcripts, after the fact and not blind.

**A measurement that explains part of the Goodfire comparison** (laptop only, post hoc). At norm 20, measured
against the full activation covariance, our vector is 6.0 units long, Goodfire's is 39.2, the two random `Σ`-shaped
vectors are about 5, and an isotropic random vector is 59. So Goodfire's direction at norm 20 pushes hard into
directions where activations barely vary, which probably explains the derailing. But the random `Σ`-shaped vectors
are as on-manifold as ours and still derailed, so being on-manifold is not what makes ours work.

## 6. Where the environments came from

All environments were written for this project by the owner's Claude sessions, except ImpossibleBench. They are
synthetic scenarios. In every one the prompt describes the task and never asks for the hack or points at it.

### Single-turn (one prompt, one answer)

- **Base set: `rlp/manif/envs.py`.** Written in an earlier study in the same parent project (the "hack manifold"
  study on Qwen3-4B). 25 hack environments and some benign controls, 150 items each. Some are built on public data:
  the code group uses MBPP problems, `quiz_key` uses MMLU questions, the chess ones use Lichess puzzles. Those data
  files are **not** in this repository (see section 9), so these environments cannot be regenerated here without
  them.
- **Conditions.** `neutral`: the loophole is visible and nothing mentions it. `tempt`: the same plus a note that the
  honest route has failed or is infeasible and what is at stake. `tempt2`: one step stronger, the grading mechanism
  restated as a fact. None instructs the hack. (The `pos` / `neg` instructed conditions in `envs.py` belong to the
  older study and are not used here.)
- **`rlp/flag/envs2.py` (24 November).** Ten more environments in the categories Tamper, Misreport, Overreach, Hedge
  and Plagiarise, written to allow whole categories to be held out.
- **`rlp/flag/envs3.py` (25 November).** `puzzle_leak` and `recon_total`.
- **Which are usable.** An environment is usable only if the frozen model hacks unprompted at an intermediate rate,
  so that the same prompt yields both hack and honest samples. Base rates were measured with `base_rate.py`
  (`results/04`, `06`). The eleven usable ones, with our category labels:
  - Game: `length_pad`, `hardcode`, `fake_cite`
  - Peek: `quiz_key`, `leak`
  - Tamper: `save_tamper`, `ckpt_swap`, `thresh_tamper`, `log_scrub`
  - Misreport: `fake_pass`
  - Plagiarise: `ref_copy`
  - `puzzle_leak` was added for steering only.
- **Which are not.** The model essentially never misreports or overreaches in one turn (`metric_report`,
  `run_summary`, `check_claim`, `scope_edit`, `refund_split`, `multi_answer`, `recon_total`: near zero). `chess_key`
  is saturated (every valid answer is a hack). Do not spend time on these without a new idea.
- **Roles on 24 November**, which is where the current vector comes from: fitting `length_pad`, `quiz_key`, `leak`,
  `hardcode`; held out `save_tamper`, `fake_pass`, `fake_cite`.

### Multi-turn (agent rollouts)

Runner: `mt.py`. Task interface: `scripts/flag/night/README_agents.md`.

- **Five scripted families, `mt_families.py`**, written for the overnight run of 24/25 November. 8 tasks each, a
  hack and a benign variant per task, labels from the working directory and the tool calls only.

  | Family | Hack variant | Label | Unsteered hack rate |
  |---|---|---|---|
  | `save_game` | a trading game whose gold target is above a proven bound on honest play | the save file disagrees with a replay of the move log | 14 of 16 |
  | `release_gate` | a latency gate that the best honest optimisation misses by 2 ms | the gate passes only with changed gate files or a bypassed auth check | 15 of 16 |
  | `leak_outputs` | the grader's expected outputs are on disk | the file is read or its contents embedded | 25 of 64 |
  | `answer_key` | an answer key is on disk, wrong on 3 hard items | the submission matches the key on those | 6 of 64 |
  | `scorer_copy` | the scorer rewards overlap with a forbidden text | overlap above a threshold | 1 of 16 |

  Only the first three have a hack rate high enough to measure a reduction, and only the first two have been
  steered with a positive result.
- **`save_game` in detail**, because the next experiments lean on it. For each game the generator computes `best`
  (gold reached by a searched honest route, a lower bound on honest play) and `bound` (a proven upper bound). The
  hack target is 140% of `bound`. The benign target is 80% of `best`. Since 26 November, `cfg["sg_frac"]` sets the
  benign target to another fraction of `best`, and games 8 to 15 exist (before that, task indices wrapped at 8, so
  "game 8" was game 0 again).
- **`ib_lcb`, `mt_ib.py`.** ImpossibleBench's Impossible-LiveCodeBench (`safety-research/impossiblebench`): coding
  tasks whose tests cannot be passed by a correct function. Its hack rate is set almost entirely by one instruction
  sentence: 0% under the published strict instruction, 48% under a "mild" one, about 90% under a plain one. Under
  "mild" the hacks are disclosed, and honest episodes mostly just run out of turns. The task data is not in this
  repository.
- **`mini_repo`, `mt_repos.py`.** 15 small synthetic repositories with one test that contradicts the README. Built
  as a fallback; lightly used.

### Goodfire's data

`data/goodfire/` (not in this repository) holds their released synthetic records and contrast definitions. We feed
their token ids to our model unchanged and take the mean difference per contrast (`gf27.py`). See section 7.

## 7. The directions

Files are in `directions/` and `results/night/`.

- **`directions/dirs_cf_36.npz`** — closed-form directions of 24 November at layer 36: `probe` (logistic), `dom`
  (difference of means), `lda`, `sig`, and random controls.
- **`directions/write_cands_36.npz`**, key `Stok_probe` — **the steering vector in use**: `Σ` times that probe.
  The probe was fitted on four single-turn environments (`length_pad`, `quiz_key`, `leak`, `hardcode`) and on no
  agent data.
- **`directions/goodfire_dirs_36.npz`** — Goodfire's contrasts rebuilt on this model; `gf_pool` is the pooled one.
  **Caveat to repeat whenever you compare:** this is our construction from their released records at our layer. It
  is not their pipeline, their pooling is not fully specified, and their final probe was selected on held-out
  rollouts that we did not reproduce.
- **`results/night/steer_table.pt`** — the 68 steering conditions of 25 November, all at norm 20. A request to the
  patched server picks a row with `extra_body={"vllm_xargs": {"vid": k}}`. Rows used in `results/08`: 0 unsteered,
  43 `sprobe_sub`, 47 `gf_sub`, 28 and 32 the random `Σ`-shaped vectors, 55 `sprobe_add`.
- **`results/night/steer_table_v8.pt` and `v8_cands.npz`** — candidates for the run in section 8.

## 8. The run in progress (started 11:47 pm ET, 25 November)

Its results are **not** in this snapshot. Code: `scripts/flag/night/v8.sh`, `v8_select.py`, `v8_cands.py`,
`v8_table.py`. The selection rules were written into `v8_select.py` before any of its data existed.

1. **Choose the vector by steering.** Five candidates, all `Σ` times a logistic probe, differing in fitting data:
   A the current vector; B all 11 single-turn environments; C the tampering environments; D the subset that detects
   best on rollout families other than the two used tonight (`ckpt_swap`, `fake_cite`, `length_pad`, `quiz_key`,
   `thresh_tamper`); E Goodfire's contrasts. Validation: `save_game` games 0 to 7, 16 hack and 16 benign episodes,
   norm 20. Rule: among candidates with at most 1 hack in 16, the highest benign solve rate.
2. **Strength curve** for the winner at norms 5, 10 and 15. Rule: the lowest norm with at most 2 hacks in 32.
3. **Difficulty curve** on new `save_game` games 8 to 15: benign target at 60%, 80%, 90% and 100% of `best`,
   unsteered against steered. This is the first test of persistence.
4. **Goodfire's direction** at norms 5 and 10.

It is a pilot: 2 seeds, 16 episodes per cell, one family carrying all the cost evidence.

## 9. Layout of this repository

This repository is an **export** of one project folder (`rlp/flag`) from a larger private working tree, made by
`scripts/flag/export_repo.sh`. That explains two things that would otherwise be confusing:

- **Paths in code and results files say `cache/flag/...`.** In this repository those files are under `results/`
  (`cache/flag/onp/x.json` → `results/onp/x.json`, `cache/flag/mt/<run>/` → `results/mt/<run>/`,
  `cache/flag/night/` → `results/night/`) and `directions/`.
- **Some results files mention other projects** (`rlp/persona`, `rlp/hackdir`, `rlp/manif`). They are earlier
  studies in the working tree and are not here, except `rlp/manif/envs.py`, which the flag code imports.

| Path | What it is |
|---|---|
| `rlp/flag/HANDOFF.md` | this file |
| `rlp/flag/results/NN_*.md` | **the record.** One file per run, with numbers, intervals and caveats. `07` and `08` are current |
| `rlp/flag/RESULTS.md`, `README.md` | index and status. The README's opening still frames the goal as detection |
| `rlp/flag/METHOD.md` | the 14B-era method. Historical |
| `results/mt/<run>/*.jsonl` | full rollout transcripts, one episode per line: `messages`, and `info` with `hit`, `kind`, `stop`, `solved`, token counts |
| `results/transcripts/rollsteer_transcripts.md` | the same task under each condition, trimmed for reading. A good first thing to look at |
| `results/onp/v5_rollsteer.json` | every number of `results/08` |
| `docs/INFRA.md` | pods, vLLM, the steering patch, pitfalls |
| `scripts/flag/night/` | pod scripts. `README_agents.md` defines the interfaces between modules |

Run names: `m1` unsteered rollouts of the night before (all five families, both variants); `v3_*` first steered
rollouts; `v4_*` controls, benign variants (`v4_sprobe_benign`) and the third family (`v4_sprobe_other`);
`v5_spec_prompt` the instruction baseline; `v8_*` the run in progress.

**Not in the repository:** activation caches, `cov_36.pt`, the benchmark and environment source data (`data/cmp`,
`data/manif`, `data/goodfire`), completions on benchmark test items, and any credentials. Ask the owner for these.
Without `cov_36.pt` and the activations you can read every result and rerun the rollout evaluations, but you cannot
refit a probe or build a new `Σ·probe`.

**Current modules** (the rest are historical): `mt.py`, `mt_families.py`, `mt_ib.py`, `mt_harvest.py`,
`mt_select.py`, `rollsteer_eval.py`, `steer_table.py`, `steer_eval.py`, `gen_multi.py`, `base_rate.py`, `envs2.py`,
`envs3.py`, `onp.py` (harvest and closed-form directions only), `gf27.py`, `q38.py`.

## 10. Practical knowledge

- **Steering runs inside vLLM**, through `scripts/flag/night/patch_vllm_steer_multi.py` (vLLM 0.34.0). Send
  `cache_salt` that names the steering row with every request, or the prefix cache will serve one condition's
  activations to another. `v3_cachecheck.py` tests this; run it after every server start.
- **Keep at most about 13 long conversations in flight on one A100.** With 20 the prefix cache collapses and
  throughput falls from about 270 to 80 generated tokens per second. This cost two hours on 25 November.
- **Give every rollout a token budget** (`--gen-budget 30000`). Steered or random conditions can ramble to the
  context limit. The evaluation applies the same rule to runs made without it; this was introduced mid-day and is
  flagged in `results/08`.
- **Pair on (task, seed).** The runner derives every turn's seed from the task key, the episode index and the turn,
  so the same episode under two conditions starts identically. Intervals are bootstraps over tasks.
- **Compare against an unsteered run on the same server** when you can. Two unsteered runs a day apart differed in
  `save_game` (14 of 16 against 10 of 16).
- **Read transcripts before believing a rate.** Every surprising number here turned out to have a mechanism visible
  in the text (rambling, giving up, an honest cache that looks like a bypass).
- **RunPod pods are sometimes dead on arrival**: status running, billing, no container. Look at the system logs in
  the first minutes and replace the pod if there are none.
- **Check that task indices mean what you think.** Indices wrap silently in most families.
- More in `docs/INFRA.md`.

## 11. Working rules

These are the owner's standing rules for agents on this project.

- **Results files are the record.** Do not change a number without rerunning the code that produced it. Label
  anything chosen after seeing data as post hoc.
- **Keep experiments lean.** One question per run. The owner has repeatedly rejected ablation grids.
- **Propose before spending.** State the plan, the cost and the time, and wait for an explicit go. A budget is not
  a go.
- **Fix selection rules before the data exists**, and write them down where they can be checked.
- **Report plainly.** Set up what is being looked at and why, then the result with its interval, then the caveats.
  Say so when something fails or is incomplete.
- **No credentials in the repository, in logs or on pods' disks.**
- When you add a result: one topic file in `rlp/flag/results/`, one line in `RESULTS.md`.

If something in this file disagrees with a results file, the results file is right and this file is stale. It was
written on the night of 25 November and nobody has promised to keep it current.

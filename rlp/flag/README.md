# Flagging reward-hacking rollouts with a direction (`rlp/flag/`)

**Question.** Can we build a single activation direction that flags reward-hacking passages in agent rollouts better
than Goodfire's reward-hacking direction (arXiv 2609.19101)? "Flags" means the following test. We teacher-force a
frozen model on a rollout, project its residual stream onto the direction, pool the projection over each passage (one
channel of one assistant step), and ask how well that score separates passages a judge labelled as hacking from clean
ones, by AUROC.

**This is the active project in the repo.** It is the only one with open work. The goal and the constraints below
were set by the owner (Niranjan) on 2026-11-14 and 2026-11-15.

- The only claim we need is that our direction beats Goodfire's and a few natural candidates (difference of means and
  a logistic probe on our data, and random directions). Keep experiments lean. Do not add ablation grids.
- The metric is per-passage detection on real rollouts of a frozen model. Telling hacker training runs from clean
  runs (the "meter" in `rlp/hackdir/`) is a different question. Persistence over RL training is a secondary goal and
  has not been started.
- The model is Qwen3-14B. The owner ruled out 4B as too small and 32B as too costly.
- No fine-tuned "hacker" model is used anywhere. Every activation comes from the frozen base model reading a
  transcript.
- The judge is `gpt-6.1-terra`, one pass per rollout.

**Status (2026-11-15).** The pipeline is built and has been run twice. Nothing is running and no pod is up.

- On Goodfire's own labelled rollouts (step 3), our trained direction was slightly worse than Goodfire's, and no
  direction beat random directions once passage length and position were controlled.
- On our own Qwen3-14B rollouts (step 4, a 90-episode pilot), after adding written reasoning to the training
  transcripts, our trained direction roughly ties Goodfire's: +0.05 (−0.05, +0.13) in matched-pair AUROC. The
  interval includes zero, so there is no confirmed win.
- **Open decision, waiting on the owner:** whether to run about 300 more episodes (roughly $15) with the comparison
  fixed in advance. See [results/03_own_rollouts.md](results/03_own_rollouts.md), "What is left".

**New direction (2026-11-23, set by the owner).** The open decision above is dropped. The 300-episode run and the
judge spot-check are not being pursued. The next recipe to try is

    L = L_steer + λ1·L_AUC + λ2·L_KL + λ3·vᵀΣ⁻¹v

- `L_steer`: adding v to the frozen model (additive, free norm) raises reward hacking across many contexts.
- `L_AUC`: v separates hack from honest passages on on-policy rollouts, with a pairwise surrogate.
- `L_KL`: adding v leaves unrelated text unchanged. `vᵀΣ⁻¹v` keeps v on the activation manifold.
- There is no necessity term in training. Necessity is never to be estimated on teacher-forced transcripts, and no
  LoRA or other fine-tuning is used. Everything is on-policy as far as cost allows (not hundreds of dollars).
- The two goals are a robust characterisation of the best linear representation of reward hacking, and a monitor
  that generalises to held-out contexts and hack types.
- The model is moving to Qwen3.8-27B, because 14B was too weak an agent. A base-rate pilot on it is done:
  [results/04_q38_base_rates.md](results/04_q38_base_rates.md). Only generation is ported to Qwen3.8 (`q38.py`); the
  reading and training code (`render.blocks`, `train.py`) is still 14B-only.
- **First run done (2026-11-24):** [results/05_onpolicy_directions.md](results/05_onpolicy_directions.md). Code:
  `onp.py` (data reading, closed-form directions, the recipe's loss, steering files) and `onp_steer.py` (steering
  table). A probe and a whitened mean difference detect held-out hacks at matched AUROC about 0.9, and subtracting a
  hack direction in free generation cuts hacking on the fitting environments by 22 to 33 points. Two things are open:
  the recipe's steering term had no signal when written as a teacher-forced log-probability, and steering on held-out
  environments is not separated from random directions.
- **Second run done (2026-11-24, evening):** [results/06_transformed_detector.md](results/06_transformed_detector.md).
  The recipe is now one detector `w` with steering along `Σw`, fitted with an on-policy steering term from prefix
  resampling (`resample.py`, `fitw.py`). Subtracting the fitted vector removes almost all hacking on environments it
  never saw. Open: repeat the fit over seeds, separate "less hacking" from "more cautious", agent rollouts. New
  environments are in `envs2.py`.
- **Third run done (2026-11-25, overnight):** [results/07_overnight.md](results/07_overnight.md), run from
  [SPEC_overnight_2026-11-24.md](SPEC_overnight_2026-11-24.md). Goodfire's direction was built on this model.
  `Σ·probe` beats it in steering (−50 points, interval excludes zero) and the probe beats it in single-turn detection,
  but on six multi-turn rollout families the two detect equally (−0.010 [−0.115, +0.085]). The fitted vector's
  advantage is mostly reproduced by a noise control, so `Σ·probe` is the steering vector to use. Rollout code is in
  `mt.py`, `mt_ib.py`, `mt_families.py`; per-request steering in `steer_table.py`, `gen_multi.py`, `steer_eval.py`;
  analysis in `mt_select.py`.
- **Open after the third run:** a flag that works on rollouts (per-step or maximum pooling with a position control),
  honest ImpossibleBench episodes that finish, steering inside rollouts, and concealed hacks. See "What is left" in
  results/07.
- **Nothing is running and no pod is up.** The next run needs the owner's approval.

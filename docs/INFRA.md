# Infrastructure notes: pods, models, judge

Read this before renting a GPU or calling the judge. It collects the things that cost time or money in earlier runs.
Each project README has its own run steps; this file holds what is shared.

## Working arrangement

- **Local (laptop).** CPU analysis, tests, and judge calls. Always use the repo's `.venv/bin/python` (Python 3.13); the
  system Python lacks `openai`. Set `HF_HUB_OFFLINE=1` for anything that only needs the tokenizer or a tiny model.
- **GPU pods (RunPod).** Generation, activation harvests, training and steering. The repo is copied to
  `/workspace/rlp` and outputs are copied back into `cache/`. There is no pinned requirements file; the setup scripts
  list the packages.
- **Secrets.** `.env` at the repo root holds `HF_TOKEN` and `OPENAI_API_KEY`. Never print or commit it. The `rlp/flag`
  pods need no secrets, so do not copy `.env` there. The persona and hackdir setup (`scripts/setup_pod.sh`) does read
  it.
- **The owner's preferences.** Propose a plan with cost and time before spending, give progress and cost updates, and
  pick the cheapest GPU that works. Report clock times in US Eastern, 12-hour format.

## Which setup script

| Project | Setup | Environment | Notes |
|---|---|---|---|
| persona, hackdir | `scripts/setup_pod.sh` | `scripts/env.sh` | vLLM 0.11, transformers 4.5x, peft; clones Wong et al.'s grader repo |
| hackdir OLMo round | `scripts/hackdir/setup_olmo.sh` | | OLMo-3.1-32B on one 96 GB GPU |
| manif | `scripts/manif/setup_manif.sh` | `scripts/manif/env.sh` | |
| flag | `scripts/flag/setup_flag.sh`, then `setup_vllm.sh` for rollouts | `scripts/flag/env.sh` | two virtual environments: reader (transformers 5.22.2) and vLLM (transformers < 5) |

## RunPod

- **Dead pods.** Some pods show RUNNING and bill, but the container never starts. A healthy pod shows image-pull logs
  within about a minute. If there are no system logs after 3 to 5 minutes, or no container after about 10, terminate
  it and create a new one, ideally in another data center. Do not terminate earlier than that.
- **Stock numbers are unreliable.** Be ready to try another GPU type, tier or disk size. A stopped pod may not
  restart because its host's GPU was taken.
- **Test the GPU first.** One community host had a GPU where `nvidia-smi` worked and torch failed. The setup scripts
  run a matrix multiply for this reason.
- **Disks.** `/workspace` is a network file system on pods with a network volume: `chown` fails there, so use
  `rsync -rlt` and not `rsync -a`. Put the virtual environment and the Hugging Face cache on the container disk
  (`/root`).
- **Stopping.** Retested 2026-11-24 on a CPU pod: the injected `RUNPOD_API_KEY` CAN stop its own pod through GraphQL (`mutation{podStop(input:{podId:"$RUNPOD_POD_ID"}){id}}` to `https://api.runpod.io/graphql`, HTTP 200, pod went to EXITED in about 20 s). `rest.runpod.io` did not resolve from inside the pod. The variables are in the main process's environment, not in ssh shells (read `/proc/1/environ`). The earlier note said the key could not stop the pod.
  automatic watchdog; `scripts/manif/watchdog.sh` is an example of one. A pod left idle bills at the full rate.
- **Stopped pods still bill for disk.** Terminate pods when a run is finished and the outputs are copied back.
- **SSH.** Launch long jobs with `setsid -f bash script.sh < /dev/null > log 2>&1`; plain `nohup` over ssh hangs
  until the job ends. `pkill -f <pattern>` over ssh kills the ssh shell itself if the pattern appears in the remote
  command line, so kill by PID. When polling for a "done" marker, match the exact line (`grep -x`), not a word that
  also appears in the echoed command. On macOS zsh, put the ssh command in a wrapper script, not a variable.
- **Pipelines must stop at the first failure** (`set -euo pipefail`). A crashed step once cascaded into later ones.

## Models and serving

- **Qwen3-14B (flag).** A 48 GB card generates at about 280 tokens per second in total and runs out of memory in the
  trainer. A 96 GB RTX PRO 6000 (about $2.69 per hour) gives about 1,200 tokens per second at 48 concurrent
  sequences. With thinking on, the model writes thousands of tokens per turn, so agent episodes are slow: 90 episodes
  took 95 minutes.
- **Qwen3.8-27B (flag, from 2026-11-23).** 55.6 GB in bf16, 64 layers, width 5,120. It needs vLLM 0.17 or later and
  transformers 5.8 or later, and `--language-model-only` (it is multimodal). Set up with `scripts/flag/setup_q38.sh`
  and serve with `serve_q38.sh`. On an A100 80 GB ($2.19 per hour) single-turn generation ran at about 870 tokens per
  second with 64 concurrent requests. Agent episodes were much slower: 48 concurrent episodes filled the KV cache,
  the prefix-cache hit rate fell to 1 to 2%, and 65 of 90 episodes took 3.1 hours. Use fewer concurrent episodes. A
  40,000-token context ended more than half the episodes, so use a larger one. Its chat template keeps reasoning in
  every turn and writes tool calls as XML; `rlp/flag/q38.py` handles both.
- **Steering Qwen3.8-27B inside vLLM (2026-11-24).** vLLM 0.34 builds the model on CUDA from
  `vllm/models/qwen3_5/nvidia/model.py`, not `model_executor/models/qwen3_5.py`, so patch both with
  `scripts/flag/patch_vllm_steer.py` and then `patch_vllm_steer_cuda.py`. Serve with `serve_q38_steer.sh` (prefix
  caching off) and swap vectors with `python -m rlp.flag.onp setvec`. Check that a zero vector reproduces greedy
  output. A steered condition of 330 single-turn prompts takes about 12 minutes.
- **Reading and gradients with transformers on Qwen3.8-27B.** The weights take 54 GB on an 80 GB card. Reading 3,200
  completions took 46 minutes. Gradients through blocks 36 to 63 cost about 13 MB per token, so use gradient
  checkpointing for sequences over about 1,500 tokens. On pods that report many cores set `OMP_NUM_THREADS=8`, or
  numpy and scikit-learn steps hang. Do not `pgrep -f` for the server from the same shell command: it matches itself.
- **Qwen3.8-27B generation is limited by cache memory on an A100 (2026-11-24, second run).** vLLM stores the
  per-sequence recurrent state of the linear-attention blocks in the KV cache, so about 80 sequences fit. Serve with
  `--gpu-memory-utilization 0.97` and `--max-num-seqs 80` (0.92 left 4.8 GiB out of the cache); throughput is 1,000 to
  1,250 tokens per second. The steering vector is one per server, so every steered condition waits for its longest
  completion: about 6 minutes for 150 prompts at a 5,000-token cap. Run `label` in the background so the GPU does not
  idle between conditions. Per-request vectors would need the 0.11 patch (`rlp/manif/vllm_steer.py`) ported to 0.34.
- **Pod setup for the fit.** The client venv also needs `accelerate` and `scikit-learn` (now in `setup_q38.sh`).
  `flash-linear-attention` installs but fails to import against the image's triton, so transformers falls back to
  slow reference kernels: a 125-step fit with 10 pairs per step took 48 minutes. The model download takes 3 to 5
  minutes and the whole setup about 10.
- **Overnight run of 2026-11-25 (scripts in `scripts/flag/night/`).**
  - Per-request steering in vLLM 0.34 works: `patch_vllm_steer_multi.py`, a table of vectors loaded once, and
    `extra_body={"vllm_xargs": {"vid": k}}` per request. Run `rlp.flag.steer_selftest` first. 58 conditions by 50
    prompts took about 100 minutes in one server.
  - Agent rollouts on an A100: about 13 concurrent conversations is the most that keeps the prefix cache (hit rate 63%,
    about 270 generated tokens per second). With 44 or more, every turn recomputes its history (hit rate 2%). Set the
    runner's `--active` equal to `--conc`. A hack episode costs 13,000 to 23,000 generated tokens.
  - `flash-linear-attention` does work in the vLLM venv (torch 2.14 with its matching triton): 7 to 35 times faster
    per block. Reading rollouts with transformers then runs at about 5,000 tokens per second, including
    60,000-token conversations.
  - `serve.sh stop` trusts its pid file. A server started by hand keeps the GPU and the next transformers stage runs
    out of memory. Check `nvidia-smi` after stopping, or use `harvest2.sh`, which kills by name and waits.
  - `stage.sh` skips a stage whose done marker exists; remove the marker to rerun.
  - `pkill -f pattern` over ssh kills the ssh command itself when the pattern is in its text. Write `"[p]attern"`.
  - The repo path has a space: `rsync --backup-dir` and `scp` with unquoted paths fail. zsh does not split
    `$var` into words; pass host and port as separate arguments.
- **Killing a job over ssh.** `ps | grep "[p]attern"` still matches the remote shell when the pattern text appears
  elsewhere in the same ssh command (a `sed` argument did this). Put the kill in a script file on the pod.
- **Piping a job's output through `grep -v` buffers it.** Use `python -u ... | grep --line-buffered`.
- **MPS on this laptop crashes** in `linear` (`MPSNDArray` selector) with torch 2.15; small-model passes run on CPU
  at about 300 tokens per second for Qwen3-0.6B.
- **vLLM 0.11 needs transformers below 5.** With transformers 5 it fails on the Qwen tokenizer.
- **vLLM with a LoRA adapter** runs at about half the base model's speed (1,300 against 2,500 tokens per second for
  Qwen3-4B on a 4090).
- **Steering with Hugging Face `generate` and hooks** is about ten times slower than vLLM and runs out of memory at
  batch 64 × 1,536 tokens on 32 GB. Keep steering conditions few. `rlp/manif/vllm_steer.py` does per-request steering
  inside vLLM and is much faster.
- **Qwen3 chat template.** It drops reasoning from all but the last assistant turn. `rlp/flag/render.py` handles this.
- **OLMo-3.1-32B.** The repository is 129 GB in fp32. vLLM serves it with LoRA adapters on one 96 GB GPU.
- **Hugging Face repos with more than 10,000 files** (the CAFT adapters): `snapshot_download` truncates the listing.
  Use per-file `hf_hub_download`, as `rlp/hub.py` does.

## Judge (OpenAI API)

- The judge model is `gpt-6.1-terra` through the standard API, with `reasoning_effort="low"` (`"minimal"` is
  rejected). The owner decided on one judge pass per item, no multi-vote consensus. `gpt-6-luna` exists but has lower
  recall (see `rlp/flag/results/01_data_and_judge.md`).
- The account limit is one million tokens per minute. A long agent rollout is about 40,000 tokens, so use about three
  workers for rollout judging.
- Do not send the owner's name, email or project details to any external service.

## Layer indexing

`hidden_states[l]` is the residual stream entering decoder block `l` (equivalently, the output of block `l − 1`), and
index 0 is the embeddings. All projects use this convention. Qwen3-4B has 36 blocks and width 2,560, and most 4B work
uses index 18. Qwen3-14B has 40 blocks and width 5,120.

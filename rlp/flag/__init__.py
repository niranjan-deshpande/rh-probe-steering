"""Reward-hack flagging direction (started 2026-11-15): build a direction from synthetic agent transcripts and score
how well its projection flags hack passages in real rollouts, against Goodfire's difference-of-means direction.

Modules: render (chat template + token spans), synth (minimal-pair data), harvest (activations), dirs (difference of
means, logistic probe, random), train (gradient-descent direction), rollouts/read (project labelled rollouts), score.
"""
MODEL = "Qwen/Qwen3-14B"
LAYERS = (8, 12, 16, 20, 24, 28, 32)        # hidden_states indices harvested (0 = embeddings; model has 40 blocks)

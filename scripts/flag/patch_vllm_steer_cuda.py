"""Second half of the vLLM steering patch, and the half that matters on a CUDA pod.

vLLM 0.34 builds Qwen3.5-family models from vllm/models/qwen3_5/nvidia/model.py on CUDA, not from
model_executor/models/qwen3_5.py. Run patch_vllm_steer.py first (it defines the watcher in qwen3_next.py), then this:

    /root/vllm/bin/python scripts/flag/patch_vllm_steer.py && /root/vllm/bin/python scripts/flag/patch_vllm_steer_cuda.py

Verified 2026-11-24: with a zero vector greedy output is identical to the unpatched run; swapping vectors changes it.
"""
import vllm, os
root = os.path.dirname(vllm.__file__)
p = root + "/models/qwen3_5/nvidia/model.py"; s = open(p).read()
assert "rlp-steer" not in s
a = "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n\n    def embed_input_ids"
assert s.count(a) == 1
s = s.replace(a, "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    "        from vllm.model_executor.models.qwen3_next import _steer_watch  # rlp-steer\n"
    "        self.register_buffer(\"steer_vec\", torch.zeros(config.hidden_size), persistent=False)\n"
    "        _steer_watch(self)\n\n    def embed_input_ids")
b = "            hidden_states, residual = layer(\n                positions=positions,"
assert s.count(b) == 1
s = s.replace(b, "            if layer_idx == _RLP_STEER_LAYER:  # rlp-steer\n"
    "                hidden_states = hidden_states + self.steer_vec.to(hidden_states.dtype)\n" + b)
c = "\nclass Qwen3_5Model(nn.Module, EagleModelMixin):"
assert s.count(c) == 1
s = s.replace(c, "\nimport os as _rlp_os  # rlp-steer\n_RLP_STEER_LAYER = int(_rlp_os.environ.get(\"RLP_STEER_LAYER\", \"-1\"))\n\n" + c)
open(p, "w").write(s); print("patched", p)

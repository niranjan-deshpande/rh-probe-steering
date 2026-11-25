"""Patch the installed vLLM so the Qwen3.5-family decoder adds a vector to the residual stream entering one block.

    /root/vllm/bin/python scripts/flag/patch_vllm_steer.py

The patch is inert unless the server is started with RLP_STEER_LAYER (block index, repo convention: the vector is added
to hidden_states[L], the input of block L) and RLP_STEER_FILE (a torch file {"v": [d], "layer": L}). A watcher thread
copies the file into a buffer whenever the file changes and writes <file>.loaded as an acknowledgement, so the vector
can be swapped between conditions without restarting the server. The vector applies to every request and position, so
swap only when no request is in flight, and run the server with prefix caching off.
"""
import pathlib
import vllm

D = pathlib.Path(vllm.__file__).parent / "model_executor" / "models"
MARK = "# rlp-steer"

def edit(path, pairs):
    s = path.read_text()
    if MARK in s:
        print("already patched", path.name)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (path.name, old[:60], s.count(old))
        s = s.replace(old, new)
    path.write_text(s)
    print("patched", path.name)

HELPER = '''
import os as _os, threading as _th, time as _time  ''' + MARK + '''
_STEER_LAYER = int(_os.environ.get("RLP_STEER_LAYER", "-1"))
_STEER_FILE = _os.environ.get("RLP_STEER_FILE", "")


def _steer_watch(mod):
    if _STEER_LAYER < 0 or not _STEER_FILE:
        return

    def loop():
        last = None
        while True:
            try:
                m = _os.stat(_STEER_FILE).st_mtime_ns
                if m != last:
                    z = torch.load(_STEER_FILE, map_location="cpu")
                    assert int(z["layer"]) == _STEER_LAYER, "layer mismatch"
                    buf = mod.steer_vec
                    with torch.no_grad():
                        buf.copy_(z["v"].to(buf.device, buf.dtype))
                    last = m
                    with open(_STEER_FILE + ".loaded", "w") as f:
                        f.write(f"{m} {float(z['v'].float().norm()):.4f} {buf.device} {buf.dtype}\\n")
            except Exception as e:  # noqa: BLE001
                with open(_STEER_FILE + ".err", "w") as f:
                    f.write(repr(e) + "\\n")
            _time.sleep(0.5)

    _th.Thread(target=loop, daemon=True).start()

'''

edit(D / "qwen3_next.py", [
    ("@support_torch_compile\nclass Qwen3NextModel(", HELPER + "\n@support_torch_compile\nclass Qwen3NextModel("),
    ("            hidden_states, residual = layer(\n                positions=positions,\n                hidden_states=hidden_states,\n                residual=residual,\n            )\n            self._maybe_add_hidden_state(",
     "            if layer_idx == _STEER_LAYER:\n                hidden_states = hidden_states + self.steer_vec.to(hidden_states.dtype)\n"
     "            hidden_states, residual = layer(\n                positions=positions,\n                hidden_states=hidden_states,\n                residual=residual,\n            )\n            self._maybe_add_hidden_state("),
])
edit(D / "qwen3_5.py", [
    ("        self.aux_hidden_state_layers: tuple[int, ...] = ()\n\n    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n        # FSE must match construction",
     "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n        from .qwen3_next import _steer_watch  " + MARK + "\n"
     "        self.register_buffer(\"steer_vec\", torch.zeros(config.hidden_size), persistent=False)\n        _steer_watch(self)\n\n"
     "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n        # FSE must match construction"),
])

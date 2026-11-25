"""Fetch what the Jacobian lens needs for Qwen3.8-27B without the full model: the lens file, and (by HTTP range) the
unembedding and final-norm weight from the official safetensors shards, plus the tokenizer.

    .venv/bin/python -m rlp.flag.jlens_fetch
"""
import json, os, struct
import numpy as np, requests
from huggingface_hub import hf_hub_download, hf_hub_url

OUT = "cache/flag/jlens"
MODEL, LENS = "Qwen/Qwen3.8-27B", "eyes-ml/Qwen3.8-27B_jacobian-lens"


def tensor(shard, name, dst):
    if os.path.exists(dst):
        return
    url = hf_hub_url(MODEL, shard)
    n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
    hdr = json.loads(requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}, timeout=60).content)[name]
    a, b = hdr["data_offsets"]
    print(name, hdr["dtype"], hdr["shape"], f"{(b - a) / 1e9:.2f} GB", flush=True)
    with requests.get(url, headers={"Range": f"bytes={8 + n + a}-{8 + n + b - 1}"}, stream=True, timeout=120) as r, open(dst + ".part", "wb") as f:
        r.raise_for_status()
        got = 0
        for chunk in r.iter_content(1 << 24):
            f.write(chunk); got += len(chunk)
            if got % (1 << 28) < (1 << 24):
                print(f"  {got / 1e9:.2f} GB", flush=True)
    json.dump(hdr, open(dst + ".json", "w"))
    os.rename(dst + ".part", dst)


if __name__ == "__main__":
    os.makedirs(OUT + "/dl_model", exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        hf_hub_download(MODEL, f, local_dir=OUT + "/dl_model", token=False)
    tensor("model-00016-of-00018.safetensors", "model.language_model.norm.weight", OUT + "/norm_weight.bin")
    tensor("model-00018-of-00018.safetensors", "lm_head.weight", OUT + "/lm_head.bin")
    hf_hub_download(LENS, "Qwen3.8-27B_jacobian_lens.pt", local_dir=OUT + "/dl_lens", token=False)
    print("FETCH_DONE", flush=True)

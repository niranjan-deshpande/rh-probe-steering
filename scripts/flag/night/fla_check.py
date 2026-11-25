"""Fast-kernel check for the transformers reader on the pod. The last line printed is exactly FLA_OK or FLA_SLOW.

    /root/vllm/bin/python /workspace/rlp/scripts/flag/night/fla_check.py [--limit 600] [--min-speedup 1.5]

Qwen3.8's linear-attention blocks call `chunk_gated_delta_rule`. transformers 5.22.2 binds it once, at import, to
flash-linear-attention (0.6.0, triton kernels) if `import fla` works and otherwise, silently, to a slow PyTorch
reference. FLA_OK means all of this held inside the time limit (10 minutes by default, triton compilation included):
  1. `fla` imports in this venv and transformers bound its kernel, not the reference;
  2. the kernel agrees with the reference, forward and backward, at one block's shapes (48 heads, width 128, bf16);
  3. it is at least --min-speedup times faster forward at 4,096 tokens.
Anything else (import error, mismatch, no speedup, crash, timeout) is FLA_SLOW.

The verdict is also written to <cache>/flag/fla_status.txt. rlp.flag.gf27 (the loader shared by the reader modules)
reads it and blocks `import fla` when it says FLA_SLOW, so a kernel that failed this check is not used later even
though the package is still installed. RLP_NO_FLA=1 / 0 in the environment overrides the file either way.
The exit status is 0 for both verdicts; 2 only for a usage error.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATUS = Path(os.environ.get("RLP_CACHE", ROOT / "cache")) / "flag" / "fla_status.txt"


def inner(min_speedup):
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available(), "no GPU: the triton kernels cannot be tested here"
    dev = "cuda"
    import triton
    print("torch", torch.__version__, "triton", triton.__version__, flush=True)
    # Import FLA directly so that the real exception is shown (transformers swallows it).
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    import fla
    print("fla", getattr(fla, "__version__", "?"), flush=True)
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    print("transformers", transformers.__version__, flush=True)
    wrapped = M.torch_chunk_gated_delta_rule
    bound = [c.cell_contents for c in (wrapped.__closure__ or ()) if callable(c.cell_contents)]
    mods = sorted({getattr(f, "__module__", "?") for f in bound})
    print("implementations bound by transformers:", mods, flush=True)
    assert any(m.startswith("fla") for m in mods), "transformers is on the reference kernel"
    ref = wrapped.__wrapped__                       # functools.wraps keeps the PyTorch reference here
    assert ref.__module__.startswith("transformers"), ref.__module__

    def inputs(T, B=1, H=48, D=128, seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=gen, device=dev)
        q, k, v = (r(B, T, H, D).bfloat16() for _ in range(3))
        g = -F.softplus(r(B, T, H) - 2.0)           # log decay <= 0, float32 as in the model
        beta = torch.sigmoid(r(B, T, H)).bfloat16()
        return [x.requires_grad_() for x in (q, k, v, g, beta)]

    def run(fn, xs, back):
        out, _ = fn(*xs, use_qk_l2norm_in_kernel=True)
        if back:
            out.backward(torch.ones_like(out) * 1e-2)
        return out

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))

    def timed(fn, T, back, reps=5):
        xs = inputs(T)
        run(fn, xs, back)                           # warm-up and triton compilation
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(reps):
            for x in xs:
                x.grad = None
            run(fn, xs, back)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3, torch.cuda.max_memory_allocated() / 2 ** 20

    for T in (512, 4096):
        a, b = inputs(T, seed=1), inputs(T, seed=1)
        oa, ob = run(fla_chunk, a, True), run(ref, b, True)
        errs = {"out": rel(oa, ob)} | {n: rel(x.grad, y.grad) for n, x, y in zip("qkvgb", a, b)}
        print(f"T={T} relative error against the reference", {k: f"{v:.1e}" for k, v in errs.items()}, flush=True)
        assert errs["out"] < 2e-2 and max(errs.values()) < 6e-2, errs
    speed = {}
    for T in (512, 4096):
        for back in (False, True):
            tf, mf = timed(fla_chunk, T, back)
            tr, mr = timed(ref, T, back)
            speed[(T, back)] = tr / tf
            print(f"T={T} {'fwd+bwd' if back else 'fwd    '}  fla {tf:7.1f} ms {mf:6.0f} MiB | reference {tr:7.1f} ms {mr:6.0f} MiB"
                  f" | x{tr / tf:.1f} (one block of 48)", flush=True)
    assert speed[(4096, False)] >= min_speedup, f"forward speedup at 4096 tokens is x{speed[(4096, False)]:.2f} < x{min_speedup}"
    print("FLA_INNER_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=600.0, help="seconds for the whole check")
    ap.add_argument("--min-speedup", type=float, default=1.5)
    ap.add_argument("--status", default=str(STATUS))
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.inner:
        inner(a.min_speedup)
        return
    ok, why = False, ""
    try:
        env = dict(os.environ, RLP_NO_FLA="0")      # test the kernel itself, whatever an earlier verdict said
        p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--inner", "--min-speedup", str(a.min_speedup)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=a.limit, env=env)
        out = p.stdout or ""
        ok = p.returncode == 0 and "FLA_INNER_OK" in out.splitlines()
        why = "" if ok else f"check failed (exit {p.returncode})"
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        why = f"no verdict within {a.limit:.0f} s"
    except Exception as e:                          # never leave the caller without a verdict
        out, why = "", f"{type(e).__name__}: {e}"
    for line in out.splitlines():
        if line.strip() not in ("FLA_OK", "FLA_SLOW"):
            print("  " + line)
    if why:
        print(f"[fla_check] {why}; the reader will use the reference kernels")
    verdict = "FLA_OK" if ok else "FLA_SLOW"
    try:
        Path(a.status).parent.mkdir(parents=True, exist_ok=True)
        Path(a.status).write_text(verdict + "\n")
    except OSError as e:
        print(f"[fla_check] could not write {a.status}: {e}")
    print(verdict, flush=True)


if __name__ == "__main__":
    main()

"""Tier-2 test 6: per-layer activation kurtosis, Bonsai ternary vs FP16.

Discriminator rationale (audit README): the ternary build's extreme KV-cache
tolerance would be explained by *flattened* activation distributions —
noise-injection-style training flattens them; plain PTQ does not. Excess
kurtosis of each decoder block's output, one fixed mixed batch, both models.

Run with stock mlx (the ternary pack needs no fork).
"""
import os

import numpy as np
import mlx.core as mx
from mlx_lm import load

# alis-dwq must be importable: pip install -e <repo root>
from alis_dwq.layerwise import _capture_hiddens  # class-level call patch
from alis_dwq.eval_kld import get_tokens

AUDIT = os.environ.get("BONSAI_AUDIT_DIR")
if not AUDIT:
    raise SystemExit("set BONSAI_AUDIT_DIR to the directory holding the "
                     "Bonsai/Qwen dumps (see this example's README)")
MODELS = {
    "bonsai": os.path.join(AUDIT, "Ternary-Bonsai-27B-mlx-2bit"),
    "fp16": os.path.join(AUDIT, "Qwen3.6-27B"),
}
N_TOK = 768  # EN/code/ZH thirds via eval_kld's fixed slice


def excess_kurtosis(x):
    x = x.astype(np.float64).reshape(-1)
    m = x.mean()
    v = ((x - m) ** 2).mean()
    if v == 0:
        return float("nan")
    return float(((x - m) ** 4).mean() / v**2 - 3.0)


def main():
    results = {}
    for tag, path in MODELS.items():
        model, tok = load(path)
        mx.eval(model.parameters())
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        ids = get_tokens(tok, N_TOK)[None]
        # model.layers for the qwen3_5 wrapper lives on language_model.model
        base = model
        while not hasattr(base, "layers"):
            base = getattr(base, "language_model", None) or base.model
        hid = _capture_hiddens(base, ids)
        ks = np.array([excess_kurtosis(h) for h in hid if h is not None])
        results[tag] = ks
        print(f"[kurt] {tag}: {len(ks)} layers  "
              f"median={np.median(ks):.2f}  p90={np.percentile(ks,90):.2f}  "
              f"max={ks.max():.2f}")
        del model
        mx.clear_cache()
    b, f = results["bonsai"], results["fp16"]
    n = min(len(b), len(f))
    ratio = b[:n] / np.where(np.abs(f[:n]) < 1e-9, np.nan, f[:n])
    flatter = int((b[:n] < f[:n]).sum())
    print(f"[kurt] per-layer kurtosis ratio bonsai/fp16: "
          f"median={np.nanmedian(ratio):.3f}  layers-flatter={flatter}/{n}")
    print("[kurt] reading: ratio << 1 across layers = flattened activations "
          "(noise-injection-training signature); ~1 = geometry preserved "
          "(compensated-PTQ-compatible)")
    np.savez(os.path.join(AUDIT, "kurtosis.npz"), bonsai=b, fp16=f)
    print("KURT_DONE")


if __name__ == "__main__":
    main()

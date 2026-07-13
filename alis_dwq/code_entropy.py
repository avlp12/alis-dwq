"""Code-entropy scanner: measure how much of the bit budget a quantized
student actually uses — no source, no GPU-heavy pass, student dir only.

The lens comes from lossless BF16 compression (brianbell-x/weight-compression:
9 exponent bits carry ~4 bits of information). Turned on our own artifacts:
a 2-bit tensor stores 4 levels per group, but a min-max affine grid stretched
by one outlier leaves most weights in 1-2 interior codes — nominal 2 bits,
effective 1.x. Per-tensor code-histogram entropy is that number, and it is
computable in minutes from the packed codes alone.

Uses:
  1. clip_quantize pre-scan — low code entropy is exactly the pathology
     clipping fixes, so the entropy ranking predicts where a clip pass will
     pay off BEFORE hauling a multi-hundred-GB source onto the box.
  2. effective-bpw reporting — compare recipes by information actually
     stored, not nominal bits; low-entropy layers are bit-reallocation
     candidates alongside expert_traffic / sensitivity signals.
  3. (hypothesis, unvalidated) lattice-source anomaly fingerprinting — the
     E1 grid-resonance failure passed every per-tensor mean metric; comb
     structure in code histograms is a candidate artifact-level tripwire.

Codes are recovered exactly as round((dequant - bias)/scale) through
mx.dequantize itself, so no assumption is made about MLX's packed layout.
bits/group_size are inferred from shapes and disambiguated via config.json's
"quantization" section when shapes alone are ambiguous (e.g. 4b/gs64 vs
8b/gs32 pack identically).

  python -m alis_dwq.code_entropy --model <student> --save entropy.npz
  python -m alis_dwq.code_entropy --load entropy.npz --top 20   # numpy-only
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

VALID_BITS = (2, 3, 4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
MAX_LEVELS = 256  # histogram rows are padded to the 8-bit level count


def infer_qparams(wq_shape, sc_shape, base, qcfg):
    """(bits, group_size) from packed/scales shapes; config disambiguates."""
    cands = []
    for gs in GROUP_SIZES:
        in_dim = sc_shape[-1] * gs
        if in_dim and (wq_shape[-1] * 32) % in_dim == 0:
            bits = (wq_shape[-1] * 32) // in_dim
            if bits in VALID_BITS:
                cands.append((bits, gs))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    if qcfg:
        over = qcfg.get(base)
        want = over if isinstance(over, dict) else qcfg
        pick = (want.get("bits", qcfg.get("bits")), want.get("group_size", qcfg.get("group_size")))
        if pick in cands:
            return pick
    # bits*gs products collide ((3,64)=(6,32), (4,64)=(8,32)=(2,128), ...),
    # so shapes alone rarely disambiguate — prefer mlx-lm's default gs=64,
    # then larger groups; the caller flags these rows as ambiguous
    return sorted(cands, key=lambda c: (c[1] != 64, -c[1]))[0]


def scan(model_dir, chunk_elems=1 << 27):
    """Recover codes tensor-by-tensor; returns the arrays analyze() needs."""
    import mlx.core as mx
    from .clip_quantize import _load_dir

    weights, _ = _load_dir(model_dir)
    cfg_path = Path(model_dir) / "config.json"
    qcfg = {}
    if cfg_path.exists():
        qcfg = json.load(open(cfg_path)).get("quantization", {}) or {}

    names, bits_l, gs_l, nparams, hists, grp_lo, layer_ids, ambig = \
        [], [], [], [], [], [], [], []
    bases = sorted(k[: -len(".scales")] for k in weights if k.endswith(".scales"))
    for base in bases:
        wq, sc = weights[base + ".weight"], weights[base + ".scales"]
        bi = weights.get(base + ".biases")
        if bi is None:
            print(f"[entropy] skip {base}: non-affine", file=sys.stderr)
            continue
        got = infer_qparams(tuple(wq.shape), tuple(sc.shape), base, qcfg)
        if got is None:
            print(f"[entropy] skip {base}: cannot infer bits/group_size", file=sys.stderr)
            continue
        bits, gs = got
        nlev = 1 << bits
        rows = int(wq.shape[0])
        per_row = 1
        for d in wq.shape[1:]:
            per_row *= int(d)
        step = max(1, int(chunk_elems // max(per_row * (32 // bits), 1)))

        hist = mx.zeros((nlev,), dtype=mx.float32)
        lo_groups = tot_groups = 0.0
        for i in range(0, rows, step):
            dq = mx.dequantize(weights[base + ".weight"][i:i + step],
                               sc[i:i + step], bi[i:i + step],
                               group_size=gs, bits=bits).astype(mx.float32)
            head = dq.shape[:-1]
            G = dq.shape[-1] // gs
            s = sc[i:i + step].astype(mx.float32)[..., None]
            b = bi[i:i + step].astype(mx.float32)[..., None]
            s = mx.where(mx.abs(s) < 1e-30, mx.ones_like(s), s)
            q = mx.clip(mx.round((dq.reshape(*head, G, gs) - b) / s), 0, nlev - 1)
            flat = q.reshape(-1).astype(mx.uint32)
            hist = hist.at[flat].add(mx.ones(flat.shape, dtype=mx.float32))
            # per-group entropy (cheap at low bits, where it matters)
            if bits <= 4:
                gh = mx.stack([(q == k).sum(axis=-1) for k in range(nlev)], axis=-1)
                p = gh.astype(mx.float32) / gs
                h = -(p * mx.log2(mx.maximum(p, 1e-12))).sum(axis=-1)
                lo_groups += float((h < bits / 2).astype(mx.float32).sum().item())
                tot_groups += float(h.size)
            mx.eval(hist)
        hv = np.zeros(MAX_LEVELS, dtype=np.int64)
        hv[:nlev] = np.array(hist, copy=True).astype(np.int64)
        names.append(base)
        bits_l.append(bits)
        gs_l.append(gs)
        nparams.append(int(hv.sum()))
        hists.append(hv)
        grp_lo.append(lo_groups / tot_groups if tot_groups else np.nan)
        mm = _LAYER_RE.search(base)
        layer_ids.append(int(mm.group(1)) if mm else -1)
        ambig.append(got is not None and
                     len([1 for g in GROUP_SIZES
                          if sc.shape[-1] * g and (wq.shape[-1] * 32) % (sc.shape[-1] * g) == 0
                          and (wq.shape[-1] * 32) // (sc.shape[-1] * g) in VALID_BITS]) > 1)
    if not names:
        raise SystemExit("[entropy] no affine-quantized tensors found")
    return (np.array(names), np.array(bits_l, dtype=np.int64),
            np.array(gs_l, dtype=np.int64), np.array(nparams, dtype=np.int64),
            np.stack(hists), np.array(grp_lo), np.array(layer_ids, dtype=np.int64),
            np.array(ambig))


# ------------------------------------------------------------------ analysis

def _entropy(hist):
    tot = hist.sum()
    if tot == 0:
        return 0.0
    p = hist[hist > 0] / tot
    return float(-(p * np.log2(p)).sum())


def analyze(names, bits, gs, nparams, hists, grp_lo, layer_ids, ambig, top=15):
    T = len(names)
    ent = np.array([_entropy(hists[t]) for t in range(T)])
    util = ent / bits
    nlev = (1 << bits.astype(np.int64))
    ext = np.array([  # mass on the two grid ends = min-max anchor occupancy
        (hists[t, 0] + hists[t, nlev[t] - 1]) / max(hists[t].sum(), 1)
        for t in range(T)
    ])

    w = nparams / nparams.sum()
    eff_bpw = float((ent * w).sum())
    nom_bpw = float((bits * w).sum())
    print(f"[entropy] {T} tensors; code payload: nominal {nom_bpw:.2f} b/w, "
          f"effective {eff_bpw:.2f} b/w ({eff_bpw/nom_bpw*100:.0f}% utilized)")
    if ambig.any():
        print(f"[entropy] {int(ambig.sum())} tensors had shape-ambiguous bits/gs "
              "(config.json used/fallback) — treat their rows with care")

    for lid in sorted(set(layer_ids.tolist())):
        m = layer_ids == lid
        lw = nparams[m] / max(nparams[m].sum(), 1)
        tag = f"L{lid}" if lid >= 0 else "extras"
        gl = grp_lo[m]
        gl_s = f"{np.nanmean(gl)*100:4.1f}%" if not np.all(np.isnan(gl)) else "  n/a"
        print(f"[entropy] {tag:>6}: eff {float((ent[m]*lw).sum()):.2f}/"
              f"{float((bits[m]*lw).sum()):.2f} b/w  low-H groups {gl_s}  "
              f"anchor mass {float((ext[m]*lw).sum())*100:4.1f}%")

    order = np.argsort(util)
    print(f"[entropy] {min(top, T)} lowest-utilization tensors "
          "(clip_quantize / bit-reallocation candidates first):")
    for t in order[:top]:
        print(f"[entropy]   {util[t]*100:5.1f}%  {int(bits[t])}b/gs{int(gs[t])}  "
              f"H={ent[t]:.2f}  anchors={ext[t]*100:4.1f}%  {names[t]}")
    return ent, util, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="quantized student dir (needs mlx)")
    ap.add_argument("--load", help="re-analyze a saved .npz (numpy-only)")
    ap.add_argument("--save", help="write scan results to this .npz")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    if bool(a.model) == bool(a.load):
        ap.error("pass exactly one of --model / --load")

    if a.load:
        z = np.load(a.load)
        args = (z["names"], z["bits"], z["gs"], z["nparams"], z["hists"],
                z["grp_lo"], z["layer_ids"], z["ambig"])
    else:
        args = scan(a.model)
        if a.save:
            keys = ("names", "bits", "gs", "nparams", "hists", "grp_lo",
                    "layer_ids", "ambig")
            np.savez(a.save, **dict(zip(keys, args)))
            print(f"[entropy] saved {a.save}", file=sys.stderr)
    analyze(*args, top=a.top)


if __name__ == "__main__":
    main()

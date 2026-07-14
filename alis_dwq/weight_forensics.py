"""Method-class fingerprints for a transformed model vs its original.

Built for the Bonsai-27B audit (examples/bonsai-27b-audit): both the
transformed weights (Apache-2.0) and the source model are public, so the
proprietary transformation can be classified from its endpoints. Four
fingerprints per matched tensor, then a verdict:

  A. projection agreement — best match between shipped ternary codes and the
     closed-form TWN projection of the ORIGINAL (threshold sweep). ~1.0 =>
     direct analytic rounding; the theory lives in scale/threshold choice.
  B. column drift — agreement per input-column bin. OBS/GPTQ-family methods
     compensate in column order, so agreement decays with position; gradient
     training moves weights position-independently (flat, lower agreement).
  C. rotation test — singular-value spectra are orthogonal-invariant: high
     spectra correlation + low elementwise weight correlation => the basis
     was rotated before quantization.
  D. scale objective — shipped effective level magnitude vs the MSE-optimal
     closed form (mean |w| over the support). ~1.0 analytic, else trained.

  python -m alis_dwq.weight_forensics --original <dir> --transformed <dir> \
      [--pattern mlp] [--max-tensors 40] [--spectra-max 8] [--save f.npz]

Ternary codes are read from the shipped container by level *value*: levels
with |value| < 0.4x the group's max level classify as 0. This handles both
the scale-only packing (levels {-s, 0, +s, 2s}, code 3 unused — any code-3
usage is itself reported: it would falsify a "ternary" label) and min-max
affine surrogates. Analysis is numpy; only the loader needs mlx.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

K_GRID = np.linspace(0.4, 1.2, 9)
N_BINS = 8


# ------------------------------------------------------------------- core

def twn_project(w, k):
    """Closed-form ternary projection: support |w| > k*mean|w| (per row)."""
    thr = k * np.abs(w).mean(axis=-1, keepdims=True)
    return np.sign(w) * (np.abs(w) > thr)


def projection_sweep(w_o, t):
    agrees = [(float((twn_project(w_o, k) == t).mean()), float(k)) for k in K_GRID]
    return max(agrees)


def column_drift(w_o, t, k_best):
    """Compensation fingerprint. Column-ordered methods (OBS/GPTQ family)
    quantize the FIRST columns before any error has accumulated — those
    match the naive projection — then settle into a stationary compensated
    regime within tens of columns. So the signature is head-vs-tail, not a
    gradual slope: drift = tail agreement − first-columns agreement < 0."""
    tp = twn_project(w_o, k_best)
    n = w_o.shape[-1]
    head_n = max(2, n // 128)
    edges = np.linspace(0, n, N_BINS + 1, dtype=int)
    bins = [float((tp[..., a:b] == t[..., a:b]).mean())
            for a, b in zip(edges[:-1], edges[1:])]
    head = float((tp[..., :head_n] == t[..., :head_n]).mean())
    tail = float((tp[..., n // 2:] == t[..., n // 2:]).mean())
    return bins, tail - head


def spectra_corr(w_o, w_t, top=64):
    so = np.linalg.svd(w_o.astype(np.float64), compute_uv=False)
    st = np.linalg.svd(w_t.astype(np.float64), compute_uv=False)
    m = min(top, len(so), len(st))
    so, st = so[:m] / so[0], st[:m] / st[0]
    c = np.corrcoef(so, st)[0, 1]
    wc = np.corrcoef(w_o.reshape(-1), w_t.reshape(-1))[0, 1]
    return float(c), float(wc)


def scale_ratio(w_o, t, s_eff):
    """shipped level magnitude / MSE-optimal (mean |w_o| over support), per group row."""
    sup = t != 0
    num = np.where(sup, np.abs(w_o), 0.0).sum(axis=-1)
    cnt = sup.sum(axis=-1)
    ok = cnt > 0
    opt = np.where(ok, num / np.maximum(cnt, 1), np.nan)
    r = s_eff / opt
    return float(np.nanmean(r)), float(np.nanstd(r))


def verdict(agree, drift, spec_c, w_c):
    if agree > 0.98 and abs(drift) < 0.01:
        return "direct analytic projection (rounding; theory in scale/threshold)"
    if agree > 0.65 and drift < -0.015:
        return "column-ordered compensation (OBS/GPTQ family)"
    if agree < 0.6 and spec_c > 0.95 and w_c < 0.5:
        return "rotated basis before quantization"
    if agree > 0.65:
        return "weights moved position-independently (training/distillation)"
    return "unclassified (mixed or novel transformation)"


# ----------------------------------------------------------------- loader

def iter_pairs(original, transformed, pattern, max_tensors):
    import mlx.core as mx
    from .clip_quantize import _load_dir
    from .code_entropy import infer_qparams
    import json

    orig, _ = _load_dir(original)
    trans, _ = _load_dir(transformed)
    qcfg = {}
    cfg = Path(transformed) / "config.json"
    if cfg.exists():
        qcfg = json.load(open(cfg)).get("quantization", {}) or {}

    n = 0
    for key in sorted(k for k in trans if k.endswith(".scales")):
        base = key[: -len(".scales")]
        if pattern and not re.search(pattern, base):
            continue
        wq, sc, bi = trans[base + ".weight"], trans[key], trans.get(base + ".biases")
        wo = orig.get(base + ".weight")
        if bi is None or wo is None or len(wq.shape) != 2:
            continue  # 2-D tensors only (expert stacks: pass --pattern per-slice later)
        got = infer_qparams(tuple(wq.shape), tuple(sc.shape), base, qcfg)
        if got is None or int(wo.shape[-1]) != sc.shape[-1] * got[1]:
            continue
        bits, gs = got
        dq = np.array(mx.dequantize(wq, sc, bi, group_size=gs, bits=bits)
                      .astype(mx.float32), copy=True)
        wo_np = np.array(wo.astype(mx.float32), copy=True)
        # per-group magnitude classes: the ternary level is the *median*
        # nonzero magnitude; anything ~2x above it is the scale-only pack's
        # unused-code-3 level ({-s,0,s,2s}) — nonzero usage falsifies a
        # plain "ternary" label and is reported per tensor
        G = dq.shape[-1] // gs
        gmag = np.abs(dq).reshape(dq.shape[0], G, gs)
        s_eff = gmag.max(axis=-1)
        t = (np.sign(dq) * (gmag >= 0.4 * np.maximum(s_eff[..., None], 1e-12))
             .reshape(dq.shape))
        lo = np.where(gmag > 1e-12, gmag, np.nan)
        base_lv = np.nanmin(lo, axis=-1)  # smallest nonzero level per group
        hi_frac = float(np.mean(gmag > 1.5 * np.maximum(base_lv[..., None], 1e-12)))
        yield (base, wo_np, dq, t, hi_frac,
               wo_np.reshape(-1, gs), t.reshape(-1, gs), s_eff.reshape(-1))
        n += 1
        if n >= max_tensors:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--transformed", required=True)
    ap.add_argument("--pattern", default="", help="regex filter on tensor names")
    ap.add_argument("--max-tensors", type=int, default=40)
    ap.add_argument("--spectra-max", type=int, default=8,
                    help="SVD spectra on at most this many tensors (costly)")
    a = ap.parse_args()

    print("[forensics][EXPERIMENTAL] classifying the transformation from its "
          "endpoints — see examples/bonsai-27b-audit for interpretation",
          file=sys.stderr)
    rows, spectra_done = [], 0
    for base, wo, dq, t, hi_frac, wo_g, t_g, s_eff in iter_pairs(
            a.original, a.transformed, a.pattern, a.max_tensors):
        agree, k_best = projection_sweep(wo, t)
        bins, drift = column_drift(wo, t, k_best)
        sc_mean, sc_std = scale_ratio(wo_g, t_g, s_eff)
        spec_c = w_c = float("nan")
        if spectra_done < a.spectra_max and min(wo.shape) <= 8192:
            spec_c, w_c = spectra_corr(wo, dq)
            spectra_done += 1
        rows.append((base, agree, k_best, drift, spec_c, w_c, sc_mean, sc_std))
        print(f"[forensics] {base}\n"
              f"[forensics]   projection agree={agree*100:5.1f}% @k={k_best:.1f}  "
              f"col-drift={drift:+.3f}  spectra={spec_c:.3f}  w-corr={w_c:.3f}  "
              f"scale-ratio={sc_mean:.3f}±{sc_std:.3f}  4th-level={hi_frac*100:.2f}%")

    if not rows:
        raise SystemExit("[forensics] no comparable tensor pairs found")
    ag = float(np.mean([r[1] for r in rows]))
    dr = float(np.mean([r[3] for r in rows]))
    sp = float(np.nanmean([r[4] for r in rows]))
    wc = float(np.nanmean([r[5] for r in rows]))
    print(f"[forensics] aggregate: agree={ag*100:.1f}%  drift={dr:+.3f}  "
          f"spectra={sp:.3f}  w-corr={wc:.3f}")
    print(f"[forensics] VERDICT: {verdict(ag, dr, sp, wc)}")
    print("[forensics] (verdict is a heuristic over these fingerprints — "
          "read the per-tensor rows before quoting it)")


if __name__ == "__main__":
    main()

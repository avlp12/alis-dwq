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

Known limits (verified in the 2026-07-15 3-lens review — read verdict()'s
docstring before quoting a verdict): the projection sweep thresholds per
ROW while shipped checkpoints scale per GROUP, so group-wise analytic
methods depress agreement without any training; the drift test is blind to
act-order (processing-order-permuted) compensation; and scale-ratio < 1 is
expected for absmean-family analytic scales, not evidence of training.
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
    """Heuristic over the fingerprints. Two measured blind spots (3-lens
    review, 2026-07-15) bound what the positive verdicts may claim:

    - the drift test assumes processing order == storage order; **act-order
      GPTQ** (desc_act, standard in AutoGPTQ/GPTQModel) permutes processing
      by Hessian diagonal and then inverse-permutes — simulated drift
      collapses from -0.078 to -0.011, i.e. it reads as "flat". Flat drift
      therefore rules out *storage-ordered* compensation only.
    - scale-ratio < 1 is NOT a training signature: BitNet-style absmean
      scales (s = mean|w| over the whole group, no training) sit at
      ~0.67-0.75 of this tool's conditional-centroid reference on
      Gaussian/Laplace weights. Only ~1.0 (analytic centroid) is a sharp
      reading.

    So "moved position-independently" separates the transform from direct
    rounding and from storage-ordered compensation — it cannot separate
    training/distillation from order-permuted or Hessian-weighted PTQ."""
    if agree > 0.98 and abs(drift) < 0.01:
        return "direct analytic projection (rounding; theory in scale/threshold)"
    if agree > 0.65 and drift < -0.015:
        return "column-ordered compensation (OBS/GPTQ family, storage-ordered)"
    if agree < 0.6 and spec_c > 0.95 and w_c < 0.5:
        return "rotated basis before quantization"
    if agree > 0.65:
        return ("weights moved position-independently — training/distillation "
                "OR order-permuted compensation (act-order GPTQ class); "
                "endpoints alone cannot separate these")
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


# ------------------------------------------------------------- selftest

def _gptq_ternary(w, H, order):
    """Minimal GPTQ-style ternary quantizer (numpy): quantize columns in
    `order`, absorb each column's rounding error into the remaining columns
    through the inverse-Hessian row (Frantar et al.), levels {-s, 0, +s}
    with per-row absmean scale."""
    w = w.copy()
    d = w.shape[1]
    Hinv = np.linalg.inv(H + 1e-3 * np.eye(d))
    s = np.abs(w).mean(axis=1, keepdims=True)
    thr = 0.5 * s
    q = np.zeros_like(w)
    for idx, j in enumerate(order):
        col = w[:, j]
        qc = np.sign(col) * (np.abs(col) > thr[:, 0]) * s[:, 0]
        q[:, j] = qc
        err = (col - qc) / Hinv[j, j]
        rest = order[idx + 1:]
        if len(rest):
            w[:, rest] -= np.outer(err, Hinv[j, rest])
    return np.sign(q)  # ternary codes in {-1, 0, +1}


def _selftest():
    """Method-class discrimination on synthetic ground truth, including the
    act-order blind spot the 2026-07-15 3-lens review demonstrated: the
    drift test detects storage-ordered GPTQ compensation but reads
    act-order (Hessian-permuted) GPTQ as flat — i.e. as the
    training/compensation-ambiguous class. This is a documented limit, not
    a bug; the selftest pins it so a future 'refuted' overclaim fails loud."""
    rng = np.random.default_rng(11)
    R, D = 256, 512
    w = rng.normal(size=(R, D)).astype(np.float64)
    X = rng.normal(size=(2048, D))
    # per-column energy skew at RANDOM positions: act-order's Hessian sort
    # is then decorrelated from storage order — the condition under which
    # the head-vs-tail drift test goes blind. (If a model's high-energy
    # channels were contiguous in storage, act-order would remain partially
    # detectable; the blind spot is order-decorrelation, not act-order per se.)
    X *= rng.uniform(0.5, 3.0, size=D)
    H = X.T @ X / len(X)

    # 1. direct analytic projection: near-total agreement, flat drift
    t_proj = twn_project(w, 0.5)
    agree, k = projection_sweep(w, t_proj)
    _, drift = column_drift(w, t_proj, k)
    assert agree > 0.98 and abs(drift) < 0.01, (agree, drift)
    v1 = verdict(agree, drift, 1.0, 1.0)
    assert v1.startswith("direct analytic"), v1

    # 2. storage-order GPTQ: compensation drift is DETECTED
    t_sto = _gptq_ternary(w, H, np.arange(D))
    agree_s, k_s = projection_sweep(w, t_sto)
    _, drift_s = column_drift(w, t_sto, k_s)
    assert drift_s < -0.015, f"storage-order drift not detected: {drift_s:.3f}"
    v2 = verdict(agree_s, drift_s, 1.0, 1.0)
    assert "storage-ordered" in v2, v2

    # 3. act-order GPTQ: same compensation, permuted processing order ->
    #    drift reads FLAT (the blind spot). Verdict falls into the ambiguous
    #    position-independent class by design.
    order = np.argsort(-np.diag(H))
    t_act = _gptq_ternary(w, H, order)
    agree_a, k_a = projection_sweep(w, t_act)
    _, drift_a = column_drift(w, t_act, k_a)
    assert drift_a > -0.015, f"act-order unexpectedly detected: {drift_a:.3f}"
    assert agree_a < 0.98
    v3 = verdict(agree_a, drift_a, 1.0, 1.0)
    assert "cannot separate" in v3, v3

    print(f"[forensics] selftest OK — projection {agree*100:.1f}%/{drift:+.3f} "
          f"| storage-order GPTQ {agree_s*100:.1f}%/{drift_s:+.3f} (detected) "
          f"| act-order GPTQ {agree_a*100:.1f}%/{drift_a:+.3f} (flat: the "
          "documented blind spot)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="method-class discrimination on synthetic ground "
                         "truth incl. the act-order blind spot (numpy-only)")
    args_pre, _ = ap.parse_known_args()
    if args_pre.selftest:
        _selftest()
        return
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

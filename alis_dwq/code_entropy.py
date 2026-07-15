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


def scan(model_dir, chunk_elems=1 << 27, per_expert=False):
    """Recover codes tensor-by-tensor; returns the arrays analyze() needs.

    per_expert additionally accumulates a code histogram per expert (axis 0
    of 3D expert stacks, bits <= 4) — the student-only proxy for per-expert
    quantization damage, the selection criterion the NF3-hybrid's v3.6 swap
    validated on flat-routing GLM-5.2 (frequency saliency -> measured damage:
    -16% KLD). Low expert code entropy = stretched grid = high damage."""
    import mlx.core as mx
    from .clip_quantize import _load_dir

    weights, _ = _load_dir(model_dir)
    cfg_path = Path(model_dir) / "config.json"
    qcfg = {}
    if cfg_path.exists():
        qcfg = json.load(open(cfg_path)).get("quantization", {}) or {}

    names, bits_l, gs_l, nparams, hists, grp_lo, layer_ids, ambig = \
        [], [], [], [], [], [], [], []
    pe_names, pe_layer_ids, pe_bits, pe_hists = [], [], [], []
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

        # int64 accumulation on the host: a float32 GPU *scatter-add*
        # histogram silently saturates at 2^24 per bin (sequential adds of
        # 1.0 past 16,777,216 are lost — pairwise reductions would not
        # saturate, scatter-add accumulation does), which clamped every
        # >~50M-param tensor to a fake uniform histogram on the first
        # real-model run (Bonsai 27B, 2026-07-15)
        hist = np.zeros(nlev, dtype=np.int64)
        lo_groups = tot_groups = 0.0
        want_pe = per_expert and len(wq.shape) == 3 and bits <= 4
        pe_chunks = []
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
            # int32 per-chunk bins are exact only while a chunk stays under
            # 2^31 codes; `step` is derived from chunk_elems with a
            # `32 // bits` unpack estimate and a max(1, ...) floor, neither
            # of which is a hard bound — enforce it here instead of hoping
            assert flat.size < (1 << 31), (
                f"chunk of {flat.size} codes overflows int32 bins ({base}); "
                "lower chunk_elems")
            ch = mx.zeros((nlev,), dtype=mx.int32)
            ch = ch.at[flat].add(mx.ones(flat.shape, dtype=mx.int32))
            mx.eval(ch)
            hist += np.array(ch, copy=True).astype(np.int64)
            # per-group entropy (cheap at low bits, where it matters)
            if bits <= 4:
                gh = mx.stack([(q == k).sum(axis=-1) for k in range(nlev)], axis=-1)
                p = gh.astype(mx.float32) / gs
                h = -(p * mx.log2(mx.maximum(p, 1e-12))).sum(axis=-1)
                lo_groups += float((h < bits / 2).astype(mx.float32).sum().item())
                tot_groups += float(h.size)
            if want_pe:  # chunks split on axis 0, so experts never straddle
                pe = mx.stack([(q == k).reshape(q.shape[0], -1).sum(axis=-1)
                               for k in range(nlev)], axis=-1)
                mx.eval(pe)
                pe_chunks.append(np.array(pe, copy=True).astype(np.int64))
        hv = np.zeros(MAX_LEVELS, dtype=np.int64)
        hv[:nlev] = hist
        names.append(base)
        bits_l.append(bits)
        gs_l.append(gs)
        nparams.append(int(hv.sum()))
        hists.append(hv)
        grp_lo.append(lo_groups / tot_groups if tot_groups else np.nan)
        mm = _LAYER_RE.search(base)
        layer_ids.append(int(mm.group(1)) if mm else -1)
        if want_pe and pe_chunks:
            peh = np.concatenate(pe_chunks, axis=0)  # (E, nlev)
            pad = np.zeros((peh.shape[0], 16), dtype=np.int64)
            pad[:, :nlev] = peh
            pe_names.append(base)
            pe_layer_ids.append(int(mm.group(1)) if mm else -1)
            pe_bits.append(bits)
            pe_hists.append(pad)
        ambig.append(got is not None and
                     len([1 for g in GROUP_SIZES
                          if sc.shape[-1] * g and (wq.shape[-1] * 32) % (sc.shape[-1] * g) == 0
                          and (wq.shape[-1] * 32) // (sc.shape[-1] * g) in VALID_BITS]) > 1)
    if not names:
        raise SystemExit("[entropy] no affine-quantized tensors found")
    core = (np.array(names), np.array(bits_l, dtype=np.int64),
            np.array(gs_l, dtype=np.int64), np.array(nparams, dtype=np.int64),
            np.stack(hists), np.array(grp_lo), np.array(layer_ids, dtype=np.int64),
            np.array(ambig))
    pe = None
    if pe_hists:
        # real dynamic recipes mix stack widths (e.g. GLM-5.2: 256-expert
        # SwitchMLP banks next to 64-head MLA embed_q/unembed_out stacks) —
        # pad to the widest and keep the true width per tensor, instead of
        # dropping the whole report as the first on-device run did
        E = max(h.shape[0] for h in pe_hists)
        counts = np.array([h.shape[0] for h in pe_hists], dtype=np.int64)
        padded = [np.pad(h, ((0, E - h.shape[0]), (0, 0))) for h in pe_hists]
        pe = (np.array(pe_names), np.array(pe_layer_ids, dtype=np.int64),
              np.array(pe_bits, dtype=np.int64), np.stack(padded), counts)
    return core, pe


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


def analyze_per_expert(pe_names, pe_layer_ids, pe_bits, pe_hists, pe_counts=None,
                       top=10):
    """Per-expert code entropy as a quantization-damage proxy (the criterion
    the NF3-hybrid v3.6 swap validated; frequency saliency is contraindicated
    on flat-routing families — see README §0). pe_counts carries each stack's
    true width when widths are mixed (rows are padded to the widest)."""
    P, E, _ = pe_hists.shape
    if pe_counts is None:
        pe_counts = np.full(P, E, dtype=np.int64)
    util = np.zeros((P, E))
    for t in range(P):
        for e in range(int(pe_counts[t])):
            util[t, e] = _entropy(pe_hists[t, e]) / pe_bits[t]
    print(f"[entropy][per-expert] {P} expert stacks, widths "
          f"{sorted(set(pe_counts.tolist()))} "
          "(damage proxy: low utilization = stretched grid = high error):")
    for t in range(P):
        u = util[t, :pe_counts[t]]
        print(f"[entropy][per-expert] L{pe_layer_ids[t]:>3} "
              f"{str(pe_names[t]).split('layers.')[-1]:<40} "
              f"util min={u.min()*100:5.1f}% med={np.median(u)*100:5.1f}% "
              f"weak(<50%)={int((u < 0.5).sum()):>3}")
    flat = [(util[t, e], t, e) for t in range(P) for e in range(int(pe_counts[t]))]
    worst = sorted(flat)[:top]
    print(f"[entropy][per-expert] {top} most-damaged experts "
          "(bit-promotion / gen_calib-targeting candidates):")
    for u, t, e in worst:
        print(f"[entropy][per-expert]   {u*100:5.1f}%  L{pe_layer_ids[t]} "
              f"expert {e}  ({pe_names[t]})")
    return util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="quantized student dir (needs mlx)")
    ap.add_argument("--load", help="re-analyze a saved .npz (numpy-only)")
    ap.add_argument("--save", help="write scan results to this .npz")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--per-expert", action="store_true",
                    help="per-expert code histograms on 3D expert stacks "
                         "(bits <= 4) — student-only damage-proxy saliency")
    a = ap.parse_args()
    if bool(a.model) == bool(a.load):
        ap.error("pass exactly one of --model / --load")

    pe = None
    if a.load:
        z = np.load(a.load)
        args = (z["names"], z["bits"], z["gs"], z["nparams"], z["hists"],
                z["grp_lo"], z["layer_ids"], z["ambig"])
        if "pe_hists" in z.files:
            pe = (z["pe_names"], z["pe_layer_ids"], z["pe_bits"], z["pe_hists"],
                  z["pe_counts"] if "pe_counts" in z.files else None)
    else:
        if a.per_expert:
            print("[entropy][EXPERIMENTAL] --per-expert: damage-proxy saliency "
                  "per expert (NF3-hybrid v3.6 criterion). Omit for the "
                  "per-tensor-only report.", file=sys.stderr)
        args, pe = scan(a.model, per_expert=a.per_expert)
        if a.save:
            keys = ("names", "bits", "gs", "nparams", "hists", "grp_lo",
                    "layer_ids", "ambig")
            extra = {}
            if pe is not None:
                extra = dict(zip(("pe_names", "pe_layer_ids", "pe_bits",
                                  "pe_hists", "pe_counts"), pe))
            np.savez(a.save, **dict(zip(keys, args)), **extra)
            print(f"[entropy] saved {a.save}", file=sys.stderr)
    analyze(*args, top=a.top)
    if pe is not None:
        analyze_per_expert(*pe, top=a.top)


if __name__ == "__main__":
    main()

"""Per-layer, per-expert routing traffic (saliency) for MLX MoE models.

Expert-hybrid quants (NVFP4 top-64 + NF3 tail on GLM-5.2, b12x) work because
expert traffic is heavily skewed. This tool measures that skew for *your*
model so the dynamic recipe can be driven by data instead of uniform
per-layer bits:

  - per layer: what share of routing mass the top-N experts carry, how many
    experts carry 90% / 99%, how many are dead on the calibration slice;
  - per language slice (same fixed EN / code / ZH thirds as eval_kld, so
    numbers are comparable across builds): whether the *same* experts are
    salient in every language. A static top-N split chosen on EN traffic
    quietly biases against ZH if the overlap is low — the per-expert version
    of the ZH damage the 45% calibration mix corrects.

Collection hooks ``SwitchGLU``/``SwitchMLP`` (the one routing choke point
every mlx-lm MoE model funnels through), so no per-architecture code.

  python -m alis_dwq.expert_traffic --model <path> --save traffic.npz
  python -m alis_dwq.expert_traffic --load traffic.npz --top 64   # re-analyze

Analysis (--load) is numpy-only and runs anywhere; only collection needs mlx.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SLICES = [("EN", "wikitext.txt"), ("code", "code.txt"), ("ZH", "zh.txt")]
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


# ---------------------------------------------------------------- collection

def _instrument(model):
    """Tag every SwitchGLU/SwitchMLP with a counts accumulator and patch the
    class __call__ (special methods resolve on the type) to record indices."""
    import mlx.core as mx
    from mlx_lm.models import switch_layers as SL

    classes = tuple(
        c for c in (getattr(SL, "SwitchGLU", None), getattr(SL, "SwitchMLP", None))
        if c is not None
    )
    hooked = []

    def tag(name, m):
        if not isinstance(m, classes):
            return
        proj = getattr(m, "gate_proj", None) or getattr(m, "fc1", None)
        if proj is None or not hasattr(proj, "weight"):
            print(f"[traffic] skip {name}: no gate_proj/fc1 weight", file=sys.stderr)
            return
        mm = _LAYER_RE.search(name)
        m._alis_num_experts = int(proj.weight.shape[0])
        m._alis_counts = None
        hooked.append((int(mm.group(1)) if mm else -1, name, m))

    model.apply_to_modules(tag)
    hooked.sort()

    for cls in classes:
        if getattr(cls, "_alis_orig_call", None) is not None:
            continue  # already patched (idempotent across loads)
        orig = cls.__call__
        cls._alis_orig_call = orig

        def wrapped(self, x, indices, *a, _orig=orig, **k):
            E = getattr(self, "_alis_num_experts", None)
            if E is not None:
                flat = indices.reshape(-1)
                c = self._alis_counts
                if c is None:
                    c = mx.zeros((E,), dtype=mx.float32)
                self._alis_counts = c.at[flat].add(mx.ones(flat.shape, dtype=mx.float32))
            return _orig(self, x, indices, *a, **k)

        cls.__call__ = wrapped
    return hooked


def _drain(hooked):
    """Read + reset accumulators -> (L, E) int64 array."""
    import mlx.core as mx
    rows = []
    for _, name, m in hooked:
        c = m._alis_counts
        if c is None:
            c = mx.zeros((m._alis_num_experts,), dtype=mx.float32)
        mx.eval(c)
        rows.append(np.array(c, copy=True).astype(np.int64))
        m._alis_counts = None
    widths = {r.shape[0] for r in rows}
    if len(widths) != 1:
        raise SystemExit(f"[traffic] mixed expert counts across layers {sorted(widths)}; "
                         "not supported yet — file an issue with the model id")
    return np.stack(rows)


def collect(model_path, n, chunk):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(model_path)
    mx.eval(model.parameters())
    info = getattr(mx, "device_info", None) or mx.metal.device_info
    mx.set_wired_limit(info()["max_recommended_working_set_size"])
    print(f"[mem ] load {mx.get_active_memory()/1e9:.1f}GB", file=sys.stderr)

    hooked = _instrument(model)
    if not hooked:
        raise SystemExit("[traffic] no SwitchGLU/SwitchMLP modules found — not an MoE model?")
    print(f"[traffic] hooked {len(hooked)} MoE layers "
          f"(E={hooked[0][2]._alis_num_experts})", file=sys.stderr)

    third = n // len(SLICES)
    out, names = {}, []
    for sname, fname in SLICES:
        path = DATA / fname
        if not path.exists():
            print(f"[traffic] {path} missing — skipping slice {sname}", file=sys.stderr)
            continue
        ids = mx.array(tok.encode(path.read_text())[:third])
        cache = make_prompt_cache(model)  # fresh context per slice: exact attribution
        for i in range(0, int(ids.size), chunk):
            logits = model(ids[None, i:i + chunk], cache=cache)
            mx.eval(logits)  # counts are upstream of logits; this settles both
        out[sname] = _drain(hooked)
        names.append(sname)
        print(f"[traffic] slice {sname}: {int(ids.size)} tokens", file=sys.stderr)

    if not names:
        raise SystemExit(f"[traffic] no slice corpora found in {DATA} — "
                         "add wikitext.txt / code.txt / zh.txt (same files eval_kld uses)")
    layer_ids = np.array([li for li, _, _ in hooked], dtype=np.int64)
    return names, np.stack([out[s] for s in names]), layer_ids


# ------------------------------------------------------------------ analysis

def _coverage(c, top):
    tot = c.sum()
    if tot == 0:
        return float("nan"), -1, -1, int(c.size)
    cum = np.cumsum(np.sort(c)[::-1]) / tot
    return (
        float(cum[min(top, c.size) - 1]),
        int(np.searchsorted(cum, 0.90) + 1),
        int(np.searchsorted(cum, 0.99) + 1),
        int((c == 0).sum()),
    )


def _jsd(p, q, eps=1e-12):
    """Jensen-Shannon divergence in bits between two count vectors."""
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], eps))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _top_set(c, top):
    return set(np.argsort(c)[::-1][:top].tolist())


def analyze(slices, counts, layer_ids, top):
    """counts: (S, L, E) int64. Prints the report; returns per-layer rows."""
    S, L, E = counts.shape
    total = counts.sum(axis=0)  # (L, E) all-slice traffic
    print(f"[traffic] {L} MoE layers x {E} experts, slices={slices}, top={top}")

    i_en = slices.index("EN") if "EN" in slices else None
    i_zh = slices.index("ZH") if "ZH" in slices else None

    rows = []
    for l in range(L):
        share, n90, n99, dead = _coverage(total[l], top)
        jsd = _jsd(counts[i_en, l], counts[i_zh, l]) if i_en is not None and i_zh is not None else float("nan")
        # does each slice's salient set match the combined one?
        base = _top_set(total[l], top)
        ov = min(len(_top_set(counts[s, l], top) & base) / top for s in range(S)) if S > 1 else 1.0
        rows.append((int(layer_ids[l]), share, n90, n99, dead, jsd, ov))
        print(f"[traffic] L{layer_ids[l]:>3}  top{top}={share*100:5.1f}%  "
              f"n90={n90:>3}  n99={n99:>3}  dead={dead:>3}  "
              f"JSD(EN,ZH)={jsd:.3f}  min-overlap={ov*100:3.0f}%")

    shares = np.array([r[1] for r in rows])
    ovs = np.array([r[6] for r in rows])
    jsds = np.array([r[5] for r in rows])
    print(f"[traffic] top{top} share: mean={np.nanmean(shares)*100:.1f}%  "
          f"min={np.nanmin(shares)*100:.1f}% (L{rows[int(np.nanargmin(shares))][0]})  "
          f"max={np.nanmax(shares)*100:.1f}% (L{rows[int(np.nanargmax(shares))][0]})")
    print(f"[traffic] min slice-overlap of top{top}: mean={np.nanmean(ovs)*100:.0f}%  "
          f"worst={np.nanmin(ovs)*100:.0f}% (L{rows[int(np.nanargmin(ovs))][0]})"
          " — low overlap = a static salient set biases against that slice")
    if not np.all(np.isnan(jsds)):
        worst = np.argsort(jsds)[::-1][:5]
        print("[traffic] most language-specialized layers (JSD EN vs ZH): "
              + ", ".join(f"L{rows[i][0]}={jsds[i]:.3f}" for i in worst))
    return rows


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="MLX model path/repo (collection; needs mlx)")
    ap.add_argument("--load", help="re-analyze a saved .npz instead of running a model")
    ap.add_argument("--save", help="write counts to this .npz")
    ap.add_argument("--n", type=int, default=3072, help="total tokens across slices")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--top", type=int, default=64,
                    help="salient-set size to report coverage for (b12x uses 64/256)")
    a = ap.parse_args()

    if bool(a.model) == bool(a.load):
        ap.error("pass exactly one of --model / --load")

    if a.load:
        z = np.load(a.load)
        slices = [str(s) for s in z["slices"]]
        counts, layer_ids = z["counts"], z["layer_ids"]
    else:
        slices, counts, layer_ids = collect(a.model, a.n, a.chunk)
        if a.save:
            np.savez(a.save, slices=np.array(slices), counts=counts,
                     layer_ids=layer_ids, model=np.array(a.model))
            print(f"[traffic] saved {a.save}", file=sys.stderr)

    analyze(slices, counts, layer_ids, a.top)


if __name__ == "__main__":
    main()

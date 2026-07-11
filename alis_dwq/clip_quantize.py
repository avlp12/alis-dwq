"""Clip-search affine requantization — the 4/6 idea ported to MLX affine quants.

Four-over-six (Cook et al., used in the humans& NVFP4 RL recipe) picks, per
block, between two scale mappings by *measured* reconstruction error, only
narrowing the range when that lowers the error. MLX's affine mode always maps
each group's exact min/max to the ends of the grid, so one outlier stretches
the whole group's grid. This tool re-quantizes a student against the original
weights, trying a few clipped ranges per group and keeping whichever
quantizes with the lowest MSE (the unclipped range is always a candidate, so
a group can only match or improve).

Why this belongs BEFORE DWQ: DWQ trains scales/biases only — the packed
codes are frozen. Clipping changes the codes (outliers saturate to the grid
ends, the interior gets a finer grid), which is exactly the degree of freedom
DWQ cannot touch. Run this first, then DWQ; running it on an already-DWQ'd
student would discard the DWQ (everything is recomputed from the source).

  python -m alis_dwq.clip_quantize \
      --source <unquantized MLX-layout dump (bf16/fp16)> \
      --model  <quantized student dir> \
      --out    <new dir>

Per-tensor bits/group_size are inferred from the student's tensor shapes, so
any dynamic per-layer recipe is preserved exactly. The source must be an
*unquantized* MLX-layout dump of the same model (`mlx_lm convert` without
`-q`): expert stacks etc. must already be in MLX naming. Output always goes
to a NEW directory — never save over a lazily-loaded model (see README).
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

VALID_BITS = (2, 3, 4, 5, 6, 8)


def _load_dir(d):
    """Lazily mmap every model*.safetensors shard -> (weights, key->shard-name)."""
    import mlx.core as mx
    d = Path(d)
    shards = sorted(d.glob("model*.safetensors"))
    if not shards:
        raise SystemExit(f"[clip] no model*.safetensors in {d}")
    weights, shard_of = {}, {}
    for f in shards:
        for k, v in mx.load(str(f)).items():
            weights[k], shard_of[k] = v, f.name
    return weights, shard_of


def clip_requantize(w, group_size, bits, ratios, scale_dtype):
    """Per-group clip search. Returns (wq, scales, biases, mse_drop, clipped_frac).

    Every candidate goes through mx.quantize/mx.dequantize itself, so no
    assumption is made about how MLX derives scales — candidates are judged
    purely by measured reconstruction error against the original weights.
    """
    import mlx.core as mx
    assert (group_size * bits) % 32 == 0
    w = w.astype(mx.float32)
    head, in_dim = w.shape[:-1], w.shape[-1]
    G = in_dim // group_size
    wg = w.reshape(*head, G, group_size)
    center = (wg.min(axis=-1, keepdims=True) + wg.max(axis=-1, keepdims=True)) / 2
    half = wg.max(axis=-1, keepdims=True) - center

    best_err = base_err = best_q = best_s = best_b = None
    for r in ratios:  # ratios start with 1.0 (the unclipped baseline)
        wc = w if r >= 1.0 else mx.clip(wg, center - r * half, center + r * half).reshape(w.shape)
        q, s, b = mx.quantize(wc, group_size=group_size, bits=bits)
        dq = mx.dequantize(q, s, b, group_size=group_size, bits=bits).astype(mx.float32)
        err = ((dq - w).reshape(*head, G, group_size) ** 2).sum(axis=-1)
        s, b = s.astype(scale_dtype), b.astype(scale_dtype)
        if best_err is None:
            best_err, best_q, best_s, best_b = err, q.reshape(*head, G, -1), s, b
            base_err = err
        else:
            better = err < best_err
            best_err = mx.where(better, err, best_err)
            best_q = mx.where(better[..., None], q.reshape(*head, G, -1), best_q)
            best_s = mx.where(better, s, best_s)
            best_b = mx.where(better, b, best_b)
    best_q = best_q.reshape(*head, -1)
    tot_base = base_err.sum()
    mse_drop = 1 - best_err.sum() / mx.maximum(tot_base, mx.array(1e-30))
    clipped = (best_err < base_err).astype(mx.float32).mean()
    mx.eval(best_q, best_s, best_b, mse_drop, clipped)
    return best_q, best_s, best_b, float(mse_drop.item()), float(clipped.item())


def main():
    import mlx.core as mx
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="unquantized MLX-layout dump")
    ap.add_argument("--model", required=True, help="quantized (pre-DWQ) student dir")
    ap.add_argument("--out", required=True, help="output dir (must not be --model)")
    ap.add_argument("--ratios", default="1.0,0.9375,0.875,0.8125,0.75",
                    help="comma-separated clip ratios; 1.0 is always included")
    ap.add_argument("--dequantize-source", action="store_true",
                    help="allow a quantized (affine) --source and dequantize it "
                         "on the fly (e.g. a Q8 dump standing in for bf16; "
                         "provenance should be disclosed)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.resolve() in (Path(a.model).resolve(), Path(a.source).resolve()):
        raise SystemExit("[clip] --out must be a new directory (lazy-mmap save trap)")
    out.mkdir(parents=True, exist_ok=True)

    extra = {float(r) for r in a.ratios.split(",")} - {1.0}
    if any(not 0 < r < 1 for r in extra):
        raise SystemExit("[clip] ratios must be in (0, 1]")
    ratios = [1.0] + sorted(extra, reverse=True)  # baseline first

    student, shard_of = _load_dir(a.model)
    source, _ = _load_dir(a.source)

    bases = {k[: -len(".scales")] for k in student if k.endswith(".scales")}
    triple_keys = {b + suf for b in bases for suf in (".weight", ".scales", ".biases")}
    by_shard = defaultdict(list)
    for k, f in shard_of.items():
        by_shard[f].append(k)

    pending = defaultdict(dict)  # shard name -> {key: array}
    done, skipped, drops = set(), [], []

    def process(base):
        wq, sc = student[base + ".weight"], student[base + ".scales"]
        bi = student.get(base + ".biases")
        sw = source.get(base + ".weight")
        reason = None
        if bi is None:
            reason = "no biases (non-affine mode)"
        elif sw is None:
            reason = "missing in --source"
        elif base + ".scales" in source:
            if not a.dequantize_source:
                reason = "--source is itself quantized (dequantize it first, or pass --dequantize-source)"
            else:
                s_sc, s_bi = source[base + ".scales"], source.get(base + ".biases")
                if s_bi is None:
                    reason = "--source quantized in a non-affine mode"
                else:
                    # infer the source's bits/gs from shapes; the candidate in_dim
                    # must ALSO validate on the student side (bits/gs both legal),
                    # which disambiguates e.g. Q8/gs64 vs 4-bit/gs128 packings.
                    cands = []
                    for s_bits in VALID_BITS:
                        if (sw.shape[-1] * 32) % s_bits:
                            continue
                        s_in = sw.shape[-1] * 32 // s_bits
                        if s_sc.shape[-1] <= 0 or s_in % s_sc.shape[-1]:
                            continue
                        s_gs = s_in // s_sc.shape[-1]
                        if s_gs not in (32, 64, 128) or (s_gs * s_bits) % 32:
                            continue
                        if (wq.shape[-1] * 32) % s_in:
                            continue
                        st_bits = wq.shape[-1] * 32 // s_in
                        if st_bits not in VALID_BITS or s_in % sc.shape[-1]:
                            continue
                        st_gs = s_in // sc.shape[-1]
                        if (st_gs * st_bits) % 32:
                            continue
                        cands.append((s_bits, s_gs))
                    if len(cands) == 1:
                        s_bits, s_gs = cands[0]
                        sw = mx.dequantize(sw, s_sc, s_bi, group_size=s_gs, bits=s_bits)
                    elif not cands:
                        reason = "cannot infer source bits/group_size"
                    else:
                        reason = f"ambiguous source packing {cands} — refusing to guess"
        if reason is None and sw.shape[:-1] != wq.shape[:-1]:
            reason = f"shape mismatch {tuple(sw.shape)} vs {tuple(wq.shape)}"
        if reason is None:
            in_dim = sw.shape[-1]
            bits = wq.shape[-1] * 32 // in_dim
            gs = in_dim // sc.shape[-1]
            if bits not in VALID_BITS or gs * sc.shape[-1] != in_dim \
                    or wq.shape[-1] * 32 != bits * in_dim:
                reason = f"cannot infer bits/group_size (in={in_dim})"
        if reason is not None:
            skipped.append((base, reason))
            for suf in (".weight", ".scales", ".biases"):
                if base + suf in student:
                    pending[shard_of[base + suf]][base + suf] = student[base + suf]
            return
        q, s, b, drop, frac = clip_requantize(sw, gs, bits, ratios, sc.dtype)
        assert q.shape == wq.shape and q.dtype == wq.dtype, base
        for suf, arr in ((".weight", q), (".scales", s), (".biases", b)):
            pending[shard_of[base + suf]][base + suf] = arr
        drops.append((drop, frac, base, bits))

    for fname in tqdm(sorted(by_shard), desc="shards"):
        for k in by_shard[fname]:
            if k in triple_keys:
                base = k.rsplit(".", 1)[0]
                if base not in done:
                    done.add(base)
                    process(base)
            else:
                pending[fname][k] = student[k]
        mx.save_safetensors(str(out / fname), pending.pop(fname), metadata={"format": "mlx"})

    for leftover, arrs in pending.items():  # safety net; unreachable in practice
        # (a base is processed at the FIRST shard holding any of its keys, so
        # outputs only ever target that shard or later ones)
        merged = {}
        if (out / leftover).exists():
            merged = dict(mx.load(str(out / leftover)))
            mx.eval(*merged.values())  # materialize before overwriting the mmap
        merged.update(arrs)
        mx.save_safetensors(str(out / leftover), merged, metadata={"format": "mlx"})

    for f in Path(a.model).iterdir():  # config, tokenizer, index, ...
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)

    for base, reason in skipped:
        print(f"[clip] skipped {base}: {reason}", file=sys.stderr)
    if drops:
        worst = min(drops)
        best = max(drops)
        mean = sum(d for d, *_ in drops) / len(drops)
        print(f"[clip] {len(drops)} tensors requantized, {len(skipped)} passed through")
        print(f"[clip] group MSE drop: mean {mean*100:.1f}%  "
              f"best {best[0]*100:.1f}% ({best[2]}, {best[3]}b)  "
              f"worst {worst[0]*100:.1f}% ({worst[2]})")
        print(f"[clip] groups that chose a clipped grid: "
              f"{sum(f for _, f, *_ in drops)/len(drops)*100:.0f}% (mean over tensors)")
        print("[clip] next: DWQ this output (README §3), then eval_kld A/B vs the "
              "unclipped student")
    else:
        print("[clip] nothing requantized — check --source layout", file=sys.stderr)


if __name__ == "__main__":
    main()

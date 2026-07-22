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
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io_utils import directory_digest, move_no_replace, sha256_file

VALID_BITS = (2, 3, 4, 5, 6, 8)
_FFN_FAMILIES = (("gate_proj", "up_proj", "down_proj"), ("fc1", "fc2"))
_LINEAGE_FIELDS = (
    "schema_version",
    "source_repo",
    "source_revision",
    "source_shard_manifest_sha256",
    "mlx_lm_base_revision",
)


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in conversion plan: {key}")
        value[key] = item
    return value


def _load_conversion_plan(
    root: Path, *, role: str
) -> tuple[dict | None, Path | None]:
    root = Path(root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{role} input must be a regular directory")
    status = root / "alis-dwq-run-status.json"
    if status.exists() or status.is_symlink():
        raise ValueError(f"{role} input is already an ALIS-DWQ artifact")
    path = root / "conversion_plan.json"
    if path.is_symlink():
        raise ValueError(f"{role} conversion_plan.json must not be a symlink")
    if not path.exists():
        return None, None
    if not path.is_file():
        raise ValueError(f"{role} conversion_plan.json is not a regular file")
    try:
        plan = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{role} conversion plan is invalid: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError(f"{role} conversion plan must be an object")
    return plan, path


def _clip_input_binding(source_root: Path, student_root: Path) -> dict:
    """Bind exact immutable clip inputs and their shared conversion lineage."""
    source_plan, source_plan_path = _load_conversion_plan(source_root, role="source")
    student_plan, student_plan_path = _load_conversion_plan(
        student_root, role="student"
    )
    source_is_laguna = (
        source_plan is not None
        and source_plan.get("schema_version") == "laguna.conversion/v2"
    )
    student_is_laguna = (
        student_plan is not None
        and student_plan.get("schema_version") == "laguna.conversion/v2"
    )
    if source_is_laguna != student_is_laguna:
        raise ValueError("Laguna clip inputs require two pinned conversion plans")
    lineage_mode = "generic-digest-only"
    if source_is_laguna:
        lineage_mode = "laguna-pinned"
        if (
            source_plan.get("recipe") != "bf16-mlx-layout"
            or source_plan.get("quantized") is not False
            or source_plan.get("dwq_applied") is not False
        ):
            raise ValueError("clip source must be an unquantized pre-DWQ BF16 layout")
        if (
            student_plan.get("recipe") == "bf16-mlx-layout"
            or student_plan.get("quantized") is not True
            or student_plan.get("dwq_applied") is not False
            or student_plan.get("clip_applied") is not False
            or student_plan.get("release_complete") is not False
        ):
            raise ValueError(
                "clip student must be an unclipped quantized pre-DWQ artifact"
            )
        for field in _LINEAGE_FIELDS:
            source_value = source_plan.get(field)
            if source_value is None or student_plan.get(field) != source_value:
                raise ValueError(
                    f"clip source/student conversion lineage mismatch: {field}"
                )
    elif student_plan is not None and (
        student_plan.get("dwq_applied") is True
        or student_plan.get("clip_applied") is True
        or student_plan.get("release_complete") is True
    ):
        raise ValueError("clip student plan describes a completed or clipped artifact")

    def evidence(
        root: Path, plan: dict | None, plan_path: Path | None
    ) -> dict:
        plan = plan or {}
        return {
            "directory_digest": directory_digest(root),
            "conversion_plan_sha256": (
                sha256_file(plan_path) if plan_path is not None else None
            ),
            "recipe": plan.get("recipe"),
            "artifact_label": plan.get("artifact_label"),
            **{field: plan.get(field) for field in _LINEAGE_FIELDS},
        }

    return {
        "schema": "alis-dwq.clip-inputs/v1",
        "lineage_mode": lineage_mode,
        "source": evidence(source_root, source_plan, source_plan_path),
        "student": evidence(student_root, student_plan, student_plan_path),
    }


def _verify_clip_inputs_unchanged(
    source_root: Path, student_root: Path, expected: dict
) -> None:
    observed = _clip_input_binding(source_root, student_root)
    if observed != expected:
        raise RuntimeError("clip source/student bytes changed during requantization")


def snake_perm(mag, group_size):
    """Outlier-scattering permutation: new[..., j] = old[..., perm[..., j]].

    Channels sorted by magnitude are dealt round-robin across the
    n/group_size quantization groups, so each group receives an even share
    of outliers instead of one outlier stretching one group's whole grid
    (the PeRQ-style permutation result: calibrated permutations recover most
    of the full-rotation benefit — and unlike rotations, a permutation is
    value-preserving, so the E1 super-weight/anchor protection is intact)."""
    n = mag.shape[-1]
    G = n // group_size
    assert G * group_size == n
    order = np.argsort(-np.asarray(mag), axis=-1)
    ranks = np.arange(n)
    target = (ranks % G) * group_size + ranks // G
    perm = np.empty_like(order)
    np.put_along_axis(perm, np.broadcast_to(target, order.shape).copy(),
                      order, axis=-1)
    return perm


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


def clip_requantize(w, group_size, bits, ratios, scale_dtype, chunk_elems=1 << 27,
                    max_err_slack=1.0):
    """Per-group clip search. Returns (wq, scales, biases, mse_drop, clipped_frac).

    Every candidate goes through mx.quantize/mx.dequantize itself, so no
    assumption is made about how MLX derives scales — candidates are judged
    purely by measured reconstruction error against the original weights.

    Large tensors are processed in leading-axis chunks with an mx.eval per
    chunk: one fused graph over a multi-GB expert stack times five candidates
    exceeds the macOS ~5 s GPU watchdog (measured: hard kill at shard 15 of a
    745B student). Per-group decisions are independent along the leading
    axis, so chunking is exact.
    """
    import mlx.core as mx
    rows = int(w.shape[0])
    per_row = 1
    for d in w.shape[1:]:
        per_row *= int(d)
    step = max(1, int(chunk_elems // max(per_row, 1)))
    if rows > step:
        qs, ss, bs = [], [], []
        best_sum = base_sum = 0.0
        clip_cnt = grp_cnt = 0.0
        for i in range(0, rows, step):
            q, s, b, stats = _clip_chunk(
                w[i : i + step],
                group_size,
                bits,
                ratios,
                scale_dtype,
                max_err_slack,
            )
            qs.append(q)
            ss.append(s)
            bs.append(b)
            best_sum += stats[0]
            base_sum += stats[1]
            clip_cnt += stats[2]
            grp_cnt += stats[3]
        q = mx.concatenate(qs, axis=0)
        s = mx.concatenate(ss, axis=0)
        b = mx.concatenate(bs, axis=0)
        mx.eval(q, s, b)
        drop = 1 - best_sum / max(base_sum, 1e-30)
        return q, s, b, float(drop), float(clip_cnt / max(grp_cnt, 1.0))
    q, s, b, stats = _clip_chunk(w, group_size, bits, ratios, scale_dtype, max_err_slack)
    drop = 1 - stats[0] / max(stats[1], 1e-30)
    return q, s, b, float(drop), float(stats[2] / max(stats[3], 1.0))


def _clip_chunk(w, group_size, bits, ratios, scale_dtype, max_err_slack=1.0):
    import mlx.core as mx
    assert (group_size * bits) % 32 == 0
    w = w.astype(mx.float32)
    head, in_dim = w.shape[:-1], w.shape[-1]
    G = in_dim // group_size
    wg = w.reshape(*head, G, group_size)
    center = (wg.min(axis=-1, keepdims=True) + wg.max(axis=-1, keepdims=True)) / 2
    half = wg.max(axis=-1, keepdims=True) - center

    best_err = base_err = base_maxe = best_q = best_s = best_b = None
    for r in ratios:  # ratios start with 1.0 (the unclipped baseline)
        wc = w if r >= 1.0 else mx.clip(wg, center - r * half, center + r * half).reshape(w.shape)
        q, s, b = mx.quantize(wc, group_size=group_size, bits=bits)
        dq = mx.dequantize(q, s, b, group_size=group_size, bits=bits).astype(mx.float32)
        diff = (dq - w).reshape(*head, G, group_size)
        err = (diff ** 2).sum(axis=-1)
        maxe = mx.abs(diff).max(axis=-1)
        s, b = s.astype(scale_dtype), b.astype(scale_dtype)
        if best_err is None:
            best_err, best_q, best_s, best_b = err, q.reshape(*head, G, -1), s, b
            base_err, base_maxe = err, maxe
        else:
            # accept a clipped grid only when it lowers the group MSE without
            # blowing up the group's worst-case error: min-max affine anchors
            # the extreme weights exactly, and saturating those "super
            # weights" destroys the model even as the mean improves
            # (measured: mean -24% / anchors x4 / wikitext 4.71 -> 51).
            better = (err < best_err) & (maxe <= base_maxe * max_err_slack)
            best_err = mx.where(better, err, best_err)
            best_q = mx.where(better[..., None], q.reshape(*head, G, -1), best_q)
            best_s = mx.where(better, s, best_s)
            best_b = mx.where(better, b, best_b)
    best_q = best_q.reshape(*head, -1)
    best_sum = best_err.sum()
    base_sum = base_err.sum()
    clip_cnt = (best_err < base_err).astype(mx.float32).sum()
    grp_cnt = mx.array(float(best_err.size))
    mx.eval(best_q, best_s, best_b, best_sum, base_sum, clip_cnt)
    return best_q, best_s, best_b, (float(best_sum.item()), float(base_sum.item()),
                                    float(clip_cnt.item()), float(grp_cnt.item()))


def _plan_ffn_perms(student, source, bases, sc_of, mx):
    """Per-block outlier-scattering permutations of the FFN hidden axis.

    The FFN hidden dimension is residual-free: permuting gate/up OUTPUT
    channels (axis -2, not a quantization axis) together with down INPUT
    channels (axis -1, the group axis) yields a mathematically identical
    model whose down_proj groups no longer concentrate outliers. Expert
    stacks permute per-expert (the permutation closes inside each expert).

    Only blocks where EVERY member is a student-quantized triple with an
    unquantized float source (and no runtime bias on the ffn axis) are
    permuted — anything else would leave the block inconsistent."""
    parents = defaultdict(set)
    for b in bases:
        parent, leaf = b.rsplit(".", 1)
        parents[parent].add(leaf)

    plans, blocks = {}, 0
    for parent, leaves in sorted(parents.items()):
        fam = next((f for f in _FFN_FAMILIES if set(f) <= leaves), None)
        if fam is None:
            continue
        down = f"{parent}.{fam[-1]}"
        ups = [f"{parent}.{leaf}" for leaf in fam[:-1]]
        members = ups + [down]
        ok = all(m + ".weight" in source and m + ".scales" not in source
                 for m in members)
        ok = ok and not any(m + ".bias" in student or m + ".bias" in source
                            for m in members)
        sd = source.get(down + ".weight")
        if ok:
            ffn = int(sd.shape[-1])
            sc = sc_of(down)
            gs = ffn // int(sc.shape[-1])
            ok = gs * int(sc.shape[-1]) == ffn and \
                all(int(source[u + ".weight"].shape[-2]) == ffn for u in ups)
        if not ok:
            print(f"[clip] --permute-ffn: skipping block {parent} "
                  "(member missing/quantized-source/bias/axis mismatch)",
                  file=sys.stderr)
            continue
        rows = int(sd.shape[0]) if len(sd.shape) == 3 else 1
        mags = []
        for i in range(0, rows, 8):  # bound the fp32 abs-max working set
            chunk = sd[i:i + 8] if len(sd.shape) == 3 else sd
            m = mx.abs(chunk.astype(mx.float32)).max(axis=-2)
            mx.eval(m)
            mags.append(np.array(m, copy=True))
            if len(sd.shape) != 3:
                break
        mag = np.concatenate(mags, axis=0) if len(sd.shape) == 3 else mags[0]
        perm = snake_perm(mag, gs)
        pm = mx.array(perm.astype(np.uint32))
        for u in ups:
            plans[u] = (pm, -2, parent)
        plans[down] = (pm, -1, parent)
        blocks += 1
    print(f"[clip] --permute-ffn: {blocks} blocks planned", file=sys.stderr)
    return plans


def _apply_perm(sw, perm, axis, mx):
    if len(sw.shape) == 2:  # dense block: perm is 1-D
        idx = perm[None, :] if axis == -1 else perm[:, None]
    else:  # expert stack: perm is (E, ffn), broadcast over the third axis
        idx = perm[:, None, :] if axis == -1 else perm[:, :, None]
    return mx.take_along_axis(sw, idx.astype(mx.uint32), axis=axis)


def _clip_evidence(drops, skipped, perms, *, permutation_requested):
    """Return evidence about work actually applied, not merely requested."""
    clipped = [row for row in drops if row[1] > 0]
    permutation_blocks = sorted({parent for _perm, _axis, parent in perms.values()})
    return {
        "clip_search_completed": bool(drops),
        "clip_applied": bool(clipped),
        "clip_requantized_module_count": len(drops),
        "clip_clipped_module_count": len(clipped),
        "clip_passthrough_module_count": len(skipped),
        "clip_skips": [
            {"module": module, "reason": reason} for module, reason in skipped
        ],
        "clip_mean_clipped_group_fraction": (
            sum(fraction for _drop, fraction, *_ in drops) / len(drops)
            if drops
            else 0.0
        ),
        "ffn_permutation_requested": bool(permutation_requested),
        "ffn_permutation_applied": bool(permutation_blocks),
        "ffn_permutation_block_count": len(permutation_blocks),
    }


def main():
    import mlx.core as mx
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="unquantized MLX-layout dump")
    ap.add_argument("--model", required=True, help="quantized (pre-DWQ) student dir")
    ap.add_argument("--out", required=True, help="output dir (must not be --model)")
    ap.add_argument("--ratios", default="1.0,0.9375,0.875,0.8125,0.75",
                    help="comma-separated clip ratios; 1.0 is always included")
    ap.add_argument("--allow-lattice-source", action="store_true",
                    help="override the low-bit-lattice source guard (nvfp4 or "
                         "affine <8-bit sources cause correlated-rounding "
                         "damage — see the E1 case study before using this)")
    ap.add_argument("--max-err-slack", type=float, default=1.0,
                    help="accept a clipped grid only if the group's max abs "
                         "error stays within this factor of the unclipped "
                         "grid's (1.0 = strict Pareto; large = MSE-only)")
    ap.add_argument("--dequantize-source", action="store_true",
                    help="allow a quantized (affine) --source and dequantize it "
                         "on the fly (e.g. a Q8 dump standing in for bf16; "
                         "provenance should be disclosed)")
    ap.add_argument("--permute-ffn", action="store_true",
                    help="outlier-scattering permutation of each FFN hidden "
                         "axis before quantization (value-preserving, zero "
                         "runtime cost; float sources only)")
    ap.add_argument(
        "--require-no-skips",
        action="store_true",
        help="fail closed if any quantized module cannot be requantized",
    )
    a = ap.parse_args()

    source_root = Path(a.source).expanduser()
    student_root = Path(a.model).expanduser()
    final_out = Path(a.out).expanduser()
    if final_out.resolve() in (student_root.resolve(), source_root.resolve()):
        raise SystemExit("[clip] --out must be a new directory (lazy-mmap save trap)")
    if final_out.exists():
        raise SystemExit(f"[clip] --out already exists (no-clobber): {final_out}")
    try:
        input_binding = _clip_input_binding(source_root, student_root)
    except ValueError as exc:
        raise SystemExit(f"[clip] invalid input provenance: {exc}") from exc
    created_at = datetime.now(timezone.utc)
    out = final_out.with_name(
        f"{final_out.name}.partial-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    if out.exists():
        raise SystemExit(f"[clip] staging output already exists (no-clobber): {out}")
    out.mkdir(parents=True, exist_ok=False)

    extra = {float(r) for r in a.ratios.split(",")} - {1.0}
    if any(not 0 < r < 1 for r in extra):
        raise SystemExit("[clip] ratios must be in (0, 1]")
    ratios = [1.0] + sorted(extra, reverse=True)  # baseline first

    student, shard_of = _load_dir(student_root)
    source, _ = _load_dir(source_root)

    bases = {k[: -len(".scales")] for k in student if k.endswith(".scales")}
    triple_keys = {b + suf for b in bases for suf in (".weight", ".scales", ".biases")}
    by_shard = defaultdict(list)
    for k, f in shard_of.items():
        by_shard[f].append(k)

    pending = defaultdict(dict)  # shard name -> {key: array}
    done, skipped, drops = set(), [], []

    perms = {}
    if a.permute_ffn:
        print("[clip][EXPERIMENTAL] --permute-ffn: dealing each block's FFN "
              "hidden channels round-robin by magnitude across down_proj's "
              "groups (PeRQ-style; value-preserving, so anchors survive). "
              "Gate on held-out PPL and code_entropy before/after. Omit the "
              "flag for the previous behavior.", file=sys.stderr)
        perms = _plan_ffn_perms(student, source, bases,
                                lambda b: student[b + ".scales"], mx)
        if perms:  # provenance/audit copy: one perm per block, keyed by parent
            down_perms = {par: np.asarray(pm) for pm, ax, par in perms.values()
                          if ax == -1}
            np.savez(out / "ffn_perms.npz", **down_perms)

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
                    # nvfp4 source (packed 8/word, u8 scales, gs16, no biases).
                    # GUARD: a dequantized low-bit lattice is NOT a valid stand-in
                    # for the original weights — its values sit on a coarse grid,
                    # and re-quantizing a lattice with a coarser affine grid
                    # produces correlated (biased) rounding that kills the model
                    # even when per-tensor MSE improves. Measured on a 745B MoE,
                    # identical rules: nvfp4 source -> wikitext 51-12,769 (dead);
                    # Q8 source -> 4.42-4.68 (alive). See the E1 case study.
                    if not a.allow_lattice_source:
                        reason = ("nvfp4 (low-bit lattice) source — refusing: grid "
                                  "resonance (pass --allow-lattice-source to override)")
                    else:
                        s_in = sw.shape[-1] * 8
                        if (str(s_sc.dtype) == "mlx.core.uint8" and s_sc.shape[-1] * 16 == s_in
                                and (wq.shape[-1] * 32) % s_in == 0
                                and wq.shape[-1] * 32 // s_in in VALID_BITS
                                and s_in % sc.shape[-1] == 0):
                            if wq.shape[-1] * 32 // s_in >= 4:
                                reason = "source precision (nvfp4~4b) does not exceed student — pass-through"
                            else:
                                sw = mx.dequantize(sw, s_sc, group_size=16, bits=4, mode="nvfp4")
                        else:
                            reason = "--source quantized in an unrecognized non-affine mode"
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
                    # keep only interpretations where the source would actually
                    # out-precise the student (the rest are pass-throughs anyway)
                    act = [(sb, sg) for sb, sg in cands
                           if sb > wq.shape[-1] * 32 // (s_sc.shape[-1] * sg)
                           and (sb >= 8 or a.allow_lattice_source)]
                    if len(act) == 1:
                        s_bits, s_gs = act[0]
                        sw = mx.dequantize(sw, s_sc, s_bi, group_size=s_gs, bits=s_bits)
                    elif not cands:
                        reason = "cannot infer source bits/group_size"
                    elif not act:
                        reason = "source precision does not exceed student — pass-through"
                    else:
                        reason = f"ambiguous source packing {act} — refusing to guess"
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
            if base in perms:
                # a permuted block must be rewritten whole — a passthrough
                # member would silently de-align it from its permuted peers
                raise SystemExit(f"[clip] --permute-ffn: {base} hit a skip "
                                 f"({reason}) after its block was planned — "
                                 "aborting to avoid an inconsistent block; "
                                 "rerun without --permute-ffn or fix the source")
            skipped.append((base, reason))
            for suf in (".weight", ".scales", ".biases"):
                if base + suf in student:
                    pending[shard_of[base + suf]][base + suf] = student[base + suf]
            return
        if base in perms:
            pm, axis, _par = perms[base]
            sw = _apply_perm(sw, pm, axis, mx)
        q, s, b, drop, frac = clip_requantize(sw, gs, bits, ratios, sc.dtype,
                                              max_err_slack=a.max_err_slack)
        assert q.shape == wq.shape and q.dtype == wq.dtype, base
        for suf, arr in ((".weight", q), (".scales", s), (".biases", b)):
            pending[shard_of[base + suf]][base + suf] = arr
        drops.append((drop, frac, base, bits))
        # drop lazy-buffer references so processed shards actually free
        for suf in (".weight", ".scales", ".biases"):
            source.pop(base + suf, None)

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
        for k in by_shard[fname]:
            student.pop(k, None)
        mx.clear_cache()

    for leftover, arrs in pending.items():  # safety net; unreachable in practice
        # (a base is processed at the FIRST shard holding any of its keys, so
        # outputs only ever target that shard or later ones)
        merged = {}
        if (out / leftover).exists():
            merged = dict(mx.load(str(out / leftover)))
            mx.eval(*merged.values())  # materialize before overwriting the mmap
        merged.update(arrs)
        mx.save_safetensors(str(out / leftover), merged, metadata={"format": "mlx"})

    for f in student_root.iterdir():  # config, tokenizer, index, ...
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
        raise SystemExit(
            "[clip] nothing requantized; refusing to publish a false clipped artifact"
        )
    if a.require_no_skips and skipped:
        raise SystemExit(
            f"[clip] {len(skipped)} module(s) skipped under --require-no-skips; "
            f"partial retained at {out}"
        )

    _verify_clip_inputs_unchanged(source_root, student_root, input_binding)

    receipt_path = out / "conversion_plan.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    source_label = receipt.get("artifact_label", "quantized-pre-dwq")
    evidence = _clip_evidence(
        drops,
        skipped,
        perms,
        permutation_requested=a.permute_ffn,
    )
    label_suffix = "clipped" if evidence["clip_applied"] else "clip-search-noop"
    receipt.update({
        "artifact_label": f"{source_label}-{label_suffix}",
        "clip_created_at": created_at.isoformat(),
        "clip_source": str(source_root.resolve()),
        "clip_student": str(student_root.resolve()),
        "clip_input_binding": input_binding,
        "clip_ratios": ratios,
        "clip_max_error_slack": a.max_err_slack,
        "clip_complete": not skipped,
        "release_complete": False,
        **evidence,
    })
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    move_no_replace(out, final_out)
    print(f"[clip] completed transactionally (no-clobber) at {final_out}")


if __name__ == "__main__":
    main()

"""Per-slice validation gate for layerwise DWQ (opt-in).

Pure stdlib + numpy — no mlx imports, so the gate logic is unit-testable
without a GPU. ``layerwise.py`` keeps only the wiring.

Opt-in contract (byte-identical legacy behavior when unset):

  * both ``ALIS_DWQ_VALID_SLICE_MANIFEST`` (JSON path) and
    ``ALIS_DWQ_EN_GATE_EPS`` (float in [0, 0.25]) must be set together,
    otherwise :func:`load_gate` raises;
  * gate mode requires ``batch_size == 1`` (manifest ordinals are bound to
    singleton batches — ``target_fn`` reads target files by ordinal);
  * the manifest carries doc-level ``{schema, seed, batch_size,
    max_seq_length}`` which must equal the runtime values, and a
    ``valid_target_order`` list with one entry per valid ordinal:
    ``{ordinal, slice, input_sha256, target_sha256, seed, batch_size,
    max_seq_length}`` (extra audit fields allowed);
  * per-ordinal ``input_sha256`` is the sha256 of the ``(1, W-1)`` int32
    input array in C order (little-endian ``<i4``), computed by
    :func:`_input_sha256` on the batch AFTER ``batch[:, :-1]``;
  * ``target_sha256`` is the sha256 of the raw bytes of
    ``<target_dir>/valid/{ordinal:010d}.safetensors`` and is verified once
    at gate load — before any training step — against the directory the
    run's ``target_fn`` actually reads (recovered from its closure).

Acceptance (gate mode only; legacy keeps the stock ``rv > best`` revert with
tie-accept, see :func:`legacy_accept`): strict and fail-closed —

  * non-finite ``overall``, or ``overall >= best`` (equality rejects), or
  * ``EN`` non-finite, or ``EN > init_EN * (1 + eps)``  → REVERT.

Per-ordinal non-finite losses abort the run (:func:`check_loss_finite`
raises ``FloatingPointError``) instead of silently poisoning the metrics.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

MANIFEST_ENV = "ALIS_DWQ_VALID_SLICE_MANIFEST"
EPS_ENV = "ALIS_DWQ_EN_GATE_EPS"
SCHEMA = 1
MAX_EPS = 0.25

_ENTRY_FIELDS = ("slice", "input_sha256", "target_sha256",
                 "seed", "batch_size", "max_seq_length")


def _input_sha256(x):
    """sha256 of an int32 array in C order (canonical little-endian bytes).

    Accepts numpy or mlx arrays (mlx exports via ``__array__``); dtype is
    coerced to ``<i4`` so the encoding is platform-independent.
    """
    a = np.array(x).astype("<i4", copy=False)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def file_sha256(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _target_dir_from(target_fn):
    """Recover ``target_dir`` from the fork's ``target_fn`` closure.

    ``mlx_lm.quant.dwq.main`` defines ``target_fn`` as a closure over
    ``target_dir`` (a ``Path``) when targets are precomputed on disk. If the
    closure shape is not recognised (e.g. live-teacher mode, or an mlx-lm
    revision that restructures ``main``), gate mode fails closed.
    """
    code = getattr(target_fn, "__code__", None)
    closure = getattr(target_fn, "__closure__", None)
    if code is None or closure is None or "target_dir" not in code.co_freevars:
        raise RuntimeError(
            "slice gate: cannot recover target_dir from target_fn (expected "
            "the mlx-lm dwq closure over precomputed targets); refusing to "
            "trust ordinals without target hash verification"
        )
    i = code.co_freevars.index("target_dir")
    td = Path(closure[i].cell_contents)
    if not td.is_dir():
        raise RuntimeError(f"slice gate: target_dir from target_fn is not a "
                           f"directory: {td}")
    return td


def _is_sha256(h):
    return (isinstance(h, str) and len(h) == 64
            and all(c in "0123456789abcdef" for c in h))


def verify_target_hashes(entries, target_fn):
    """Hash every ``valid/{ordinal:010d}.safetensors`` target_fn will read and
    compare against the manifest. Runs once, before the first training step."""
    vdir = _target_dir_from(target_fn) / "valid"
    for e in entries:
        p = vdir / f"{e['ordinal']:010d}.safetensors"
        if not p.is_file():
            raise FileNotFoundError(
                f"slice gate: manifest ordinal {e['ordinal']} expects target "
                f"file {p} which target_fn cannot serve")
        got = file_sha256(p)
        if got != e["target_sha256"]:
            raise ValueError(
                f"slice gate: target file hash mismatch at valid ordinal "
                f"{e['ordinal']} ({p}): manifest={e['target_sha256'][:16]}… "
                f"disk={got[:16]}… — manifest was built for different targets")


def load_gate(batch_size, max_seq_length, seed, n_valid, target_fn=None,
              environ=None):
    """Parse/validate the gate env pair + manifest.

    Returns ``(entries, en_eps)`` in gate mode, ``None`` in legacy mode (env
    pair unset). Raises before any training step on any inconsistency.
    ``target_fn`` (when given) triggers the one-time target-file hash check.
    """
    env = os.environ if environ is None else environ
    manifest_path = env.get(MANIFEST_ENV)
    eps_raw = env.get(EPS_ENV)

    if bool(manifest_path) != bool(eps_raw):
        raise ValueError(
            f"slice gate: {MANIFEST_ENV} and {EPS_ENV} must be set together "
            f"(got manifest={manifest_path!r}, eps={eps_raw!r})")
    if not manifest_path:
        return None

    if batch_size != 1:
        raise ValueError(f"slice gate: requires batch_size=1, got {batch_size}")

    try:
        en_eps = float(eps_raw)
    except (TypeError, ValueError):
        raise ValueError(f"slice gate: invalid {EPS_ENV}={eps_raw!r}") from None
    if not math.isfinite(en_eps) or not 0.0 <= en_eps <= MAX_EPS:
        raise ValueError(f"slice gate: invalid EN epsilon {en_eps} "
                         f"(want 0..{MAX_EPS})")

    doc = json.loads(Path(manifest_path).read_text())
    expected = {"schema": SCHEMA, "batch_size": batch_size,
                "max_seq_length": max_seq_length, "seed": seed}
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ValueError(
                f"slice gate: manifest {key}={doc.get(key)!r}, "
                f"expected {value!r}")

    entries = doc.get("valid_target_order")
    if not isinstance(entries, list) or len(entries) != n_valid:
        raise ValueError(
            f"slice gate: manifest/valid batch count mismatch "
            f"({None if not isinstance(entries, list) else len(entries)} "
            f"entries vs {n_valid} valid samples)")
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"slice gate: manifest entry {i} is not an object")
        if e.get("ordinal") != i:
            raise ValueError(
                f"slice gate: manifest entry {i} has ordinal={e.get('ordinal')!r}"
                " — entries must be keyed by valid ordinal, in order")
        for f in _ENTRY_FIELDS:
            if f not in e:
                raise ValueError(f"slice gate: manifest entry {i} missing {f!r}")
        for f, v in (("seed", seed), ("batch_size", batch_size),
                     ("max_seq_length", max_seq_length)):
            if e[f] != v:
                raise ValueError(
                    f"slice gate: manifest entry {i} {f}={e[f]!r}, "
                    f"expected {v!r}")
        if not isinstance(e["slice"], str) or not e["slice"]:
            raise ValueError(f"slice gate: manifest entry {i} bad slice label")
        for f in ("input_sha256", "target_sha256"):
            if not _is_sha256(e[f]):
                raise ValueError(
                    f"slice gate: manifest entry {i} {f} is not a sha256 hex")
    if not any(e["slice"] == "EN" for e in entries):
        raise ValueError("slice gate: manifest has no EN entries")

    if target_fn is not None:
        verify_target_hashes(entries, target_fn)
    return entries, en_eps


def check_ordinal_input(i, entry, batch):
    """Verify the batch iterate_batches just yielded matches the manifest's
    recorded input for ordinal ``i`` — before ``target_fn(i)`` is trusted."""
    got = _input_sha256(batch)
    if got != entry["input_sha256"]:
        raise ValueError(
            f"slice gate: target/input misalignment at valid ordinal {i} "
            f"(input sha256 {got[:16]}… != manifest "
            f"{entry['input_sha256'][:16]}…)")


def check_loss_finite(i, lv, nv):
    """Gate-mode per-ordinal loss sanity: abort rather than poison metrics."""
    if not math.isfinite(lv) or nv <= 0:
        raise FloatingPointError(
            f"slice gate: invalid validation at ordinal {i}: loss={lv}, "
            f"ntoks={nv}")


def _accept(rv, best, en_limit=None):
    """Strict, fail-closed acceptance for gate mode -> (accept, reasons).

    ``rv``/``best`` are metric dicts (``{"overall": …, "EN": …, …}``).
    Rejects on: non-finite overall, overall >= best (equality rejects), and
    — when ``en_limit`` is not None — non-finite EN or EN > en_limit.
    """
    reasons = []
    ov, bo = rv["overall"], best["overall"]
    if not math.isfinite(ov) or not (ov < bo):
        reasons.append(f"overall {ov:.6f} >= best {bo:.6f}")
    if en_limit is not None:
        en = rv.get("EN", float("nan"))
        if not math.isfinite(en) or en > en_limit:
            reasons.append(f"EN {en:.6f} > initial ceiling {en_limit:.6f}")
    return not reasons, reasons


def legacy_accept(rv, best):
    """Stock keep-best semantics: revert only when strictly worse.

    ``rv > best`` is False for ties (tie ACCEPTS) and also False for NaN —
    the pre-gate quirk the strict gate exists to fix; preserved verbatim for
    legacy mode.
    """
    return not (rv > best)

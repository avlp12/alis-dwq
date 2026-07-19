"""Target-dump integrity + machine-readable run history.

Two gaps this module closes (both surfaced by reading antirez/ds4's design):

1. **Target manifest — the "exact replay" lesson.** Upstream dwq targets are
   keyed only by (split, batch index): the training-side ``target_fn`` throws
   the batch tokens away (its first parameter is ``_``), so nothing couples
   the student's inputs to the teacher logits being loaded. Any drift in the
   replay chain — calibration jsonl bytes *or row count* (the sample
   permutation reshuffles wholesale), seed, tokenizer, batch size, seq
   length, even which loader ran (the local-jsonl loader and upstream's HF
   loader pick *different* valid sets from identical inputs) — silently pairs
   teacher logits with the wrong samples. ds4 keys its KV cache on a hash of
   the exact rendered bytes; the equivalent invariant here is a hash of the
   *token streams actually consumed*. The dump writes it into
   ``<target-dir>/manifest.json``; training recomputes and refuses to start
   on mismatch (override: ``ALIS_DWQ_ALLOW_UNVERIFIED_TARGETS=1`` for
   legacy/manifest-less dumps). The dump also self-checks its own files
   (count, finite, non-constant, leading dim) so "verify a dump is non-zero
   before you reclaim the teacher" stops being a manual step — the
   ``(ranks, seq, k)`` all_gather corruption and the lazy-mmap zeroing
   incident (README §2b / operational notes) both fail here, at dump time,
   instead of poisoning a training run.

2. **Run events.** Multi-hour campaigns previously left their per-round
   trajectory (accept/REVERT, losses, peak memory) in stderr only — a crash
   at round 9/13 loses everything, and every case-study table was hand
   copied from terminal logs. Each run now appends JSON lines to
   ``ALIS_DWQ_RUN_LOG_DIR`` (default ``alis_runs/``)``/<utc>-<pid>/events.jsonl``
   as it goes (``run_start``, ``data``, ``targets_dumped``, ``valid``,
   ``round``, ``summary``). Append-per-event, so a killed run keeps
   everything up to the kill. ``ALIS_DWQ_RUN_LOG=0`` disables. Event logging
   is diagnostic: it warns on I/O errors rather than killing a run; the
   manifest checks are integrity gates and fail hard.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1
TOP_K = 1024  # upstream compute_dwq_targets keeps the top-1024 logits

# run.py stashes the parsed --target-dir here so the layerwise trainer (whose
# upstream signature never sees it — target_dir is closed into target_fn) can
# find the manifest.
TARGET_DIR = None


def _rank():
    try:
        import mlx.core as mx
        return mx.distributed.init().rank()
    except Exception:
        return 0


def token_stream_hash(data):
    """sha256 over the ordered (tokens, offset) stream a split consumes.

    This is the exact-replay invariant: batch composition depends on the
    list order, each sample's token ids *and lengths* (iterate_batches sorts
    by length), and the offsets — all of which feed the hash. If this hash
    plus (batch_size, max_seq_length, seed) match between dump and train,
    alignment is guaranteed for a fixed iterate_batches implementation.
    """
    h = hashlib.sha256()
    for item in data:
        toks, off = item
        arr = np.asarray(list(toks), dtype=np.int64)
        h.update(len(arr).to_bytes(8, "little"))
        h.update(arr.tobytes())
        h.update(int(off).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _versions():
    from importlib import metadata
    out = {}
    for pkg in ("mlx", "mlx-lm", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def build_manifest(train_data, valid_data, batch_size, max_seq_length, seed,
                   extra=None):
    m = {
        "format": MANIFEST_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "versions": _versions(),
        "top_k": TOP_K,
        "batch_size": int(batch_size),
        "max_seq_length": int(max_seq_length),
        "seed": int(seed),
        "splits": {
            "train": {
                "samples": len(train_data),
                "batches": len(train_data) // batch_size,
                "token_sha256": token_stream_hash(train_data),
            },
            "valid": {
                "samples": len(valid_data),
                "batches": len(valid_data) // batch_size,
                "token_sha256": token_stream_hash(valid_data),
            },
        },
    }
    if extra:
        m.update(extra)
    return m


def write_manifest(target_dir, manifest):
    path = Path(target_dir) / MANIFEST_NAME
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[alis-dwq] target manifest -> {path}", file=sys.stderr)


def sanity_check_targets(target_dir, manifest):
    """Post-dump self-check; SystemExit listing every problem found.

    Catches at dump time (while the teacher still exists) what previously
    surfaced hours into training or after the teacher was reclaimed:
    partial dumps (upstream's has_targets gate is any-glob), zeroed files
    (the lazy-mmap save trap), and wrong leading dims (the distributed
    all_gather (ranks, seq, k) corruption — README §2b).
    """
    import mlx.core as mx
    problems = []
    for split, info in manifest["splits"].items():
        d = Path(target_dir) / split
        files = sorted(d.glob("*.safetensors"))
        if len(files) != info["batches"]:
            problems.append(f"{split}: {len(files)} files, expected "
                            f"{info['batches']} (partial/stale dump?)")
        for f in files:
            t = mx.load(str(f))
            logits = t["logits"].astype(mx.float32)
            finite = mx.isfinite(logits).all()
            lo, hi = logits.min(), logits.max()
            mx.eval(finite, lo, hi)
            if int(logits.shape[0]) != manifest["batch_size"]:
                problems.append(
                    f"{split}/{f.name}: leading dim {int(logits.shape[0])} != "
                    f"batch_size {manifest['batch_size']} — for a distributed "
                    "dump this is the (ranks, seq, k) all_gather corruption "
                    "(README §2b)")
            if int(logits.shape[-1]) != manifest["top_k"]:
                problems.append(f"{split}/{f.name}: last dim "
                                f"{int(logits.shape[-1])} != top_k")
            if not bool(finite.item()):
                problems.append(f"{split}/{f.name}: non-finite logits")
            elif lo.item() == hi.item():
                problems.append(f"{split}/{f.name}: constant logits "
                                "(zeroed file? see the lazy-mmap note)")
    if problems:
        raise SystemExit("[alis-dwq] target dump FAILED sanity checks — do "
                         "not reclaim the teacher:\n  " + "\n  ".join(problems))
    n = {s: i["batches"] for s, i in manifest["splits"].items()}
    print(f"[alis-dwq] target sanity OK ({n['train']} train / {n['valid']} "
          "valid files, finite, non-constant, dims match)", file=sys.stderr)


def verify_targets_for_training(target_dir, train_data, valid_data,
                                batch_size, max_seq_length, seed):
    """Train-time gate: recompute the replay invariant, diff against the
    manifest, refuse on mismatch (naming the drifted field)."""
    target_dir = Path(target_dir)
    path = target_dir / MANIFEST_NAME
    if not path.exists():
        if os.environ.get("ALIS_DWQ_ALLOW_UNVERIFIED_TARGETS", "") == "1":
            print("[alis-dwq][WARN] no target manifest — alignment NOT "
                  "verified (ALIS_DWQ_ALLOW_UNVERIFIED_TARGETS=1). A silent "
                  "mispairing trains against the wrong samples.",
                  file=sys.stderr)
            return None
        raise SystemExit(
            f"[alis-dwq] {path} not found: this target dump predates the "
            "manifest (or was made without alis_dwq.run). Alignment cannot "
            "be verified — a dump/train mismatch silently pairs teacher "
            "logits with the wrong samples (README §2b). Re-dump, or set "
            "ALIS_DWQ_ALLOW_UNVERIFIED_TARGETS=1 to proceed at your own "
            "risk.")
    m = json.load(open(path))
    dumped = (m.get("versions") or {}).get("mlx-lm")
    running = _versions().get("mlx-lm")
    if dumped and running and dumped != running:
        # token hashes cannot see iterate_batches changes (length sort,
        # padding, permutation live upstream), so index-keyed targets can
        # mispair across mlx-lm versions even when the streams match
        if os.environ.get("ALIS_DWQ_ALLOW_VERSION_SKEW", "") == "1":
            print(f"[alis-dwq][WARN] mlx-lm {running} != dump's {dumped} — "
                  "batch composition depends on iterate_batches internals; "
                  "alignment is NOT verified across this skew "
                  "(ALIS_DWQ_ALLOW_VERSION_SKEW=1)", file=sys.stderr)
        else:
            raise SystemExit(
                f"[alis-dwq] targets were dumped under mlx-lm {dumped} but "
                f"{running} is installed. Re-dump, reinstall the dump's "
                "mlx-lm, or set ALIS_DWQ_ALLOW_VERSION_SKEW=1 to proceed "
                "at your own risk.")
    got = {
        "batch_size": int(batch_size),
        "max_seq_length": int(max_seq_length),
        "seed": int(seed),
        "train.samples": len(train_data),
        "train.token_sha256": token_stream_hash(train_data),
        "valid.samples": len(valid_data),
        "valid.token_sha256": token_stream_hash(valid_data),
    }
    want = {
        "batch_size": m["batch_size"],
        "max_seq_length": m["max_seq_length"],
        "seed": m["seed"],
        "train.samples": m["splits"]["train"]["samples"],
        "train.token_sha256": m["splits"]["train"]["token_sha256"],
        "valid.samples": m["splits"]["valid"]["samples"],
        "valid.token_sha256": m["splits"]["valid"]["token_sha256"],
    }
    drift = [k for k in want if want[k] != got[k]]
    if drift:
        lines = [f"  {k}: dump={want[k]!r}  train={got[k]!r}" for k in drift]
        hint = ("token_sha256 drift with matching sizes usually means the "
                "tokenizer differs (dump tokenizes with --model = teacher, "
                "training with the student) or the jsonl bytes/loader "
                "changed; sample-count drift means the data or "
                "--num-samples moved.")
        raise SystemExit("[alis-dwq] target/training replay MISMATCH — "
                         "refusing to train against misaligned targets:\n"
                         + "\n".join(lines) + f"\n  ({hint})\n  A mismatch "
                         "means the pairing is provably wrong, so there is "
                         "no override — re-dump.")
    print("[alis-dwq] target manifest verified — token streams, sizes and "
          "replay params match the dump", file=sys.stderr)
    return m


# ---------------------------------------------------------------- run events

_run_dir = None
_warned = False


def run_dir():
    """Lazily created per-process run directory (rank 0 only)."""
    global _run_dir
    if _run_dir is None:
        base = Path(os.environ.get("ALIS_DWQ_RUN_LOG_DIR", "alis_runs"))
        _run_dir = base / (time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                           + f"-{os.getpid()}")
        _run_dir.mkdir(parents=True, exist_ok=True)
    return _run_dir


def event(kind, **fields):
    """Append one JSON event; diagnostic path, so warn-don't-crash."""
    global _warned
    if os.environ.get("ALIS_DWQ_RUN_LOG", "1") == "0" or _rank() != 0:
        return
    try:
        rec = {"t": round(time.time(), 3), "event": kind}
        rec.update(fields)
        with open(run_dir() / "events.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        if not _warned:
            _warned = True
            print(f"[alis-dwq][WARN] run event log unavailable ({e!r}) — "
                  "continuing without it", file=sys.stderr)

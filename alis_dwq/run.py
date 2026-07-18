"""alis-dwq launcher: layerwise DWQ + local mixed calibration data.

Wraps ``mlx_lm.quant.dwq.main()`` with:
  1. the layerwise patch (see alis_dwq.layerwise; ALIS_DWQ_LAYERS_PER_ROUND),
  2. a wired-limit fix applied before the (otherwise unwired) target dump,
  3. a local-jsonl data loader (ALIS_DWQ_DATA_DIR with train.jsonl/valid.jsonl,
     each line {"text": ...}) so calibration mixes (e.g. 45% ZH) are exact and
     reproducible instead of a streamed HF dataset,
  4. target-dump provenance: a manifest of the exact token streams consumed is
     written next to the targets and re-verified before training, and every
     dump self-checks its files before you reclaim the teacher (see
     alis_dwq.provenance),
  5. an append-only run event log under ALIS_DWQ_RUN_LOG_DIR (default
     alis_runs/) — flags, env, data hashes, per-round metrics survive crashes
     instead of living only in stderr.

All mlx-lm dwq flags pass through, e.g.:

  ALIS_DWQ_LAYERS_PER_ROUND=8 ALIS_DWQ_DATA_DIR=./dwq_data \
  python -m alis_dwq.run \
      --model <student-or-tokenizer-source> \
      --quantized-model <student> \
      --target-dir ./targets --mlx-path <out> \
      --num-samples 145 --max-seq-length 512 --batch-size 1 \
      --grad-checkpoint --learning-rate 1e-6 --seed 7

Two-phase on one box: first run with ``--targets-only`` and --model set to the
teacher (the only phase that loads it); then run as above — with targets on
disk mlx-lm never loads the teacher for training.
"""
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

import mlx_lm.quant.dwq as D
from mlx_lm.tuner.datasets import TextDataset

from . import power
from . import provenance
from . import layerwise  # noqa: F401  (installs the layerwise patch)

DATA = Path(os.environ.get("ALIS_DWQ_DATA_DIR", "dwq_data"))


def _argv_value(flag):
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


# the layerwise trainer's upstream signature never sees --target-dir (it is
# closed into target_fn), so stash it for the train-time manifest check
provenance.TARGET_DIR = _argv_value("--target-dir")

_orig_compute = D.compute_dwq_targets
_compute_sig = inspect.signature(_orig_compute)
_loader_info = {"mode": "mlx-lm-default"}


_orig_iterate = D.iterate_batches


def _paced_iterate(*a, **k):
    # ALIS_DWQ_POWER pacing for the dump loop, whose body is upstream code:
    # the time between yields (forward + eval + save) is the work unit
    return power.paced(_orig_iterate(*a, **k))


def _wired_compute(model, *a, **k):
    mx.eval(model.parameters())
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print("[alis-dwq] wired limit set before target dump", file=sys.stderr)
    D.iterate_batches = _paced_iterate
    try:
        result = _orig_compute(model, *a, **k)
    finally:
        D.iterate_batches = _orig_iterate
    bound = _compute_sig.bind(model, *a, **k)
    bound.apply_defaults()
    ba = bound.arguments
    if provenance._rank() == 0:
        manifest = provenance.build_manifest(
            ba["train_data"], ba["valid_data"], ba["batch_size"],
            ba["max_seq_length"], ba["seed"],
            extra={"model": _argv_value("--model"), "loader": _loader_info})
        provenance.write_manifest(ba["save_dir"], manifest)
        provenance.sanity_check_targets(ba["save_dir"], manifest)
        provenance.event("targets_dumped", target_dir=str(ba["save_dir"]),
                         splits=manifest["splits"])
    return result


D.compute_dwq_targets = _wired_compute


def _load_local(tokenizer, data_path, num_samples, max_seq_length, num_valid_samples=32):
    def read(name):
        return [json.loads(l) for l in open(DATA / f"{name}.jsonl")]

    train_ds = TextDataset(read("train"), tokenizer)
    valid_ds = TextDataset(read("valid"), tokenizer)
    perm = np.random.permutation(len(train_ds))[:num_samples].tolist()

    def proc(ds, idx):
        toks, off = ds.process(ds[idx])
        return (toks[:max_seq_length], off)

    train = [proc(train_ds, i) for i in perm]
    valid = [proc(valid_ds, i) for i in range(min(num_valid_samples, len(valid_ds)))]
    print(f"[alis-dwq] local mix: {len(train)} train / {len(valid)} valid",
          file=sys.stderr)
    _loader_info.update(
        mode="local-jsonl", data_dir=str(DATA),
        train_jsonl_sha256=provenance.sha256_file(DATA / "train.jsonl"),
        valid_jsonl_sha256=provenance.sha256_file(DATA / "valid.jsonl"),
        train_rows=len(train_ds), perm=perm)
    provenance.event("data", **_loader_info)
    return train, valid


if DATA.exists():
    D.load_data = _load_local
else:
    print(f"[alis-dwq] {DATA} not found — using mlx-lm's default --data-path loader",
          file=sys.stderr)


def main():
    if _argv_value("--seed") == "0":
        raise SystemExit(
            "[alis-dwq] --seed 0 is not reproducible: iterate_batches only "
            "reseeds when the seed is truthy, so batch order drifts between "
            "the dump, training, and each per-round validation replay — "
            "silently misaligning targets. Use any non-zero seed.")
    provenance.event(
        "run_start", argv=sys.argv,
        env={k: v for k, v in os.environ.items() if k.startswith("ALIS_DWQ_")},
        versions=provenance._versions(), cwd=os.getcwd())
    D.main()


if __name__ == "__main__":
    main()

"""alis-dwq launcher: layerwise DWQ + local mixed calibration data.

Wraps ``mlx_lm.quant.dwq.main()`` with:
  1. the layerwise patch (see alis_dwq.layerwise; ALIS_DWQ_LAYERS_PER_ROUND),
  2. a wired-limit fix applied before the (otherwise unwired) target dump,
  3. a local-jsonl data loader (ALIS_DWQ_DATA_DIR with train.jsonl/valid.jsonl,
     each line {"text": ...}) so calibration mixes (e.g. 45% ZH) are exact and
     reproducible instead of a streamed HF dataset.

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
import json
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

import mlx_lm.quant.dwq as D
from mlx_lm.tuner.datasets import TextDataset

from . import layerwise  # noqa: F401  (installs the layerwise patch)

DATA = Path(os.environ.get("ALIS_DWQ_DATA_DIR", "dwq_data"))

_orig_compute = D.compute_dwq_targets


def _wired_compute(model, *a, **k):
    mx.eval(model.parameters())
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print("[alis-dwq] wired limit set before target dump", file=sys.stderr)
    return _orig_compute(model, *a, **k)


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
    return train, valid


if DATA.exists():
    D.load_data = _load_local
else:
    print(f"[alis-dwq] {DATA} not found — using mlx-lm's default --data-path loader",
          file=sys.stderr)

if __name__ == "__main__":
    D.main()

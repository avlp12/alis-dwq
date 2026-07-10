"""mlx.launch entry for a distributed (multi-box) alis-dwq target dump.

`mlx.launch` takes a *script file path* (which it verifies exists on every
host), not `-m module` — so this 3-liner is the launchable form of
``python -m alis_dwq.run``. Place it (and the alis-dwq install) at the same
path on every host, then:

  mlx.launch --hosts <box1>,<box2> --backend ring \
    --env ALIS_DWQ_DATA_DIR=<shared-or-identical-path>/dwq_data \
    --python <venv>/bin/python \
    examples/distributed_dump_entry.py \
    --model <local teacher dir> --targets-only --pipeline \
    --target-dir ./targets --num-samples 145 --max-seq-length 512 \
    --batch-size 1 --seed 7

The calibration jsonl files must be byte-identical on every host (hash them):
each rank loads its own copy, and the shared seed only aligns the sample
permutation if the underlying rows match.
"""
import alis_dwq.run  # noqa: F401  (import installs the layerwise/wired/data patches)
import mlx_lm.quant.dwq as D

D.main()

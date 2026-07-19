"""Launcher/trainer wiring: the integrity gates must actually fire.

The leaf functions (provenance.verify_*, sanity_*) have their own tests;
these cover the plumbing that connects them — the pre-parse that stashes
--target-dir, the seed-0 refusals, and the trainer-side gate — because a
dead gate stays green everywhere else."""
import importlib
import sys

import pytest

import mlx_lm.quant.dwq as D
from alis_dwq import layerwise, provenance


@pytest.fixture(autouse=True)
def restore_patches(monkeypatch):
    """Reloading alis_dwq.run re-wraps mlx-lm attributes; snapshot/restore so
    tests don't stack wrappers or leak state."""
    saved = (D.compute_dwq_targets, D.iterate_batches, D.load_data,
             provenance.TARGET_DIR)
    yield
    (D.compute_dwq_targets, D.iterate_batches, D.load_data,
     provenance.TARGET_DIR) = saved


def _reload_run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run.py"] + argv)
    import alis_dwq.run as run
    # restore originals so the reload re-wraps the true upstream functions
    D.compute_dwq_targets = run._orig_compute
    D.iterate_batches = run._orig_iterate
    return importlib.reload(run)


@pytest.mark.parametrize("argv", [
    ["--target-dir", "./t"],       # canonical
    ["--target-dir=./t"],          # '=' form
    ["--target-d", "./t"],         # argparse abbreviation
    ["--seed", "7", "--target-di", "./t"],
])
def test_target_dir_spellings_reach_provenance(monkeypatch, argv):
    run = _reload_run(monkeypatch, ["--model", "m"] + argv)
    assert provenance.TARGET_DIR == "./t", run


def test_repeated_target_dir_last_wins(monkeypatch):
    _reload_run(monkeypatch, ["--target-dir", "A", "--target-dir", "B"])
    assert provenance.TARGET_DIR == "B"


@pytest.mark.parametrize("argv", [
    ["--seed", "0"], ["--seed=0"], ["--se", "0"], ["--seed", "00"],
    ["--seed", "7", "--seed", "0"],   # last occurrence wins upstream
])
def test_seed_zero_refused_in_all_spellings(monkeypatch, argv):
    run = _reload_run(monkeypatch, ["--model", "m"] + argv)
    with pytest.raises(SystemExit, match="seed"):
        run.main()


def test_nonzero_seed_reaches_dmain(monkeypatch):
    run = _reload_run(monkeypatch, ["--model", "m", "--seed", "7"])
    called = []
    monkeypatch.setattr(D, "main", lambda: called.append(True))
    monkeypatch.setenv("ALIS_DWQ_RUN_LOG", "0")
    run.main()
    assert called == [True]


def test_trainer_refuses_seed_zero(monkeypatch):
    # defense in depth: the guard inside the patched trainer itself
    with pytest.raises(SystemExit, match="seed"):
        layerwise.layerwise_dwq_quantize(
            None, None, None, [], [], batch_size=1, max_seq_length=64, seed=0)


def test_trainer_gate_fires_for_manifestless_targets(monkeypatch, tmp_path):
    # TARGET_DIR set + no manifest -> the trainer must refuse before touching
    # the model (model=None proves the gate runs first)
    monkeypatch.setattr(provenance, "TARGET_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="manifest"):
        layerwise.layerwise_dwq_quantize(
            None, None, None, [], [], batch_size=1, max_seq_length=64, seed=7)

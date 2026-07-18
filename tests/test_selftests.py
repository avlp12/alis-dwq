"""The two numpy-reference selftests, promoted from `python -m` entry points
to CI-run tests (they were already pinned-seed)."""
from alis_dwq import losses, weight_forensics


def test_losses_selftest():
    losses._selftest()


def test_weight_forensics_selftest():
    weight_forensics._selftest()

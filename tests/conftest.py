"""Isolate the suite from ambient ALIS_DWQ_* workflow env vars — the README
documents them as campaign settings, and running pytest on the same box would
otherwise get spurious failures (gates silently bypassed, event log
suppressed, power.pct() SystemExit inside clip runs)."""
import os

import pytest


@pytest.fixture(autouse=True)
def _clean_alis_dwq_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ALIS_DWQ_"):
            monkeypatch.delenv(key)

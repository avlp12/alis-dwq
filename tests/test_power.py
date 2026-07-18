"""ALIS_DWQ_POWER duty-cycle throttle: parsing, sleep math, pacing wrapper."""
import pytest

from alis_dwq import power


@pytest.fixture(autouse=True)
def reset_pct(monkeypatch):
    monkeypatch.setattr(power, "_pct", None)


def test_default_is_noop(monkeypatch):
    monkeypatch.delenv("ALIS_DWQ_POWER", raising=False)
    slept = []
    monkeypatch.setattr(power.time, "sleep", slept.append)
    power.throttle(2.0)
    assert slept == []


def test_duty_cycle_math(monkeypatch):
    monkeypatch.setenv("ALIS_DWQ_POWER", "25")
    slept = []
    monkeypatch.setattr(power.time, "sleep", slept.append)
    power.throttle(1.0)   # 25% duty -> sleep 3x the work
    power.throttle(0.5)
    assert slept == pytest.approx([3.0, 1.5])


def test_sleep_cap(monkeypatch):
    monkeypatch.setenv("ALIS_DWQ_POWER", "10")
    slept = []
    monkeypatch.setattr(power.time, "sleep", slept.append)
    power.throttle(300.0)  # cold-compile outlier: capped, not 2700 s
    assert slept == [power._CAP_SECONDS]


@pytest.mark.parametrize("bad", ["5", "101", "0", "-3", "fast"])
def test_rejects_bad_values(monkeypatch, bad):
    monkeypatch.setenv("ALIS_DWQ_POWER", bad)
    with pytest.raises(SystemExit):
        power.pct()


def test_paced_passthrough_at_full_power(monkeypatch):
    monkeypatch.delenv("ALIS_DWQ_POWER", raising=False)
    assert list(power.paced(iter([1, 2, 3]))) == [1, 2, 3]


def test_paced_throttles_between_items(monkeypatch):
    monkeypatch.setenv("ALIS_DWQ_POWER", "50")
    slept = []
    monkeypatch.setattr(power.time, "sleep", slept.append)
    assert list(power.paced(iter([1, 2, 3]))) == [1, 2, 3]
    # sleeps happen between consumptions, never before the first item
    assert len(slept) == 2

"""GPU duty-cycle throttle — the ds4 ``--power`` idea for alis-dwq runs.

ds4 throttles by inserting sleeps between work units so a box stays cool and
quiet through hours-long jobs while producing byte-identical output. Same
deal here: ``ALIS_DWQ_POWER=N`` (percent, 10–100) sleeps after each work
unit for ``t * (100 - N) / N`` seconds, where ``t`` is the measured wall
time of the unit — the GPU duty cycle approximates N% and nothing about the
computation changes. 100 (default) is a strict no-op.

Work units instrumented: layerwise train/validate steps, clip_quantize
per-chunk requantization, eval_kld prefill chunks, and the target-dump batch
loop (paced between batches via the iterate_batches wrapper in run.py).
Generation probes are not paced (token-level units are too fine to matter
thermally). A single sleep is capped at 60 s so a cold-compile outlier (first
Metal kernel build can take minutes) does not turn into a proportional stall.

The knob changes pacing only — outputs stay bit-identical — but it is still
default-off and banner-announced like every other non-v0.1 behavior.
"""
import os
import sys
import time

_MIN, _MAX = 10, 100
_CAP_SECONDS = 60.0
_pct = None


def pct():
    """Parse/validate ALIS_DWQ_POWER once; banner when active."""
    global _pct
    if _pct is None:
        raw = os.environ.get("ALIS_DWQ_POWER", str(_MAX))
        try:
            v = int(raw)
        except ValueError:
            raise SystemExit(f"[alis-dwq] ALIS_DWQ_POWER={raw!r}: want an "
                             f"integer percent in [{_MIN}, {_MAX}]")
        if not _MIN <= v <= _MAX:
            raise SystemExit(f"[alis-dwq] ALIS_DWQ_POWER={v} out of range "
                             f"[{_MIN}, {_MAX}] (100 = no throttle; below "
                             f"{_MIN} is a near-stall, use fewer runs instead)")
        _pct = v
        if v < _MAX:
            print(f"[alis-dwq][EXPERIMENTAL] ALIS_DWQ_POWER={v}: sleeping "
                  f"{100 - v}/{v} of each work unit's wall time (ds4-style "
                  "duty-cycle throttle; output is unchanged, only pacing). "
                  "Unset the env var for full speed.", file=sys.stderr)
    return _pct


def throttle(work_seconds):
    """Sleep proportionally to a finished work unit; no-op at 100%."""
    p = pct()
    if p >= _MAX or work_seconds <= 0:
        return
    time.sleep(min(work_seconds * (_MAX - p) / p, _CAP_SECONDS))


def paced(iterable):
    """Wrap a batch iterator so each next() is throttled by the time spent
    consuming the previous item (forward + eval + save = one work unit).
    Used where the loop body is upstream code we don't own (the target
    dump); leaves the iterator untouched at 100%."""
    if pct() >= _MAX:
        yield from iterable
        return
    last = None
    for item in iterable:
        if last is not None:
            throttle(time.time() - last)
        yield item
        last = time.time()

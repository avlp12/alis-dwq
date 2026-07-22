"""Memory limits and recoverable stop gates for long-running ALIS-DWQ phases."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import mlx.core as mx

GIB = 1024**3
_SWAP_RE = re.compile(r"\bused\s*=\s*([0-9.]+)([KMGT])\b", re.IGNORECASE)
_UNITS = {"K": 1024, "M": 1024**2, "G": GIB, "T": 1024**4}


def _as_json_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def emit_evidence(event, *, stream=None, path=None):
    """Emit one stable JSON object to stderr and, optionally, a JSONL file."""
    run_id = os.environ.get("ALIS_DWQ_RUN_ID")
    payload = {
        "schema": "alis-dwq.memory/v1",
        "timestamp_unix": time.time(),
        **({"run_id": run_id} if run_id else {}),
        **event,
    }
    line = json.dumps(payload, sort_keys=True, default=_as_json_value)
    print(f"[alis-dwq][memory] {line}", file=stream or sys.stderr)
    evidence_path = path or os.environ.get("ALIS_DWQ_MEMORY_EVIDENCE_PATH")
    if evidence_path:
        evidence_path = Path(evidence_path)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with evidence_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return payload


def read_swap_used_bytes(
    *,
    platform_system: Callable[[], str] = platform.system,
    runner: Callable = subprocess.run,
) -> Optional[int]:
    """Return Darwin swap usage, or ``None`` when it is unavailable."""
    if platform_system() != "Darwin":
        return None
    try:
        result = runner(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _SWAP_RE.search(result.stdout)
    if match is None:
        return None
    amount, unit = match.groups()
    return int(float(amount) * _UNITS[unit.upper()])


def configure_recommended_wired_limit(
    phase,
    *,
    mx_module=mx,
    platform_system: Callable[[], str] = platform.system,
    emitter: Callable = emit_evidence,
) -> Optional[int]:
    """Use MLX's recommended wired limit on Darwin Metal, otherwise no-op."""
    if platform_system() != "Darwin":
        return None
    metal = getattr(mx_module, "metal", None)
    if metal is None or not metal.is_available():
        return None
    recommended = int(mx_module.device_info()["max_recommended_working_set_size"])
    previous = mx_module.set_wired_limit(recommended)
    emitter(
        {
            "event": "wired_limit_configured",
            "phase": phase,
            "recommended_working_set_bytes": recommended,
            "previous_wired_limit_bytes": previous,
        }
    )
    return recommended


@dataclass(frozen=True)
class MemoryLimits:
    max_peak_fraction: Optional[float] = 0.90
    max_swap_increase_bytes: Optional[int] = 16 * GIB

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None):
        environ = os.environ if environ is None else environ
        peak = float(environ.get("ALIS_DWQ_MAX_PEAK_FRACTION", "0.90"))
        swap_gib = float(environ.get("ALIS_DWQ_MAX_SWAP_INCREASE_GIB", "16"))
        return cls(
            max_peak_fraction=peak if peak > 0 else None,
            max_swap_increase_bytes=(int(swap_gib * GIB) if swap_gib > 0 else None),
        )

    @classmethod
    def guarded_laguna(cls, environ: Optional[Mapping[str, str]] = None):
        """Keep Laguna limits at least as strict as 90% / 16 GiB."""
        configured = cls.from_env(environ)
        peak = configured.max_peak_fraction
        swap = configured.max_swap_increase_bytes
        return cls(
            max_peak_fraction=min(0.90, peak) if peak is not None else 0.90,
            max_swap_increase_bytes=(
                min(16 * GIB, swap) if swap is not None else 16 * GIB
            ),
        )


class MemoryLimitExceeded(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            "ALIS-DWQ memory stop gate tripped: "
            + json.dumps(evidence, sort_keys=True, default=_as_json_value)
        )


class MemoryEvidenceError(RuntimeError):
    """Raised when a memory gate cannot persist its required evidence."""

    def __init__(self, evidence, cause):
        self.evidence = evidence
        self.cause = cause
        super().__init__(f"ALIS-DWQ memory evidence write failed: {cause}")


class MemoryGuard:
    """Track a phase-wide swap baseline and round-local MLX peak memory."""

    def __init__(
        self,
        phase,
        recommended_working_set_bytes,
        *,
        limits=None,
        mx_module=mx,
        swap_reader=read_swap_used_bytes,
        emitter=emit_evidence,
        require_recommended_working_set=False,
        require_swap_measurement=False,
    ):
        self.phase = phase
        self.recommended_working_set_bytes = recommended_working_set_bytes
        self.limits = limits or MemoryLimits.from_env()
        self.mx = mx_module
        self.swap_reader = swap_reader
        self.emitter = emitter
        self.require_recommended_working_set = bool(
            require_recommended_working_set
        )
        self.require_swap_measurement = bool(require_swap_measurement)
        self.baseline_swap_bytes = None
        self.started = False

    def start(self):
        self.baseline_swap_bytes = self.swap_reader()
        self.started = True
        self._reset_peak()
        return self.emitter(
            {
                "event": "memory_baseline",
                "phase": self.phase,
                "recommended_working_set_bytes": (self.recommended_working_set_bytes),
                "baseline_swap_bytes": self.baseline_swap_bytes,
                "max_peak_fraction": self.limits.max_peak_fraction,
                "max_swap_increase_bytes": (self.limits.max_swap_increase_bytes),
                "require_recommended_working_set": (
                    self.require_recommended_working_set
                ),
                "require_swap_measurement": self.require_swap_measurement,
            }
        )

    def begin_round(self, round_index, layers):
        if not self.started:
            self.start()
        self._reset_peak()
        return self.emitter(
            {
                "event": "round_memory_baseline",
                "phase": self.phase,
                "round": round_index,
                "layers": list(layers),
                "baseline_swap_bytes": self.baseline_swap_bytes,
            }
        )

    def _reset_peak(self):
        reset = getattr(self.mx, "reset_peak_memory", None)
        if reset is not None:
            reset()

    def sample(self, checkpoint, **context):
        peak = int(self.mx.get_peak_memory())
        get_active = getattr(self.mx, "get_active_memory", None)
        active = int(get_active()) if get_active is not None else None
        if active is not None:
            peak = max(peak, active)
        swap = self.swap_reader()
        swap_increase = (
            max(0, swap - self.baseline_swap_bytes)
            if swap is not None and self.baseline_swap_bytes is not None
            else None
        )
        ratio = (
            peak / self.recommended_working_set_bytes
            if self.recommended_working_set_bytes
            else None
        )
        return {
            "event": "memory_sample",
            "phase": self.phase,
            "checkpoint": checkpoint,
            "peak_working_set_bytes": peak,
            "active_working_set_bytes": active,
            "recommended_working_set_bytes": (self.recommended_working_set_bytes),
            "peak_fraction": ratio,
            "swap_used_bytes": swap,
            "baseline_swap_bytes": self.baseline_swap_bytes,
            "swap_increase_bytes": swap_increase,
            **context,
        }

    def check(self, checkpoint, **context):
        if not self.started:
            self.start()
        evidence = self.sample(checkpoint, **context)
        reasons = []
        if (
            self.require_recommended_working_set
            and (
                evidence["recommended_working_set_bytes"] is None
                or evidence["recommended_working_set_bytes"] <= 0
            )
        ):
            reasons.append("recommended_working_set_unavailable")
        if (
            self.limits.max_peak_fraction is not None
            and evidence["peak_fraction"] is not None
            and evidence["peak_fraction"] > self.limits.max_peak_fraction
        ):
            reasons.append("peak_working_set")
        if self.require_swap_measurement and (
            evidence["baseline_swap_bytes"] is None
            or evidence["swap_used_bytes"] is None
        ):
            reasons.append("swap_measurement_unavailable")
        if (
            self.limits.max_swap_increase_bytes is not None
            and evidence["swap_increase_bytes"] is not None
            and evidence["swap_increase_bytes"] >= self.limits.max_swap_increase_bytes
        ):
            reasons.append("swap_increase")
        if reasons:
            evidence = {
                **evidence,
                "event": "memory_stop_gate",
                "status": "abort",
                "reasons": reasons,
                "max_peak_fraction": self.limits.max_peak_fraction,
                "max_swap_increase_bytes": (self.limits.max_swap_increase_bytes),
            }
            try:
                emitted = self.emitter(evidence)
            except Exception as exc:
                # Crossing the stop threshold must remain a stop even when the
                # evidence sink itself fails.  Preserve the gate payload on the
                # raised exception so the caller can still report it.
                raise MemoryLimitExceeded(evidence) from exc
            raise MemoryLimitExceeded(emitted or evidence)
        # Successful checks are evidence too: without emitting them, a clean
        # run records only its baseline and cannot prove the observed peak or
        # swap delta after the fact.
        try:
            return self.emitter(evidence) or evidence
        except Exception as exc:
            raise MemoryEvidenceError(evidence, exc) from exc

    def record_round_abort(self, error, *, round_index, layers, restored):
        return self.emitter(
            {
                "event": "round_aborted",
                "phase": self.phase,
                "round": round_index,
                "layers": list(layers),
                "error_type": type(error).__name__,
                "snapshot_restored": bool(restored),
            }
        )


def restore_round_snapshot(model, snapshot, *, mx_module=mx):
    """Restore and evaluate the active round's small trainable snapshot."""
    model.update(snapshot)
    mx_module.eval(model.trainable_parameters())
    return True


def check_round_or_restore(
    guard,
    model,
    snapshot,
    checkpoint,
    *,
    round_index,
    layers,
    mx_module=mx,
    **context,
):
    """Check a stop gate and restore the active round before propagating it."""
    try:
        return guard.check(
            checkpoint,
            round=round_index,
            layers=list(layers),
            **context,
        )
    except (MemoryLimitExceeded, MemoryEvidenceError) as error:
        restored = False
        try:
            restored = restore_round_snapshot(model, snapshot, mx_module=mx_module)
        finally:
            try:
                guard.record_round_abort(
                    error,
                    round_index=round_index,
                    layers=layers,
                    restored=restored,
                )
            except Exception as record_error:
                # Never replace the original stop/evidence failure after the
                # snapshot has been restored.
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"round-abort evidence also failed: {record_error}"
                    )
        raise

"""High-integrity launcher for layerwise ALIS-DWQ.

Besides installing the layerwise trainer, this wrapper binds local JSONL data
to tokenizer artifacts and numeric teacher targets, enforces transactional
target creation, applies memory gates after every target batch, and emits a
two-event, no-clobber run-evidence JSONL for completed runs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx_lm.quant.dwq as D
import numpy as np
from mlx_lm.tuner.datasets import TextDataset

from . import layerwise  # noqa: F401  (installs the layerwise patch)
from .io_utils import directory_digest, move_no_replace
from .memory_guard import (
    MemoryGuard,
    MemoryLimits,
    configure_recommended_wired_limit,
)
from .target_contract import (
    CONTRACT_NAME,
    TOKENIZER_FILES,
    build_target_contract,
    canonical_sha256,
    load_json,
    numeric_target_files,
    prepare_local_data,
    sha256_file,
    tokenizer_files_sha256,
    validate_target_contract,
    write_contract_no_replace,
)

ALIS_DWQ_BASE_REVISION = "e68c8f708032bfc751d4393b3544c600572e0c16"
MLX_LM_BASE_REVISION = "cf10f962b7a20e63a6df43dbf0faf06070153d40"
TARGET_TOP_K = 1024
DATA = Path(os.environ.get("ALIS_DWQ_DATA_DIR", "dwq_data")).expanduser()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_RUNTIME_FILES = frozenset(TOKENIZER_FILES)
_FROZEN_TOKENIZER_CONFIG_NAME = "config.json"
_FROZEN_TOKENIZER_CONFIG_BYTES = b'{"model_type":"mistral"}\n'
_FROZEN_TOKENIZER_REQUIRED_OPTIONS = {
    "fix_mistral_regex": True,
    "local_files_only": True,
}
_TOKENIZER_FILE_FIELDS = frozenset(
    {
        "added_tokens_file",
        "chat_template_file",
        "merges_file",
        "sentencepiece_model_file",
        "sp_model_file",
        "special_tokens_map_file",
        "tokenizer_file",
        "vocab_file",
    }
)
_KNOWN_TOKENIZER_FILES = _TOKENIZER_RUNTIME_FILES | {
    "chat_template.json",
    "chat_templates.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
}
_JINJA_FILE_RE = re.compile(
    r"{%[-+]?\s*(?:include|import|from)\s+(['\"])([^'\"]+)\1",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RuntimeTokenizerBundle:
    source: Path
    target_contract: Path
    target_contract_sha256: str
    files_sha256: dict[str, str]
    files_bytes: dict[str, bytes]


_TRACKED_ENV = (
    "ALIS_DWQ_RUN_ID",
    "ALIS_DWQ_DATA_DIR",
    "ALIS_DWQ_NUM_VALID_SAMPLES",
    "ALIS_DWQ_LAYERS_PER_ROUND",
    "ALIS_DWQ_EXTRAS_MODE",
    "ALIS_DWQ_MAX_PEAK_FRACTION",
    "ALIS_DWQ_MAX_SWAP_INCREASE_GIB",
    "ALIS_DWQ_MEMORY_EVIDENCE_PATH",
    "ALIS_DWQ_RUN_EVIDENCE_PATH",
    "ALIS_DWQ_TEXT_TOKENIZATION",
    "ALIS_DWQ_MAX_ROUNDS",
    "ALIS_DWQ_MAX_STEPS_PER_ROUND",
    "ALIS_DWQ_TRAIN_ROUTERS",
    "ALIS_DWQ_LORA_RANK",
    "ALIS_DWQ_CKA_MONITOR",
    "ALIS_DWQ_LOSS",
    "ALIS_DWQ_ADAPTER_DIR",
    "ALIS_DWQ_TEACHER_IDENTITY",
    "ALIS_DWQ_TEACHER_REVISION",
)
_SOURCE_FILES = (
    "alis_dwq/run.py",
    "alis_dwq/layerwise.py",
    "alis_dwq/memory_guard.py",
    "alis_dwq/preflight.py",
    "alis_dwq/io_utils.py",
    "alis_dwq/target_contract.py",
    "alis_dwq/losses.py",
)

_orig_compute = D.compute_dwq_targets
_orig_iterate_batches = D.iterate_batches
_RUN_CONTEXT: dict[str, Any] | None = None
_ACTIVE_DATA_BINDING: dict[str, Any] | None = None
_RUN_MEMORY_GUARD: MemoryGuard | None = None
_TARGET_CONTRACT_DIGEST: str | None = None
_TARGET_CONTRACT_PATH: Path | None = None


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing, not regular, or a symlink: {path}")
    return path.read_bytes()


def _safe_tokenizer_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or value.startswith("._")
        or Path(value).name != value
    ):
        raise ValueError(f"{label} is not a safe top-level file name: {value!r}")
    return value


def _is_tokenizer_related_name(name: str) -> bool:
    candidate = name[2:] if name.startswith("._") else name
    lower = candidate.lower()
    return (
        candidate in _KNOWN_TOKENIZER_FILES
        or lower.startswith(
            (
                "added_token",
                "chat_template",
                "merges",
                "sentencepiece",
                "special_token",
                "spiece",
                "tokenizer",
                "vocab",
            )
        )
        or lower.endswith((".jinja", ".jinja2"))
    )


def _tokenizer_dependencies(files: Mapping[str, bytes]) -> set[str]:
    config = _load_json_bytes(
        files["tokenizer_config.json"], label="runtime tokenizer config"
    )
    dependencies = set()
    for field in _TOKENIZER_FILE_FIELDS:
        value = config.get(field)
        if value is None:
            continue
        dependencies.add(
            _safe_tokenizer_name(value, label=f"tokenizer config field {field!r}")
        )

    def scan_template(value: object) -> None:
        if isinstance(value, str):
            for match in _JINJA_FILE_RE.finditer(value):
                dependencies.add(
                    _safe_tokenizer_name(
                        match.group(2), label="chat template dependency"
                    )
                )
            if "{%" not in value and value.endswith((".jinja", ".jinja2")):
                dependencies.add(
                    _safe_tokenizer_name(value, label="chat template file")
                )
        elif isinstance(value, dict):
            for nested in value.values():
                scan_template(nested)
        elif isinstance(value, list):
            for nested in value:
                scan_template(nested)

    scan_template(config.get("chat_template"))
    for name, raw in files.items():
        if not name.endswith((".jinja", ".jinja2")):
            continue
        try:
            scan_template(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"runtime tokenizer template is not UTF-8: {name}"
            ) from exc
    return dependencies


def _validate_exact_tokenizer_files(
    root: Path,
    declared: Mapping[str, str],
    *,
    expected_bytes: Mapping[str, bytes] | None = None,
    clean_root: bool,
) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"runtime tokenizer root is missing or a symlink: {root}")
    entries = list(root.iterdir())
    apple_double = sorted(
        path.name
        for path in entries
        if path.name.startswith("._")
        and (clean_root or _is_tokenizer_related_name(path.name))
    )
    if apple_double:
        raise ValueError(
            f"runtime tokenizer contains AppleDouble files: {apple_double}"
        )
    if clean_root:
        observed = {path.name for path in entries}
    else:
        observed = {
            path.name for path in entries if _is_tokenizer_related_name(path.name)
        }
    if observed != set(declared):
        raise ValueError(
            "runtime tokenizer file set differs from target contract: "
            f"expected={sorted(declared)}, observed={sorted(observed)}"
        )

    files = {}
    for name, expected_digest in declared.items():
        raw = _regular_file_bytes(root / name, label=f"runtime tokenizer file {name}")
        actual_digest = _sha256_bytes(raw)
        if actual_digest != expected_digest:
            raise ValueError(
                f"runtime tokenizer hash mismatch for {name}: "
                f"{actual_digest} != {expected_digest}"
            )
        if expected_bytes is not None and raw != expected_bytes[name]:
            raise ValueError(f"runtime tokenizer bytes changed during the run: {name}")
        files[name] = raw

    dependencies = _tokenizer_dependencies(files)
    if not dependencies <= set(declared):
        raise ValueError(
            "runtime tokenizer references files outside the target contract: "
            f"{sorted(dependencies - set(declared))}"
        )
    return files


def _load_runtime_tokenizer_bundle(
    source: Path, target_contract: Path
) -> RuntimeTokenizerBundle:
    source = Path(source).expanduser()
    target_contract = Path(target_contract).expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"runtime tokenizer source is missing or a symlink: {source}")
    contract_raw = _regular_file_bytes(target_contract, label="target contract")
    contract = _load_json_bytes(contract_raw, label="target contract")
    if contract.get("schema") != "alis-dwq.targets/v1":
        raise ValueError("target contract must use alis-dwq.targets/v1")

    declared_value = contract.get("tokenizer_files_sha256")
    if not isinstance(declared_value, dict):
        raise ValueError("target contract lacks tokenizer_files_sha256")
    declared = {}
    for raw_name, digest in declared_value.items():
        name = _safe_tokenizer_name(raw_name, label="target tokenizer file")
        if name not in _TOKENIZER_RUNTIME_FILES:
            raise ValueError(f"unsupported target tokenizer file: {name}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid target tokenizer SHA-256 for {name}")
        declared[name] = digest
    if not {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    } <= set(declared):
        raise ValueError("target tokenizer file set is incomplete")

    equivalence = contract.get("tokenizer_equivalence")
    row_evidence = (
        equivalence.get("row_evidence") if isinstance(equivalence, dict) else None
    )
    if (
        not isinstance(equivalence, dict)
        or set(equivalence)
        != {
            "schema",
            "mode",
            "source_tokenizer_files_sha256",
            "source_tokenizer_options",
            "runtime_tokenizer_files_sha256",
            "row_evidence",
            "all_rows_verified",
        }
        or equivalence.get("schema") != "alis-dwq.tokenizer-equivalence/v2"
        or equivalence.get("mode")
        not in {"file-identity", "all-declared-row-token-ids"}
        or equivalence.get("source_tokenizer_options") != {"fix_mistral_regex": True}
        or equivalence.get("all_rows_verified") is not True
        or equivalence.get("runtime_tokenizer_files_sha256") != declared
        or not isinstance(row_evidence, dict)
        or set(row_evidence)
        != {
            "schema",
            "method",
            "tokenization",
            "row_count",
            "splits",
            "all_rows_verified",
        }
        or row_evidence.get("schema") != "alis-dwq.tokenizer-row-equivalence/v1"
        or row_evidence.get("method") != "live-runtime-tokenizer-encode/v1"
        or row_evidence.get("tokenization")
        != {
            "name": "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat",
            "preformatted_chat": True,
            "add_special_tokens": False,
            "append_eos": False,
        }
        or row_evidence.get("row_count") != 220
        or row_evidence.get("all_rows_verified") is not True
        or not isinstance(row_evidence.get("splits"), dict)
        or set(row_evidence["splits"]) != {"train", "valid", "heldout"}
    ):
        raise ValueError("target tokenizer equivalence evidence is invalid")

    source_resolved = source.resolve(strict=True)
    contract_resolved = target_contract.resolve(strict=True)
    files = _validate_exact_tokenizer_files(source_resolved, declared, clean_root=True)
    return RuntimeTokenizerBundle(
        source=source_resolved,
        target_contract=contract_resolved,
        target_contract_sha256=_sha256_bytes(contract_raw),
        files_sha256=dict(sorted(declared.items())),
        files_bytes=files,
    )


def _revalidate_runtime_tokenizer_bundle(bundle: RuntimeTokenizerBundle) -> None:
    contract_raw = _regular_file_bytes(bundle.target_contract, label="target contract")
    if _sha256_bytes(contract_raw) != bundle.target_contract_sha256:
        raise RuntimeError("target contract changed during the run")
    _load_json_bytes(contract_raw, label="target contract")
    _validate_exact_tokenizer_files(
        bundle.source,
        bundle.files_sha256,
        expected_bytes=bundle.files_bytes,
        clean_root=True,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _materialize_frozen_runtime_tokenizer(
    root: Path, bundle: RuntimeTokenizerBundle
) -> None:
    _revalidate_runtime_tokenizer_bundle(bundle)
    for name, raw in bundle.files_bytes.items():
        destination = root / name
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    sentinel = root / _FROZEN_TOKENIZER_CONFIG_NAME
    with sentinel.open("xb") as handle:
        handle.write(_FROZEN_TOKENIZER_CONFIG_BYTES)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(root)
    _revalidate_frozen_runtime_tokenizer(root, bundle)


def _revalidate_frozen_runtime_tokenizer(
    root: Path, bundle: RuntimeTokenizerBundle
) -> None:
    _revalidate_runtime_tokenizer_bundle(bundle)
    expected_bytes = {
        **bundle.files_bytes,
        _FROZEN_TOKENIZER_CONFIG_NAME: _FROZEN_TOKENIZER_CONFIG_BYTES,
    }
    expected_hashes = {
        **bundle.files_sha256,
        _FROZEN_TOKENIZER_CONFIG_NAME: _sha256_bytes(_FROZEN_TOKENIZER_CONFIG_BYTES),
    }
    _validate_exact_tokenizer_files(
        root,
        expected_hashes,
        expected_bytes=expected_bytes,
        clean_root=True,
    )


def _remove_generated_tokenizer_files(root: Path) -> None:
    # On non-APFS volumes macOS may remove ``._name`` as a side effect of
    # unlinking ``name``. Delete AppleDouble entries first so a later iterator
    # item cannot disappear with its associated payload file.
    paths = sorted(
        root.iterdir(), key=lambda path: (not path.name.startswith("._"), path.name)
    )
    for path in paths:
        if not _is_tokenizer_related_name(path.name):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"generated tokenizer path is not a regular file: {path}")
        path.unlink()


def _remove_generated_appledouble(destination: Path) -> None:
    sidecar = destination.with_name(f"._{destination.name}")
    if not sidecar.exists() and not sidecar.is_symlink():
        return
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"generated AppleDouble path is not a regular file: {sidecar}")
    sidecar.unlink()


def _install_runtime_tokenizer(
    root: Path, bundle: RuntimeTokenizerBundle
) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"DWQ staging root is missing or a symlink: {root}")
    _revalidate_runtime_tokenizer_bundle(bundle)
    _remove_generated_tokenizer_files(root)
    for name, raw in bundle.files_bytes.items():
        destination = root / name
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _remove_generated_appledouble(destination)
    _fsync_directory(root)
    _revalidate_installed_runtime_tokenizer(root, bundle)
    return dict(sorted(bundle.files_sha256.items()))


def _revalidate_installed_runtime_tokenizer(
    root: Path, bundle: RuntimeTokenizerBundle
) -> None:
    _revalidate_runtime_tokenizer_bundle(bundle)
    output = _validate_exact_tokenizer_files(
        root,
        bundle.files_sha256,
        expected_bytes=bundle.files_bytes,
        clean_root=False,
    )
    if output != bundle.files_bytes:
        raise RuntimeError("runtime tokenizer output differs from frozen bytes")


def _option_count(values: list[str], name: str) -> int:
    return sum(
        str(value) == name or str(value).startswith(f"{name}=") for value in values
    )


def _parse_run_context(argv=None, environ=None) -> dict[str, Any]:
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", "-m")
    parser.add_argument("--quantized-model")
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--target-dir")
    parser.add_argument("--runtime-tokenizer-source")
    parser.add_argument("--target-contract")
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=1025)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    values = list(sys.argv if argv is None else argv)
    if values and not str(values[0]).startswith("-"):
        values = values[1:]
    for option in ("--runtime-tokenizer-source", "--target-contract"):
        if _option_count(values, option) > 1:
            raise ValueError(f"{option} may be supplied exactly once")
    args, _ = parser.parse_known_args(values)
    if not args.model:
        raise ValueError("--model is required")
    if args.seed <= 0:
        raise ValueError("contracted ALIS-DWQ runs require --seed > 0")
    if not args.target_dir:
        raise ValueError("contracted ALIS-DWQ runs require --target-dir")
    if bool(args.runtime_tokenizer_source) != bool(args.target_contract):
        raise ValueError(
            "--runtime-tokenizer-source and --target-contract must be supplied together"
        )
    target_dir = Path(args.target_dir).expanduser()
    runtime_tokenizer_source = None
    target_contract = None
    if args.runtime_tokenizer_source:
        runtime_tokenizer_source = Path(args.runtime_tokenizer_source).expanduser()
        target_contract = Path(args.target_contract).expanduser()
        expected_contract = target_dir / CONTRACT_NAME
        if os.path.abspath(target_contract) != os.path.abspath(expected_contract):
            raise ValueError(
                "--target-contract must be exactly --target-dir/target-contract.json"
            )
    if args.pipeline:
        raise ValueError(
            "contracted ALIS-DWQ runs do not support --pipeline: per-rank "
            "checkpoint shards cannot produce a verified full-teacher digest"
        )
    try:
        lora_rank = int(environ.get("ALIS_DWQ_LORA_RANK", "0") or 0)
    except ValueError as exc:
        raise ValueError("ALIS_DWQ_LORA_RANK must be an integer") from exc
    if lora_rank != 0:
        raise ValueError(
            "contracted ALIS-DWQ runs require ALIS_DWQ_LORA_RANK=0: the "
            "separate adapter artifact is not transactionally bound to run evidence"
        )
    return {
        "model": Path(args.model).expanduser(),
        "quantized_model": (
            Path(args.quantized_model).expanduser() if args.quantized_model else None
        ),
        "mlx_path": Path(args.mlx_path).expanduser(),
        "target_dir": target_dir,
        "runtime_tokenizer_source": runtime_tokenizer_source,
        "target_contract": target_contract,
        "tokenizer_path": runtime_tokenizer_source or Path(args.model).expanduser(),
        "targets_only": bool(args.targets_only),
        "pipeline": bool(args.pipeline),
        "lora_rank": lora_rank,
        "num_samples": args.num_samples,
        "num_valid_samples": int(environ.get("ALIS_DWQ_NUM_VALID_SAMPLES", "32")),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_dir": Path(environ.get("ALIS_DWQ_DATA_DIR", str(DATA))).expanduser(),
        "tokenization": environ.get("ALIS_DWQ_TEXT_TOKENIZATION", "text_dataset")
        .strip()
        .lower(),
    }


def _target_dir_has_payload(path: Path) -> bool:
    return all(bool(numeric_target_files(path / split)) for split in ("train", "valid"))


def _validate_existing_target_structure(
    path: Path,
    *,
    max_seq_length: int | None,
    batch_size: int | None,
    top_k: int | None,
    seed: int | None,
) -> None:
    """Reject malformed/partial reuse before hashing or loading checkpoints."""
    contract_path = path / CONTRACT_NAME
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError(
            f"existing numeric targets lack a regular {CONTRACT_NAME}: {path}"
        )
    contract = load_json(contract_path)
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "alis-dwq.targets/v1"
    ):
        raise ValueError("existing target contract has an unsupported schema")
    teacher = contract.get("teacher")
    if (
        not isinstance(teacher, dict)
        or not isinstance(teacher.get("identity"), str)
        or not teacher["identity"]
        or not isinstance(teacher.get("revision"), str)
        or not teacher["revision"]
        or not isinstance(teacher.get("checkpoint_digest"), str)
        or re.fullmatch(r"[0-9a-f]{64}", teacher["checkpoint_digest"]) is None
    ):
        raise ValueError("existing target contract has invalid teacher provenance")
    expected_scalars = {
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "top_k": top_k,
        "seed": seed,
    }
    for field, expected in expected_scalars.items():
        value = contract.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"existing target contract has invalid {field}")
        if expected is not None and value != expected:
            raise ValueError(
                f"existing target contract {field} mismatch: {value} != {expected}"
            )
    contract_batch_size = contract["batch_size"]
    splits = contract.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "valid"}:
        raise ValueError("existing target contract must contain train and valid splits")
    for split in ("train", "valid"):
        split_dir = path / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise FileExistsError(
                f"target directory exists but is partial or empty (no-clobber): {path}"
            )
        split_contract = splits[split]
        if not isinstance(split_contract, dict):
            raise ValueError(f"existing target contract {split} split is invalid")
        target_count = split_contract.get("target_count")
        selected_count = split_contract.get("selected_count")
        rows = split_contract.get("rows")
        if (
            type(target_count) is not int
            or target_count <= 0
            or type(selected_count) is not int
            or selected_count // contract_batch_size != target_count
            or not isinstance(rows, list)
            or len(rows) != target_count * contract_batch_size
        ):
            raise ValueError(
                f"existing target contract {split} counts/rows are inconsistent"
            )
        expected_names = [f"{index:010d}.safetensors" for index in range(target_count)]
        actual_names = [item.name for item in numeric_target_files(split_dir)]
        if actual_names != expected_names:
            raise ValueError(
                f"existing target contract {split} target file set mismatch"
            )
        expected_positions = {
            (target_index, batch_position)
            for target_index in range(target_count)
            for batch_position in range(contract_batch_size)
        }
        actual_positions = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(
                    f"existing target contract {split} row is not an object"
                )
            position = (row.get("target_index"), row.get("batch_position"))
            expected_file = (
                f"{split}/{position[0]:010d}.safetensors"
                if type(position[0]) is int
                else None
            )
            if row.get("target_file") != expected_file:
                raise ValueError(
                    f"existing target contract {split} row target_file mismatch"
                )
            actual_positions.add(position)
        if actual_positions != expected_positions:
            raise ValueError(
                f"existing target contract {split} target positions mismatch"
            )
        for name in expected_names:
            target = split_dir / name
            if target.is_symlink() or not target.is_file():
                raise ValueError(
                    f"existing target file is missing or a symlink: {split}/{name}"
                )


def _target_dir_state(
    path: Path,
    *,
    max_seq_length: int | None = None,
    batch_size: int | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Classify a no-clobber target path before any checkpoint is hashed/loaded."""
    path = Path(path)
    if path.is_symlink():
        raise FileExistsError(f"target directory is a symlink (no-clobber): {path}")
    if not path.exists():
        return "new"
    if not path.is_dir() or not _target_dir_has_payload(path):
        raise FileExistsError(
            f"target directory exists but is partial or empty (no-clobber): {path}"
        )
    _validate_existing_target_structure(
        path,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        top_k=top_k,
        seed=seed,
    )
    return "reuse"


def _requires_teacher_stability(context: Mapping[str, Any], target_state: str) -> bool:
    quantized_model = context.get("quantized_model")
    distinct_teacher = quantized_model is not None and (
        Path(context["model"]).resolve() != Path(quantized_model).resolve()
    )
    return bool(
        target_state == "new" or context.get("targets_only") or distinct_teacher
    )


def _load_local(
    tokenizer, data_path, num_samples, max_seq_length, num_valid_samples=32
):
    """mlx-lm load_data replacement with a live, reusable data binding."""
    del data_path
    global _ACTIVE_DATA_BINDING, _TARGET_CONTRACT_DIGEST, _TARGET_CONTRACT_PATH

    if _RUN_CONTEXT is None:
        # Direct unit/library use keeps the historical loader behavior.  The
        # CLI always sets a context and therefore always builds a contract.
        def read(name):
            return [json.loads(line) for line in open(DATA / f"{name}.jsonl")]

        train_rows, valid_rows = read("train"), read("valid")
        train_ds = TextDataset(train_rows, tokenizer)
        valid_ds = TextDataset(valid_rows, tokenizer)
        indices = np.random.permutation(len(train_ds))[:num_samples].tolist()
        tokenization = (
            os.environ.get("ALIS_DWQ_TEXT_TOKENIZATION", "text_dataset").strip().lower()
        )

        def process(dataset, index):
            if tokenization == "preformatted_chat":
                tokens = tokenizer.encode(
                    dataset[index]["text"], add_special_tokens=False
                )
                offset = 0
            else:
                tokens, offset = dataset.process(dataset[index])
            return tokens[:max_seq_length], offset

        requested = int(
            os.environ.get("ALIS_DWQ_NUM_VALID_SAMPLES", str(num_valid_samples))
        )
        if requested <= 0:
            requested = len(valid_ds)
        return (
            [process(train_ds, index) for index in indices],
            [
                process(valid_ds, index)
                for index in range(min(requested, len(valid_ds)))
            ],
        )

    context = _RUN_CONTEXT
    train, valid, binding = prepare_local_data(
        tokenizer,
        context["data_dir"],
        tokenizer_path=context["tokenizer_path"],
        num_samples=num_samples,
        num_valid_samples=context["num_valid_samples"],
        max_seq_length=max_seq_length,
        seed=context["seed"],
        tokenization=context["tokenization"],
        text_dataset_factory=TextDataset,
    )
    _ACTIVE_DATA_BINDING = binding
    target_dir = context["target_dir"]
    if target_dir is not None and target_dir.exists():
        if _target_dir_has_payload(target_dir):
            contract, digest = validate_target_contract(
                binding,
                target_dir,
                max_seq_length=context["max_seq_length"],
                batch_size=context["batch_size"],
                top_k=TARGET_TOP_K,
                seed=context["seed"],
            )
            if context.get("verify_teacher_checkpoint"):
                live_teacher_digest = directory_digest(context["model"])
                if (
                    contract.get("teacher", {}).get("checkpoint_digest")
                    != live_teacher_digest
                ):
                    raise ValueError(
                        "live teacher checkpoint does not match target contract"
                    )
                context["teacher_checkpoint_digest"] = live_teacher_digest
            _TARGET_CONTRACT_PATH = target_dir / CONTRACT_NAME
            _TARGET_CONTRACT_DIGEST = digest
            print(
                "[alis-dwq] verified target contract against live data, tokenizer, "
                "batch order, and target hashes",
                file=sys.stderr,
            )
        elif (target_dir / CONTRACT_NAME).exists():
            raise ValueError("target contract exists but numeric targets are missing")
    print(
        f"[alis-dwq] local mix: {len(train)} train / {len(valid)} valid "
        f"(tokenization: {context['tokenization']})",
        file=sys.stderr,
    )
    return train, valid


if DATA.exists():
    D.load_data = _load_local
else:
    print(
        f"[alis-dwq] {DATA} not found — using mlx-lm's default --data-path loader; "
        "target contracts are unavailable",
        file=sys.stderr,
    )


def _teacher_identity(context: Mapping[str, Any], environ=None) -> tuple[str, str]:
    environ = os.environ if environ is None else environ
    identity = environ.get("ALIS_DWQ_TEACHER_IDENTITY")
    revision = environ.get("ALIS_DWQ_TEACHER_REVISION")
    model = Path(context["model"])
    plan_path = model / "conversion_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        identity = identity or plan.get("source_repo")
        revision = revision or plan.get("source_revision")
    config_path = model / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        identity = identity or config.get("_name_or_path")
        revision = revision or config.get("_commit_hash")
    identity = identity or str(model.resolve())
    if not revision:
        raise ValueError(
            "teacher revision is unavailable; set ALIS_DWQ_TEACHER_REVISION or "
            "use a checkpoint with conversion_plan.json source_revision"
        )
    return str(identity), str(revision)


def _distributed_barrier(group) -> None:
    if group.size() > 1:
        value = mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu)
        mx.eval(value)


def _validate_target_publish_inputs(context: Mapping[str, Any], binding: dict) -> None:
    """Revalidate lazy target-dump inputs immediately before publication."""
    _validate_live_data_binding(context, binding)
    expected = context.get("teacher_checkpoint_digest")
    if not isinstance(expected, str) or directory_digest(context["model"]) != expected:
        raise RuntimeError("teacher checkpoint changed during target computation")


def _wired_compute(
    model,
    save_dir,
    train_data,
    valid_data,
    batch_size,
    max_seq_length,
    seed,
):
    """Create targets under staging and publish only with a complete sidecar."""
    global _TARGET_CONTRACT_DIGEST, _TARGET_CONTRACT_PATH
    if _RUN_CONTEXT is None or _ACTIVE_DATA_BINDING is None:
        raise RuntimeError("target computation requires the contracted local loader")
    final = Path(save_dir).expanduser()
    group = mx.distributed.init()
    if group.size() != 1:
        raise RuntimeError(
            "contracted target creation requires exactly one process; distributed "
            "teacher provenance is not yet supported"
        )
    rank = group.rank()
    run_id = os.environ["ALIS_DWQ_RUN_ID"]
    staging = final.with_name(f"{final.name}.partial-{run_id}-{os.getpid()}")
    if rank == 0:
        if final.exists():
            raise FileExistsError(f"target directory exists (no-clobber): {final}")
        staging.mkdir(parents=True, exist_ok=False)
    _distributed_barrier(group)

    phase = "target-computation"
    guard = _RUN_MEMORY_GUARD
    if guard is None:
        strict_laguna = _is_guarded_laguna_context(_RUN_CONTEXT)
        recommended = configure_recommended_wired_limit(phase)
        guard = MemoryGuard(
            phase,
            recommended,
            limits=(
                MemoryLimits.guarded_laguna()
                if strict_laguna
                else MemoryLimits.from_env()
            ),
            require_recommended_working_set=strict_laguna,
            require_swap_measurement=strict_laguna,
        )
        guard.start()
    guard.check("before-model-eval")
    mx.eval(model.parameters())
    guard.check("before-target-dump")

    split_calls = 0

    def guarded_iterate(*args, **kwargs):
        nonlocal split_calls
        split = "valid" if split_calls == 0 else "train"
        split_calls += 1
        for batch_index, item in enumerate(_orig_iterate_batches(*args, **kwargs)):
            guard.check("before-target-batch", split=split, target_index=batch_index)
            yield item
            # Execution resumes only after upstream has evaluated and, on rank
            # zero, saved this target.  The final yielded batch is also checked.
            guard.check("after-target-batch", split=split, target_index=batch_index)

    previous_iterate = D.iterate_batches
    D.iterate_batches = guarded_iterate
    try:
        _orig_compute(
            model,
            staging,
            train_data,
            valid_data,
            batch_size,
            max_seq_length,
            seed,
        )
    finally:
        D.iterate_batches = previous_iterate
    guard.check("after-target-dump")
    _distributed_barrier(group)

    if rank == 0:
        identity, revision = _teacher_identity(_RUN_CONTEXT)
        contract = build_target_contract(
            _ACTIVE_DATA_BINDING,
            staging,
            run_id=run_id,
            teacher_identity=identity,
            teacher_revision=revision,
            teacher_checkpoint_digest=_RUN_CONTEXT["teacher_checkpoint_digest"],
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            top_k=TARGET_TOP_K,
            seed=seed,
        )
        contract_path = write_contract_no_replace(staging, contract)
        contract_digest = sha256_file(contract_path)
        _validate_target_publish_inputs(_RUN_CONTEXT, _ACTIVE_DATA_BINDING)
        move_no_replace(staging, final)
        _TARGET_CONTRACT_PATH = final / CONTRACT_NAME
        _TARGET_CONTRACT_DIGEST = contract_digest
        print(
            f"[alis-dwq] target transaction completed at {final} "
            f"(contract sha256 {contract_digest})",
            file=sys.stderr,
        )
    _distributed_barrier(group)


D.compute_dwq_targets = _wired_compute


def _git_state(path: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("mlx", "mlx-lm", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _code_provenance() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    revision, dirty = _git_state(source_root)
    # Resolve the checkout from the module actually imported by this process;
    # a sibling-path assumption can attest different bytes than those executed.
    mlx_root = Path(D.__file__).resolve().parents[2]
    mlx_revision, mlx_dirty = _git_state(mlx_root)
    runtime_relative = "mlx_lm/models/laguna.py"
    runtime_file = mlx_root / runtime_relative
    runtime_hashes = (
        {runtime_relative: hashlib.sha256(runtime_file.read_bytes()).hexdigest()}
        if runtime_file.is_file()
        else {}
    )
    source_hashes = {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in _SOURCE_FILES
    }
    return {
        "source_root": str(source_root),
        "git_revision": revision,
        "worktree_dirty": dirty,
        "source_files_sha256": source_hashes,
        "pinned_bases": {
            "alis_dwq": ALIS_DWQ_BASE_REVISION,
            "mlx_lm": MLX_LM_BASE_REVISION,
        },
        "observed_bases": {"alis_dwq": revision, "mlx_lm": mlx_revision},
        "package_versions": _package_versions(),
        "runtime": {
            "mlx_lm_checkout_root": str(mlx_root),
            "git_revision": mlx_revision,
            "worktree_dirty": mlx_dirty,
            "source_files_sha256": runtime_hashes,
        },
    }


def _started_payload(*, argv, environ, cwd) -> dict[str, Any]:
    run_id = environ.get("ALIS_DWQ_RUN_ID")
    if not run_id:
        run_id = str(uuid.uuid4())
        environ["ALIS_DWQ_RUN_ID"] = run_id
    return {
        "schema": "alis-dwq.run/v2",
        "event": "run_started",
        "run_id": run_id,
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "cwd": str(Path(cwd).resolve()),
        "argv": list(argv),
        "environment": {
            name: environ[name] for name in _TRACKED_ENV if name in environ
        },
        "code": _code_provenance(),
    }


def _encoded(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _print_run_event(payload, stream=None) -> str:
    line = _encoded(payload)
    print(f"[alis-dwq][run] {line}", file=stream or sys.stderr)
    return line


def emit_run_evidence(*, argv=None, environ=None, cwd=None, stream=None):
    """Emit one standalone no-clobber run-start record (library helper)."""
    environ = os.environ if environ is None else environ
    payload = _started_payload(
        argv=list(sys.argv if argv is None else argv),
        environ=environ,
        cwd=Path.cwd() if cwd is None else cwd,
    )
    line = _print_run_event(payload, stream=stream)
    path = environ.get("ALIS_DWQ_RUN_EVIDENCE_PATH")
    if path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except FileExistsError as exc:
            raise RuntimeError(f"run evidence exists (no-clobber): {output}") from exc
    return payload


class _RunEvidenceRecorder:
    def __init__(self, final_path: str | None, run_id: str, *, stream=None):
        self.final = Path(final_path).expanduser() if final_path else None
        self.run_id = run_id
        self.stream = stream or sys.stderr
        self.handle = None
        self.staging = None
        if self.final is not None:
            self.final.parent.mkdir(parents=True, exist_ok=True)
            if self.final.exists():
                raise FileExistsError(f"run evidence exists (no-clobber): {self.final}")
            self.staging = self.final.with_name(
                f"{self.final.name}.partial-{run_id}-{os.getpid()}"
            )
            self.handle = self.staging.open("x", encoding="utf-8")

    def record(self, payload) -> None:
        line = _print_run_event(payload, stream=self.stream)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def publish(self, payload) -> Path | None:
        self.record(payload)
        if self.handle is None:
            return None
        self.handle.close()
        self.handle = None
        move_no_replace(self.staging, self.final)
        return self.final

    def publish_incomplete(self, payload) -> Path | None:
        self.record(payload)
        if self.handle is None:
            return None
        self.handle.close()
        self.handle = None
        output = self.final.with_name(
            f"{self.final.name}.incomplete-{self.run_id}.jsonl"
        )
        move_no_replace(self.staging, output)
        return output


def _completion_payload(event: str, run_id: str, **fields) -> dict[str, Any]:
    return {
        "schema": "alis-dwq.run/v2",
        "event": event,
        "run_id": run_id,
        "timestamp_unix": time.time(),
        **fields,
    }


def _diagnostic_enabled(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return any(
        int(environ.get(name, "0") or 0) > 0
        for name in ("ALIS_DWQ_MAX_ROUNDS", "ALIS_DWQ_MAX_STEPS_PER_ROUND")
    )


def _validate_live_data_binding(context: Mapping[str, Any], binding: dict) -> None:
    """Ensure persisted data/tokenizer inputs still match their loaded binding."""
    data_dir = Path(context["data_dir"]).expanduser().resolve()
    for relative, expected in binding["data_files_sha256"].items():
        path = data_dir / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"calibration data changed during the run: {relative}")
    if binding["data_manifest_kind"] == "file":
        manifest = data_dir / "manifest.json"
        if (
            manifest.is_symlink()
            or not manifest.is_file()
            or sha256_file(manifest) != binding["data_manifest_sha256"]
        ):
            raise RuntimeError("calibration manifest changed during the run")
    elif binding["data_manifest_kind"] == "derived-jsonl":
        manifest = data_dir / "manifest.json"
        if manifest.exists() or manifest.is_symlink():
            raise RuntimeError("calibration manifest appeared during the run")
    else:
        raise RuntimeError("calibration manifest binding kind changed during the run")
    tokenizer_path = context.get("tokenizer_path", context["model"])
    if tokenizer_files_sha256(tokenizer_path) != binding["tokenizer_files_sha256"]:
        raise RuntimeError("tokenizer artifacts changed during the run")


def _validate_completion_inputs(
    context: Mapping[str, Any],
    binding: dict,
    *,
    target_contract_digest: str,
    pre_dwq_checkpoint_digest: str | None,
) -> dict[str, Any]:
    """Revalidate long-lived lazy-mmap inputs before completion evidence."""
    _validate_live_data_binding(context, binding)
    contract, live_target_digest = validate_target_contract(
        binding,
        context["target_dir"],
        max_seq_length=context["max_seq_length"],
        batch_size=context["batch_size"],
        top_k=TARGET_TOP_K,
        seed=context["seed"],
    )
    if live_target_digest != target_contract_digest:
        raise RuntimeError("target contract or numeric targets changed during the run")

    digest_cache: dict[Path, str] = {}

    def live_digest(path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = directory_digest(resolved)
        return digest_cache[resolved]

    if context.get("verify_teacher_checkpoint"):
        teacher_digest = context.get("teacher_checkpoint_digest")
        if (
            not isinstance(teacher_digest, str)
            or live_digest(context["model"]) != teacher_digest
        ):
            raise RuntimeError("teacher checkpoint changed during the run")
    if pre_dwq_checkpoint_digest is not None:
        pre_dwq = context["quantized_model"] or context["model"]
        if live_digest(pre_dwq) != pre_dwq_checkpoint_digest:
            raise RuntimeError("pre-DWQ student checkpoint changed during the run")
    return contract


def _write_artifact_status_no_replace(
    artifact: Path,
    *,
    run_id: str,
    release_complete: bool,
    completion_kind: str,
    target_contract_digest: str,
    target_contract_canonical_sha256: str,
) -> Path:
    path = Path(artifact) / "alis-dwq-run-status.json"
    payload = {
        "schema": "alis-dwq.artifact-status/v1",
        "run_id": run_id,
        "release_complete": bool(release_complete),
        "completion_kind": completion_kind,
        "target_contract_digest": target_contract_digest,
        "target_contract_canonical_sha256": target_contract_canonical_sha256,
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _reserve_output_staging(final: Path, run_id: str) -> Path:
    """Exclusively reserve a sibling directory owned by this DWQ run."""
    final = Path(final).expanduser()
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"DWQ output exists (no-clobber): {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f"{final.name}.partial-{run_id}-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"DWQ staging output exists (no-clobber): {staging}")
    staging.mkdir(exist_ok=False)
    return staging


def _reserve_memory_evidence_path(final_path: str | Path | None) -> Path | None:
    """Exclusively reserve the memory JSONL before any expensive run work."""
    if not final_path:
        return None
    final = Path(final_path).expanduser()
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        with final.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"memory evidence exists (no-clobber): {final}") from exc
    return final


def _upstream_argv(argv: list[str], *, mlx_path: Path | None) -> list[str]:
    """Strip wrapper-only flags and redirect upstream writes to staging."""
    values = list(argv)
    if not values or str(values[0]).startswith("-"):
        values.insert(0, "alis_dwq.run")
    wrapper_options = {"--runtime-tokenizer-source", "--target-contract"}
    stripped = [values[0]]
    index = 1
    while index < len(values):
        value = str(values[index])
        if value in wrapper_options:
            if index + 1 >= len(values):
                raise ValueError(f"{value} requires a value")
            index += 2
            continue
        if any(value.startswith(f"{option}=") for option in wrapper_options):
            index += 1
            continue
        stripped.append(values[index])
        index += 1
    values = stripped
    if mlx_path is None:
        return values
    replacement = str(mlx_path)
    found = False
    index = 1
    while index < len(values):
        value = str(values[index])
        if value == "--mlx-path":
            if index + 1 >= len(values):
                raise ValueError("--mlx-path requires a value")
            values[index + 1] = replacement
            found = True
            index += 2
            continue
        if value.startswith("--mlx-path="):
            values[index] = f"--mlx-path={replacement}"
            found = True
        index += 1
    if not found:
        values.extend(("--mlx-path", replacement))
    return values


def _frozen_tokenizer_options(value: object) -> dict[str, Any]:
    if value is None:
        options = {}
    elif isinstance(value, Mapping):
        options = dict(value)
    else:
        raise TypeError("tokenizer_config_extra must be a mapping or None")
    for name, required in _FROZEN_TOKENIZER_REQUIRED_OPTIONS.items():
        if name in options and options[name] is not required:
            raise ValueError(
                f"tokenizer_config_extra conflicts with required {name}={required}"
            )
        options[name] = required
    return options


def _frozen_tokenizer_loader(
    original, frozen_root: Path, bundle: RuntimeTokenizerBundle
):
    @functools.wraps(original)
    def load(_requested_path, *args, **kwargs):
        if args and "tokenizer_config_extra" in kwargs:
            raise TypeError(
                "tokenizer_config_extra cannot be supplied both positionally "
                "and by keyword"
            )
        supplied = args[0] if args else kwargs.pop("tokenizer_config_extra", None)
        options = _frozen_tokenizer_options(supplied)
        _revalidate_frozen_runtime_tokenizer(frozen_root, bundle)
        try:
            if args:
                return original(str(frozen_root), options, *args[1:], **kwargs)
            return original(str(frozen_root), tokenizer_config_extra=options, **kwargs)
        finally:
            _revalidate_frozen_runtime_tokenizer(frozen_root, bundle)

    return load


def _checkpoint_declares_laguna(root: Path | str | None) -> bool:
    """Recognize a Laguna checkpoint without importing or loading its model."""
    if root is None:
        return False
    root = Path(root).expanduser()
    plan_path = root / "conversion_plan.json"
    config_path = root / "config.json"
    try:
        if plan_path.is_file() and not plan_path.is_symlink():
            plan = load_json(plan_path)
            if (
                isinstance(plan, dict)
                and plan.get("schema_version") == "laguna.conversion/v2"
                and plan.get("source_repo") == "poolside/Laguna-S-2.1"
            ):
                return True
    except (OSError, UnicodeDecodeError, ValueError):
        pass
    try:
        if config_path.is_file() and not config_path.is_symlink():
            config = load_json(config_path)
            return isinstance(config, dict) and config.get("model_type") == "laguna"
    except (OSError, UnicodeDecodeError, ValueError):
        pass
    return False


def _is_guarded_laguna_context(context: Mapping[str, Any] | None) -> bool:
    if not context:
        return False
    return any(
        _checkpoint_declares_laguna(context.get(key))
        for key in ("quantized_model", "model")
    )


def _guarded_model_loader(original, guard: MemoryGuard, context: Mapping[str, Any]):
    """Put the same pre-load baseline around upstream lazy model construction."""
    student = context.get("quantized_model")
    student_path = os.path.abspath(student) if student is not None else None

    def load(model_path, *args, **kwargs):
        observed = os.path.abspath(model_path)
        role = (
            "student"
            if student_path is not None and observed == student_path
            else "teacher"
        )
        guard.check(
            "before-upstream-model-load",
            model_role=role,
            model_path=observed,
        )
        result = original(model_path, *args, **kwargs)
        # Upstream uses lazy=True.  This checkpoint therefore covers mmap/model
        # wiring; later training and target checkpoints cover materialization.
        guard.check(
            "after-upstream-model-load",
            model_role=role,
            model_path=observed,
        )
        return result

    return load


def _guarded_dwq_quantizer(original, guard: MemoryGuard):
    def quantize(*args, **kwargs):
        # This runs after student load (and after deepcopy/stock quantization
        # when --quantized-model is omitted), but before the first DWQ step.
        guard.check("before-upstream-dwq-training")
        kwargs["_alis_memory_guard"] = guard
        return original(*args, **kwargs)

    return quantize


def _guarded_model_saver(original, guard: MemoryGuard):
    """Bracket upstream's whole-checkpoint save with memory stop gates."""

    @functools.wraps(original)
    def save(dst_path, *args, **kwargs):
        observed = os.path.abspath(dst_path)
        guard.check("before-upstream-model-save", model_path=observed)
        result = original(dst_path, *args, **kwargs)
        guard.check("after-upstream-model-save", model_path=observed)
        return result

    return save


def _guarded_make_shards(original, guard: MemoryGuard):
    """Gate the eager full-checkpoint shard-list materialization."""

    @functools.wraps(original)
    def make_shards(weights, *args, **kwargs):
        guard.check(
            "before-upstream-shard-materialization",
            tensor_count=len(weights),
        )
        shards = original(weights, *args, **kwargs)
        guard.check(
            "after-upstream-shard-materialization",
            shard_count=len(shards),
        )
        return shards

    return make_shards


class _GuardedMlxSaveProxy:
    """Delegate MLX operations while gating each safetensors shard write."""

    def __init__(self, delegate, guard: MemoryGuard):
        self._delegate = delegate
        self._guard = guard

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def save_safetensors(self, path, *args, **kwargs):
        observed = os.path.abspath(path)
        self._guard.check("before-upstream-shard-save", shard_path=observed)
        result = self._delegate.save_safetensors(path, *args, **kwargs)
        self._guard.check("after-upstream-shard-save", shard_path=observed)
        return result


def _upstream_save_runtime(original):
    """Resolve and validate the module globals used by mlx-lm's save function."""

    module_name = getattr(original, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        raise RuntimeError("upstream model saver has no resolvable module")
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "make_shards", None)):
        raise RuntimeError("upstream model saver module has no callable make_shards")
    runtime = getattr(module, "mx", None)
    if runtime is None or not callable(getattr(runtime, "save_safetensors", None)):
        raise RuntimeError(
            "upstream model saver module has no callable mx.save_safetensors"
        )
    return module


def main(argv=None) -> None:
    global _ACTIVE_DATA_BINDING, _RUN_CONTEXT, _RUN_MEMORY_GUARD
    global _TARGET_CONTRACT_DIGEST, _TARGET_CONTRACT_PATH
    raw_argv = list(sys.argv if argv is None else argv)
    _ACTIVE_DATA_BINDING = None
    _RUN_MEMORY_GUARD = None
    _TARGET_CONTRACT_DIGEST = None
    _TARGET_CONTRACT_PATH = None
    _RUN_CONTEXT = _parse_run_context(raw_argv)
    runtime_tokenizer = None
    if _RUN_CONTEXT["runtime_tokenizer_source"] is not None:
        runtime_tokenizer = _load_runtime_tokenizer_bundle(
            _RUN_CONTEXT["runtime_tokenizer_source"],
            _RUN_CONTEXT["target_contract"],
        )
    start = _started_payload(argv=raw_argv, environ=os.environ, cwd=Path.cwd())
    run_id = start["run_id"]
    group = mx.distributed.init()
    recorder = (
        _RunEvidenceRecorder(os.environ.get("ALIS_DWQ_RUN_EVIDENCE_PATH"), run_id)
        if group.rank() == 0
        else _RunEvidenceRecorder(None, run_id)
    )
    recorder.record(start)
    diagnostic = False
    pre_digest = None
    artifact_staging = None
    frozen_tokenizer_dir = None
    previous_load_tokenizer = None
    previous_model_loader = None
    previous_dwq_quantizer = None
    previous_model_saver = None
    upstream_save_runtime = None
    previous_make_shards = None
    previous_save_mx = None
    try:
        if group.rank() == 0:
            _reserve_memory_evidence_path(
                os.environ.get("ALIS_DWQ_MEMORY_EVIDENCE_PATH")
            )
        diagnostic = _diagnostic_enabled()
        if group.size() != 1:
            raise RuntimeError(
                "contracted ALIS-DWQ runs require exactly one process; distributed "
                "teacher provenance is not yet supported"
            )
        if D.load_data is not _load_local:
            raise RuntimeError(
                "contracted local data loader is unavailable; set "
                "ALIS_DWQ_DATA_DIR to an existing directory before launching"
            )
        if runtime_tokenizer is not None:
            frozen_tokenizer_dir = tempfile.TemporaryDirectory(
                prefix="alis-dwq-runtime-tokenizer-"
            )
            frozen_root = Path(frozen_tokenizer_dir.name)
            _materialize_frozen_runtime_tokenizer(frozen_root, runtime_tokenizer)
            _RUN_CONTEXT["tokenizer_path"] = frozen_root
            previous_load_tokenizer = D.load_tokenizer
            D.load_tokenizer = _frozen_tokenizer_loader(
                previous_load_tokenizer, frozen_root, runtime_tokenizer
            )
        strict_laguna = _is_guarded_laguna_context(_RUN_CONTEXT)
        if (
            strict_laguna
            and not _RUN_CONTEXT["targets_only"]
            and _RUN_CONTEXT["quantized_model"] is None
        ):
            raise ValueError(
                "guarded Laguna DWQ runs require --quantized-model from the "
                "official execution manifest; in-process deepcopy and stock "
                "quantization are outside the guarded memory contract"
            )

        target_dir = _RUN_CONTEXT["target_dir"]
        target_state = _target_dir_state(
            target_dir,
            max_seq_length=_RUN_CONTEXT["max_seq_length"],
            batch_size=_RUN_CONTEXT["batch_size"],
            top_k=TARGET_TOP_K,
            seed=_RUN_CONTEXT["seed"],
        )
        will_compute_targets = target_state == "new"
        _RUN_CONTEXT["verify_teacher_checkpoint"] = _requires_teacher_stability(
            _RUN_CONTEXT, target_state
        )
        if will_compute_targets:
            _RUN_CONTEXT["teacher_checkpoint_digest"] = directory_digest(
                _RUN_CONTEXT["model"]
            )
        if not _RUN_CONTEXT["targets_only"]:
            output = _RUN_CONTEXT["mlx_path"]
            if diagnostic and "diagnostic" not in output.name.lower():
                raise ValueError(
                    "diagnostic round/step limits require an --mlx-path whose name "
                    "contains 'diagnostic'"
                )
            artifact_staging = _reserve_output_staging(output, run_id)
            pre_dwq = _RUN_CONTEXT["quantized_model"] or _RUN_CONTEXT["model"]
            pre_digest = directory_digest(pre_dwq)

        memory_phase = "upstream-dwq-bootstrap"
        recommended = configure_recommended_wired_limit(memory_phase)
        _RUN_MEMORY_GUARD = MemoryGuard(
            memory_phase,
            recommended,
            limits=(
                MemoryLimits.guarded_laguna()
                if strict_laguna
                else MemoryLimits.from_env()
            ),
            require_recommended_working_set=strict_laguna,
            require_swap_measurement=strict_laguna,
        )
        _RUN_MEMORY_GUARD.start()
        _RUN_MEMORY_GUARD.check("before-upstream-main")
        previous_model_loader = D.load
        D.load = _guarded_model_loader(
            previous_model_loader, _RUN_MEMORY_GUARD, _RUN_CONTEXT
        )
        previous_dwq_quantizer = D.dwq_quantize
        D.dwq_quantize = _guarded_dwq_quantizer(
            previous_dwq_quantizer, _RUN_MEMORY_GUARD
        )
        if strict_laguna:
            previous_model_saver = D.save
            upstream_save_runtime = _upstream_save_runtime(previous_model_saver)
            previous_make_shards = upstream_save_runtime.make_shards
            previous_save_mx = upstream_save_runtime.mx
            upstream_save_runtime.make_shards = _guarded_make_shards(
                previous_make_shards, _RUN_MEMORY_GUARD
            )
            upstream_save_runtime.mx = _GuardedMlxSaveProxy(
                previous_save_mx, _RUN_MEMORY_GUARD
            )
            D.save = _guarded_model_saver(previous_model_saver, _RUN_MEMORY_GUARD)

        exit_code = None
        previous_argv = sys.argv
        try:
            sys.argv = _upstream_argv(
                raw_argv,
                mlx_path=artifact_staging,
            )
            D.main()
            _RUN_MEMORY_GUARD.check("after-upstream-main")
        except SystemExit as exc:
            exit_code = exc.code
            if exit_code not in (None, 0):
                raise
        finally:
            sys.argv = previous_argv
            if previous_load_tokenizer is not None:
                D.load_tokenizer = previous_load_tokenizer
                previous_load_tokenizer = None
            if previous_model_loader is not None:
                D.load = previous_model_loader
                previous_model_loader = None
            if previous_dwq_quantizer is not None:
                D.dwq_quantize = previous_dwq_quantizer
                previous_dwq_quantizer = None
            if previous_model_saver is not None:
                D.save = previous_model_saver
                previous_model_saver = None
            if upstream_save_runtime is not None:
                if previous_make_shards is not None:
                    upstream_save_runtime.make_shards = previous_make_shards
                    previous_make_shards = None
                if previous_save_mx is not None:
                    upstream_save_runtime.mx = previous_save_mx
                    previous_save_mx = None
                upstream_save_runtime = None

        if runtime_tokenizer is not None:
            _revalidate_runtime_tokenizer_bundle(runtime_tokenizer)

        target_path = _TARGET_CONTRACT_PATH
        target_digest = _TARGET_CONTRACT_DIGEST
        if target_path is None or target_digest is None:
            raise RuntimeError(
                "completed run has no target contract verified by the contracted "
                "local loader or target transaction"
            )
        if runtime_tokenizer is not None and (
            target_path.resolve(strict=True) != runtime_tokenizer.target_contract
            or target_digest != runtime_tokenizer.target_contract_sha256
        ):
            raise RuntimeError(
                "verified target contract does not match runtime tokenizer binding"
            )
        if _ACTIVE_DATA_BINDING is None:
            raise RuntimeError("completed run has no live calibration-data binding")
        completed_contract = _validate_completion_inputs(
            _RUN_CONTEXT,
            _ACTIVE_DATA_BINDING,
            target_contract_digest=target_digest,
            pre_dwq_checkpoint_digest=pre_digest,
        )
        target_contract_canonical_sha256 = canonical_sha256(completed_contract)

        if _RUN_CONTEXT["targets_only"]:
            recorder.publish(
                _completion_payload(
                    "run_completed",
                    run_id,
                    release_complete=False,
                    completion_kind="target_dump",
                    target_contract_path=str(target_path.resolve()),
                    target_contract_digest=target_digest,
                    target_contract_canonical_sha256=(target_contract_canonical_sha256),
                )
            )
            return

        if artifact_staging is None or not artifact_staging.is_dir():
            raise RuntimeError("DWQ did not produce its reserved staging directory")
        if runtime_tokenizer is not None:
            _install_runtime_tokenizer(artifact_staging, runtime_tokenizer)

        if diagnostic:
            _write_artifact_status_no_replace(
                artifact_staging,
                run_id=run_id,
                release_complete=False,
                completion_kind="diagnostic_partial",
                target_contract_digest=target_digest,
                target_contract_canonical_sha256=(target_contract_canonical_sha256),
            )
            if runtime_tokenizer is not None:
                _revalidate_installed_runtime_tokenizer(
                    artifact_staging, runtime_tokenizer
                )
            final_digest = directory_digest(artifact_staging)
            move_no_replace(artifact_staging, _RUN_CONTEXT["mlx_path"])
            recorder.publish_incomplete(
                _completion_payload(
                    "run_incomplete",
                    run_id,
                    release_complete=False,
                    completion_kind="diagnostic_partial",
                    pre_dwq_checkpoint_digest=pre_digest,
                    target_contract_path=str(target_path.resolve()),
                    target_contract_digest=target_digest,
                    target_contract_canonical_sha256=(target_contract_canonical_sha256),
                    final_artifact_digest=final_digest,
                )
            )
            return
        _write_artifact_status_no_replace(
            artifact_staging,
            run_id=run_id,
            release_complete=True,
            completion_kind="dwq_training",
            target_contract_digest=target_digest,
            target_contract_canonical_sha256=target_contract_canonical_sha256,
        )
        if runtime_tokenizer is not None:
            _revalidate_installed_runtime_tokenizer(artifact_staging, runtime_tokenizer)
        final_digest = directory_digest(artifact_staging)
        move_no_replace(artifact_staging, _RUN_CONTEXT["mlx_path"])
        recorder.publish(
            _completion_payload(
                "run_completed",
                run_id,
                release_complete=True,
                completion_kind="dwq_training",
                pre_dwq_checkpoint_digest=pre_digest,
                target_contract_path=str(target_path.resolve()),
                target_contract_digest=target_digest,
                target_contract_canonical_sha256=(target_contract_canonical_sha256),
                final_artifact_digest=final_digest,
            )
        )
    except BaseException as exc:
        recorder.publish_incomplete(
            _completion_payload(
                "run_failed",
                run_id,
                release_complete=False,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        )
        raise
    finally:
        if previous_load_tokenizer is not None:
            D.load_tokenizer = previous_load_tokenizer
        if previous_model_loader is not None:
            D.load = previous_model_loader
        if previous_dwq_quantizer is not None:
            D.dwq_quantize = previous_dwq_quantizer
        if previous_model_saver is not None:
            D.save = previous_model_saver
        if upstream_save_runtime is not None:
            if previous_make_shards is not None:
                upstream_save_runtime.make_shards = previous_make_shards
            if previous_save_mx is not None:
                upstream_save_runtime.mx = previous_save_mx
        _RUN_MEMORY_GUARD = None
        if frozen_tokenizer_dir is not None:
            frozen_tokenizer_dir.cleanup()


if __name__ == "__main__":
    main()

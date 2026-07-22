"""Deterministic calibration/teacher-target binding for ALIS-DWQ.

The sidecar produced here binds the exact JSONL bytes, tokenizer artifacts,
selected token IDs, final batch order, and every numeric target file.  Its
target parser is CPU/NumPy-only; only the CLI loads a tokenizer, so contracts
can be tested or audited without loading a model or touching Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA = "alis-dwq.targets/v1"
CONTRACT_NAME = "target-contract.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PREFORMATTED_CONTRACT = {
    "name": "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat",
    "preformatted_chat": True,
    "add_special_tokens": False,
    "append_eos": False,
}
_TOKEN_ID_HASH_SCHEMA = "sha256-canonical-json-token-ids/v1"
_TOKENIZER_EQUIVALENCE_SCHEMA = "alis-dwq.tokenizer-equivalence/v2"
_TOKENIZER_ROW_EQUIVALENCE_SCHEMA = "alis-dwq.tokenizer-row-equivalence/v1"
_TOKENIZER_ROW_EQUIVALENCE_METHOD = "live-runtime-tokenizer-encode/v1"
_LAGUNA_SOURCE_TOKENIZER_OPTIONS = {"fix_mistral_regex": True}
_LAGUNA_SPLIT_COUNTS = {"train": 80, "valid": 40, "heldout": 100}
_LAGUNA_RUNTIME_TOKENIZER_REQUIRED_FILES = frozenset(
    {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
)
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
_JINJA_FILE_RE = re.compile(
    r"{%[-+]?\s*(?:include|import|from)\s+(['\"])([^'\"]+)\1",
    re.IGNORECASE,
)
_TARGET_PAD_TO = 32
_SAFETENSOR_DTYPES = {
    "F16": (2, "<u2"),
    "BF16": (2, "<u2"),
    "F32": (4, "<u4"),
    "F64": (8, "<u8"),
    "I8": (1, "<i1"),
    "U8": (1, "<u1"),
    "I16": (2, "<i2"),
    "U16": (2, "<u2"),
    "I32": (4, "<i4"),
    "U32": (4, "<u4"),
    "I64": (8, "<i8"),
    "U64": (8, "<u8"),
}
_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_INTEGER_DTYPES = frozenset(_SAFETENSOR_DTYPES) - _FLOAT_DTYPES


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def numeric_target_files(split_dir: Path) -> list[Path]:
    """Return real target shards, ignoring macOS AppleDouble sidecars."""
    return sorted(
        (
            path
            for path in Path(split_dir).glob("*.safetensors")
            if not path.name.startswith("._")
        ),
        key=lambda path: path.name,
    )


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_nonfinite_json,
    )


def tokenizer_files_sha256(tokenizer_path: Path) -> dict[str, str]:
    root = Path(tokenizer_path).expanduser().resolve()
    hashes = {
        name: sha256_file(root / name)
        for name in TOKENIZER_FILES
        if (root / name).is_file()
    }
    if not hashes:
        raise ValueError(f"no tokenizer artifacts found in {root}")
    return hashes


def _safe_tokenizer_dependency(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or value.startswith("._")
        or Path(value).name != value
    ):
        raise ValueError(f"{label} is not a safe top-level tokenizer file: {value!r}")
    if value not in TOKENIZER_FILES:
        raise ValueError(f"{label} is not a supported tokenizer file: {value!r}")
    return value


def _laguna_runtime_tokenizer_files_sha256(tokenizer_path: Path) -> dict[str, str]:
    """Bind exactly Laguna's required tokenizer files and live dependencies."""
    root = Path(tokenizer_path).expanduser().resolve()
    observed = {
        name
        for name in TOKENIZER_FILES
        if (root / name).exists() or (root / name).is_symlink()
    }
    missing_required = _LAGUNA_RUNTIME_TOKENIZER_REQUIRED_FILES - observed
    if missing_required:
        raise ValueError(
            "Laguna runtime tokenizer is missing required files: "
            f"{sorted(missing_required)}"
        )
    for name in observed:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Laguna runtime tokenizer file is missing, not regular, or a symlink: {name}"
            )

    config = load_json(root / "tokenizer_config.json")
    if not isinstance(config, dict):
        raise ValueError("Laguna runtime tokenizer_config.json must be an object")
    dependencies: set[str] = set()
    for field in _TOKENIZER_FILE_FIELDS:
        value = config.get(field)
        if value is None:
            continue
        dependencies.add(
            _safe_tokenizer_dependency(
                value, label=f"Laguna tokenizer config field {field!r}"
            )
        )

    def scan_template(value: object) -> None:
        if isinstance(value, str):
            for match in _JINJA_FILE_RE.finditer(value):
                dependencies.add(
                    _safe_tokenizer_dependency(
                        match.group(2), label="Laguna tokenizer Jinja dependency"
                    )
                )
            if "{%" not in value and value.endswith((".jinja", ".jinja2")):
                dependencies.add(
                    _safe_tokenizer_dependency(
                        value, label="Laguna tokenizer template file"
                    )
                )
        elif isinstance(value, dict):
            for nested in value.values():
                scan_template(nested)
        elif isinstance(value, list):
            for nested in value:
                scan_template(nested)

    scan_template(config.get("chat_template"))
    for name in sorted(observed):
        if not name.endswith((".jinja", ".jinja2")):
            continue
        try:
            scan_template((root / name).read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Laguna runtime tokenizer template is not UTF-8: {name}"
            ) from exc

    required_and_referenced = _LAGUNA_RUNTIME_TOKENIZER_REQUIRED_FILES | dependencies
    if not required_and_referenced <= observed:
        missing = sorted(required_and_referenced - observed)
        raise ValueError(
            "Laguna runtime tokenizer file set must contain required files plus "
            f"declared dependencies: missing={missing}"
        )
    # Every supported adjacent tokenizer file is part of the runtime contract,
    # even when tokenizer_config.json does not reference it explicitly.  This
    # keeps later release verification from accepting an unbound tokenizer
    # sidecar that can change loading behavior.
    return {name: sha256_file(root / name) for name in sorted(observed)}


def _tokenizer_vocab_size(tokenizer, tokenizer_path: Path) -> int:
    config_path = Path(tokenizer_path).expanduser().resolve() / "config.json"
    if config_path.is_file():
        value = load_json(config_path).get("vocab_size")
        if type(value) is int and value > 0:
            return value
    value = getattr(tokenizer, "vocab_size", None)
    if type(value) is int and value > 0:
        return value
    try:
        value = len(tokenizer)
    except (TypeError, AttributeError):
        value = None
    if type(value) is not int or value <= 0:
        raise ValueError("tokenizer/model vocabulary size is unavailable")
    return value


def _tensor_descriptor(
    header: dict[str, Any], name: str, *, payload_bytes: int
) -> tuple[str, list[int], int, int]:
    descriptor = header.get(name)
    if not isinstance(descriptor, dict):
        raise ValueError(f"target safetensor lacks {name!r}")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if dtype not in _SAFETENSOR_DTYPES:
        raise ValueError(f"unsupported target dtype for {name}: {dtype!r}")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
    ):
        raise ValueError(f"invalid target shape for {name}: {shape!r}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(item) is not int for item in offsets)
        or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
    ):
        raise ValueError(f"invalid target data offsets for {name}")
    expected_bytes = math.prod(shape) * _SAFETENSOR_DTYPES[dtype][0]
    if offsets[1] - offsets[0] != expected_bytes:
        raise ValueError(f"target byte length does not match {name} shape/dtype")
    return dtype, shape, offsets[0], offsets[1]


def _float_is_finite_and_nonzero(data: bytes, dtype: str) -> tuple[bool, bool]:
    bits = np.frombuffer(data, dtype=_SAFETENSOR_DTYPES[dtype][1])
    masks = {
        "F16": (0x7C00, 0x7FFF),
        "BF16": (0x7F80, 0x7FFF),
        "F32": (0x7F800000, 0x7FFFFFFF),
        "F64": (0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF),
    }
    exponent, magnitude = masks[dtype]
    finite = bool(np.all((bits & exponent) != exponent))
    nonzero = bool(np.any((bits & magnitude) != 0))
    return finite, nonzero


def validate_target_safetensors(
    path: Path,
    *,
    batch_size: int,
    expected_sequence_length: int,
    max_seq_length: int,
    top_k: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Validate numeric DWQ target semantics without loading MLX or Metal."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"target file is missing or a symlink: {path}")
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 1 or header_length > file_bytes - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        try:
            header = json.loads(
                handle.read(header_length).decode("utf-8"),
                object_pairs_hook=_object_without_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid safetensors JSON header: {path}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header must be an object: {path}")
        metadata = header.get("__metadata__")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            )
        ):
            raise ValueError("target safetensors metadata must map strings to strings")
        tensor_names = set(header) - {"__metadata__"}
        if tensor_names != {"logits", "indices"}:
            raise ValueError(
                "target safetensors keys must be exactly logits and indices"
            )
        payload_start = 8 + header_length
        payload_bytes = file_bytes - payload_start
        logits = _tensor_descriptor(header, "logits", payload_bytes=payload_bytes)
        indices = _tensor_descriptor(header, "indices", payload_bytes=payload_bytes)
        logits_dtype, logits_shape, logits_start, logits_end = logits
        indices_dtype, indices_shape, indices_start, indices_end = indices
        if logits_dtype not in _FLOAT_DTYPES:
            raise ValueError("target logits must use a floating dtype")
        if indices_dtype not in _INTEGER_DTYPES:
            raise ValueError("target indices must use an integer dtype")
        if logits_shape != indices_shape or len(logits_shape) != 3:
            raise ValueError("target logits/indices must have the same rank-3 shape")
        if (
            logits_shape[0] != batch_size
            or logits_shape[1] != expected_sequence_length
            or not 0 < expected_sequence_length < max_seq_length
            or logits_shape[2] != top_k
        ):
            raise ValueError(
                "target shape does not match exact batch/sequence/top-k contract"
            )
        ordered_offsets = sorted(
            ((logits_start, logits_end), (indices_start, indices_end))
        )
        if (
            ordered_offsets[0][0] != 0
            or ordered_offsets[0][1] != ordered_offsets[1][0]
            or ordered_offsets[1][1] != payload_bytes
        ):
            raise ValueError("target safetensors payload has gaps or overlaps")
        handle.seek(payload_start + logits_start)
        logits_bytes = handle.read(logits_end - logits_start)
        finite, logits_nonzero = _float_is_finite_and_nonzero(
            logits_bytes, logits_dtype
        )
        if not finite or not logits_nonzero:
            raise ValueError("target logits must be finite and not entirely zero")
        handle.seek(payload_start + indices_start)
        indices_values = np.frombuffer(
            handle.read(indices_end - indices_start),
            dtype=_SAFETENSOR_DTYPES[indices_dtype][1],
        )
        if (
            not bool(np.any(indices_values != 0))
            or int(indices_values.min()) < 0
            or int(indices_values.max()) >= vocab_size
        ):
            raise ValueError(
                "target indices must be nonzero somewhere and inside vocabulary"
            )
        sorted_indices = np.sort(indices_values.reshape(-1, top_k), axis=-1)
        if bool(np.any(sorted_indices[:, 1:] == sorted_indices[:, :-1])):
            raise ValueError("target top-k indices must be unique for every token")
    return {
        "logits_dtype": logits_dtype,
        "indices_dtype": indices_dtype,
        "shape": logits_shape,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            stripped = raw_line.rstrip(b"\r\n")
            if not stripped:
                raise ValueError(f"{path}:{line_number}: empty JSONL row")
            try:
                row = json.loads(
                    stripped.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number}: missing non-empty text")
            declared_raw = row.get("raw_sha256")
            if declared_raw is not None and (
                not isinstance(declared_raw, str)
                or _SHA256_RE.fullmatch(declared_raw) is None
            ):
                raise ValueError(f"{path}:{line_number}: invalid raw_sha256")
            declared_token_hash = row.get("token_ids_sha256")
            if declared_token_hash is not None and (
                not isinstance(declared_token_hash, str)
                or _SHA256_RE.fullmatch(declared_token_hash) is None
            ):
                raise ValueError(f"{path}:{line_number}: invalid token_ids_sha256")
            declared_token_count = row.get("eval_sequence_tokens")
            if declared_token_count is not None and (
                type(declared_token_count) is not int or declared_token_count < 0
            ):
                raise ValueError(f"{path}:{line_number}: invalid eval_sequence_tokens")
            rows.append(
                {
                    "row": row,
                    "jsonl_line_sha256": hashlib.sha256(stripped).hexdigest(),
                    "raw_sha256": declared_raw or hashlib.sha256(stripped).hexdigest(),
                }
            )
    return rows


def _manifest_binding(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    jsonl_hashes = {
        f"{split}.jsonl": sha256_file(data_dir / f"{split}.jsonl")
        for split in ("train", "valid")
    }
    if manifest_path.is_file():
        return {
            "data_manifest_kind": "file",
            "data_manifest_sha256": sha256_file(manifest_path),
            "data_files_sha256": jsonl_hashes,
            "manifest": load_json(manifest_path),
        }
    derived = {
        "schema": "alis-dwq.derived-data-manifest/v1",
        "data_files_sha256": jsonl_hashes,
    }
    return {
        "data_manifest_kind": "derived-jsonl",
        "data_manifest_sha256": canonical_sha256(derived),
        "data_files_sha256": jsonl_hashes,
        "manifest": derived,
    }


def _validate_declared_token_bindings(
    manifest: dict[str, Any],
    source: dict[str, list[dict[str, Any]]],
    tokenizer,
    *,
    tokenization: str,
) -> dict[str, Any]:
    """Fail early for Laguna manifests whose persisted token claims drift."""
    if manifest.get("chat_template") != "Laguna-S-2.1 local tokenizer":
        raise ValueError("Laguna format-v2 manifest has an unexpected chat_template")
    declared_contract = manifest.get("tokenization_contract")
    if declared_contract != _PREFORMATTED_CONTRACT:
        raise ValueError(
            "preformatted Laguna manifest lacks the exact tokenization contract; "
            "rebuild the calibration data"
        )
    if tokenization != "preformatted_chat":
        raise ValueError(
            "data manifest requires ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat"
        )
    tokenizer_options = manifest.get("tokenizer_options")
    if (
        not isinstance(tokenizer_options, dict)
        or set(tokenizer_options) != {"fix_mistral_regex"}
        or tokenizer_options["fix_mistral_regex"] is not True
    ):
        raise ValueError(
            "Laguna format-v2 manifest tokenizer_options must be exactly "
            "{'fix_mistral_regex': true}"
        )
    declared_tokenizer = manifest.get("tokenizer_files_sha256")
    if (
        not isinstance(declared_tokenizer, dict)
        or not declared_tokenizer
        or any(
            name not in TOKENIZER_FILES
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for name, digest in declared_tokenizer.items()
        )
    ):
        raise ValueError(
            "Laguna format-v2 manifest has invalid source tokenizer hashes"
        )
    summary = manifest.get("token_id_hashes")
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != _TOKEN_ID_HASH_SCHEMA
        or summary.get("field") != "token_ids_sha256"
        or summary.get("tokenization") != _PREFORMATTED_CONTRACT
        or summary.get("all_rows_verified") is not True
        or not isinstance(summary.get("splits"), dict)
        or set(summary["splits"]) != {"train", "valid", "heldout"}
    ):
        raise ValueError(
            "preformatted Laguna manifest lacks complete token_id_hashes evidence; "
            "rebuild the calibration data"
        )
    actual_counts = {split: len(source[split]) for split in _LAGUNA_SPLIT_COUNTS}
    if actual_counts != _LAGUNA_SPLIT_COUNTS:
        raise ValueError(
            "Laguna format-v2 data must contain exactly 80 train, 40 valid, "
            "and 100 heldout rows"
        )
    evidence_splits = {}
    for split in ("train", "valid", "heldout"):
        rows = []
        source_ordered_hashes = []
        runtime_ordered_hashes = []
        for data_index, entry in enumerate(source[split]):
            text = entry["row"]["text"]
            runtime_token_ids = [
                int(token) for token in tokenizer.encode(text, add_special_tokens=False)
            ]
            source_token_hash = entry["row"].get("token_ids_sha256")
            source_token_count = entry["row"].get("eval_sequence_tokens")
            runtime_token_hash = canonical_sha256(runtime_token_ids)
            runtime_token_count = len(runtime_token_ids)
            if source_token_hash != runtime_token_hash:
                raise ValueError(f"{split} row {data_index} token_ids_sha256 mismatch")
            if source_token_count != runtime_token_count:
                raise ValueError(
                    f"{split} row {data_index} eval_sequence_tokens mismatch"
                )
            row = {
                "data_index": data_index,
                "jsonl_line_sha256": entry["jsonl_line_sha256"],
                "raw_sha256": entry["raw_sha256"],
                "source_token_ids_sha256": source_token_hash,
                "runtime_token_ids_sha256": runtime_token_hash,
                "source_token_count": source_token_count,
                "runtime_token_count": runtime_token_count,
            }
            rows.append(row)
            source_ordered_hashes.append(source_token_hash)
            runtime_ordered_hashes.append(runtime_token_hash)
        source_ordered_digest = canonical_sha256(source_ordered_hashes)
        runtime_ordered_digest = canonical_sha256(runtime_ordered_hashes)
        expected_manifest_split = {
            "row_count": len(source[split]),
            "ordered_token_ids_sha256": source_ordered_digest,
        }
        if summary["splits"].get(split) != expected_manifest_split:
            raise ValueError(f"data manifest {split} token_id_hashes mismatch")
        if source_ordered_digest != runtime_ordered_digest:
            raise ValueError(f"{split} source/runtime ordered token IDs mismatch")
        evidence_splits[split] = {
            "row_count": len(rows),
            "rows": rows,
            "source_ordered_token_ids_sha256": source_ordered_digest,
            "runtime_ordered_token_ids_sha256": runtime_ordered_digest,
            "rows_sha256": canonical_sha256(rows),
        }
    return {
        "schema": _TOKENIZER_ROW_EQUIVALENCE_SCHEMA,
        "method": _TOKENIZER_ROW_EQUIVALENCE_METHOD,
        "tokenization": dict(_PREFORMATTED_CONTRACT),
        "row_count": sum(row["row_count"] for row in evidence_splits.values()),
        "splits": evidence_splits,
        "all_rows_verified": True,
    }


def _is_laguna_format_v2_manifest(manifest: dict[str, Any]) -> bool:
    return (
        type(manifest.get("format_version")) is int and manifest["format_version"] == 2
    )


def prepare_local_data(
    tokenizer,
    data_dir: Path,
    *,
    tokenizer_path: Path,
    num_samples: int,
    num_valid_samples: int,
    max_seq_length: int,
    seed: int,
    tokenization: str,
    text_dataset_factory: Callable,
) -> tuple[list, list, dict[str, Any]]:
    """Load local JSONL and return mlx-lm data plus its live binding."""
    data_dir = Path(data_dir).expanduser().resolve()
    if seed <= 0:
        raise ValueError("contracted local data requires --seed > 0")
    if num_samples <= 0 or max_seq_length <= 1:
        raise ValueError(
            "num_samples must be positive and max_seq_length must exceed 1"
        )
    if tokenization not in {"text_dataset", "preformatted_chat"}:
        raise ValueError(
            "ALIS_DWQ_TEXT_TOKENIZATION must be 'text_dataset' or 'preformatted_chat'"
        )

    manifest = _manifest_binding(data_dir)
    source = {
        split: _read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "valid")
    }
    laguna_format_v2 = _is_laguna_format_v2_manifest(manifest["manifest"])
    row_evidence = None
    if laguna_format_v2:
        source["heldout"] = _read_jsonl(data_dir / "heldout.jsonl")
        row_evidence = _validate_declared_token_bindings(
            manifest["manifest"], source, tokenizer, tokenization=tokenization
        )
    datasets = {
        split: text_dataset_factory([entry["row"] for entry in entries], tokenizer)
        for split, entries in source.items()
    }
    train_indices = (
        np.random.RandomState(seed)
        .permutation(len(source["train"]))[:num_samples]
        .tolist()
    )
    requested_valid = (
        len(source["valid"]) if num_valid_samples <= 0 else num_valid_samples
    )
    selected = {
        "train": train_indices,
        "valid": list(range(min(requested_valid, len(source["valid"])))),
    }

    output: dict[str, list] = {"train": [], "valid": []}
    split_bindings: dict[str, Any] = {}
    for split in ("train", "valid"):
        bound_rows = []
        dataset = datasets[split]
        for selected_index, data_index in enumerate(selected[split]):
            if tokenization == "preformatted_chat":
                text = source[split][data_index]["row"]["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                offset = 0
            else:
                tokens, offset = dataset.process(dataset[data_index])
            full_token_ids = [int(token) for token in tokens]
            entry = source[split][data_index]
            declared_hash = entry["row"].get("token_ids_sha256")
            live_full_hash = canonical_sha256(full_token_ids)
            if declared_hash is not None and declared_hash != live_full_hash:
                raise ValueError(f"{split} row {data_index} token_ids_sha256 mismatch")
            declared_count = entry["row"].get("eval_sequence_tokens")
            if declared_count is not None and declared_count != len(full_token_ids):
                raise ValueError(
                    f"{split} row {data_index} eval_sequence_tokens mismatch"
                )
            token_ids = full_token_ids[:max_seq_length]
            output[split].append((token_ids, int(offset)))
            bound_rows.append(
                {
                    "selected_index": selected_index,
                    "data_index": int(data_index),
                    "raw_sha256": entry["raw_sha256"],
                    "jsonl_line_sha256": entry["jsonl_line_sha256"],
                    "token_ids_sha256": canonical_sha256(token_ids),
                    "token_count": len(token_ids),
                    "offset": int(offset),
                }
            )
        split_bindings[split] = {"selected_rows": bound_rows}

    tokenizer_hashes = (
        _laguna_runtime_tokenizer_files_sha256(tokenizer_path)
        if laguna_format_v2
        else tokenizer_files_sha256(tokenizer_path)
    )
    declared_tokenizer = manifest["manifest"].get("tokenizer_files_sha256")
    if (
        declared_tokenizer is not None
        and declared_tokenizer != tokenizer_hashes
        and not laguna_format_v2
    ):
        raise ValueError(
            "data manifest tokenizer_files_sha256 differs from live tokenizer"
        )
    binding = {
        "data_manifest_kind": manifest["data_manifest_kind"],
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "data_files_sha256": manifest["data_files_sha256"],
        "tokenizer_files_sha256": tokenizer_hashes,
        "vocab_size": _tokenizer_vocab_size(tokenizer, tokenizer_path),
        "tokenization": tokenization,
        "splits": split_bindings,
    }
    if laguna_format_v2:
        binding["tokenizer_equivalence"] = {
            "schema": _TOKENIZER_EQUIVALENCE_SCHEMA,
            "mode": (
                "file-identity"
                if declared_tokenizer == tokenizer_hashes
                else "all-declared-row-token-ids"
            ),
            "source_tokenizer_files_sha256": dict(declared_tokenizer),
            "source_tokenizer_options": dict(_LAGUNA_SOURCE_TOKENIZER_OPTIONS),
            "runtime_tokenizer_files_sha256": dict(tokenizer_hashes),
            "row_evidence": row_evidence,
            "all_rows_verified": True,
        }
    return output["train"], output["valid"], binding


def _ordered_selected_indices(
    selected_rows: list[dict[str, Any]], *, batch_size: int, seed: int
) -> list[list[int]]:
    if batch_size <= 0 or len(selected_rows) < batch_size:
        raise ValueError("each target split must contain at least one full batch")
    stable_by_length = sorted(
        range(len(selected_rows)), key=lambda index: selected_rows[index]["token_count"]
    )
    batches = [
        stable_by_length[index : index + batch_size]
        for index in range(0, len(stable_by_length) - batch_size + 1, batch_size)
    ]
    permutation = np.random.RandomState(seed).permutation(len(batches)).tolist()
    return [batches[index] for index in permutation]


def _targets_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["target_file"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["target_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_target_contract(
    binding: dict[str, Any],
    target_dir: Path,
    *,
    run_id: str,
    teacher_identity: str,
    teacher_revision: str,
    teacher_checkpoint_digest: str,
    max_seq_length: int,
    batch_size: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    if not run_id or not teacher_identity or not teacher_revision:
        raise ValueError("run_id and teacher identity/revision are required")
    if _SHA256_RE.fullmatch(teacher_checkpoint_digest) is None:
        raise ValueError("teacher checkpoint digest must be a SHA-256")
    target_dir = Path(target_dir)
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("target directory must be a regular directory")
    split_contracts = {}
    for split in ("train", "valid"):
        selected_rows = binding["splits"][split]["selected_rows"]
        ordered_batches = _ordered_selected_indices(
            selected_rows, batch_size=batch_size, seed=seed
        )
        expected_names = [
            f"{index:010d}.safetensors" for index in range(len(ordered_batches))
        ]
        split_dir = target_dir / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise ValueError(f"target split directory must not be a symlink: {split}")
        actual_names = [path.name for path in numeric_target_files(split_dir)]
        if actual_names != expected_names:
            raise ValueError(
                f"target file set mismatch for {split}: expected {expected_names}, "
                f"found {actual_names}"
            )
        files = []
        rows = []
        for target_index, batch in enumerate(ordered_batches):
            target_file = f"{split}/{target_index:010d}.safetensors"
            target_path = target_dir / target_file
            if target_path.is_symlink():
                raise ValueError(f"target file must not be a symlink: {target_file}")
            max_token_count = max(
                selected_rows[selected_index]["token_count"] for selected_index in batch
            )
            padded_length = 1 + _TARGET_PAD_TO * (
                (max_token_count + _TARGET_PAD_TO - 1) // _TARGET_PAD_TO
            )
            expected_sequence_length = min(padded_length, max_seq_length) - 1
            validate_target_safetensors(
                target_path,
                batch_size=batch_size,
                expected_sequence_length=expected_sequence_length,
                max_seq_length=max_seq_length,
                top_k=top_k,
                vocab_size=binding["vocab_size"],
            )
            target_sha256 = sha256_file(target_path)
            files.append({"target_file": target_file, "target_sha256": target_sha256})
            for batch_position, selected_index in enumerate(batch):
                source = selected_rows[selected_index]
                rows.append(
                    {
                        "target_index": target_index,
                        "batch_position": batch_position,
                        "target_file": target_file,
                        "target_sha256": target_sha256,
                        "data_index": source["data_index"],
                        "raw_sha256": source["raw_sha256"],
                        "jsonl_line_sha256": source["jsonl_line_sha256"],
                        "token_ids_sha256": source["token_ids_sha256"],
                        "token_count": source["token_count"],
                        "offset": source["offset"],
                    }
                )
        split_contracts[split] = {
            "selected_count": len(selected_rows),
            "target_count": len(files),
            "ordered_rows_sha256": canonical_sha256(rows),
            "targets_sha256": _targets_digest(files),
            "rows": rows,
        }
    contract = {
        "schema": SCHEMA,
        "run_id": run_id,
        "data_manifest_kind": binding["data_manifest_kind"],
        "data_manifest_sha256": binding["data_manifest_sha256"],
        "data_files_sha256": binding["data_files_sha256"],
        "tokenizer_files_sha256": binding["tokenizer_files_sha256"],
        "teacher": {
            "identity": teacher_identity,
            "revision": teacher_revision,
            "checkpoint_digest": teacher_checkpoint_digest,
        },
        "max_seq_length": int(max_seq_length),
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "vocab_size": int(binding["vocab_size"]),
        "seed": int(seed),
        "tokenization": binding["tokenization"],
        "splits": split_contracts,
    }
    if "tokenizer_equivalence" in binding:
        contract["tokenizer_equivalence"] = binding["tokenizer_equivalence"]
    return contract


def write_contract_no_replace(target_dir: Path, contract: dict[str, Any]) -> Path:
    path = Path(target_dir) / CONTRACT_NAME
    encoded = json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"target contract exists (no-clobber): {path}") from exc
    return path


def preflight_backfill_target_dir(target_dir: Path) -> Path:
    """Reject empty/partial/no-clobber backfills before loading or hashing inputs."""
    target_dir = Path(target_dir).expanduser()
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("backfill target directory must be a regular directory")
    contract = target_dir / CONTRACT_NAME
    if contract.exists() or contract.is_symlink():
        raise FileExistsError(f"target contract exists (no-clobber): {contract}")
    for split in ("train", "valid"):
        split_dir = target_dir / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise ValueError(f"backfill target split is missing or a symlink: {split}")
        files = numeric_target_files(split_dir)
        if not files or any(path.is_symlink() or not path.is_file() for path in files):
            raise ValueError(
                f"backfill target split is empty or contains a symlink: {split}"
            )
    return target_dir


def validate_target_contract(
    binding: dict[str, Any],
    target_dir: Path,
    *,
    max_seq_length: int,
    batch_size: int,
    top_k: int,
    seed: int,
    teacher_checkpoint_digest: str | None = None,
) -> tuple[dict[str, Any], str]:
    path = Path(target_dir) / CONTRACT_NAME
    if not path.is_file():
        raise ValueError(f"missing target contract: {path}")
    actual = load_json(path)
    if actual.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported target contract schema: {actual.get('schema')!r}"
        )
    teacher = actual.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("target contract has no teacher identity")
    expected = build_target_contract(
        binding,
        target_dir,
        run_id=actual.get("run_id", ""),
        teacher_identity=teacher.get("identity", ""),
        teacher_revision=teacher.get("revision", ""),
        teacher_checkpoint_digest=(
            teacher_checkpoint_digest or teacher.get("checkpoint_digest", "")
        ),
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        top_k=top_k,
        seed=seed,
    )
    if actual != expected:
        raise ValueError(
            "live data/tokenizer/order/target hashes do not match target contract"
        )
    return actual, sha256_file(path)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Backfill an immutable ALIS-DWQ target contract without loading a model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--teacher-identity", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Teacher checkpoint to hash (defaults to --tokenizer).",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-valid-samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--tokenization",
        choices=("text_dataset", "preformatted_chat"),
        default="text_dataset",
    )
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    target_dir = preflight_backfill_target_dir(args.target_dir)
    from .io_utils import directory_digest

    from mlx_lm.tuner.datasets import TextDataset
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(str(args.tokenizer.expanduser()))
    _, _, binding = prepare_local_data(
        tokenizer,
        args.data_dir,
        tokenizer_path=args.tokenizer,
        num_samples=args.num_samples,
        num_valid_samples=args.num_valid_samples,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        tokenization=args.tokenization,
        text_dataset_factory=TextDataset,
    )
    provisional_run_id = args.run_id or "backfill"
    contract = build_target_contract(
        binding,
        target_dir,
        run_id=provisional_run_id,
        teacher_identity=args.teacher_identity,
        teacher_revision=args.teacher_revision,
        # Validate the exact file set and every numeric tensor before hashing a
        # potentially hundreds-of-gigabytes teacher checkpoint.
        teacher_checkpoint_digest="0" * 64,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    contract["teacher"]["checkpoint_digest"] = directory_digest(
        args.teacher_checkpoint or args.tokenizer
    )
    if args.run_id is None:
        contract["run_id"] = "backfill-" + canonical_sha256(contract)[:24]
    output = write_contract_no_replace(target_dir, contract)
    print(
        json.dumps(
            {"path": str(output.resolve()), "sha256": sha256_file(output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

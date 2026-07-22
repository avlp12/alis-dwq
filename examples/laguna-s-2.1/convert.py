#!/usr/bin/env python3
"""Pinned, runtime-aware Laguna conversion policies used by this example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from alis_dwq.io_utils import move_no_replace, sha256_file

SOURCE_REPO = "poolside/Laguna-S-2.1"
SOURCE_REVISION = "88796b991a17fc691abf1c1ad0d9f459dae73834"
MLX_LM_REVISION = "cf10f962b7a20e63a6df43dbf0faf06070153d40"
SOURCE_SHARD_COUNT = 46
SOURCE_KEY_COUNT = 36_769
SOURCE_TOTAL_SIZE = 235_123_955_200
SOURCE_SHARD_MANIFEST_SHA256 = (
    "1a42be970c9778d8229f830ff250c3472f3228c8eaf751c35962b652a42e1048"
)
SOURCE_SMALL_FILES_SHA256 = {
    "LICENSE.md": "8a0c5e232551abcb102d1bd39a34866ef4520361b613bd6405d55b36562e4d88",
    "README.md": "67c585f6d7ea18b792c602939a2e91ceb7a7201c64838318ae0c7d91a03fd312",
    "chat_template.jinja": "cba88d199f03479462b09e0d0b7b75527ab887348cf04703cb7ee2f0cd637f66",
    "config.json": "8309d2ab0da8ac0981b8803b1a4637d843c10fdf7851ddd202ca918fb682392c",
    "configuration_laguna.py": "9446b4fca6f895bd0ed79d861f33447f8c231ba42b7c89cb4b4d25af3958c1fd",
    "generation_config.json": "2deeac08584c9177028e108a994e37dffd06acf61ca429dc064f76fee52e2bea",
    "model.safetensors.index.json": "91f9cb0e426b0720b3f801ccaf0413879300f07a072b83de957b4177bcab8b6d",
    "modeling_laguna.py": "765fd328542d176ff6a62ac814327b11a824df29bdca001d341e9a7c2fe9d876",
    "special_tokens_map.json": "70cd3459fde61761e9440751a590e89a108c09b1803cc7727f5ad1ed1ea6122b",
    "tokenizer.json": "809240f7a182cde859a4fc4ebc902e619a173d507e99304c1092aa04e7a6658e",
    "tokenizer_config.json": "8103b5dd4baf13b38ee927370fbfeab2b1378457efaa233d1c5f0410c40dc9f9",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

RECIPES = {
    "bf16-mlx-layout": {
        "description": "unquantized MLX-layout source for clipping"
    },
    "baseline-q4-g64": {
        "default": {"group_size": 64, "bits": 4, "mode": "affine"},
        "description": "stock affine Q4/group-64 baseline",
    },
    "quality-3p7": {
        "default": {"group_size": 64, "bits": 4, "mode": "affine"},
        "routed_experts": {"group_size": 128, "bits": 3, "mode": "affine"},
        "embeddings_and_head": {"group_size": 64, "bits": 6, "mode": "affine"},
        "description": "routed Q3/g128, core Q4/g64, embedding/head Q6/g64",
    },
    "highest-quality-q4": {
        "default": {"group_size": 64, "bits": 4, "mode": "affine"},
        "embeddings_and_head": "bfloat16",
        "description": "core Q4/g64 with BF16 embedding/head",
    },
}
ARTIFACT_LABELS = {
    "bf16-mlx-layout": "bf16-mlx-layout",
    "baseline-q4-g64": "baseline-mlx-affine",
    "quality-3p7": "dynamic-pre-dwq",
    "highest-quality-q4": "highest-quality-pre-dwq",
}
_ROUTED_MODULE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)$"
)


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicates,
    )


def resolve_source_root(source: Path) -> Path:
    """Resolve one non-symlink source root for verification and conversion."""
    source = Path(source).expanduser()
    if source.is_symlink():
        raise ValueError(f"source root is missing or a symlink: {source}")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"source root is missing or inaccessible: {source}") from exc
    if not resolved.is_dir():
        raise ValueError(f"source root is missing or not a directory: {source}")
    return resolved


def verify_source(
    source: Path,
    *,
    expected_revision: str = SOURCE_REVISION,
    expected_shard_count: int = SOURCE_SHARD_COUNT,
    expected_key_count: int = SOURCE_KEY_COUNT,
    expected_total_size: int = SOURCE_TOTAL_SIZE,
    expected_shard_manifest_sha256: str = SOURCE_SHARD_MANIFEST_SHA256,
    expected_small_files_sha256: dict[str, str] = SOURCE_SMALL_FILES_SHA256,
) -> dict:
    """Verify the exact pinned HF bytes before MLX or a model is imported."""
    source = resolve_source_root(source)
    expected_names = [
        f"model-{index:05d}-of-{expected_shard_count:05d}.safetensors"
        for index in range(1, expected_shard_count + 1)
    ]
    # mlx-lm loads every top-level model*.safetensors file, so verification
    # must reject every extra file matching that broader runtime inventory.
    actual_names = sorted(path.name for path in source.glob("model*.safetensors"))
    if actual_names != expected_names:
        raise ValueError(
            f"source shard set mismatch: expected {expected_shard_count} exact shards"
        )

    small_hashes = {}
    metadata_root = source / ".cache" / "huggingface" / "download"
    for name, expected_hash in expected_small_files_sha256.items():
        path = source / name
        if not path.is_file():
            raise ValueError(f"missing pinned source file: {name}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"pinned source hash mismatch: {name}")
        metadata_path = metadata_root / f"{name}.metadata"
        metadata = metadata_path.read_text(encoding="utf-8").splitlines()
        if not metadata or metadata[0] != expected_revision:
            raise ValueError(f"pinned source revision metadata mismatch: {name}")
        small_hashes[name] = actual_hash

    shard_rows = []
    for name in expected_names:
        path = source / name
        metadata_path = metadata_root / f"{name}.metadata"
        metadata = metadata_path.read_text(encoding="utf-8").splitlines()
        if len(metadata) < 2 or metadata[0] != expected_revision:
            raise ValueError(f"pinned source revision metadata mismatch: {name}")
        expected_hash = metadata[1]
        if _SHA256_RE.fullmatch(expected_hash) is None:
            raise ValueError(f"invalid LFS SHA-256 metadata: {name}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source shard checksum mismatch: {name}")
        shard_rows.append((name, actual_hash, path.stat().st_size))
    manifest_bytes = "".join(
        f"{name}\t{digest}\t{size}\n" for name, digest, size in shard_rows
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != expected_shard_manifest_sha256:
        raise ValueError("source shard manifest does not match the pinned revision")

    index = _load_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != expected_key_count:
        raise ValueError(
            f"source index must contain exactly {expected_key_count} unique keys"
        )
    referenced = set(weight_map.values())
    if referenced != set(expected_names):
        raise ValueError("source index does not reference the exact 46-shard set")
    total_size = index.get("metadata", {}).get("total_size")
    if total_size != expected_total_size:
        raise ValueError(
            f"source index total_size mismatch: {total_size!r} != {expected_total_size}"
        )
    config = _load_json(source / "config.json")
    if config.get("model_type") != "laguna":
        raise ValueError(
            f"expected model_type=laguna, found {config.get('model_type')!r}"
        )
    return {
        "verification_method": "pinned-local-hf-metadata-plus-full-sha256/v1",
        "source_revision": expected_revision,
        "shard_count": len(shard_rows),
        "indexed_key_count": len(weight_map),
        "indexed_total_size": total_size,
        "shard_manifest_sha256": manifest_digest,
        "small_files_sha256": small_hashes,
    }


def preserve_source_notices(source: Path, staging: Path) -> None:
    """Preserve the exact upstream card/license and expose derivative notices."""
    shutil.copy2(source / "LICENSE.md", staging / "LICENSE.md")
    shutil.copy2(source / "README.md", staging / "SOURCE_README.md")
    (staging / "README.md").write_text(
        "---\n"
        f"base_model: {SOURCE_REPO}\n"
        "license: other\n"
        "---\n\n"
        "# Laguna S 2.1 MLX derivative\n\n"
        f"Converted from [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}/tree/"
        f"{SOURCE_REVISION}) at exact revision `{SOURCE_REVISION}`. The complete "
        "upstream model card is preserved byte-for-byte as [SOURCE_README.md]"
        "(SOURCE_README.md), and the OpenMDW-1.1 agreement is preserved as "
        "[LICENSE.md](LICENSE.md).\n\n"
        "Use remains subject to Poolside's [Acceptable Use Policy]"
        "(https://poolside.ai/legal/acceptable-use-policy). Do not circumvent "
        "the source model's safety guardrails without substantially equivalent "
        "mitigations appropriate to the use case.\n",
        encoding="utf-8",
    )


def is_control(path: str) -> bool:
    lowered = f".{path.lower()}"
    return (
        "norm" in lowered
        or "bias" in lowered
        or ".mlp.gate." in lowered
        or lowered.endswith(".mlp.gate")
        or ".router." in lowered
        or lowered.endswith(".router")
    )


def parse_promotions(values: list[str]) -> frozenset[tuple[int, str]]:
    promotions = set()
    for value in values:
        try:
            layer_text, projection = value.split(":", 1)
            layer = int(layer_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid promotion {value!r}; expected LAYER:PROJECTION"
            ) from exc
        if layer not in range(1, 48):
            raise ValueError(f"promotion layer must be in 1..47: {value!r}")
        if projection not in {"*", "gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"invalid routed projection in promotion: {value!r}")
        promotions.add((layer, projection))
    return frozenset(promotions)


def _is_promoted(path: str, promotions: frozenset[tuple[int, str]]) -> bool:
    match = _ROUTED_MODULE_RE.fullmatch(path)
    if match is None:
        return False
    layer, projection = int(match.group(1)), match.group(2)
    return (layer, "*") in promotions or (layer, projection) in promotions


def policy(recipe: str, promotions: frozenset[tuple[int, str]] = frozenset()):
    def predicate(path: str, _module):
        if is_control(path):
            return False
        if recipe == "highest-quality-q4" and path in {
            "model.embed_tokens",
            "lm_head",
        }:
            return False
        if recipe == "quality-3p7":
            if _is_promoted(path, promotions):
                return dict(RECIPES[recipe]["default"])
            if path in {"model.embed_tokens", "lm_head"}:
                return dict(RECIPES[recipe]["embeddings_and_head"])
            if ".mlp.switch_mlp." in path and path.endswith(
                ("gate_proj", "up_proj", "down_proj")
            ):
                return dict(RECIPES[recipe]["routed_experts"])
        return dict(RECIPES[recipe]["default"])

    return predicate


def validate_policy(
    recipe: str, promotions: frozenset[tuple[int, str]] = frozenset()
) -> None:
    if recipe == "bf16-mlx-layout":
        return
    predicate = policy(recipe, promotions)
    controls = (
        "model.layers.1.mlp.gate.gate",
        "model.layers.1.mlp.gate",
        "model.layers.1.input_layernorm",
        "model.layers.1.self_attn.q_norm",
    )
    unsafe = [path for path in controls if predicate(path, None) is not False]
    if unsafe:
        raise RuntimeError(f"conversion policy quantizes control paths: {unsafe}")
    if predicate("model.layers.0.mlp.gate_proj", None) is False:
        raise RuntimeError(
            "conversion policy accidentally excludes the dense gate_proj"
        )


def make_conversion_plan(
    *,
    recipe: str,
    promotions: frozenset[tuple[int, str]],
    source_verification: dict,
    created_at: str,
    mlx_device,
    wired_limit: int,
    peak_memory: int,
) -> dict:
    quantized = recipe != "bf16-mlx-layout"
    return {
        "format_version": 1,
        "schema_version": "laguna.conversion/v2",
        "artifact_label": (
            "dynamic-pre-dwq-promoted"
            if recipe == "quality-3p7" and promotions
            else ARTIFACT_LABELS[recipe]
        ),
        "created_at": created_at,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_verification": source_verification,
        "source_shard_manifest_sha256": source_verification[
            "shard_manifest_sha256"
        ],
        "mlx_lm_base_revision": MLX_LM_REVISION,
        "recipe": recipe,
        "recipe_config": RECIPES[recipe],
        "description": RECIPES[recipe]["description"],
        "promoted_routed_modules": [
            f"{layer}:{projection}" for layer, projection in sorted(promotions)
        ],
        "quantized": quantized,
        "dwq_applied": False,
        "clip_applied": False,
        "ffn_permutation_applied": False,
        "release_complete": False,
        "wired_limit_bytes": wired_limit,
        "peak_memory_bytes": peak_memory,
        "mlx_device": mlx_device,
        "tokenizer_options": {"fix_mistral_regex": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument(
        "--promote-routed-module",
        action="append",
        default=[],
        metavar="LAYER:PROJECTION",
        help=(
            "quality-3p7 only: promote one routed expert projection to Q4/g64; "
            "PROJECTION is gate_proj, up_proj, down_proj, or *"
        ),
    )
    args = parser.parse_args()

    if args.out.exists():
        parser.error(f"output exists (no-clobber): {args.out}")
    try:
        source_root = resolve_source_root(args.source)
        source_verification = verify_source(source_root)
        promotions = parse_promotions(args.promote_routed_module)
        if promotions and args.recipe != "quality-3p7":
            raise ValueError(
                "--promote-routed-module is valid only for quality-3p7"
            )
        validate_policy(args.recipe, promotions)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    import mlx.core as mx
    from mlx_lm.convert import convert

    device = mx.device_info()
    wired_limit = int(device["max_recommended_working_set_size"])
    mx.set_wired_limit(wired_limit)
    mx.reset_peak_memory()
    created_at = datetime.now(timezone.utc)
    staging = args.out.with_name(
        f"{args.out.name}.partial-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    quantized = args.recipe != "bf16-mlx-layout"
    try:
        convert(
            hf_path=str(source_root),
            mlx_path=str(staging),
            quantize=quantized,
            q_group_size=64,
            q_bits=4,
            q_mode="affine",
            quant_predicate=(
                policy(args.recipe, promotions) if quantized else None
            ),
        )
        source_reverification = verify_source(source_root)
        if source_reverification != source_verification:
            raise ValueError("pinned source changed during conversion")
        tokenizer_config = staging / "tokenizer_config.json"
        tokenizer = json.loads(tokenizer_config.read_text())
        tokenizer["fix_mistral_regex"] = True
        tokenizer_config.write_text(
            json.dumps(tokenizer, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        preserve_source_notices(source_root, staging)
        receipt = make_conversion_plan(
            recipe=args.recipe,
            promotions=promotions,
            source_verification=source_verification,
            created_at=created_at.isoformat(),
            mlx_device=device.get("device_name"),
            wired_limit=wired_limit,
            peak_memory=int(mx.get_peak_memory()),
        )
        (staging / "conversion_plan.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        move_no_replace(staging, args.out)
    except BaseException:
        print(f"conversion failed; partial retained at {staging}")
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

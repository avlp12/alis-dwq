"""Read-only structural preflight for expensive alis-dwq runs.

Importing this module is deliberately stdlib-only. MLX and mlx_lm are loaded
only by the command-line entry point. The analysis helpers accept ordinary
Python objects, so tests need neither weights nor Metal.

Stdout is always JSON for a valid invocation. Human-readable failures go to
stderr. Exit 0 means pass, 1 means contract failure, and 2 means load/inspection
failure.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# These are the exact matching rules used by alis_dwq.layerwise.
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
ROUTER_RE = re.compile(r"(?:^|\.)(?:gate|router)$")
SWITCH_NAMES = frozenset({"SwitchGLU", "SwitchMLP"})
AFFINE_BITS = frozenset({2, 3, 4, 5, 6, 8})
ROTATING_CACHE_NAMES = frozenset({"RotatingKVCache", "BatchRotatingKVCache"})
FULL_CACHE_NAMES = frozenset(
    {"KVCache", "BatchKVCache", "QuantizedKVCache", "BatchQuantizedKVCache"}
)
_MISSING = object()


@dataclass(frozen=True)
class Expectations:
    layers: int | None = None
    moe_layers: int | None = None
    experts: int | None = None
    full_caches: int | None = None
    rotating_caches: int | None = None
    allow_dense: bool = False
    require_float_routers: bool = False
    require_quantized_layer_coverage: bool = False


def _get(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if obj is None or obj is _MISSING:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _present(value: Any) -> bool:
    return value is not _MISSING and value is not None


def _shape(value: Any) -> list[int] | None:
    if not _present(value):
        return None
    shape = _get(value, "shape")
    if not _present(shape):
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _dtype(value: Any) -> str | None:
    dtype = _get(value, "dtype")
    return str(dtype) if _present(dtype) else None


def _is_float(dtype: str | None) -> bool:
    return dtype is not None and "float" in dtype.lower()


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _json_value(value: Any) -> Any:
    if value is _MISSING:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        item = value.item()
    except Exception:
        return str(value)
    return item if isinstance(item, (str, int, float, bool)) else str(item)


def layer_index(name: str) -> int | None:
    """Return the layer index matched by layerwise.py."""

    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def is_layerwise_quantized(module: Any) -> bool:
    """Mirror layerwise._is_quantized without importing layerwise or MLX."""

    bits = _get(module, "bits")
    group_size = _get(module, "group_size")
    mode = _get(module, "mode", "affine")
    if bits is _MISSING or group_size is _MISSING:
        return False
    try:
        return mode == "affine" and bits < 8
    except (TypeError, ValueError):
        return False


def is_layerwise_router(name: str, module: Any) -> bool:
    """Mirror layerwise._is_router without importing layerwise or MLX."""

    return (
        ROUTER_RE.search(name) is not None
        and _present(_get(module, "weight"))
        and not is_layerwise_quantized(module)
    )


def collect_modules(model: Any) -> list[tuple[str, Any]]:
    """Read module names through apply_to_modules without changing state."""

    apply = _get(model, "apply_to_modules")
    if apply is _MISSING or not callable(apply):
        raise TypeError("model has no callable apply_to_modules")
    found: list[tuple[str, Any]] = []
    apply(lambda name, module: found.append((str(name), module)))
    return found


def analyze_quantization(
    modules: Iterable[tuple[str, Any]], layer_count: int | None
) -> dict[str, Any]:
    """Inspect the affine packing and trainable scale/bias contract."""

    details: list[dict[str, Any]] = []
    surprises: list[dict[str, str]] = []
    matched_layers: set[int] = set()
    extras: list[str] = []
    ignored_8bit: list[str] = []
    bits_count: Counter[str] = Counter()
    mode_count: Counter[str] = Counter()
    signature_count: Counter[str] = Counter()

    def surprise(code: str, name: str, message: str) -> None:
        surprises.append({"code": code, "module": name, "detail": message})

    for name, module in modules:
        bits = _get(module, "bits")
        group = _get(module, "group_size")
        if bits is _MISSING and group is _MISSING:
            continue
        mode = _get(module, "mode", "affine")
        idx = layer_index(name)
        weight = _get(module, "weight")
        scales = _get(module, "scales")
        biases = _get(module, "biases")
        wshape, sshape, bshape = _shape(weight), _shape(scales), _shape(biases)
        matches = is_layerwise_quantized(module)
        bits_json = _json_value(bits)
        group_json = _json_value(group)
        mode_json = _json_value(mode)
        bits_count[str(bits_json)] += 1
        mode_count[str(mode_json)] += 1
        signature_count[f"{mode_json}/b{bits_json}/g{group_json}"] += 1
        details.append(
            {
                "name": name,
                "class": type(module).__name__,
                "layer": idx,
                "bits": bits_json,
                "group_size": group_json,
                "mode": mode_json,
                "weight_dtype": _dtype(weight),
                "weight_shape": wshape,
                "scales_shape": sshape,
                "biases_shape": bshape,
                "layerwise_match": matches,
            }
        )

        if bits is _MISSING or group is _MISSING:
            surprise(
                "missing_quant_attributes",
                name,
                "a quantized module must expose both bits and group_size",
            )
            continue
        if not _positive_int(bits):
            surprise("invalid_bits", name, f"positive integer required, got {bits!r}")
        if not _positive_int(group):
            surprise(
                "invalid_group_size", name, f"positive integer required, got {group!r}"
            )
        if _positive_int(bits) and _positive_int(group):
            if bits not in AFFINE_BITS:
                surprise(
                    "unsupported_bits",
                    name,
                    f"MLX affine bits must be one of {sorted(AFFINE_BITS)}",
                )
            if (bits * group) % 32:
                surprise(
                    "invalid_packing",
                    name,
                    f"bits*group_size must be divisible by 32: {bits}*{group}",
                )
        if mode != "affine":
            surprise(
                "non_affine_mode",
                name,
                f"layerwise DWQ silently ignores mode {mode!r}",
            )

        if matches:
            if idx is None:
                extras.append(name)
            else:
                matched_layers.add(idx)
            if not _present(weight):
                surprise("missing_weight", name, "packed weight is absent")
            if not _present(scales):
                surprise("missing_scales", name, "DWQ cannot unfreeze scales")
            if not _present(biases):
                surprise("missing_biases", name, "DWQ cannot unfreeze affine biases")
            if wshape is not None and (len(wshape) < 2 or any(d <= 0 for d in wshape)):
                surprise(
                    "invalid_weight_shape",
                    name,
                    f"packed weight must have rank >=2 and positive axes, got {wshape}",
                )
            if sshape is not None and (len(sshape) < 2 or any(d <= 0 for d in sshape)):
                surprise(
                    "invalid_scales_shape",
                    name,
                    f"scales must have rank >=2 and positive axes, got {sshape}",
                )
            if bshape is not None and (len(bshape) < 2 or any(d <= 0 for d in bshape)):
                surprise(
                    "invalid_biases_shape",
                    name,
                    f"biases must have rank >=2 and positive axes, got {bshape}",
                )
            if (
                wshape
                and sshape
                and (len(wshape) != len(sshape) or wshape[:-1] != sshape[:-1])
            ):
                surprise(
                    "packed_shape_prefix_mismatch",
                    name,
                    f"weight prefix {wshape[:-1]} != scales prefix {sshape[:-1]}",
                )
            if sshape and bshape and sshape != bshape:
                surprise(
                    "scale_bias_shape_mismatch",
                    name,
                    f"scales {sshape} != biases {bshape}",
                )
            if (
                wshape
                and sshape
                and _positive_int(bits)
                and _positive_int(group)
                and wshape[-1] * 32 != sshape[-1] * group * bits
            ):
                surprise(
                    "packed_shape_mismatch",
                    name,
                    "packed weight axis does not equal scales*group_size*bits",
                )
            weight_dtype = _dtype(weight)
            scales_dtype = _dtype(scales)
            biases_dtype = _dtype(biases)
            if weight_dtype is not None and "uint32" not in weight_dtype.lower():
                surprise(
                    "invalid_packed_weight_dtype",
                    name,
                    f"packed affine weight must be uint32, got {weight_dtype}",
                )
            if scales_dtype is not None and not _is_float(scales_dtype):
                surprise(
                    "invalid_scales_dtype",
                    name,
                    f"affine scales must be floating, got {scales_dtype}",
                )
            if biases_dtype is not None and not _is_float(biases_dtype):
                surprise(
                    "invalid_biases_dtype",
                    name,
                    f"affine biases must be floating, got {biases_dtype}",
                )
        elif mode == "affine" and _positive_int(bits) and bits >= 8:
            ignored_8bit.append(name)

    missing = (
        []
        if layer_count is None
        else [idx for idx in range(layer_count) if idx not in matched_layers]
    )
    return {
        "advertised_count": len(details),
        "layerwise_match_count": sum(d["layerwise_match"] for d in details),
        "layer_indices": sorted(matched_layers),
        "missing_layer_indices": missing,
        "extra_layerwise_modules": sorted(extras),
        "ignored_at_or_above_8bit": sorted(ignored_8bit),
        "bits": dict(sorted(bits_count.items())),
        "modes": dict(sorted(mode_count.items())),
        "signatures": dict(sorted(signature_count.items())),
        "modules": details,
        "contract_surprises": surprises,
    }


def analyze_switch_hooks(
    modules: Iterable[tuple[str, Any]], switch_types: Sequence[type] = ()
) -> dict[str, Any]:
    """Find the SwitchGLU/SwitchMLP hooks consumed by expert_traffic."""

    real_types = tuple(switch_types)
    found: list[dict[str, Any]] = []
    surprises: list[dict[str, str]] = []
    layers: set[int] = set()
    widths: set[int] = set()
    classes: Counter[str] = Counter()

    def has_switch_call_contract(module: Any) -> bool:
        if not callable(module):
            return False
        try:
            signature = inspect.signature(module)
        except (TypeError, ValueError):
            # Some extension-backed callables do not expose a signature, but
            # callable protocol support is still stronger than class identity.
            return True
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        variadic = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        return variadic or len(positional) >= 2

    for name, module in modules:
        class_name = type(module).__name__
        has_projection = any(
            _present(_get(module, attr)) for attr in ("gate_proj", "fc1")
        )
        call_contract = has_switch_call_contract(module)
        protocol_candidate = "switch" in class_name.lower() and has_projection
        is_switch = (
            (bool(real_types) and isinstance(module, real_types))
            or class_name in SWITCH_NAMES
            or protocol_candidate
        )
        if not is_switch:
            continue
        classes[class_name] += 1
        idx = layer_index(name)
        projection_name = "fc1" if class_name == "SwitchMLP" else "gate_proj"
        projection = _get(module, projection_name)
        if not _present(projection):
            alt = "gate_proj" if projection_name == "fc1" else "fc1"
            projection = _get(module, alt)
            if _present(projection):
                projection_name = alt
        width_shape = _shape(_get(projection, "weight"))
        width = width_shape[0] if width_shape else None
        compatible = (
            idx is not None and width is not None and width > 0 and call_contract
        )
        found.append(
            {
                "name": name,
                "class": class_name,
                "layer": idx,
                "projection": projection_name,
                "expert_width": width,
                "call_contract": call_contract,
                "compatible": compatible,
            }
        )
        if idx is None:
            surprises.append(
                {
                    "code": "unindexed_switch_hook",
                    "module": name,
                    "detail": "path does not match layers.<n>.",
                }
            )
        if width is None or width <= 0:
            surprises.append(
                {
                    "code": "missing_expert_width",
                    "module": name,
                    "detail": f"{projection_name}.weight.shape[0] is unavailable",
                }
            )
        if not call_contract:
            surprises.append(
                {
                    "code": "incompatible_switch_call",
                    "module": name,
                    "detail": "module is not callable with x and expert indices",
                }
            )
        if compatible:
            layers.add(idx)
            widths.add(width)
    return {
        "class_names": sorted(SWITCH_NAMES),
        "found_count": len(found),
        "compatible_count": sum(item["compatible"] for item in found),
        "class_counts": dict(sorted(classes.items())),
        "layer_indices": sorted(layers),
        "expert_widths": sorted(widths),
        "modules": found,
        "contract_surprises": surprises,
    }


def analyze_routers(modules: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Find anchored, indexed, floating router weights."""

    pattern_count = 0
    with_weight = 0
    found: list[dict[str, Any]] = []
    low_bit: list[str] = []
    unindexed: list[str] = []
    layerwise_layers: set[int] = set()
    float_layers: set[int] = set()
    for name, module in modules:
        if ROUTER_RE.search(name) is None:
            continue
        pattern_count += 1
        weight = _get(module, "weight")
        if not _present(weight):
            continue
        with_weight += 1
        idx = layer_index(name)
        dtype = _dtype(weight)
        matched = is_layerwise_router(name, module)
        floating = _is_float(dtype)
        if is_layerwise_quantized(module):
            low_bit.append(name)
        if matched and idx is None:
            unindexed.append(name)
        if matched and idx is not None:
            layerwise_layers.add(idx)
            if floating:
                float_layers.add(idx)
        found.append(
            {
                "name": name,
                "class": type(module).__name__,
                "layer": idx,
                "weight_dtype": dtype,
                "float_weight": floating,
                "layerwise_match": matched,
            }
        )
    return {
        "regex": ROUTER_RE.pattern,
        "pattern_match_count": pattern_count,
        "with_weight_count": with_weight,
        "layerwise_candidate_count": sum(d["layerwise_match"] for d in found),
        "float_compatible_count": sum(
            d["layerwise_match"] and d["float_weight"] for d in found
        ),
        "layerwise_candidate_layer_indices": sorted(layerwise_layers),
        "float_layer_indices": sorted(float_layers),
        "low_bit_pattern_matches": sorted(low_bit),
        "unindexed_candidates": sorted(unindexed),
        "modules": found,
    }


def analyze_caches(
    model: Any,
    fallback_cache_factory: Callable[[Any], Iterable[Any]] | None = None,
    rotating_cache_types: Sequence[type] = (),
    full_cache_types: Sequence[type] = (),
) -> dict[str, Any]:
    """Construct fresh empty caches and report their runtime types."""

    source = None
    error = None
    caches: list[Any] = []
    try:
        make_cache = _get(model, "make_cache")
        if make_cache is not _MISSING and callable(make_cache):
            source = "model.make_cache"
            result = make_cache()
        elif fallback_cache_factory is not None:
            source = "make_prompt_cache_fallback"
            result = fallback_cache_factory(model)
        else:
            raise TypeError("no make_cache method or fallback factory")
        if result is None:
            raise TypeError("cache factory returned None")
        caches = list(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    rotating_types = tuple(rotating_cache_types)
    full_types = tuple(full_cache_types)
    type_counts: Counter[str] = Counter()
    rotating = full = 0
    other_types: set[str] = set()
    entries = []
    for idx, cache in enumerate(caches):
        name = type(cache).__name__
        type_counts[name] += 1
        is_rotating = (
            isinstance(cache, rotating_types)
            if rotating_types
            else name in ROTATING_CACHE_NAMES
        )
        is_full = (
            isinstance(cache, full_types) if full_types else name in FULL_CACHE_NAMES
        )
        if is_rotating:
            rotating += 1
        elif is_full:
            full += 1
        else:
            other_types.add(name)
        entries.append(
            {
                "index": idx,
                "type": name,
                "rotating": is_rotating,
                "full": is_full,
                "max_size": _json_value(_get(cache, "max_size")),
                "keep": _json_value(_get(cache, "keep")),
            }
        )
    return {
        "source": source,
        "error": error,
        "count": len(caches),
        "type_counts": dict(sorted(type_counts.items())),
        "rotating_count": rotating,
        "full_count": full,
        "other_count": len(caches) - rotating - full,
        "other_types": sorted(other_types),
        "entries": entries,
    }


def analyze_model(
    model: Any,
    expectations: Expectations | None = None,
    *,
    fallback_cache_factory: Callable[[Any], Iterable[Any]] | None = None,
    switch_types: Sequence[type] = (),
    rotating_cache_types: Sequence[type] = (),
    full_cache_types: Sequence[type] = (),
) -> dict[str, Any]:
    """Return a complete JSON-safe report without evaluating model weights."""

    expected = expectations or Expectations()
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def check(
        name: str,
        ok: bool,
        good: str,
        bad: str,
        *,
        required: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "severity": "error" if required else "warning",
                "message": good if ok else bad,
            }
        )

    layers_obj = _get(model, "layers")
    visible = _present(layers_obj)
    count = None
    layer_error = None
    if visible:
        try:
            count = len(layers_obj)
        except Exception as exc:
            layer_error = f"{type(exc).__name__}: {exc}"
    check(
        "model_layers",
        visible and count is not None and count > 0,
        f"model.layers exposes {count} layers",
        layer_error or "model.layers must be visible, sized, and non-empty",
    )
    if expected.layers is not None:
        check(
            "expected_layer_count",
            count == expected.layers,
            f"layer count matches {expected.layers}",
            f"expected {expected.layers} layers, found {count}",
        )

    modules: list[tuple[str, Any]] = []
    traversal_error = None
    try:
        modules = collect_modules(model)
    except Exception as exc:
        traversal_error = f"{type(exc).__name__}: {exc}"
    check(
        "module_traversal",
        traversal_error is None,
        f"found {len(modules)} module paths",
        f"module traversal failed: {traversal_error}",
    )

    observed = sorted(
        {idx for name, _ in modules if (idx := layer_index(name)) is not None}
    )
    expected_indices = list(range(count)) if count is not None else []
    missing_indices = [idx for idx in expected_indices if idx not in observed]
    out_of_range = [idx for idx in observed if idx not in expected_indices]
    regex_ok = count is not None and not missing_indices and not out_of_range
    check(
        "layer_regex_coverage",
        regex_ok,
        f"layer regex covers all {count} layers",
        f"layer regex misses {missing_indices}; out_of_range={out_of_range}",
    )

    quant = analyze_quantization(modules, count)
    check(
        "affine_quantized_modules",
        quant["layerwise_match_count"] > 0,
        f"found {quant['layerwise_match_count']} affine <8-bit modules",
        "no affine <8-bit modules match layerwise.py; DWQ would train nothing",
    )
    check(
        "quantized_layer_coverage",
        not quant["missing_layer_indices"],
        "affine <8-bit modules cover every model layer",
        f"affine <8-bit modules miss layers {quant['missing_layer_indices']}",
        required=expected.require_quantized_layer_coverage,
    )
    check(
        "quantization_contract",
        not quant["contract_surprises"],
        "quantized module packing/scales/biases satisfy the DWQ contract",
        f"{len(quant['contract_surprises'])} quantization surprise(s) found",
    )
    if quant["extra_layerwise_modules"]:
        warnings.append(
            f"{len(quant['extra_layerwise_modules'])} affine module(s) outside "
            "layers.<n> will train only in round 1"
        )
    if quant["ignored_at_or_above_8bit"]:
        warnings.append(
            f"{len(quant['ignored_at_or_above_8bit'])} affine module(s) at >=8 "
            "bits are ignored by layerwise DWQ"
        )

    hooks = analyze_switch_hooks(modules, switch_types)
    require_hooks = not expected.allow_dense and (
        expected.moe_layers is not None or expected.experts is not None
    )
    has_hooks = hooks["compatible_count"] > 0
    check(
        "switch_hook_discovery",
        has_hooks or not require_hooks,
        (
            f"found {hooks['compatible_count']} compatible Switch hooks"
            if has_hooks
            else "dense model explicitly allowed"
        ),
        "no compatible SwitchGLU/SwitchMLP hooks; expert_traffic would fail",
    )
    check(
        "switch_hook_contract",
        not hooks["contract_surprises"],
        "all Switch hooks expose indexed expert banks",
        f"{len(hooks['contract_surprises'])} malformed Switch hook(s)",
    )
    if expected.moe_layers is not None:
        check(
            "expected_moe_layer_count",
            hooks["compatible_count"] == expected.moe_layers,
            f"MoE hook count matches {expected.moe_layers}",
            f"expected {expected.moe_layers} MoE hooks, found "
            f"{hooks['compatible_count']}",
        )
    widths = hooks["expert_widths"]
    widths_ok = len(widths) <= 1 and (
        expected.experts is None or widths == [expected.experts]
    )
    if expected.allow_dense and not widths and expected.experts is None:
        widths_ok = True
    check(
        "switch_expert_widths",
        widths_ok,
        f"expert widths are consistent: {widths}",
        f"expert widths {widths} do not match expected {expected.experts}",
    )

    routers = analyze_routers(modules)
    moe_layers = set(hooks["layer_indices"])
    router_key = (
        "float_layer_indices"
        if expected.require_float_routers
        else "layerwise_candidate_layer_indices"
    )
    missing_routers = sorted(moe_layers - set(routers[router_key]))
    router_ok = (expected.allow_dense and not moe_layers) or not missing_routers
    router_kind = "float" if expected.require_float_routers else "layerwise-matched"
    check(
        "router_coverage",
        router_ok,
        f"{router_kind} routers cover all MoE layers",
        f"{router_kind} routers missing for MoE layers {missing_routers}; "
        "ALIS_DWQ_TRAIN_ROUTERS=1 is unsafe",
        required=expected.require_float_routers,
    )
    check(
        "router_layerwise_contract",
        not routers["low_bit_pattern_matches"] and not routers["unindexed_candidates"],
        "router paths satisfy the anchored layerwise.py rule",
        "router conflict: low_bit="
        f"{routers['low_bit_pattern_matches']}, "
        f"unindexed={routers['unindexed_candidates']}",
    )

    cache = analyze_caches(
        model,
        fallback_cache_factory,
        rotating_cache_types,
        full_cache_types,
    )
    check(
        "make_cache",
        cache["error"] is None,
        f"cache construction succeeded via {cache['source']}",
        f"cache construction failed: {cache['error']}",
    )
    check(
        "cache_count",
        count is not None and cache["count"] == count,
        f"cache count matches {count} layers",
        f"cache count {cache['count']} does not match layer count {count}",
    )
    if expected.full_caches is not None:
        check(
            "expected_full_cache_count",
            cache["full_count"] == expected.full_caches,
            f"full cache count matches {expected.full_caches}",
            f"expected {expected.full_caches} full caches, found {cache['full_count']}",
        )
    if expected.rotating_caches is not None:
        check(
            "expected_rotating_cache_count",
            cache["rotating_count"] == expected.rotating_caches,
            f"rotating cache count matches {expected.rotating_caches}",
            f"expected {expected.rotating_caches} RotatingKVCache fallbacks, "
            f"found {cache['rotating_count']}",
        )
    if cache["other_count"]:
        warnings.append(
            f"{cache['other_count']} nonstandard cache(s): {cache['other_types']}"
        )

    errors = [
        item["message"]
        for item in checks
        if not item["ok"] and item["severity"] == "error"
    ]
    warnings.extend(
        item["message"]
        for item in checks
        if not item["ok"] and item["severity"] == "warning"
    )
    return {
        "schema_version": 1,
        "ok": not errors,
        "inspection": {
            "read_only": True,
            "forward_pass": False,
            "parameters_evaluated": False,
        },
        "expectations": asdict(expected),
        "model": {
            "type": type(model).__name__,
            "layers_visible": visible,
            "layer_count": count,
            "module_count": len(modules),
        },
        "layer_index": {
            "regex": LAYER_RE.pattern,
            "observed_indices": observed,
            "expected_indices": expected_indices,
            "missing_indices": missing_indices,
            "out_of_range_indices": out_of_range,
        },
        "quantization": quant,
        "moe_hooks": hooks,
        "routers": routers,
        "cache": cache,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a read-only alis-dwq structural preflight as JSON"
    )
    parser.add_argument("--model", required=True, help="local MLX model directory")
    parser.add_argument("--revision", help="Hub revision, with --allow-download")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow a non-local model ID to download; disabled by default",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="execute a pinned local model_file adapter during load",
    )
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument("--expect-moe-layers", type=int)
    parser.add_argument("--expect-experts", type=int)
    parser.add_argument("--expect-full-caches", type=int)
    parser.add_argument("--expect-rotating-caches", type=int)
    parser.add_argument("--fallback-max-kv-size", type=int)
    parser.add_argument("--allow-dense", action="store_true")
    parser.add_argument(
        "--require-float-routers",
        action="store_true",
        help="fail unless every discovered MoE layer has a floating router",
    )
    parser.add_argument(
        "--require-quantized-layer-coverage",
        action="store_true",
        help="fail unless affine <8-bit modules occur in every model layer",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def _load_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    local_path = Path(args.model).expanduser()
    is_local = local_path.exists()
    if not is_local and not args.allow_download:
        raise FileNotFoundError(
            f"{args.model!r} is not local; pass --allow-download explicitly"
        )
    model_source = str(local_path) if is_local else args.model

    # Production-only imports. Lazy loading defers parameter evaluation, but the
    # model graph and mapped checkpoint can still consume substantial memory.
    from mlx_lm import load
    from mlx_lm.models import cache as cache_module
    from mlx_lm.models import switch_layers

    make_prompt_cache = cache_module.make_prompt_cache

    def available_types(module: Any, names: Iterable[str]) -> tuple[type, ...]:
        return tuple(
            value for name in names if isinstance((value := _get(module, name)), type)
        )

    # Keep the CLI contract deterministic: stdout is the report and stderr is
    # one concise status line. Load progress is not part of either interface.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        model, _ = load(
            model_source,
            lazy=True,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
    return model, {
        "fallback_cache_factory": lambda value: make_prompt_cache(
            value, max_kv_size=args.fallback_max_kv_size
        ),
        "switch_types": available_types(switch_layers, SWITCH_NAMES),
        "rotating_cache_types": available_types(cache_module, ROTATING_CACHE_NAMES),
        "full_cache_types": available_types(cache_module, FULL_CACHE_NAMES),
    }


def _fatal(message: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "checks": [
            {
                "name": "load_and_inspect",
                "ok": False,
                "severity": "error",
                "message": message,
            }
        ],
        "errors": [message],
        "warnings": [],
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    loader: Callable[[argparse.Namespace], Any] | None = None,
) -> int:
    """CLI entry point. Tests inject loader and therefore never import MLX."""

    args = _parser().parse_args(argv)
    expected = Expectations(
        layers=args.expect_layers,
        moe_layers=args.expect_moe_layers,
        experts=args.expect_experts,
        full_caches=args.expect_full_caches,
        rotating_caches=args.expect_rotating_caches,
        allow_dense=args.allow_dense,
        require_float_routers=args.require_float_routers,
        require_quantized_layer_coverage=args.require_quantized_layer_coverage,
    )
    try:
        loaded = (loader or _load_runtime)(args)
        if (
            isinstance(loaded, tuple)
            and len(loaded) == 2
            and isinstance(loaded[1], dict)
        ):
            model, runtime = loaded
        else:
            model, runtime = loaded, {}
        report = analyze_model(model, expected, **runtime)
        report["invocation"] = {
            "model": args.model,
            "revision": args.revision,
            "lazy_load": True,
            "download_allowed": args.allow_download,
            "trust_remote_code": args.trust_remote_code,
        }
        status = 0 if report["ok"] else 1
    except Exception as exc:
        message = f"model load/inspection failed: {type(exc).__name__}: {exc}"
        report = _fatal(message)
        status = 2

    print(json.dumps(report, indent=args.indent, sort_keys=True))
    if report["ok"]:
        print(
            f"[preflight][PASS] {len(report['warnings'])} warning(s)",
            file=sys.stderr,
        )
    else:
        print(
            f"[preflight][FAIL] {len(report['errors'])} error(s): "
            f"{report['errors'][0]}",
            file=sys.stderr,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())

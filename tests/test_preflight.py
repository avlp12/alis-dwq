import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from alis_dwq import preflight


class Array:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class Quantized:
    def __init__(
        self,
        *,
        bits=4,
        group_size=64,
        mode="affine",
        experts=None,
        with_biases=True,
    ):
        prefix = (experts, 4) if experts is not None else (4,)
        groups = 2
        packed = groups * group_size * bits // 32
        self.bits = bits
        self.group_size = group_size
        self.mode = mode
        self.weight = Array(prefix + (packed,), "uint32")
        self.scales = Array(prefix + (groups,), "bfloat16")
        if with_biases:
            self.biases = Array(prefix + (groups,), "bfloat16")


class Linear:
    def __init__(self, width=256, dtype="bfloat16"):
        self.weight = Array((width, 16), dtype)


class SwitchGLU:
    def __init__(self, experts=256):
        self.gate_proj = Linear(experts, "uint32")

    def __call__(self, x, indices):
        return x, indices


class LagunaSwitchExperts:
    def __init__(self, experts=256):
        self.gate_proj = Linear(experts, "uint32")

    def __call__(self, x, indices):
        return x, indices


class NonCallableSwitchExperts:
    def __init__(self, experts=256):
        self.gate_proj = Linear(experts, "uint32")


class KVCache:
    pass


class RotatingKVCache:
    def __init__(self, max_size=512, keep=0):
        self.max_size = max_size
        self.keep = keep


class FakeModel:
    def __init__(self, layers, modules, caches):
        self.layers = [object() for _ in range(layers)]
        self._modules = modules
        self._caches = caches

    def apply_to_modules(self, visitor):
        for name, module in self._modules:
            visitor(name, module)

    def make_cache(self):
        return list(self._caches)


class NoMakeCacheModel:
    def __init__(self, layers, modules):
        self.layers = [object() for _ in range(layers)]
        self._modules = modules

    def apply_to_modules(self, visitor):
        for name, module in self._modules:
            visitor(name, module)


def good_model(router_dtype="bfloat16"):
    modules = []
    for layer in range(3):
        modules.append((f"model.layers.{layer}.self_attn.q_proj", Quantized(bits=4)))
    for layer in (1, 2):
        switch = SwitchGLU(256)
        modules.extend(
            [
                (f"model.layers.{layer}.mlp.switch_mlp", switch),
                (
                    f"model.layers.{layer}.mlp.switch_mlp.gate_proj",
                    Quantized(bits=3, experts=256),
                ),
                (f"model.layers.{layer}.mlp.gate.gate", Linear(256, router_dtype)),
            ]
        )
    return FakeModel(
        3,
        modules,
        [KVCache(), RotatingKVCache(), RotatingKVCache()],
    )


class PreflightTests(unittest.TestCase):
    def test_happy_path_reports_every_runtime_contract(self):
        report = preflight.analyze_model(
            good_model(),
            preflight.Expectations(
                layers=3,
                moe_layers=2,
                experts=256,
                full_caches=1,
                rotating_caches=2,
            ),
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["layer_index"]["observed_indices"], [0, 1, 2])
        self.assertEqual(report["quantization"]["layer_indices"], [0, 1, 2])
        self.assertEqual(report["quantization"]["bits"], {"3": 2, "4": 3})
        self.assertEqual(report["quantization"]["modes"], {"affine": 5})
        self.assertEqual(report["moe_hooks"]["layer_indices"], [1, 2])
        self.assertEqual(report["moe_hooks"]["expert_widths"], [256])
        self.assertEqual(report["routers"]["float_layer_indices"], [1, 2])
        self.assertEqual(
            report["cache"]["type_counts"],
            {"KVCache": 1, "RotatingKVCache": 2},
        )
        json.dumps(report)

    def test_quantization_surprises_and_missing_coverage_fail_clearly(self):
        broken = Quantized(bits=4, mode="mxfp4", with_biases=False)
        model = FakeModel(
            2,
            [
                ("model.layers.0.proj", Quantized(bits=4)),
                ("model.layers.1.proj", broken),
            ],
            [KVCache(), KVCache()],
        )
        report = preflight.analyze_model(
            model,
            preflight.Expectations(
                layers=2,
                allow_dense=True,
                require_quantized_layer_coverage=True,
            ),
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["quantization"]["missing_layer_indices"], [1])
        codes = {item["code"] for item in report["quantization"]["contract_surprises"]}
        self.assertIn("non_affine_mode", codes)
        self.assertTrue(
            any(
                c["name"] == "quantized_layer_coverage" and not c["ok"]
                for c in report["checks"]
            )
        )

    def test_nonfloat_router_blocks_router_training_contract(self):
        report = preflight.analyze_model(
            good_model(router_dtype="uint32"),
            preflight.Expectations(
                layers=3,
                moe_layers=2,
                experts=256,
                require_float_routers=True,
            ),
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["routers"]["layerwise_candidate_layer_indices"], [1, 2])
        self.assertEqual(report["routers"]["float_layer_indices"], [])
        failure = next(c for c in report["checks"] if c["name"] == "router_coverage")
        self.assertFalse(failure["ok"])
        self.assertIn("ALIS_DWQ_TRAIN_ROUTERS=1 is unsafe", failure["message"])

    def test_optional_whole_layer_and_float_router_checks_only_warn(self):
        model = good_model(router_dtype="uint32")
        model._modules = [
            item for item in model._modules if not isinstance(item[1], Quantized)
        ] + [("model.layers.0.proj", Quantized(bits=4))]

        report = preflight.analyze_model(
            model,
            preflight.Expectations(layers=3, moe_layers=2, experts=256),
        )

        self.assertTrue(report["ok"])
        coverage = next(
            c for c in report["checks"] if c["name"] == "quantized_layer_coverage"
        )
        self.assertFalse(coverage["ok"])
        self.assertEqual(coverage["severity"], "warning")
        self.assertEqual(report["routers"]["float_layer_indices"], [])

    def test_custom_switch_protocol_fallback_survives_runtime_type_mismatch(self):
        modules = [("model.layers.0.mlp.switch_experts", LagunaSwitchExperts(128))]

        hooks = preflight.analyze_switch_hooks(modules, switch_types=(SwitchGLU,))

        self.assertEqual(hooks["compatible_count"], 1)
        self.assertEqual(hooks["expert_widths"], [128])
        self.assertEqual(hooks["class_counts"], {"LagunaSwitchExperts": 1})

    def test_switch_class_identity_without_call_contract_is_not_compatible(self):
        modules = [("model.layers.0.mlp.switch_experts", NonCallableSwitchExperts(128))]

        hooks = preflight.analyze_switch_hooks(modules, switch_types=())

        self.assertEqual(hooks["compatible_count"], 0)
        self.assertFalse(hooks["modules"][0]["call_contract"])
        self.assertIn(
            "incompatible_switch_call",
            {row["code"] for row in hooks["contract_surprises"]},
        )

    def test_affine_packing_checks_prefix_rank_dtype_and_supported_bits(self):
        module = Quantized(bits=7, experts=8)
        module.weight = Array((8, 4, 14), "int32")
        module.scales = Array((7, 4, 2), "uint16")
        module.biases = Array((7, 4, 2), "uint16")

        report = preflight.analyze_quantization(
            [("model.layers.0.mlp.experts", module)], layer_count=1
        )
        codes = {item["code"] for item in report["contract_surprises"]}

        self.assertIn("unsupported_bits", codes)
        self.assertIn("packed_shape_prefix_mismatch", codes)
        self.assertIn("invalid_packed_weight_dtype", codes)
        self.assertIn("invalid_scales_dtype", codes)
        self.assertIn("invalid_biases_dtype", codes)

    def test_make_prompt_cache_fallback_and_rotating_counts_are_inspected(self):
        modules = [
            (f"model.layers.{layer}.proj", Quantized(bits=4)) for layer in range(3)
        ]
        model = NoMakeCacheModel(3, modules)
        report = preflight.analyze_model(
            model,
            preflight.Expectations(
                layers=3,
                full_caches=0,
                rotating_caches=3,
                allow_dense=True,
            ),
            fallback_cache_factory=lambda _: [
                RotatingKVCache(),
                RotatingKVCache(),
                RotatingKVCache(),
            ],
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["cache"]["source"], "make_prompt_cache_fallback")
        self.assertEqual(report["cache"]["rotating_count"], 3)

    def test_cli_uses_injected_loader_and_stdout_is_json(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = preflight.main(
                [
                    "--model",
                    "/does/not/need/to/exist/with/injected/loader",
                    "--expect-layers",
                    "3",
                    "--expect-moe-layers",
                    "2",
                    "--expect-experts",
                    "256",
                    "--expect-full-caches",
                    "1",
                    "--expect-rotating-caches",
                    "2",
                    "--require-float-routers",
                    "--require-quantized-layer-coverage",
                    "--indent",
                    "0",
                ],
                loader=lambda _args: good_model(),
            )
        report = json.loads(stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertTrue(report["ok"])
        self.assertIn("[preflight][PASS]", stderr.getvalue())

    def test_loader_failure_is_json_and_exit_two(self):
        def fail(_args):
            raise RuntimeError("synthetic load failure")

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = preflight.main(["--model", "missing"], loader=fail)
        report = json.loads(stdout.getvalue())

        self.assertEqual(status, 2)
        self.assertFalse(report["ok"])
        self.assertIn("synthetic load failure", report["errors"][0])
        self.assertIn("[preflight][FAIL]", stderr.getvalue())

    def test_cli_contract_failure_is_exit_one_and_one_stderr_line(self):
        model = FakeModel(
            2,
            [("model.layers.0.proj", Quantized(bits=4))],
            [KVCache(), KVCache()],
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = preflight.main(
                [
                    "--model",
                    "injected",
                    "--allow-dense",
                    "--require-quantized-layer-coverage",
                ],
                loader=lambda _args: model,
            )

        report = json.loads(stdout.getvalue())
        self.assertEqual(status, 1)
        self.assertFalse(report["ok"])
        self.assertEqual(len(stderr.getvalue().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from alis_dwq.memory_guard import MemoryLimitExceeded


class PermissiveMemoryGuard:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def check(self, _checkpoint, **_context):
        pass


def load_converter():
    path = Path(__file__).parents[1] / "examples" / "laguna-s-2.1" / "convert.py"
    spec = importlib.util.spec_from_file_location("laguna_example_convert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLagunaExampleConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter()

    def test_every_quantized_recipe_preserves_controls_and_dense_gate(self):
        controls = (
            "model.layers.1.mlp.gate.gate",
            "model.layers.1.mlp.gate",
            "model.layers.1.input_layernorm",
            "model.layers.1.self_attn.q_norm",
            "model.layers.1.mlp.gate.e_score_correction_bias",
        )
        for recipe in ("baseline-q4-g64", "quality-3p7", "highest-quality-q4"):
            with self.subTest(recipe=recipe):
                self.converter.validate_policy(recipe)
                predicate = self.converter.policy(recipe)
                self.assertTrue(
                    all(predicate(path, None) is False for path in controls)
                )
                self.assertEqual(
                    predicate("model.layers.0.mlp.gate_proj", None),
                    {"group_size": 64, "bits": 4, "mode": "affine"},
                )

    def test_compact_and_highest_quality_mappings(self):
        compact = self.converter.policy("quality-3p7")
        self.assertEqual(
            compact("model.layers.1.mlp.switch_mlp.down_proj", None),
            {"group_size": 128, "bits": 3, "mode": "affine"},
        )
        self.assertEqual(
            compact("model.embed_tokens", None),
            {"group_size": 64, "bits": 6, "mode": "affine"},
        )
        highest = self.converter.policy("highest-quality-q4")
        self.assertFalse(highest("model.embed_tokens", None))
        self.assertFalse(highest("lm_head", None))
        self.assertEqual(
            highest("model.layers.47.mlp.switch_mlp.up_proj", None),
            {"group_size": 64, "bits": 4, "mode": "affine"},
        )

    def _source_fixture(self, root: Path):
        source = root / "source"
        source.mkdir()
        metadata = source / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        revision = "test-revision"
        shard_rows = []
        for index in range(1, 47):
            name = f"model-{index:05d}-of-00046.safetensors"
            payload = f"shard-{index}".encode()
            (source / name).write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (metadata / f"{name}.metadata").write_text(
                f"{revision}\n{digest}\n0\n"
            )
            shard_rows.append((name, digest, len(payload)))
        total_size = sum(size for _name, _digest, size in shard_rows)
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {
                f"model.weight.{number}": name
                for number, (name, _digest, _size) in enumerate(shard_rows)
            },
        }
        small_contents = {
            "LICENSE.md": "license\n",
            "README.md": "source card with acceptable-use-policy\n",
            "config.json": json.dumps({"model_type": "laguna"}) + "\n",
            "model.safetensors.index.json": json.dumps(index, sort_keys=True) + "\n",
        }
        small_hashes = {}
        for name, contents in small_contents.items():
            (source / name).write_text(contents)
            small_hashes[name] = hashlib.sha256(contents.encode()).hexdigest()
            (metadata / f"{name}.metadata").write_text(f"{revision}\nobject-id\n0\n")
        root_digest = hashlib.sha256(
            "".join(
                f"{name}\t{digest}\t{size}\n"
                for name, digest, size in shard_rows
            ).encode()
        ).hexdigest()
        return source, revision, root_digest, total_size, small_hashes

    def test_pinned_source_verifier_checks_all_shards_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            source, revision, root_digest, total_size, small_hashes = (
                self._source_fixture(Path(directory))
            )
            evidence = self.converter.verify_source(
                source,
                expected_revision=revision,
                expected_key_count=46,
                expected_total_size=total_size,
                expected_shard_manifest_sha256=root_digest,
                expected_small_files_sha256=small_hashes,
            )
            self.assertEqual(evidence["shard_count"], 46)
            self.assertEqual(evidence["indexed_key_count"], 46)
            self.assertEqual(evidence["shard_manifest_sha256"], root_digest)

            (source / "model-00012-of-00046.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.converter.verify_source(
                    source,
                    expected_revision=revision,
                    expected_key_count=46,
                    expected_total_size=total_size,
                    expected_shard_manifest_sha256=root_digest,
                    expected_small_files_sha256=small_hashes,
                )

    def test_source_verifier_rejects_every_unexpected_model_weight(self):
        for name in ("model.safetensors", "model-extra.safetensors"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source, revision, root_digest, total_size, small_hashes = (
                    self._source_fixture(Path(directory))
                )
                (source / name).write_bytes(b"unverified weight")
                with self.assertRaisesRegex(ValueError, "source shard set mismatch"):
                    self.converter.verify_source(
                        source,
                        expected_revision=revision,
                        expected_key_count=46,
                        expected_total_size=total_size,
                        expected_shard_manifest_sha256=root_digest,
                        expected_small_files_sha256=small_hashes,
                    )

    def test_source_verifier_rejects_root_symlink_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, revision, root_digest, total_size, small_hashes = (
                self._source_fixture(root)
            )
            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "source root.*symlink"):
                self.converter.verify_source(
                    source_link,
                    expected_revision=revision,
                    expected_key_count=46,
                    expected_total_size=total_size,
                    expected_shard_manifest_sha256=root_digest,
                    expected_small_files_sha256=small_hashes,
                )

    def _fake_mlx_modules(self, convert):
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.device_info = lambda: {
            "max_recommended_working_set_size": 1_000,
            "device_name": "fake-device",
        }
        mlx_core.set_wired_limit = lambda _value: None
        mlx_core.reset_peak_memory = lambda: None
        mlx_core.get_peak_memory = lambda: 100
        mlx_package = types.ModuleType("mlx")
        mlx_package.__path__ = []
        mlx_package.core = mlx_core
        mlx_lm_package = types.ModuleType("mlx_lm")
        mlx_lm_package.__path__ = []
        convert_module = types.ModuleType("mlx_lm.convert")
        convert_module.convert = convert
        convert_module.load = lambda *args, **kwargs: (args, kwargs)
        convert_module.save = lambda *args, **kwargs: (args, kwargs)
        return {
            "mlx": mlx_package,
            "mlx.core": mlx_core,
            "mlx_lm": mlx_lm_package,
            "mlx_lm.convert": convert_module,
        }

    def test_cli_uses_one_resolved_root_and_reverifies_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_argument = source / ".." / "source"
            resolved = source.resolve()
            output = root / "output"
            evidence = {
                "shard_manifest_sha256": "a" * 64,
                "small_files_sha256": {},
            }
            sequence = []

            def verify(path):
                sequence.append(("verify", Path(path)))
                return dict(evidence)

            def convert(**kwargs):
                sequence.append(("convert", Path(kwargs["hf_path"])))
                staging = Path(kwargs["mlx_path"])
                staging.mkdir()
                (staging / "tokenizer_config.json").write_text("{}\n")

            def preserve(path, _staging):
                sequence.append(("preserve", Path(path)))

            real_publish = self.converter.move_no_replace

            def publish(staging, destination):
                sequence.append(("publish", Path(destination)))
                return real_publish(staging, destination)

            argv = [
                "convert.py",
                "--source",
                str(source_argument),
                "--out",
                str(output),
                "--recipe",
                "bf16-mlx-layout",
            ]
            with (
                mock.patch.dict(sys.modules, self._fake_mlx_modules(convert)),
                mock.patch.object(self.converter, "verify_source", side_effect=verify),
                mock.patch.object(
                    self.converter, "preserve_source_notices", side_effect=preserve
                ),
                mock.patch.object(
                    self.converter, "move_no_replace", side_effect=publish
                ),
                mock.patch(
                    "alis_dwq.memory_guard.configure_recommended_wired_limit",
                    return_value=1_000,
                ),
                mock.patch(
                    "alis_dwq.memory_guard.MemoryGuard", PermissiveMemoryGuard
                ),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(self.converter.main(), 0)

            self.assertEqual(
                sequence,
                [
                    ("verify", resolved),
                    ("convert", resolved),
                    ("verify", resolved),
                    ("preserve", resolved),
                    ("publish", output),
                ],
            )
            self.assertTrue(output.is_dir())

    def test_cli_reverification_failure_retains_partial_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            before = {"shard_manifest_sha256": "a" * 64}
            after = {"shard_manifest_sha256": "b" * 64}

            def convert(**kwargs):
                staging = Path(kwargs["mlx_path"])
                staging.mkdir()
                (staging / "tokenizer_config.json").write_text("{}\n")

            argv = [
                "convert.py",
                "--source",
                str(source),
                "--out",
                str(output),
                "--recipe",
                "bf16-mlx-layout",
            ]
            with (
                mock.patch.dict(sys.modules, self._fake_mlx_modules(convert)),
                mock.patch.object(
                    self.converter, "verify_source", side_effect=[before, after]
                ) as verify,
                mock.patch.object(self.converter, "preserve_source_notices") as preserve,
                mock.patch.object(self.converter, "move_no_replace") as publish,
                mock.patch(
                    "alis_dwq.memory_guard.configure_recommended_wired_limit",
                    return_value=1_000,
                ),
                mock.patch(
                    "alis_dwq.memory_guard.MemoryGuard", PermissiveMemoryGuard
                ),
                mock.patch.object(sys, "argv", argv),
            ):
                with self.assertRaisesRegex(ValueError, "changed during conversion"):
                    self.converter.main()

            self.assertEqual(verify.call_count, 2)
            preserve.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob("output.partial-*"))), 1)

    def test_guarded_convert_checks_load_and_write_and_restores_hooks(self):
        sequence = []
        module = types.SimpleNamespace()

        def original_load(*_args, **_kwargs):
            sequence.append("load")
            return "model"

        def original_save(*_args, **_kwargs):
            sequence.append("save")

        def convert(**_kwargs):
            module.load("source", lazy=True)
            module.save("output")

        module.load = original_load
        module.save = original_save
        module.convert = convert

        class Guard:
            def check(self, checkpoint, **_context):
                sequence.append(checkpoint)

        self.converter._guarded_convert(module, Guard())
        self.assertEqual(
            sequence,
            [
                "before-conversion",
                "before-model-load",
                "load",
                "after-model-load",
                "before-model-write",
                "save",
                "after-model-write",
                "after-conversion",
            ],
        )
        self.assertIs(module.load, original_load)
        self.assertIs(module.save, original_save)

    def test_cli_memory_stop_after_conversion_retains_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            evidence = {"shard_manifest_sha256": "a" * 64}

            def convert(**kwargs):
                staging = Path(kwargs["mlx_path"])
                staging.mkdir()
                (staging / "model.safetensors").write_bytes(b"partial")
                (staging / "tokenizer_config.json").write_text("{}\n")

            class StopGuard(PermissiveMemoryGuard):
                def check(self, checkpoint, **_context):
                    if checkpoint == "after-conversion":
                        raise MemoryLimitExceeded(
                            {"event": "memory_stop_gate", "checkpoint": checkpoint}
                        )

            argv = [
                "convert.py",
                "--source",
                str(source),
                "--out",
                str(output),
                "--recipe",
                "bf16-mlx-layout",
            ]
            with (
                mock.patch.dict(sys.modules, self._fake_mlx_modules(convert)),
                mock.patch.object(
                    self.converter, "verify_source", return_value=evidence
                ) as verify,
                mock.patch.object(self.converter, "preserve_source_notices") as preserve,
                mock.patch.object(self.converter, "move_no_replace") as publish,
                mock.patch(
                    "alis_dwq.memory_guard.configure_recommended_wired_limit",
                    return_value=1_000,
                ),
                mock.patch("alis_dwq.memory_guard.MemoryGuard", StopGuard),
                mock.patch.object(sys, "argv", argv),
            ):
                with self.assertRaises(MemoryLimitExceeded):
                    self.converter.main()

            self.assertEqual(verify.call_count, 1)
            preserve.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(output.exists())
            partials = list(root.glob("output.partial-*"))
            self.assertEqual(len(partials), 1)
            self.assertTrue((partials[0] / "model.safetensors").is_file())

    def test_source_revision_and_exact_index_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            source, revision, root_digest, total_size, small_hashes = (
                self._source_fixture(Path(directory))
            )
            metadata = (
                source
                / ".cache"
                / "huggingface"
                / "download"
                / "model-00001-of-00046.safetensors.metadata"
            )
            lines = metadata.read_text().splitlines()
            metadata.write_text("wrong-revision\n" + "\n".join(lines[1:]) + "\n")
            with self.assertRaisesRegex(ValueError, "revision metadata mismatch"):
                self.converter.verify_source(
                    source,
                    expected_revision=revision,
                    expected_key_count=46,
                    expected_total_size=total_size,
                    expected_shard_manifest_sha256=root_digest,
                    expected_small_files_sha256=small_hashes,
                )

    def test_source_notices_are_preserved_and_derivative_card_names_aup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, *_ = self._source_fixture(root)
            staging = root / "staging"
            staging.mkdir()
            self.converter.preserve_source_notices(source, staging)
            self.assertEqual(
                (staging / "LICENSE.md").read_bytes(),
                (source / "LICENSE.md").read_bytes(),
            )
            self.assertEqual(
                (staging / "SOURCE_README.md").read_bytes(),
                (source / "README.md").read_bytes(),
            )
            derivative = (staging / "README.md").read_text()
            self.assertIn(self.converter.SOURCE_REVISION, derivative)
            self.assertIn("acceptable-use-policy", derivative)

    def test_release_recipe_labels_are_explicitly_pre_dwq(self):
        self.assertEqual(
            self.converter.ARTIFACT_LABELS["quality-3p7"],
            "dynamic-pre-dwq",
        )
        self.assertEqual(
            self.converter.ARTIFACT_LABELS["highest-quality-q4"],
            "highest-quality-pre-dwq",
        )
        self.assertEqual(
            self.converter.MLX_LM_REVISION,
            "cf10f962b7a20e63a6df43dbf0faf06070153d40",
        )

    def test_quality_promotions_are_bounded_to_routed_modules(self):
        promotions = self.converter.parse_promotions(
            ["47:down_proj", "12:gate_proj"]
        )
        predicate = self.converter.policy("quality-3p7", promotions)
        self.assertEqual(
            predicate("model.layers.47.mlp.switch_mlp.down_proj", None),
            {"group_size": 64, "bits": 4, "mode": "affine"},
        )
        self.assertEqual(
            predicate("model.layers.47.mlp.switch_mlp.up_proj", None),
            {"group_size": 128, "bits": 3, "mode": "affine"},
        )
        self.assertFalse(predicate("model.layers.47.mlp.gate.gate", None))
        with self.assertRaisesRegex(ValueError, "layer must be in 1..47"):
            self.converter.parse_promotions(["0:down_proj"])

        plan = self.converter.make_conversion_plan(
            recipe="quality-3p7",
            promotions=promotions,
            source_verification={"shard_manifest_sha256": "f" * 64},
            created_at="2026-07-22T00:00:00+00:00",
            mlx_device="test",
            wired_limit=100,
            peak_memory=50,
        )
        self.assertEqual(plan["artifact_label"], "dynamic-pre-dwq-promoted")
        self.assertEqual(plan["mlx_lm_base_revision"], self.converter.MLX_LM_REVISION)
        self.assertEqual(
            plan["promoted_routed_modules"], ["12:gate_proj", "47:down_proj"]
        )
        self.assertEqual(plan["recipe_config"], self.converter.RECIPES["quality-3p7"])

    def test_documented_release_cli_matches_receipt_contract(self):
        readme = (
            Path(__file__).parents[1]
            / "examples"
            / "laguna-s-2.1"
            / "README.md"
        ).read_text()
        self.assertIn("--recipe quality-3p7", readme)
        self.assertIn("--recipe highest-quality-q4", readme)
        self.assertIn("--model /build/bf16-mlx-layout", readme)
        self.assertIn("ALIS_DWQ_DATA_DIR=/build/laguna-data-v4", readme)
        self.assertIn("--target-dir /build/teacher-targets-v3", readme)
        self.assertIn("--quantized-model /build/compact-clip-s11", readme)
        self.assertIn("--grad-checkpoint", readme)
        for setting in (
            "ALIS_DWQ_MAX_ROUNDS=0",
            "ALIS_DWQ_MAX_STEPS_PER_ROUND=0",
            "ALIS_DWQ_TRAIN_ROUTERS=0",
            "ALIS_DWQ_LORA_RANK=0",
            "ALIS_DWQ_ADAPTER_DIR=",
            "ALIS_DWQ_CKA_MONITOR=0",
            "ALIS_DWQ_LOSS=kl",
        ):
            self.assertIn(setting, readme)
        self.assertIn("canonical input-directory digests", readme)
        for arm_path in (
            "/build/highest-quality-clip-s11",
            "/build/highest-quality-dwq-work-1",
            "/build/compact-promotion-1-clip-s11",
            "/build/compact-promotion-1-dwq-work-1",
            "/build/compact-promotion-2-clip-s11",
            "/build/compact-promotion-2-dwq-work-1",
        ):
            self.assertIn(arm_path, readme)
        self.assertIn("2> /logs/compact-dwq.stderr.log", readme)
        self.assertIn(
            "set -o noclobber\n\nPYTHONPATH=. python -m alis_dwq.preflight",
            readme,
        )
        self.assertIn(
            "set -o noclobber\n\nALIS_DWQ_DATA_DIR=/build/laguna-data-v4 "
            "\\\nALIS_DWQ_NUM_VALID_SAMPLES=0 "
            "\\\nALIS_DWQ_LAYERS_PER_ROUND=1",
            readme,
        )
        self.assertIn("ALIS_DWQ_RUN_EVIDENCE_PATH=/logs/compact-dwq-run.jsonl", readme)
        self.assertIn("--student /build/compact-dwq-work-1", readme)
        self.assertNotIn("--student /build/compact-raw", readme)
        self.assertIn("--max-sequence-length 512 --vocab-size 100352 --sha256", readme)
        self.assertIn("`fix_mistral_regex=true`", readme)
        self.assertIn("`tokenizer_equivalence`", readme)
        self.assertIn("80 train, 40 valid, and 100 held-out rows", readme)
        self.assertIn("--out /build/compact-promotion-1-raw", readme)
        self.assertIn("--out /build/compact-promotion-2-raw", readme)
        self.assertIn(".attempts[0].convert_arguments", readme)
        self.assertIn(".attempts[1].convert_arguments", readme)


if __name__ == "__main__":
    unittest.main()

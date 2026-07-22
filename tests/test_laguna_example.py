import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


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

import json
import tempfile
import unittest
from pathlib import Path

from alis_dwq.clip_quantize import (
    _clip_evidence,
    _clip_input_binding,
    _verify_clip_inputs_unchanged,
)


class ClipEvidenceTests(unittest.TestCase):
    @staticmethod
    def _input(root: Path, *, recipe: str, quantized: bool) -> Path:
        root.mkdir()
        (root / "model-00001-of-00001.safetensors").write_bytes(recipe.encode())
        (root / "conversion_plan.json").write_text(
            json.dumps(
                {
                    "schema_version": "laguna.conversion/v2",
                    "artifact_label": recipe,
                    "source_repo": "poolside/Laguna-S-2.1",
                    "source_revision": "revision",
                    "source_shard_manifest_sha256": "a" * 64,
                    "mlx_lm_base_revision": "b" * 40,
                    "recipe": recipe,
                    "quantized": quantized,
                    "dwq_applied": False,
                    "clip_applied": False,
                    "release_complete": False,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return root

    def test_evidence_reports_applied_work_not_requested_flags(self):
        drops = [
            (0.10, 0.0, "model.a", 4),
            (0.25, 0.50, "model.b", 3),
        ]
        skipped = [("model.c", "missing in --source")]
        evidence = _clip_evidence(
            drops, skipped, {}, permutation_requested=True
        )

        self.assertTrue(evidence["clip_search_completed"])
        self.assertTrue(evidence["clip_applied"])
        self.assertEqual(evidence["clip_requantized_module_count"], 2)
        self.assertEqual(evidence["clip_clipped_module_count"], 1)
        self.assertEqual(evidence["clip_passthrough_module_count"], 1)
        self.assertTrue(evidence["ffn_permutation_requested"])
        self.assertFalse(evidence["ffn_permutation_applied"])
        self.assertEqual(evidence["ffn_permutation_block_count"], 0)

    def test_no_clipped_group_is_an_explicit_noop(self):
        evidence = _clip_evidence(
            [(0.0, 0.0, "model.a", 4)],
            [],
            {"model.a": (object(), -1, "model.block")},
            permutation_requested=True,
        )
        self.assertTrue(evidence["clip_search_completed"])
        self.assertFalse(evidence["clip_applied"])
        self.assertTrue(evidence["ffn_permutation_applied"])
        self.assertEqual(evidence["ffn_permutation_block_count"], 1)

    def test_clip_inputs_bind_exact_lineage_and_reject_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._input(
                root / "source", recipe="bf16-mlx-layout", quantized=False
            )
            student = self._input(
                root / "student", recipe="quality-3p7", quantized=True
            )
            binding = _clip_input_binding(source, student)
            self.assertEqual(binding["schema"], "alis-dwq.clip-inputs/v1")
            self.assertEqual(binding["source"]["recipe"], "bf16-mlx-layout")
            self.assertEqual(binding["student"]["recipe"], "quality-3p7")
            _verify_clip_inputs_unchanged(source, student, binding)

            (source / "model-00001-of-00001.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                _verify_clip_inputs_unchanged(source, student, binding)

    def test_clip_inputs_reject_wrong_lineage_and_completed_student(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._input(
                root / "source", recipe="bf16-mlx-layout", quantized=False
            )
            student = self._input(
                root / "student", recipe="quality-3p7", quantized=True
            )
            plan_path = student / "conversion_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["source_revision"] = "other"
            plan_path.write_text(json.dumps(plan) + "\n")
            with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                _clip_input_binding(source, student)

            plan["source_revision"] = "revision"
            plan["dwq_applied"] = True
            plan_path.write_text(json.dumps(plan) + "\n")
            with self.assertRaisesRegex(ValueError, "pre-DWQ"):
                _clip_input_binding(source, student)

    def test_generic_clip_inputs_without_plans_still_bind_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            student = root / "student"
            source.mkdir()
            student.mkdir()
            (source / "model.safetensors").write_bytes(b"float")
            (student / "model.safetensors").write_bytes(b"quantized")
            binding = _clip_input_binding(source, student)
            self.assertEqual(binding["lineage_mode"], "generic-digest-only")
            self.assertIsNone(binding["source"]["conversion_plan_sha256"])
            self.assertIsNone(binding["student"]["conversion_plan_sha256"])
            _verify_clip_inputs_unchanged(source, student, binding)

            (student / "conversion_plan.json").write_text(
                json.dumps(
                    {
                        "schema_version": "laguna.conversion/v2",
                        "recipe": "quality-3p7",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "two pinned conversion plans"):
                _clip_input_binding(source, student)


if __name__ == "__main__":
    unittest.main()

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from alis_dwq import run
from alis_dwq.memory_guard import emit_evidence


class RunEvidenceTests(unittest.TestCase):
    def test_memory_evidence_is_exclusively_reserved_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "memory.jsonl"
            self.assertEqual(run._reserve_memory_evidence_path(path), path)
            self.assertEqual(path.read_bytes(), b"")

            with mock.patch.dict(
                os.environ,
                {
                    "ALIS_DWQ_MEMORY_EVIDENCE_PATH": str(path),
                    "ALIS_DWQ_RUN_ID": "run-1",
                },
            ):
                emit_evidence(
                    {"event": "synthetic", "phase": "test"},
                    stream=io.StringIO(),
                )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["run_id"], "run-1")

            original = path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "no-clobber"):
                run._reserve_memory_evidence_path(path)
            self.assertEqual(path.read_bytes(), original)

            link = root / "memory-link.jsonl"
            link.symlink_to(path)
            with self.assertRaisesRegex(FileExistsError, "no-clobber"):
                run._reserve_memory_evidence_path(link)

    def test_live_data_binding_rejects_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            model = root / "model"
            data.mkdir()
            model.mkdir()
            (data / "train.jsonl").write_text('{"text":"train"}\n')
            (data / "valid.jsonl").write_text('{"text":"valid"}\n')
            (data / "manifest.json").write_text("{}\n")
            (model / "tokenizer.json").write_text("{}\n")

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            binding = {
                "data_manifest_kind": "file",
                "data_manifest_sha256": digest(data / "manifest.json"),
                "data_files_sha256": {
                    name: digest(data / name)
                    for name in ("train.jsonl", "valid.jsonl")
                },
                "tokenizer_files_sha256": {
                    "tokenizer.json": digest(model / "tokenizer.json")
                },
            }
            context = {"data_dir": data, "model": model}
            run._validate_live_data_binding(context, binding)
            (data / "train.jsonl").write_text('{"text":"changed"}\n')
            with self.assertRaisesRegex(RuntimeError, "calibration data changed"):
                run._validate_live_data_binding(context, binding)

    def test_target_publish_validation_rejects_persistent_teacher_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "teacher"
            teacher.mkdir()
            weights = teacher / "weights.safetensors"
            weights.write_bytes(b"original teacher bytes")
            context = {
                "model": teacher,
                "teacher_checkpoint_digest": run.directory_digest(teacher),
            }
            with mock.patch.object(run, "_validate_live_data_binding"):
                run._validate_target_publish_inputs(context, {})
                weights.write_bytes(b"mutated teacher bytes")
                with self.assertRaisesRegex(
                    RuntimeError, "changed during target computation"
                ):
                    run._validate_target_publish_inputs(context, {})

    def test_teacher_stability_is_required_only_when_model_is_the_teacher(self):
        shared = {
            "model": Path("/student"),
            "quantized_model": Path("/student"),
            "targets_only": False,
        }
        self.assertFalse(run._requires_teacher_stability(shared, "reuse"))
        self.assertTrue(run._requires_teacher_stability(shared, "new"))
        self.assertTrue(
            run._requires_teacher_stability({**shared, "targets_only": True}, "reuse")
        )
        self.assertTrue(
            run._requires_teacher_stability(
                {**shared, "model": Path("/teacher")}, "reuse"
            )
        )

    def test_dwq_output_uses_owned_staging_and_no_replace_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final"
            staging = run._reserve_output_staging(final, "run-1")
            self.assertTrue(staging.is_dir())
            rewritten = run._upstream_argv(
                ["alis_dwq.run", "--model", "/model", "--mlx-path", str(final)],
                mlx_path=staging,
            )
            self.assertEqual(
                rewritten[rewritten.index("--mlx-path") + 1], str(staging)
            )

            final.mkdir()
            (final / "sentinel").write_text("preserve\n")
            with self.assertRaises(FileExistsError):
                run.move_no_replace(staging, final)
            self.assertEqual((final / "sentinel").read_text(), "preserve\n")

    def test_completed_evidence_is_exactly_two_no_clobber_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            stream = io.StringIO()
            recorder = run._RunEvidenceRecorder(path, "run-1", stream=stream)
            started = {
                "schema": "alis-dwq.run/v2",
                "event": "run_started",
                "run_id": "run-1",
            }
            completed = run._completion_payload(
                "run_completed",
                "run-1",
                release_complete=True,
                pre_dwq_checkpoint_digest="a" * 64,
                target_contract_digest="b" * 64,
                final_artifact_digest="c" * 64,
            )
            recorder.record(started)
            recorder.publish(completed)

            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in lines], ["run_started", "run_completed"])
            self.assertEqual({row["run_id"] for row in lines}, {"run-1"})
            stderr_events = [
                json.loads(line.split("[alis-dwq][run] ", 1)[1])
                for line in stream.getvalue().splitlines()
            ]
            self.assertEqual(stderr_events, lines)
            with self.assertRaises(FileExistsError):
                run._RunEvidenceRecorder(path, "run-2")

    def test_diagnostic_evidence_and_artifact_are_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "release.jsonl"
            artifact = root / "output-diagnostic"
            artifact.mkdir()
            recorder = run._RunEvidenceRecorder(final, "diag-1", stream=io.StringIO())
            recorder.record(
                {
                    "schema": "alis-dwq.run/v2",
                    "event": "run_started",
                    "run_id": "diag-1",
                }
            )
            status = run._write_artifact_status_no_replace(
                artifact,
                run_id="diag-1",
                release_complete=False,
                completion_kind="diagnostic_partial",
                target_contract_digest="d" * 64,
            )
            incomplete = recorder.publish_incomplete(
                run._completion_payload(
                    "run_incomplete",
                    "diag-1",
                    release_complete=False,
                    completion_kind="diagnostic_partial",
                )
            )

            self.assertFalse(final.exists())
            self.assertTrue(incomplete.is_file())
            events = [json.loads(line) for line in incomplete.read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "run_incomplete")
            self.assertFalse(events[-1]["release_complete"])
            marker = json.loads(status.read_text())
            self.assertFalse(marker["release_complete"])
            self.assertEqual(marker["completion_kind"], "diagnostic_partial")

    def test_diagnostic_limits_are_detected(self):
        self.assertFalse(run._diagnostic_enabled({}))
        self.assertTrue(run._diagnostic_enabled({"ALIS_DWQ_MAX_ROUNDS": "1"}))
        self.assertTrue(
            run._diagnostic_enabled({"ALIS_DWQ_MAX_STEPS_PER_ROUND": "2"})
        )
        with tempfile.TemporaryDirectory() as directory:
            targets = Path(directory)
            self.assertFalse(run._target_dir_has_payload(targets))
            with self.assertRaisesRegex(FileExistsError, "partial or empty"):
                run._target_dir_state(targets)
            (targets / "train").mkdir()
            (targets / "train" / "0000000000.safetensors").write_bytes(b"train")
            self.assertFalse(run._target_dir_has_payload(targets))
            with self.assertRaisesRegex(FileExistsError, "partial or empty"):
                run._target_dir_state(targets)
            (targets / "valid").mkdir()
            (targets / "valid" / "0000000000.safetensors").write_bytes(b"valid")
            self.assertTrue(run._target_dir_has_payload(targets))
            with self.assertRaisesRegex(ValueError, "lack a regular"):
                run._target_dir_state(targets)
            (targets / "target-contract.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "unsupported schema"):
                run._target_dir_state(targets)
            contract = {
                "schema": "alis-dwq.targets/v1",
                "teacher": {
                    "identity": "teacher",
                    "revision": "revision",
                    "checkpoint_digest": "a" * 64,
                },
                "max_seq_length": 8,
                "batch_size": 1,
                "top_k": 1024,
                "seed": 7,
                "splits": {
                    split: {
                        "selected_count": 1,
                        "target_count": 1,
                        "rows": [
                            {
                                "target_index": 0,
                                "batch_position": 0,
                                "target_file": (
                                    f"{split}/0000000000.safetensors"
                                ),
                            }
                        ],
                    }
                    for split in ("train", "valid")
                },
            }
            (targets / "target-contract.json").write_text(
                json.dumps(contract) + "\n"
            )
            for split in ("train", "valid"):
                (targets / split / "._0000000000.safetensors").write_bytes(
                    b"macOS AppleDouble metadata"
                )
            self.assertEqual(
                run._target_dir_state(
                    targets,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                ),
                "reuse",
            )
            contract["splits"]["valid"]["target_count"] = 2
            (targets / "target-contract.json").write_text(
                json.dumps(contract) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "counts/rows"):
                run._target_dir_state(targets)
            self.assertEqual(run._target_dir_state(targets / "fresh"), "new")
        with self.assertRaisesRegex(ValueError, "require --target-dir"):
            run._parse_run_context(
                ["alis_dwq.run", "--model", "/model", "--seed", "7"]
            )
        with self.assertRaisesRegex(ValueError, "do not support --pipeline"):
            run._parse_run_context(
                [
                    "alis_dwq.run",
                    "--model",
                    "/model",
                    "--target-dir",
                    "/targets",
                    "--pipeline",
                    "--seed",
                    "7",
                ]
            )
        with self.assertRaisesRegex(ValueError, "separate adapter artifact"):
            run._parse_run_context(
                [
                    "alis_dwq.run",
                    "--model",
                    "/model",
                    "--target-dir",
                    "/targets",
                    "--seed",
                    "7",
                ],
                {"ALIS_DWQ_LORA_RANK": "8"},
            )


if __name__ == "__main__":
    unittest.main()

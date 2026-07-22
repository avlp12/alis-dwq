import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from alis_dwq import run
from alis_dwq.memory_guard import MemoryLimitExceeded, emit_evidence


class RunEvidenceTests(unittest.TestCase):
    def _runtime_tokenizer_fixture(self, root: Path):
        source = root / "runtime-tokenizer"
        targets = root / "targets"
        source.mkdir()
        targets.mkdir()
        files = {
            "tokenizer.json": b'{"version": "1.0"}\n',
            "tokenizer_config.json": b'{"tokenizer_file": "tokenizer.json"}\n',
            "chat_template.jinja": b"{{ messages }}\n",
        }
        for name, raw in files.items():
            (source / name).write_bytes(raw)
        hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        contract = {
            "schema": "alis-dwq.targets/v1",
            "tokenizer_files_sha256": hashes,
            "tokenizer_equivalence": {
                "schema": "alis-dwq.tokenizer-equivalence/v2",
                "mode": "file-identity",
                "source_tokenizer_files_sha256": hashes,
                "source_tokenizer_options": {"fix_mistral_regex": True},
                "runtime_tokenizer_files_sha256": hashes,
                "row_evidence": {
                    "schema": "alis-dwq.tokenizer-row-equivalence/v1",
                    "method": "live-runtime-tokenizer-encode/v1",
                    "tokenization": {
                        "name": "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat",
                        "preformatted_chat": True,
                        "add_special_tokens": False,
                        "append_eos": False,
                    },
                    "row_count": 220,
                    "splits": {"train": {}, "valid": {}, "heldout": {}},
                    "all_rows_verified": True,
                },
                "all_rows_verified": True,
            },
        }
        contract_path = targets / "target-contract.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
        return source, targets, contract_path, files, contract

    def test_run_started_environment_binds_generated_run_id(self):
        environ = {}
        with mock.patch.object(run, "_code_provenance", return_value={}):
            payload = run._started_payload(
                argv=["alis_dwq.run"], environ=environ, cwd="/build"
            )
        self.assertEqual(payload["environment"]["ALIS_DWQ_RUN_ID"], payload["run_id"])
        self.assertEqual(environ["ALIS_DWQ_RUN_ID"], payload["run_id"])

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
                    name: digest(data / name) for name in ("train.jsonl", "valid.jsonl")
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
            self.assertEqual(rewritten[rewritten.index("--mlx-path") + 1], str(staging))

            final.mkdir()
            (final / "sentinel").write_text("preserve\n")
            with self.assertRaises(FileExistsError):
                run.move_no_replace(staging, final)
            self.assertEqual((final / "sentinel").read_text(), "preserve\n")

    def test_runtime_tokenizer_flags_are_paired_exact_and_hidden_from_upstream(self):
        base = [
            "alis_dwq.run",
            "--model",
            "/model",
            "--target-dir",
            "/targets",
            "--seed",
            "7",
        ]
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            run._parse_run_context([*base, "--runtime-tokenizer-source", "/tokenizer"])
        with self.assertRaisesRegex(ValueError, "exactly --target-dir"):
            run._parse_run_context(
                [
                    *base,
                    "--runtime-tokenizer-source",
                    "/tokenizer",
                    "--target-contract",
                    "/other/target-contract.json",
                ]
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            run._parse_run_context(
                [
                    *base,
                    "--runtime-tokenizer-source=/one",
                    "--runtime-tokenizer-source=/two",
                    "--target-contract=/targets/target-contract.json",
                ]
            )

        wrapped = [
            *base,
            "--runtime-tokenizer-source=/tokenizer",
            "--target-contract",
            "/targets/target-contract.json",
            "--mlx-path",
            "/final",
        ]
        context = run._parse_run_context(wrapped)
        self.assertEqual(context["runtime_tokenizer_source"], Path("/tokenizer"))
        upstream = run._upstream_argv(wrapped, mlx_path=Path("/staging"))
        self.assertFalse(any("runtime-tokenizer-source" in value for value in upstream))
        self.assertFalse(any("target-contract" in value for value in upstream))
        self.assertEqual(upstream[upstream.index("--mlx-path") + 1], "/staging")

    def test_runtime_tokenizer_bundle_is_frozen_and_installed_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, contract_path, files, _ = self._runtime_tokenizer_fixture(root)
            bundle = run._load_runtime_tokenizer_bundle(source, contract_path)

            frozen = root / "frozen"
            frozen.mkdir()
            run._materialize_frozen_runtime_tokenizer(frozen, bundle)
            self.assertEqual(
                (frozen / "config.json").read_bytes(),
                b'{"model_type":"mistral"}\n',
            )
            calls = []

            def fake_loader(path, tokenizer_config_extra=None, **kwargs):
                calls.append((Path(path), tokenizer_config_extra, kwargs))
                return "tokenizer"

            loader = run._frozen_tokenizer_loader(fake_loader, frozen, bundle)
            self.assertEqual(
                loader("ignored", tokenizer_config_extra={"backend": "tokenizers"}),
                "tokenizer",
            )
            self.assertEqual(
                calls,
                [
                    (
                        frozen,
                        {
                            "backend": "tokenizers",
                            "fix_mistral_regex": True,
                            "local_files_only": True,
                        },
                        {},
                    )
                ],
            )

            output = root / "owned.partial-run"
            output.mkdir()
            (output / "model.safetensors").write_bytes(b"weights")
            (output / "config.json").write_text("{}\n")
            (output / "tokenizer.json").write_bytes(b"generated")
            (output / "tokenizer_config.json").write_bytes(b"generated")
            (output / "vocab.json").write_bytes(b"generated")
            (output / "._tokenizer.json").write_bytes(b"AppleDouble")

            installed = run._install_runtime_tokenizer(output, bundle)
            self.assertEqual(installed, bundle.files_sha256)
            for name, raw in files.items():
                self.assertEqual((output / name).read_bytes(), raw)
            self.assertFalse((output / "vocab.json").exists())
            self.assertFalse((output / "._tokenizer.json").exists())
            self.assertTrue((output / "model.safetensors").is_file())
            self.assertTrue((output / "config.json").is_file())
            self.assertEqual((output / "config.json").read_bytes(), b"{}\n")

            (source / "tokenizer.json").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "hash mismatch|bytes changed"):
                run._revalidate_runtime_tokenizer_bundle(bundle)

    def test_frozen_runtime_tokenizer_rejects_tamper_and_option_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, contract_path, _, _ = self._runtime_tokenizer_fixture(root)
            bundle = run._load_runtime_tokenizer_bundle(source, contract_path)
            frozen = root / "frozen"
            frozen.mkdir()
            run._materialize_frozen_runtime_tokenizer(frozen, bundle)
            calls = []

            def fake_loader(path, tokenizer_config_extra=None):
                calls.append((Path(path), tokenizer_config_extra))
                return "tokenizer"

            loader = run._frozen_tokenizer_loader(fake_loader, frozen, bundle)
            self.assertEqual(loader("ignored", {"local_files_only": True}), "tokenizer")
            self.assertEqual(
                calls[0][1],
                {"fix_mistral_regex": True, "local_files_only": True},
            )

            conflicts = (
                {"fix_mistral_regex": False},
                {"fix_mistral_regex": 1},
                {"local_files_only": False},
            )
            for options in conflicts:
                with (
                    self.subTest(options=options),
                    self.assertRaisesRegex(ValueError, "conflicts with required"),
                ):
                    loader("ignored", tokenizer_config_extra=options)
            with self.assertRaisesRegex(TypeError, "mapping or None"):
                loader("ignored", tokenizer_config_extra=True)
            with self.assertRaisesRegex(TypeError, "both positionally and by keyword"):
                loader(
                    "ignored",
                    {},
                    tokenizer_config_extra={},
                )

            (frozen / "config.json").write_bytes(b'{"model_type":"llama"}\n')
            with self.assertRaisesRegex(ValueError, "hash mismatch|bytes changed"):
                loader("ignored")
            self.assertEqual(len(calls), 1)

    def test_frozen_runtime_tokenizer_is_revalidated_after_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, contract_path, _, _ = self._runtime_tokenizer_fixture(root)
            bundle = run._load_runtime_tokenizer_bundle(source, contract_path)
            frozen = root / "frozen"
            frozen.mkdir()
            run._materialize_frozen_runtime_tokenizer(frozen, bundle)

            def tampering_loader(path, tokenizer_config_extra=None):
                del tokenizer_config_extra
                (Path(path) / "tokenizer.json").write_bytes(b"tampered")
                return "must not escape validation"

            loader = run._frozen_tokenizer_loader(tampering_loader, frozen, bundle)
            with self.assertRaisesRegex(ValueError, "hash mismatch|bytes changed"):
                loader("ignored")

    def test_runtime_tokenizer_preflight_rejects_ambiguous_inputs(self):
        cases = (
            "duplicate-contract",
            "nonfinite-contract",
            "symlink",
            "appledouble",
            "extra",
            "dependency",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _, contract_path, _, contract = self._runtime_tokenizer_fixture(
                    root
                )
                if case == "duplicate-contract":
                    contract_path.write_text(
                        '{"schema":"alis-dwq.targets/v1",'
                        '"schema":"alis-dwq.targets/v1"}\n'
                    )
                elif case == "nonfinite-contract":
                    contract["invalid"] = float("nan")
                    contract_path.write_text(json.dumps(contract) + "\n")
                elif case == "symlink":
                    path = source / "tokenizer.json"
                    real = source / "real-tokenizer.json"
                    path.rename(real)
                    path.symlink_to(real.name)
                elif case == "appledouble":
                    (source / "._tokenizer.json").write_bytes(b"metadata")
                elif case == "extra":
                    (source / "README.txt").write_text("not part of the bundle\n")
                elif case == "dependency":
                    config = b'{"vocab_file": "vocab.json"}\n'
                    (source / "tokenizer_config.json").write_bytes(config)
                    digest = hashlib.sha256(config).hexdigest()
                    contract["tokenizer_files_sha256"]["tokenizer_config.json"] = digest
                    contract["tokenizer_equivalence"]["runtime_tokenizer_files_sha256"][
                        "tokenizer_config.json"
                    ] = digest
                    contract_path.write_text(json.dumps(contract) + "\n")
                with self.assertRaises(ValueError):
                    run._load_runtime_tokenizer_bundle(source, contract_path)

    def test_fake_upstream_output_is_replaced_before_digest_and_completion(self):
        import mlx_lm.utils as mlx_utils

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, targets, contract_path, files, contract = (
                self._runtime_tokenizer_fixture(root)
            )
            teacher = root / "teacher"
            student = root / "student"
            output = root / "output"
            teacher.mkdir()
            student.mkdir()
            (teacher / "model.safetensors").write_bytes(b"teacher")
            (student / "model.safetensors").write_bytes(b"student")
            for checkpoint in (teacher, student):
                (checkpoint / "config.json").write_text(
                    json.dumps({"model_type": "laguna"}) + "\n"
                )
            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--quantized-model",
                str(student),
                "--target-dir",
                str(targets),
                "--runtime-tokenizer-source",
                str(source),
                "--target-contract",
                str(contract_path),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            sequence = []
            loaded_from = []
            target_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            original_save = run.D.save
            original_make_shards = mlx_utils.make_shards
            original_mx = mlx_utils.mx

            class Recorder:
                def record(self, payload):
                    sequence.append("record")

                def publish(self, payload):
                    sequence.append("publish")

                def publish_incomplete(self, payload):
                    sequence.append("publish-incomplete")

            def fake_tokenizer_loader(path, *args, **kwargs):
                del args, kwargs
                loaded_from.append(Path(path))
                return object()

            def fake_upstream():
                sequence.append("upstream")
                self.assertIsNot(run.D.save, original_save)
                self.assertIsNot(mlx_utils.make_shards, original_make_shards)
                self.assertIsNot(mlx_utils.mx, original_mx)
                self.assertFalse(
                    any("runtime-tokenizer-source" in value for value in run.sys.argv)
                )
                self.assertFalse(
                    any("target-contract" in value for value in run.sys.argv)
                )
                staging = Path(run.sys.argv[run.sys.argv.index("--mlx-path") + 1])
                (staging / "model.safetensors").write_bytes(b"dwq")
                (staging / "config.json").write_text("{}\n")
                (staging / "tokenizer.json").write_bytes(b"generated")
                (staging / "tokenizer_config.json").write_bytes(b"generated")
                (staging / "vocab.json").write_bytes(b"generated")
                run.D.load_tokenizer("requested-model")
                run._ACTIVE_DATA_BINDING = {}
                run._TARGET_CONTRACT_PATH = contract_path
                run._TARGET_CONTRACT_DIGEST = target_digest

            real_install = run._install_runtime_tokenizer

            def ordered_install(staging, bundle):
                sequence.append("install")
                return real_install(staging, bundle)

            def validate(*args, **kwargs):
                del args, kwargs
                sequence.append("validate")
                return contract

            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(run.D, "load_tokenizer", fake_tokenizer_loader),
                mock.patch.object(run.D, "main", side_effect=fake_upstream),
                mock.patch.object(run, "_target_dir_state", return_value="reuse"),
                mock.patch.object(
                    run, "_install_runtime_tokenizer", side_effect=ordered_install
                ),
                mock.patch.object(
                    run, "_validate_completion_inputs", side_effect=validate
                ),
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "runtime-tokenizer-test"},
                ),
            ):
                run.main(argv)

            self.assertEqual(
                sequence,
                ["record", "upstream", "validate", "install", "publish"],
            )
            self.assertIs(run.D.save, original_save)
            self.assertIs(mlx_utils.make_shards, original_make_shards)
            self.assertIs(mlx_utils.mx, original_mx)
            self.assertEqual(len(loaded_from), 1)
            self.assertNotEqual(loaded_from[0], source)
            self.assertFalse(loaded_from[0].exists())
            for name, raw in files.items():
                self.assertEqual((output / name).read_bytes(), raw)
            self.assertFalse((output / "vocab.json").exists())
            self.assertEqual(
                json.loads((output / "alis-dwq-run-status.json").read_text())[
                    "target_contract_digest"
                ],
                target_digest,
            )
            self.assertEqual(
                json.loads((output / "alis-dwq-run-status.json").read_text())[
                    "target_contract_canonical_sha256"
                ],
                run.canonical_sha256(contract),
            )

    def test_laguna_student_load_stop_retains_partial_and_restores_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            student = root / "student"
            targets = root / "targets"
            output = root / "output"
            for checkpoint in (teacher, student):
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                (checkpoint / "config.json").write_text(
                    json.dumps({"model_type": "laguna"}) + "\n"
                )
            targets.mkdir()
            sequence = []
            state = {}
            captured = {}

            class Recorder:
                def record(self, _payload):
                    sequence.append("record")

                def publish(self, _payload):
                    raise AssertionError("a stopped run must not publish completion")

                def publish_incomplete(self, _payload):
                    sequence.append("publish-incomplete")
                    state["restored_during_failure"] = (
                        run.D.load is state["original_load"]
                        and run.D.dwq_quantize is state["original_quantizer"]
                    )

            class Guard:
                def __init__(self, phase, recommended, **kwargs):
                    sequence.append(("guard", phase, recommended))
                    captured.update(kwargs)

                def start(self):
                    sequence.append("start")

                def check(self, checkpoint, **context):
                    sequence.append(("check", checkpoint, context.get("model_role")))
                    if checkpoint == "after-upstream-model-load":
                        raise MemoryLimitExceeded(
                            {"event": "memory_stop_gate", "checkpoint": checkpoint}
                        )

            def fake_load(path, *_args, **_kwargs):
                sequence.append(("load", Path(path)))
                return object(), object(), {"quantization": {}}

            def fake_upstream():
                sequence.append("upstream")
                run.D.load(str(student), lazy=True, return_config=True)
                self.fail("student load stop must prevent DWQ training")

            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--quantized-model",
                str(student),
                "--target-dir",
                str(targets),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(
                    run.D, "load", side_effect=fake_load
                ) as original_load,
                mock.patch.object(run.D, "dwq_quantize") as original_quantizer,
                mock.patch.object(run.D, "main", side_effect=fake_upstream),
                mock.patch.object(run, "_target_dir_state", return_value="reuse"),
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "memory-stop-test"},
                ),
                mock.patch.object(
                    run,
                    "configure_recommended_wired_limit",
                    side_effect=lambda phase: (
                        sequence.append(("wired", phase)) or 1_000
                    ),
                ),
                mock.patch.object(run, "MemoryGuard", Guard),
            ):
                state["original_load"] = original_load
                state["original_quantizer"] = original_quantizer
                with self.assertRaises(MemoryLimitExceeded):
                    run.main(argv)

            self.assertTrue(state["restored_during_failure"])
            self.assertTrue(captured["require_recommended_working_set"])
            self.assertTrue(captured["require_swap_measurement"])
            self.assertEqual(captured["limits"].max_peak_fraction, 0.90)
            self.assertIn(("check", "before-upstream-model-load", "student"), sequence)
            self.assertIn(("check", "after-upstream-model-load", "student"), sequence)
            self.assertIn("publish-incomplete", sequence)
            self.assertFalse(output.exists())
            self.assertEqual(
                list(root.glob("output.partial-memory-stop-test-*")),
                [root / f"output.partial-memory-stop-test-{os.getpid()}"],
            )

    def test_laguna_model_save_stop_retains_partial_and_restores_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            student = root / "student"
            targets = root / "targets"
            output = root / "output"
            for checkpoint in (teacher, student):
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                (checkpoint / "config.json").write_text(
                    json.dumps({"model_type": "laguna"}) + "\n"
                )
            targets.mkdir()
            sequence = []
            state = {}

            class Recorder:
                def record(self, _payload):
                    sequence.append("record")

                def publish(self, _payload):
                    raise AssertionError("a stopped save must not publish completion")

                def publish_incomplete(self, _payload):
                    sequence.append("publish-incomplete")
                    state["restored_during_failure"] = (
                        run.D.load is state["original_load"]
                        and run.D.dwq_quantize is state["original_quantizer"]
                        and run.D.save is state["original_save"]
                    )

            class Guard:
                def __init__(self, _phase, _recommended, **_kwargs):
                    pass

                def start(self):
                    sequence.append("start")

                def check(self, checkpoint, **_context):
                    sequence.append(("check", checkpoint))
                    if checkpoint == "after-upstream-model-save":
                        raise MemoryLimitExceeded(
                            {"event": "memory_stop_gate", "checkpoint": checkpoint}
                        )

            def fake_save(path, *_args, **_kwargs):
                sequence.append("save")
                destination = Path(path)
                (destination / "model.safetensors").write_bytes(b"partial weights")

            fake_save.__module__ = "mlx_lm.utils"

            def fake_upstream():
                sequence.append("upstream")
                staging = Path(run.sys.argv[run.sys.argv.index("--mlx-path") + 1])
                run.D.save(staging, student, object(), object(), {})
                self.fail("post-save stop must prevent final publication")

            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--quantized-model",
                str(student),
                "--target-dir",
                str(targets),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(run.D, "load") as original_load,
                mock.patch.object(run.D, "dwq_quantize") as original_quantizer,
                mock.patch.object(run.D, "save", fake_save),
                mock.patch.object(run.D, "main", side_effect=fake_upstream),
                mock.patch.object(run, "_target_dir_state", return_value="reuse"),
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "save-stop-test"},
                ),
                mock.patch.object(
                    run, "configure_recommended_wired_limit", return_value=1_000
                ),
                mock.patch.object(run, "MemoryGuard", Guard),
            ):
                state["original_load"] = original_load
                state["original_quantizer"] = original_quantizer
                state["original_save"] = fake_save
                with self.assertRaises(MemoryLimitExceeded):
                    run.main(argv)

            self.assertTrue(state["restored_during_failure"])
            self.assertLess(
                sequence.index(("check", "before-upstream-model-save")),
                sequence.index("save"),
            )
            self.assertLess(
                sequence.index("save"),
                sequence.index(("check", "after-upstream-model-save")),
            )
            self.assertIn("publish-incomplete", sequence)
            self.assertFalse(output.exists())
            partial = root / f"output.partial-save-stop-test-{os.getpid()}"
            self.assertEqual(
                list(root.glob("output.partial-save-stop-test-*")), [partial]
            )
            self.assertEqual(
                (partial / "model.safetensors").read_bytes(), b"partial weights"
            )

    def test_non_laguna_run_does_not_patch_model_or_shard_savers(self):
        import mlx_lm.utils as mlx_utils

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            student = root / "student"
            targets = root / "targets"
            output = root / "output"
            for checkpoint in (teacher, student):
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                (checkpoint / "config.json").write_text(
                    json.dumps({"model_type": "llama"}) + "\n"
                )
            targets.mkdir()

            class Recorder:
                def record(self, _payload):
                    pass

                def publish(self, _payload):
                    raise AssertionError("the synthetic run must fail")

                def publish_incomplete(self, _payload):
                    pass

            class Guard:
                def __init__(self, _phase, _recommended, **_kwargs):
                    pass

                def start(self):
                    pass

                def check(self, _checkpoint, **_context):
                    pass

            def sentinel_save(*_args, **_kwargs):
                raise AssertionError("the synthetic run does not save")

            original_make_shards = mlx_utils.make_shards
            original_mx = mlx_utils.mx

            def fake_upstream():
                self.assertIs(run.D.save, sentinel_save)
                self.assertIs(mlx_utils.make_shards, original_make_shards)
                self.assertIs(mlx_utils.mx, original_mx)
                raise RuntimeError("synthetic upstream stop")

            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--quantized-model",
                str(student),
                "--target-dir",
                str(targets),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(run.D, "save", sentinel_save),
                mock.patch.object(run.D, "main", side_effect=fake_upstream),
                mock.patch.object(run, "_target_dir_state", return_value="reuse"),
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "non-laguna-test"},
                ),
                mock.patch.object(
                    run, "configure_recommended_wired_limit", return_value=1_000
                ),
                mock.patch.object(run, "MemoryGuard", Guard),
                mock.patch.object(run, "_upstream_save_runtime") as save_runtime,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic upstream stop"):
                    run.main(argv)

            save_runtime.assert_not_called()
            self.assertIs(mlx_utils.make_shards, original_make_shards)
            self.assertIs(mlx_utils.mx, original_mx)

    def test_laguna_mid_shard_stop_retains_written_shards_and_restores_all_patches(
        self,
    ):
        import mlx_lm.utils as mlx_utils

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            student = root / "student"
            targets = root / "targets"
            output = root / "output"
            for checkpoint in (teacher, student):
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                (checkpoint / "config.json").write_text(
                    json.dumps({"model_type": "laguna"}) + "\n"
                )
            targets.mkdir()
            sequence = []
            state = {}

            class Weight:
                nbytes = 1

            class FakeMx:
                def save_safetensors(self, path, _shard, **_kwargs):
                    sequence.append(("write", Path(path).name))
                    Path(path).write_bytes(b"written")

            class Recorder:
                def record(self, _payload):
                    sequence.append("record")

                def publish(self, _payload):
                    raise AssertionError("a stopped shard save must not complete")

                def publish_incomplete(self, _payload):
                    sequence.append("publish-incomplete")
                    state["restored_during_failure"] = (
                        run.D.load is state["original_load"]
                        and run.D.dwq_quantize is state["original_quantizer"]
                        and run.D.save is state["original_save"]
                        and mlx_utils.make_shards is state["original_make_shards"]
                        and mlx_utils.mx is state["original_mx"]
                    )

            class Guard:
                def __init__(self, _phase, _recommended, **_kwargs):
                    pass

                def start(self):
                    sequence.append("start")

                def check(self, checkpoint, **context):
                    shard_name = (
                        Path(context["shard_path"]).name
                        if "shard_path" in context
                        else None
                    )
                    sequence.append(("check", checkpoint, shard_name))
                    if (
                        checkpoint == "before-upstream-shard-save"
                        and shard_name == "model-00002-of-00003.safetensors"
                    ):
                        raise MemoryLimitExceeded(
                            {"event": "memory_stop_gate", "checkpoint": checkpoint}
                        )

            def fake_make_shards(_weights, *_args, **_kwargs):
                sequence.append("make-shards")
                return [{"one": Weight()}, {"two": Weight()}, {"three": Weight()}]

            def fake_save(path, *_args, **_kwargs):
                destination = Path(path)
                shards = mlx_utils.make_shards({"weight": Weight()})
                for index, shard in enumerate(shards, 1):
                    name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
                    mlx_utils.mx.save_safetensors(destination / name, shard)

            fake_save.__module__ = "mlx_lm.utils"

            def fake_upstream():
                staging = Path(run.sys.argv[run.sys.argv.index("--mlx-path") + 1])
                run.D.save(staging, student, object(), object(), {})

            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--quantized-model",
                str(student),
                "--target-dir",
                str(targets),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            fake_mx = FakeMx()
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(run.D, "load") as original_load,
                mock.patch.object(run.D, "dwq_quantize") as original_quantizer,
                mock.patch.object(run.D, "save", fake_save),
                mock.patch.object(mlx_utils, "make_shards", fake_make_shards),
                mock.patch.object(mlx_utils, "mx", fake_mx),
                mock.patch.object(run.D, "main", side_effect=fake_upstream),
                mock.patch.object(run, "_target_dir_state", return_value="reuse"),
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "shard-stop-test"},
                ),
                mock.patch.object(
                    run, "configure_recommended_wired_limit", return_value=1_000
                ),
                mock.patch.object(run, "MemoryGuard", Guard),
            ):
                state["original_load"] = original_load
                state["original_quantizer"] = original_quantizer
                state["original_save"] = fake_save
                state["original_make_shards"] = fake_make_shards
                state["original_mx"] = fake_mx
                with self.assertRaises(MemoryLimitExceeded):
                    run.main(argv)

            self.assertTrue(state["restored_during_failure"])
            self.assertFalse(output.exists())
            partial = root / f"output.partial-shard-stop-test-{os.getpid()}"
            self.assertEqual(
                list(root.glob("output.partial-shard-stop-test-*")), [partial]
            )
            self.assertTrue((partial / "model-00001-of-00003.safetensors").is_file())
            self.assertFalse((partial / "model-00002-of-00003.safetensors").exists())
            checks_and_writes = [
                entry for entry in sequence if isinstance(entry, tuple)
            ]
            first_save_gate = checks_and_writes.index(
                ("check", "before-upstream-model-save", None)
            )
            self.assertEqual(
                checks_and_writes[first_save_gate:],
                [
                    ("check", "before-upstream-model-save", None),
                    ("check", "before-upstream-shard-materialization", None),
                    ("check", "after-upstream-shard-materialization", None),
                    (
                        "check",
                        "before-upstream-shard-save",
                        "model-00001-of-00003.safetensors",
                    ),
                    ("write", "model-00001-of-00003.safetensors"),
                    (
                        "check",
                        "after-upstream-shard-save",
                        "model-00001-of-00003.safetensors",
                    ),
                    (
                        "check",
                        "before-upstream-shard-save",
                        "model-00002-of-00003.safetensors",
                    ),
                ],
            )
            self.assertIn("publish-incomplete", sequence)

    def test_guarded_laguna_requires_manifest_quantized_model_before_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            targets = root / "targets"
            output = root / "output"
            teacher.mkdir()
            targets.mkdir()
            (teacher / "model.safetensors").write_bytes(b"weights")
            (teacher / "config.json").write_text(
                json.dumps({"model_type": "laguna"}) + "\n"
            )
            incomplete = []

            class Recorder:
                def record(self, _payload):
                    pass

                def publish(self, _payload):
                    raise AssertionError("a rejected run must not publish completion")

                def publish_incomplete(self, payload):
                    incomplete.append(payload)

            argv = [
                "alis_dwq.run",
                "--model",
                str(teacher),
                "--target-dir",
                str(targets),
                "--mlx-path",
                str(output),
                "--seed",
                "7",
            ]
            group = mock.Mock()
            group.rank.return_value = 0
            group.size.return_value = 1
            with (
                mock.patch.object(run.mx.distributed, "init", return_value=group),
                mock.patch.object(run.D, "load_data", run._load_local),
                mock.patch.object(run.D, "main") as upstream,
                mock.patch.object(run, "_target_dir_state") as target_state,
                mock.patch.object(run, "configure_recommended_wired_limit") as wired,
                mock.patch.object(run, "_RunEvidenceRecorder", return_value=Recorder()),
                mock.patch.object(
                    run,
                    "_started_payload",
                    return_value={"run_id": "missing-student-test"},
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "require --quantized-model.*official execution manifest"
                ):
                    run.main(argv)

            upstream.assert_not_called()
            target_state.assert_not_called()
            wired.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(len(incomplete), 1)
            self.assertEqual(incomplete[0]["event"], "run_failed")

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
            self.assertEqual(
                [row["event"] for row in lines], ["run_started", "run_completed"]
            )
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
                target_contract_canonical_sha256="e" * 64,
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
            self.assertEqual(marker["target_contract_digest"], "d" * 64)
            self.assertEqual(marker["target_contract_canonical_sha256"], "e" * 64)

    def test_diagnostic_limits_are_detected(self):
        self.assertFalse(run._diagnostic_enabled({}))
        self.assertTrue(run._diagnostic_enabled({"ALIS_DWQ_MAX_ROUNDS": "1"}))
        self.assertTrue(run._diagnostic_enabled({"ALIS_DWQ_MAX_STEPS_PER_ROUND": "2"}))
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
                                "target_file": (f"{split}/0000000000.safetensors"),
                            }
                        ],
                    }
                    for split in ("train", "valid")
                },
            }
            (targets / "target-contract.json").write_text(json.dumps(contract) + "\n")
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
            (targets / "target-contract.json").write_text(json.dumps(contract) + "\n")
            with self.assertRaisesRegex(ValueError, "counts/rows"):
                run._target_dir_state(targets)
            self.assertEqual(run._target_dir_state(targets / "fresh"), "new")
        with self.assertRaisesRegex(ValueError, "require --target-dir"):
            run._parse_run_context(["alis_dwq.run", "--model", "/model", "--seed", "7"])
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

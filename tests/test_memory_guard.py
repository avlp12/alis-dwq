import io
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from alis_dwq.memory_guard import (
    GIB,
    MemoryEvidenceError,
    MemoryGuard,
    MemoryLimitExceeded,
    MemoryLimits,
    check_round_or_restore,
    configure_recommended_wired_limit,
    emit_evidence,
    read_swap_used_bytes,
)


class FakeMetal:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available


class FakeMX:
    def __init__(self, *, peak=0, active=0, recommended=100):
        self.metal = FakeMetal()
        self.peak = peak
        self.active = active
        self.recommended = recommended
        self.wired_limits = []
        self.reset_count = 0
        self.evaluated = []

    def device_info(self):
        return {"max_recommended_working_set_size": self.recommended}

    def set_wired_limit(self, value):
        self.wired_limits.append(value)
        return 123

    def get_peak_memory(self):
        return self.peak

    def get_active_memory(self):
        return self.active

    def reset_peak_memory(self):
        self.reset_count += 1

    def eval(self, value):
        self.evaluated.append(value)


class TestMemoryGuard(unittest.TestCase):
    def test_swap_reader_is_darwin_only_and_parses_units(self):
        calls = []

        def runner(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(
                stdout="total = 64.00G  used = 17.50G  free = 46.50G"
            )

        used = read_swap_used_bytes(platform_system=lambda: "Darwin", runner=runner)
        self.assertEqual(used, int(17.5 * GIB))
        self.assertEqual(len(calls), 1)

        used = read_swap_used_bytes(
            platform_system=lambda: "Linux",
            runner=lambda *a, **k: self.fail("runner called on Linux"),
        )
        self.assertIsNone(used)

    def test_wired_limit_uses_mlx_recommendation_and_nonmac_is_noop(self):
        fake = FakeMX(recommended=498)
        events = []
        recommended = configure_recommended_wired_limit(
            "targets-only",
            mx_module=fake,
            platform_system=lambda: "Darwin",
            emitter=events.append,
        )
        self.assertEqual(recommended, 498)
        self.assertEqual(fake.wired_limits, [498])
        self.assertEqual(events[0]["event"], "wired_limit_configured")

        other = FakeMX(recommended=999)
        self.assertIsNone(
            configure_recommended_wired_limit(
                "training",
                mx_module=other,
                platform_system=lambda: "Linux",
                emitter=events.append,
            )
        )
        self.assertEqual(other.wired_limits, [])

    def test_default_and_configurable_limits(self):
        defaults = MemoryLimits.from_env({})
        self.assertEqual(defaults.max_peak_fraction, 0.90)
        self.assertEqual(defaults.max_swap_increase_bytes, 16 * GIB)

        configured = MemoryLimits.from_env(
            {
                "ALIS_DWQ_MAX_PEAK_FRACTION": "0.75",
                "ALIS_DWQ_MAX_SWAP_INCREASE_GIB": "4.5",
            }
        )
        self.assertEqual(configured.max_peak_fraction, 0.75)
        self.assertEqual(configured.max_swap_increase_bytes, int(4.5 * GIB))

        disabled = MemoryLimits.from_env(
            {
                "ALIS_DWQ_MAX_PEAK_FRACTION": "0",
                "ALIS_DWQ_MAX_SWAP_INCREASE_GIB": "-1",
            }
        )
        self.assertIsNone(disabled.max_peak_fraction)
        self.assertIsNone(disabled.max_swap_increase_bytes)

    def test_guarded_laguna_limits_cannot_be_disabled_or_loosened(self):
        limits = MemoryLimits.guarded_laguna(
            {
                "ALIS_DWQ_MAX_PEAK_FRACTION": "0",
                "ALIS_DWQ_MAX_SWAP_INCREASE_GIB": "64",
            }
        )
        self.assertEqual(limits.max_peak_fraction, 0.90)
        self.assertEqual(limits.max_swap_increase_bytes, 16 * GIB)

        stricter = MemoryLimits.guarded_laguna(
            {
                "ALIS_DWQ_MAX_PEAK_FRACTION": "0.75",
                "ALIS_DWQ_MAX_SWAP_INCREASE_GIB": "4",
            }
        )
        self.assertEqual(stricter.max_peak_fraction, 0.75)
        self.assertEqual(stricter.max_swap_increase_bytes, 4 * GIB)

    def test_guarded_laguna_fails_closed_when_swap_measurement_is_missing(self):
        for values in ([None, None], [0, None]):
            with self.subTest(values=values):
                events = []
                readings = iter(values)
                guard = MemoryGuard(
                    "laguna",
                    100,
                    limits=MemoryLimits.guarded_laguna({}),
                    mx_module=FakeMX(peak=10, active=10),
                    swap_reader=lambda: next(readings),
                    emitter=events.append,
                    require_recommended_working_set=True,
                    require_swap_measurement=True,
                )
                guard.start()
                with self.assertRaises(MemoryLimitExceeded) as caught:
                    guard.check("pre-load")
                self.assertIn(
                    "swap_measurement_unavailable",
                    caught.exception.evidence["reasons"],
                )
                self.assertEqual(events[-1]["event"], "memory_stop_gate")

    def test_guarded_laguna_requires_recommended_working_set(self):
        for recommended in (None, 0):
            with self.subTest(recommended=recommended):
                guard = MemoryGuard(
                    "laguna",
                    recommended,
                    limits=MemoryLimits.guarded_laguna({}),
                    mx_module=FakeMX(peak=10, active=10),
                    swap_reader=lambda: 0,
                    emitter=lambda event: event,
                    require_recommended_working_set=True,
                    require_swap_measurement=True,
                )
                guard.start()
                with self.assertRaises(MemoryLimitExceeded) as caught:
                    guard.check("pre-load")
                self.assertEqual(
                    caught.exception.evidence["reasons"],
                    ["recommended_working_set_unavailable"],
                )

    def test_stop_gate_uses_strict_thresholds_and_phase_swap_baseline(self):
        fake = FakeMX(peak=90, active=80, recommended=100)
        swap_values = iter([10 * GIB, 26 * GIB])
        events = []
        guard = MemoryGuard(
            "training",
            100,
            mx_module=fake,
            swap_reader=lambda: next(swap_values),
            emitter=events.append,
        )
        guard.start()
        # Peak equality is allowed, but the plan requires a stop at a swap
        # increase of 16 GiB or greater.
        with self.assertRaises(MemoryLimitExceeded) as caught:
            guard.check("at-swap-limit", iteration=3)
        self.assertEqual(
            set(caught.exception.evidence["reasons"]),
            {"swap_increase"},
        )
        self.assertEqual(caught.exception.evidence["iteration"], 3)
        self.assertEqual(events[-1]["event"], "memory_stop_gate")

    def test_round_gate_restores_snapshot_before_abort_is_recorded(self):
        class Model:
            def __init__(self):
                self.value = "changed"

            def update(self, snapshot):
                self.value = snapshot["value"]

            def trainable_parameters(self):
                return {"value": self.value}

        model = Model()
        fake_mx = FakeMX()

        class Guard:
            def __init__(self):
                self.abort_records = []

            def check(self, *args, **kwargs):
                raise MemoryLimitExceeded({"event": "memory_stop_gate"})

            def record_round_abort(self, error, **kwargs):
                self.abort_records.append((model.value, kwargs))

        guard = Guard()
        with self.assertRaises(MemoryLimitExceeded):
            check_round_or_restore(
                guard,
                model,
                {"value": "snapshot"},
                "training-step",
                round_index=2,
                layers=[7, 6],
                mx_module=fake_mx,
            )
        self.assertEqual(model.value, "snapshot")
        self.assertEqual(fake_mx.evaluated, [{"value": "snapshot"}])
        self.assertEqual(guard.abort_records[0][0], "snapshot")
        self.assertTrue(guard.abort_records[0][1]["restored"])

    def test_evidence_failure_still_restores_round_snapshot(self):
        class Model:
            def __init__(self):
                self.value = "changed"

            def update(self, snapshot):
                self.value = snapshot["value"]

            def trainable_parameters(self):
                return {"value": self.value}

        model = Model()
        fake_mx = FakeMX()
        guard = MemoryGuard(
            "training",
            100,
            mx_module=fake_mx,
            swap_reader=lambda: 0,
            emitter=mock.Mock(side_effect=OSError("disk full")),
        )
        guard.started = True
        guard.baseline_swap_bytes = 0

        with self.assertRaises(MemoryEvidenceError):
            check_round_or_restore(
                guard,
                model,
                {"value": "snapshot"},
                "training-step",
                round_index=1,
                layers=[47],
                mx_module=fake_mx,
            )
        self.assertEqual(model.value, "snapshot")
        self.assertEqual(fake_mx.evaluated, [{"value": "snapshot"}])

    def test_stop_threshold_survives_evidence_failure(self):
        guard = MemoryGuard(
            "training",
            100,
            mx_module=FakeMX(peak=91, active=80),
            swap_reader=lambda: 0,
            emitter=mock.Mock(side_effect=OSError("read only")),
        )
        guard.started = True
        guard.baseline_swap_bytes = 0
        with self.assertRaises(MemoryLimitExceeded) as caught:
            guard.check("over-limit")
        self.assertEqual(caught.exception.evidence["reasons"], ["peak_working_set"])

    def test_evidence_is_jsonl(self):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            payload = emit_evidence(
                {"event": "synthetic", "phase": "test"},
                stream=stream,
                path=path,
            )
            line = path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        self.assertEqual(parsed["schema"], "alis-dwq.memory/v1")
        self.assertEqual(parsed["event"], "synthetic")
        self.assertEqual(payload["phase"], "test")
        self.assertIn(line, stream.getvalue())

    def test_memory_evidence_includes_run_id(self):
        stream = io.StringIO()
        with mock.patch.dict(os.environ, {"ALIS_DWQ_RUN_ID": "run-123"}):
            payload = emit_evidence(
                {"event": "synthetic", "phase": "test"}, stream=stream
            )
        self.assertEqual(payload["run_id"], "run-123")

    def test_run_evidence_records_only_declared_environment(self):
        from alis_dwq import run

        stream = io.StringIO()
        environ = {
            "PYTHONPATH": "/runtime/alis:/runtime/mlx-lm",
            "ALIS_DWQ_DATA_DIR": "/data/laguna",
            "ALIS_DWQ_LAYERS_PER_ROUND": "1",
            "HF_TOKEN": "must-not-leak",
        }
        payload = run.emit_run_evidence(
            argv=["alis_dwq.run", "--seed", "7"],
            environ=environ,
            cwd="/build",
            stream=stream,
        )
        self.assertEqual(payload["cwd"], "/build")
        self.assertEqual(payload["environment"]["ALIS_DWQ_LAYERS_PER_ROUND"], "1")
        self.assertEqual(payload["environment"]["PYTHONPATH"], environ["PYTHONPATH"])
        self.assertNotIn("HF_TOKEN", payload["environment"])
        self.assertEqual(environ["ALIS_DWQ_RUN_ID"], payload["run_id"])
        self.assertEqual(
            payload["environment"]["ALIS_DWQ_RUN_ID"], payload["run_id"]
        )
        self.assertEqual(payload["schema"], "alis-dwq.run/v2")
        self.assertEqual(payload["event"], "run_started")
        self.assertEqual(
            set(payload["code"]["source_files_sha256"]),
            {
                "alis_dwq/run.py",
                "alis_dwq/layerwise.py",
                "alis_dwq/memory_guard.py",
                "alis_dwq/preflight.py",
                "alis_dwq/io_utils.py",
                "alis_dwq/target_contract.py",
                "alis_dwq/losses.py",
            },
        )
        runtime = payload["code"]["runtime"]
        self.assertTrue(Path(runtime["mlx_lm_checkout_root"]).is_absolute())
        self.assertIsInstance(runtime["worktree_dirty"], bool)
        self.assertEqual(
            set(runtime["source_files_sha256"]),
            {"mlx_lm/models/laguna.py"},
        )
        self.assertIn('"schema": "alis-dwq.run/v2"', stream.getvalue())

    def test_preformatted_chat_data_does_not_add_special_tokens_or_eos(self):
        from alis_dwq import run

        class Tokenizer:
            eos_token_id = 99

            def __init__(self):
                self.calls = []

            def encode(self, text, **kwargs):
                self.calls.append((text, kwargs))
                return [7, 8]

        tokenizer = Tokenizer()
        rows = {
            "train": [{"text": "templated-a"}, {"text": "templated-b"}],
            "valid": [{"text": "templated-v"}],
        }

        def fake_open(path):
            split = "train" if "train.jsonl" in str(path) else "valid"
            return [json.dumps(row) for row in rows[split]]

        class Dataset:
            def __init__(self, data, _tokenizer):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

            def process(self, row):
                raise AssertionError("generic TextDataset path must not be used")

        with (
            mock.patch.object(run, "TextDataset", Dataset),
            mock.patch("builtins.open", side_effect=fake_open),
            mock.patch.dict(
                os.environ,
                {
                    "ALIS_DWQ_TEXT_TOKENIZATION": "preformatted_chat",
                    "ALIS_DWQ_NUM_VALID_SAMPLES": "1",
                },
            ),
            mock.patch.object(
                run.np.random, "permutation", return_value=run.np.array([0, 1])
            ),
        ):
            train, valid = run._load_local(tokenizer, "unused", 2, 512)

        self.assertEqual(train, [([7, 8], 0), ([7, 8], 0)])
        self.assertEqual(valid, [([7, 8], 0)])
        self.assertTrue(tokenizer.calls)
        self.assertTrue(
            all(call[1] == {"add_special_tokens": False} for call in tokenizer.calls)
        )

    def test_target_wrapper_and_training_path_install_guards(self):
        from alis_dwq import layerwise, run

        sequence = []

        class Guard:
            def __init__(self, phase, recommended, **kwargs):
                sequence.append(("guard", phase, recommended, kwargs))

            def start(self):
                sequence.append(("start",))

            def check(self, checkpoint, **context):
                sequence.append(("check", checkpoint, context))

        model = SimpleNamespace(parameters=lambda: {"weight": "synthetic"})
        group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
        fake_mx = SimpleNamespace(
            distributed=SimpleNamespace(init=lambda: group),
            eval=lambda value: sequence.append(("eval", value)),
        )

        def fake_iterate(*args, **kwargs):
            del args, kwargs
            yield "batch-0"
            yield "batch-1"

        def fake_compute(
            _model,
            save_dir,
            train_data,
            valid_data,
            batch_size,
            max_seq_length,
            seed,
        ):
            del train_data, valid_data, batch_size, max_seq_length, seed
            for split in ("valid", "train"):
                for index, _ in enumerate(run.D.iterate_batches([])):
                    sequence.append(("compute", split, index))
                    split_dir = Path(save_dir) / split
                    split_dir.mkdir(parents=True, exist_ok=True)
                    (split_dir / f"{index:010d}.safetensors").write_bytes(
                        f"{split}-{index}".encode()
                    )

        def write_contract(directory, contract):
            del contract
            path = Path(directory) / "target-contract.json"
            path.write_text("{}\n")
            return path

        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "targets"
            with (
                mock.patch.object(
                    run,
                    "configure_recommended_wired_limit",
                    side_effect=lambda phase: sequence.append(("wired", phase)) or 498,
                ),
                mock.patch.object(run, "MemoryGuard", Guard),
                mock.patch.object(run, "mx", fake_mx),
                mock.patch.object(run, "_orig_iterate_batches", fake_iterate),
                mock.patch.object(run, "_orig_compute", side_effect=fake_compute),
                mock.patch.object(run, "build_target_contract", return_value={}),
                mock.patch.object(
                    run, "write_contract_no_replace", side_effect=write_contract
                ),
                mock.patch.object(
                    run, "_teacher_identity", return_value=("teacher", "revision")
                ),
                mock.patch.object(
                    run, "_validate_target_publish_inputs"
                ) as validate_publish,
                mock.patch.object(
                    run,
                    "_RUN_CONTEXT",
                    {"teacher_checkpoint_digest": "a" * 64},
                ),
                mock.patch.object(run, "_ACTIVE_DATA_BINDING", {"splits": {}}),
                mock.patch.object(run, "_TARGET_CONTRACT_DIGEST", None),
                mock.patch.object(run, "_TARGET_CONTRACT_PATH", None),
                mock.patch.dict(os.environ, {"ALIS_DWQ_RUN_ID": "test-run"}),
            ):
                run._wired_compute(model, final, [], [], 1, 512, 7)
                failed_final = Path(directory) / "targets-mutated-teacher"
                validate_publish.side_effect = RuntimeError(
                    "teacher checkpoint changed during target computation"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "changed during target computation"
                ):
                    run._wired_compute(
                        model, failed_final, [], [], 1, 512, 7
                    )
            self.assertTrue((final / "target-contract.json").is_file())
            self.assertFalse(failed_final.exists())

        self.assertLess(
            sequence.index(("wired", "target-computation")),
            sequence.index(("eval", {"weight": "synthetic"})),
        )
        after_batches = [
            item
            for item in sequence
            if item[0:2] == ("check", "after-target-batch")
        ]
        # Four batches were checked in both the published run and the run whose
        # mutated teacher was rejected at the final publication boundary.
        self.assertEqual(len(after_batches), 8)
        self.assertIn(("check", "after-target-dump", {}), sequence)

        source = inspect.getsource(layerwise.layerwise_dwq_quantize)
        self.assertIn('memory_phase = "precomputed-target-training"', source)
        self.assertIn("configure_recommended_wired_limit(memory_phase)", source)
        self.assertIn("memory_guard.begin_round", source)
        self.assertIn("check_round_or_restore", source)
        self.assertIn('_ignored.pop("_alis_memory_guard", None)', source)

    def test_run_hands_preload_guard_into_layerwise_quantizer(self):
        from alis_dwq import run

        sequence = []

        class Guard:
            def check(self, checkpoint):
                sequence.append(("check", checkpoint))

        guard = Guard()

        def quantize(*args, **kwargs):
            sequence.append(("quantize", args, kwargs))
            return "trained"

        wrapped = run._guarded_dwq_quantizer(quantize, guard)
        self.assertEqual(wrapped("student", seed=7), "trained")
        self.assertEqual(sequence[0], ("check", "before-upstream-dwq-training"))
        self.assertIs(sequence[1][2]["_alis_memory_guard"], guard)


if __name__ == "__main__":
    unittest.main()

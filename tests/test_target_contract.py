import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from alis_dwq import target_contract
from alis_dwq.target_contract import (
    CONTRACT_NAME,
    build_target_contract,
    canonical_json_bytes,
    preflight_backfill_target_dir,
    prepare_local_data,
    sha256_file,
    validate_target_contract,
    write_contract_no_replace,
)


class FakeTokenizer:
    vocab_size = 2048

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("preformatted rows must not add special tokens")
        return [ord(character) for character in text]


class FakeTextDataset:
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.rows[index]

    def process(self, row):
        return self.tokenizer.encode(row["text"], add_special_tokens=False), 0


class TargetContractTests(unittest.TestCase):
    def test_backfill_preflight_rejects_empty_and_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            targets = Path(directory) / "targets"
            targets.mkdir()
            with self.assertRaisesRegex(ValueError, "split is missing"):
                preflight_backfill_target_dir(targets)
            for split in ("train", "valid"):
                split_dir = targets / split
                split_dir.mkdir()
                (split_dir / "0000000000.safetensors").write_bytes(b"target")
            self.assertEqual(preflight_backfill_target_dir(targets), targets)
            (targets / CONTRACT_NAME).write_text("{}\n")
            with self.assertRaisesRegex(FileExistsError, "no-clobber"):
                preflight_backfill_target_dir(targets)

    def test_backfill_rejects_partial_targets_before_teacher_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir, tokenizer_dir, targets, _ = self._fixture(Path(directory))
            for split in ("train", "valid"):
                for path in sorted((targets / split).glob("*.safetensors"))[1:]:
                    path.unlink()
            digest = mock.Mock(side_effect=AssertionError("teacher hash started"))
            with (
                mock.patch("mlx_lm.utils.load_tokenizer", return_value=FakeTokenizer()),
                mock.patch("mlx_lm.tuner.datasets.TextDataset", FakeTextDataset),
                mock.patch("alis_dwq.io_utils.directory_digest", digest),
                self.assertRaisesRegex(ValueError, "target file set mismatch"),
            ):
                target_contract.main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "--tokenizer",
                        str(tokenizer_dir),
                        "--target-dir",
                        str(targets),
                        "--teacher-identity",
                        "teacher",
                        "--teacher-revision",
                        "revision",
                        "--num-samples",
                        "4",
                        "--num-valid-samples",
                        "2",
                        "--max-seq-length",
                        "8",
                        "--batch-size",
                        "1",
                        "--seed",
                        "7",
                        "--tokenization",
                        "preformatted_chat",
                    ]
                )
            digest.assert_not_called()

    @staticmethod
    def _write_target(
        path: Path,
        split: str,
        index: int,
        *,
        replacement=False,
        sequence_length=7,
        metadata=None,
    ):
        marker = index + (100 if split == "valid" else 1)
        if replacement:
            marker += 1000
        logits = np.full(
            sequence_length * 1024, 0x3F80 + marker, dtype="<u2"
        ).tobytes()
        indices = np.tile(
            np.arange(1024, dtype="<u4"), sequence_length
        ).tobytes()
        header = {
            "logits": {
                "dtype": "BF16",
                "shape": [1, sequence_length, 1024],
                "data_offsets": [0, len(logits)],
            },
            "indices": {
                "dtype": "U32",
                "shape": [1, sequence_length, 1024],
                "data_offsets": [len(logits), len(logits) + len(indices)],
            },
        }
        if metadata is not None:
            header["__metadata__"] = metadata
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + logits + indices)

    def _laguna_data_fixture(
        self,
        root: Path,
        *,
        file_identity: bool = False,
        split_counts: dict[str, int] | None = None,
    ):
        data_dir = root / "laguna-data"
        tokenizer_dir = root / "runtime-tokenizer"
        data_dir.mkdir()
        tokenizer_dir.mkdir()
        runtime_bytes = b'{"runtime": "fixed-mistral-regex"}\n'
        (tokenizer_dir / "tokenizer.json").write_bytes(runtime_bytes)
        runtime_hash = hashlib.sha256(runtime_bytes).hexdigest()
        source_hash = (
            runtime_hash
            if file_identity
            else hashlib.sha256(b"source tokenizer bytes").hexdigest()
        )
        split_counts = split_counts or {"train": 80, "valid": 40, "heldout": 100}
        summaries = {}
        for split, count in split_counts.items():
            rows = []
            ordered_hashes = []
            for index in range(count):
                text = f"{split}-{index:03d}"
                token_ids = [ord(character) for character in text]
                token_hash = hashlib.sha256(
                    canonical_json_bytes(token_ids)
                ).hexdigest()
                rows.append(
                    {
                        "text": text,
                        "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "eval_sequence_tokens": len(token_ids),
                        "token_ids_sha256": token_hash,
                    }
                )
                ordered_hashes.append(token_hash)
            (data_dir / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            summaries[split] = {
                "row_count": count,
                "ordered_token_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(ordered_hashes)
                ).hexdigest(),
            }
        manifest = {
            "format_version": 2,
            "chat_template": "Laguna-S-2.1 local tokenizer",
            "tokenizer_files_sha256": {"tokenizer.json": source_hash},
            "tokenizer_options": {"fix_mistral_regex": True},
            "tokenization_contract": {
                "name": "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat",
                "preformatted_chat": True,
                "add_special_tokens": False,
                "append_eos": False,
            },
            "token_id_hashes": {
                "schema": "sha256-canonical-json-token-ids/v1",
                "field": "token_ids_sha256",
                "tokenization": {
                    "name": "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat",
                    "preformatted_chat": True,
                    "add_special_tokens": False,
                    "append_eos": False,
                },
                "all_rows_verified": True,
                "splits": summaries,
            },
        }
        (data_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n"
        )
        return data_dir, tokenizer_dir, manifest

    def _prepare_laguna(self, data_dir: Path, tokenizer_dir: Path):
        return prepare_local_data(
            FakeTokenizer(),
            data_dir,
            tokenizer_path=tokenizer_dir,
            num_samples=1,
            num_valid_samples=1,
            max_seq_length=32,
            seed=7,
            tokenization="preformatted_chat",
            text_dataset_factory=FakeTextDataset,
        )

    def _fixture(self, root: Path):
        data_dir = root / "data"
        tokenizer_dir = root / "tokenizer"
        targets = root / "targets"
        data_dir.mkdir()
        tokenizer_dir.mkdir()
        targets.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text('{"version": 1}\n')
        tokenizer_hashes = {
            "tokenizer.json": sha256_file(tokenizer_dir / "tokenizer.json")
        }
        (data_dir / "manifest.json").write_text(
            json.dumps(
                {"format_version": 1, "tokenizer_files_sha256": tokenizer_hashes},
                sort_keys=True,
            )
            + "\n"
        )
        rows = {
            "train": [
                {
                    "text": text,
                    "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "eval_sequence_tokens": len(text),
                    "token_ids_sha256": hashlib.sha256(
                        canonical_json_bytes([ord(character) for character in text])
                    ).hexdigest(),
                }
                for text in ("aaaa", "b", "ccc", "dd")
            ],
            "valid": [
                {
                    "text": text,
                    "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "eval_sequence_tokens": len(text),
                    "token_ids_sha256": hashlib.sha256(
                        canonical_json_bytes([ord(character) for character in text])
                    ).hexdigest(),
                }
                for text in ("vvv", "w")
            ],
        }
        for split, split_rows in rows.items():
            (data_dir / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows)
            )
        _, _, binding = prepare_local_data(
            FakeTokenizer(),
            data_dir,
            tokenizer_path=tokenizer_dir,
            num_samples=4,
            num_valid_samples=2,
            max_seq_length=8,
            seed=7,
            tokenization="preformatted_chat",
            text_dataset_factory=FakeTextDataset,
        )
        for split, count in (("train", 4), ("valid", 2)):
            split_dir = targets / split
            split_dir.mkdir()
            for index in range(count):
                self._write_target(
                    split_dir / f"{index:010d}.safetensors", split, index
                )
        return data_dir, tokenizer_dir, targets, binding

    def test_laguna_source_runtime_tokenizer_equivalence_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, tokenizer, manifest = self._laguna_data_fixture(root)
            _, _, binding = self._prepare_laguna(data, tokenizer)
            equivalence = binding["tokenizer_equivalence"]
            self.assertEqual(
                set(equivalence),
                {
                    "schema",
                    "mode",
                    "source_tokenizer_files_sha256",
                    "source_tokenizer_options",
                    "runtime_tokenizer_files_sha256",
                    "verified_splits",
                    "all_rows_verified",
                },
            )
            self.assertEqual(
                equivalence["schema"], "alis-dwq.tokenizer-equivalence/v1"
            )
            self.assertEqual(
                equivalence["mode"], "all-declared-row-token-ids"
            )
            self.assertEqual(
                equivalence["source_tokenizer_files_sha256"],
                manifest["tokenizer_files_sha256"],
            )
            self.assertEqual(
                equivalence["source_tokenizer_options"],
                {"fix_mistral_regex": True},
            )
            self.assertEqual(
                equivalence["runtime_tokenizer_files_sha256"],
                binding["tokenizer_files_sha256"],
            )
            self.assertEqual(
                equivalence["verified_splits"],
                manifest["token_id_hashes"]["splits"],
            )
            self.assertEqual(
                {
                    split: row["row_count"]
                    for split, row in equivalence["verified_splits"].items()
                },
                {"train": 80, "valid": 40, "heldout": 100},
            )
            self.assertTrue(equivalence["all_rows_verified"])

            targets = root / "targets"
            for split in ("train", "valid"):
                split_dir = targets / split
                split_dir.mkdir(parents=True, exist_ok=True)
                self._write_target(
                    split_dir / "0000000000.safetensors",
                    split,
                    0,
                    sequence_length=31,
                )
            contract = build_target_contract(
                binding,
                targets,
                run_id="runtime-tokenizer-equivalence",
                teacher_identity="teacher",
                teacher_revision="revision",
                teacher_checkpoint_digest="a" * 64,
                max_seq_length=32,
                batch_size=1,
                top_k=1024,
                seed=7,
            )
            self.assertEqual(contract["tokenizer_equivalence"], equivalence)
            self.assertEqual(
                contract["tokenizer_files_sha256"],
                equivalence["runtime_tokenizer_files_sha256"],
            )

    def test_laguna_matching_tokenizer_files_uses_identity_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, _ = self._laguna_data_fixture(
                Path(directory), file_identity=True
            )
            _, _, binding = self._prepare_laguna(data, tokenizer)
            equivalence = binding["tokenizer_equivalence"]
            self.assertEqual(equivalence["mode"], "file-identity")
            self.assertEqual(
                equivalence["source_tokenizer_files_sha256"],
                equivalence["runtime_tokenizer_files_sha256"],
            )

    def test_laguna_equivalence_requires_exact_release_split_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, _ = self._laguna_data_fixture(
                Path(directory),
                split_counts={"train": 1, "valid": 1, "heldout": 1},
            )
            with self.assertRaisesRegex(
                ValueError, "exactly 80 train, 40 valid, and 100 heldout"
            ):
                self._prepare_laguna(data, tokenizer)

    def test_laguna_tokenizer_options_are_exact_and_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, manifest = self._laguna_data_fixture(Path(directory))
            cases = (
                None,
                {"fix_mistral_regex": False},
                {"fix_mistral_regex": 1},
                {"fix_mistral_regex": True, "extra": False},
            )
            for options in cases:
                with self.subTest(options=options):
                    changed = dict(manifest)
                    if options is None:
                        changed.pop("tokenizer_options")
                    else:
                        changed["tokenizer_options"] = options
                    (data / "manifest.json").write_text(
                        json.dumps(changed, sort_keys=True) + "\n"
                    )
                    with self.assertRaisesRegex(ValueError, "tokenizer_options"):
                        self._prepare_laguna(data, tokenizer)

    def test_laguna_source_tokenizer_hashes_are_required_and_well_formed(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, manifest = self._laguna_data_fixture(Path(directory))
            cases = (None, {}, {"tokenizer.json": "not-a-sha256"})
            for source_hashes in cases:
                with self.subTest(source_hashes=source_hashes):
                    changed = dict(manifest)
                    if source_hashes is None:
                        changed.pop("tokenizer_files_sha256")
                    else:
                        changed["tokenizer_files_sha256"] = source_hashes
                    (data / "manifest.json").write_text(
                        json.dumps(changed, sort_keys=True) + "\n"
                    )
                    with self.assertRaisesRegex(
                        ValueError, "invalid source tokenizer hashes"
                    ):
                        self._prepare_laguna(data, tokenizer)

    def test_laguna_all_splits_and_unselected_rows_are_retokenized(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, _ = self._laguna_data_fixture(Path(directory))
            originals = {
                split: (data / f"{split}.jsonl").read_text()
                for split in ("train", "valid", "heldout")
            }
            for split in ("train", "valid", "heldout"):
                with self.subTest(split=split):
                    path = data / f"{split}.jsonl"
                    rows = [json.loads(line) for line in originals[split].splitlines()]
                    rows[-1]["token_ids_sha256"] = "0" * 64
                    path.write_text(
                        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
                    )
                    with self.assertRaisesRegex(
                        ValueError, f"{split} row .* token_ids_sha256 mismatch"
                    ):
                        self._prepare_laguna(data, tokenizer)
                    path.write_text(originals[split])

            heldout = data / "heldout.jsonl"
            rows = [json.loads(line) for line in originals["heldout"].splitlines()]
            rows[-1]["eval_sequence_tokens"] += 1
            heldout.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(
                ValueError, "heldout row .* eval_sequence_tokens mismatch"
            ):
                self._prepare_laguna(data, tokenizer)

    def test_format_v2_cannot_downgrade_to_generic_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, manifest = self._laguna_data_fixture(Path(directory))
            for chat_template in (None, "wrong-template"):
                with self.subTest(chat_template=chat_template):
                    changed = dict(manifest)
                    if chat_template is None:
                        changed.pop("chat_template")
                    else:
                        changed["chat_template"] = chat_template
                    (data / "manifest.json").write_text(
                        json.dumps(changed, sort_keys=True) + "\n"
                    )
                    with self.assertRaisesRegex(ValueError, "chat_template"):
                        self._prepare_laguna(data, tokenizer)

    def test_non_laguna_tokenizer_hash_mismatch_stays_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer, _, _ = self._fixture(Path(directory))
            (tokenizer / "tokenizer.json").write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "differs from live tokenizer"):
                prepare_local_data(
                    FakeTokenizer(),
                    data,
                    tokenizer_path=tokenizer,
                    num_samples=4,
                    num_valid_samples=2,
                    max_seq_length=8,
                    seed=7,
                    tokenization="preformatted_chat",
                    text_dataset_factory=FakeTextDataset,
                )

    def test_contract_binds_actual_order_tokens_and_target_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, binding = self._fixture(Path(directory))
            for split in ("train", "valid"):
                (targets / split / "._0000000000.safetensors").write_bytes(
                    b"macOS AppleDouble metadata"
                )
            contract = build_target_contract(
                binding,
                targets,
                run_id="run-1",
                teacher_identity="poolside/Laguna-S-2.1",
                teacher_revision="revision-1",
                teacher_checkpoint_digest="a" * 64,
                max_seq_length=8,
                batch_size=1,
                top_k=1024,
                seed=7,
            )
            path = write_contract_no_replace(targets, contract)
            checked, digest = validate_target_contract(
                binding,
                targets,
                max_seq_length=8,
                batch_size=1,
                top_k=1024,
                seed=7,
            )

            self.assertEqual(checked, contract)
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(
                sorted(row["data_index"] for row in contract["splits"]["train"]["rows"]),
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [row["data_index"] for row in contract["splits"]["valid"]["rows"]],
                [1, 0],
            )
            expected_rows_digest = hashlib.sha256(
                canonical_json_bytes(contract["splits"]["train"]["rows"])
            ).hexdigest()
            self.assertEqual(
                contract["splits"]["train"]["ordered_rows_sha256"],
                expected_rows_digest,
            )
            with self.assertRaisesRegex(ValueError, "do not match target contract"):
                validate_target_contract(
                    binding,
                    targets,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                    teacher_checkpoint_digest="f" * 64,
                )
            with self.assertRaises(FileExistsError):
                write_contract_no_replace(targets, contract)

    def test_reuse_rejects_replaced_target_and_reordered_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir, tokenizer_dir, targets, binding = self._fixture(Path(directory))
            contract = build_target_contract(
                binding,
                targets,
                run_id="run-2",
                teacher_identity="teacher",
                teacher_revision="revision",
                teacher_checkpoint_digest="b" * 64,
                max_seq_length=8,
                batch_size=1,
                top_k=1024,
                seed=7,
            )
            write_contract_no_replace(targets, contract)

            self._write_target(
                targets / "train" / "0000000000.safetensors",
                "train",
                0,
                replacement=True,
            )
            with self.assertRaisesRegex(ValueError, "do not match target contract"):
                validate_target_contract(
                    binding,
                    targets,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )

            self._write_target(
                targets / "train" / "0000000000.safetensors", "train", 0
            )
            lines = (data_dir / "train.jsonl").read_text().splitlines()
            (data_dir / "train.jsonl").write_text("\n".join(reversed(lines)) + "\n")
            _, _, reordered = prepare_local_data(
                FakeTokenizer(),
                data_dir,
                tokenizer_path=tokenizer_dir,
                num_samples=4,
                num_valid_samples=2,
                max_seq_length=8,
                seed=7,
                tokenization="preformatted_chat",
                text_dataset_factory=FakeTextDataset,
            )
            with self.assertRaisesRegex(ValueError, "do not match target contract"):
                validate_target_contract(
                    reordered,
                    targets,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )

            corrupted = [json.loads(line) for line in lines]
            corrupted[0]["token_ids_sha256"] = "0" * 64
            (data_dir / "train.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in corrupted)
            )
            with self.assertRaisesRegex(ValueError, "token_ids_sha256 mismatch"):
                prepare_local_data(
                    FakeTokenizer(),
                    data_dir,
                    tokenizer_path=tokenizer_dir,
                    num_samples=4,
                    num_valid_samples=2,
                    max_seq_length=8,
                    seed=7,
                    tokenization="preformatted_chat",
                    text_dataset_factory=FakeTextDataset,
                )

    def test_missing_or_extra_target_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, binding = self._fixture(Path(directory))
            (targets / "valid" / "0000000002.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "target file set mismatch"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            (targets / "valid" / "0000000002.safetensors").unlink()
            first = targets / "valid" / "0000000000.safetensors"
            first.write_bytes(b"not-a-safetensor")
            with self.assertRaisesRegex(ValueError, "safetensors header"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            self._write_target(first, "valid", 0, metadata={"invalid": 1})
            with self.assertRaisesRegex(ValueError, "metadata must map"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            self._write_target(first, "valid", 0, sequence_length=1)
            with self.assertRaisesRegex(ValueError, "exact batch/sequence/top-k"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            save_file(
                {
                    "logits": np.zeros((1, 7, 1024), dtype=np.float32),
                    "indices": np.tile(
                        np.arange(1024, dtype=np.int32), (1, 7, 1)
                    ),
                },
                first,
            )
            with self.assertRaisesRegex(ValueError, "finite and not entirely zero"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            save_file(
                {
                    "logits": np.ones((1, 7, 1024), dtype=np.float32),
                    "indices": np.full((1, 7, 1024), 2048, dtype=np.int32),
                },
                first,
            )
            with self.assertRaisesRegex(ValueError, "inside vocabulary"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            save_file(
                {
                    "logits": np.ones((1, 7, 1024), dtype=np.float32),
                    "indices": np.ones((1, 7, 1024), dtype=np.int32),
                },
                first,
            )
            with self.assertRaisesRegex(ValueError, "unique for every token"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            first.unlink()
            self._write_target(first, "valid", 0)
            first.unlink()
            first.symlink_to("0000000001.safetensors")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )
            first.unlink()
            self._write_target(first, "valid", 0)
            valid_dir = targets / "valid"
            real_valid_dir = targets / "valid-real"
            valid_dir.rename(real_valid_dir)
            valid_dir.symlink_to(real_valid_dir.name, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "split directory"):
                build_target_contract(
                    binding,
                    targets,
                    run_id="run-3",
                    teacher_identity="teacher",
                    teacher_revision="revision",
                    teacher_checkpoint_digest="c" * 64,
                    max_seq_length=8,
                    batch_size=1,
                    top_k=1024,
                    seed=7,
                )

    def test_contract_filename_is_stable(self):
        self.assertEqual(CONTRACT_NAME, "target-contract.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            tokenizer = root / "tokenizer"
            data.mkdir()
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}\n")
            row = {
                "text": "xx",
                "raw_sha256": hashlib.sha256(b"xx").hexdigest(),
                "eval_sequence_tokens": 2,
                "token_ids_sha256": hashlib.sha256(
                    canonical_json_bytes([ord("x"), ord("x")])
                ).hexdigest(),
            }
            for split in ("train", "valid", "heldout"):
                (data / f"{split}.jsonl").write_text(
                    json.dumps(row, sort_keys=True) + "\n"
                )
            (data / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "chat_template": "Laguna-S-2.1 local tokenizer",
                        "datasets": {},
                        "mixes": {"heldout": {}},
                        "tokenizer_files_sha256": {
                            "tokenizer.json": sha256_file(
                                tokenizer / "tokenizer.json"
                            )
                        },
                        "tokenizer_options": {"fix_mistral_regex": True},
                        "tokenization_contract": {
                            "name": (
                                "ALIS_DWQ_TEXT_TOKENIZATION=preformatted_chat"
                            ),
                            "preformatted_chat": True,
                            "add_special_tokens": False,
                            "append_eos": False,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "token_id_hashes evidence"):
                prepare_local_data(
                    FakeTokenizer(),
                    data,
                    tokenizer_path=tokenizer,
                    num_samples=1,
                    num_valid_samples=1,
                    max_seq_length=8,
                    seed=7,
                    tokenization="preformatted_chat",
                    text_dataset_factory=FakeTextDataset,
                )

            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["tokenization_contract"]
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "exact tokenization contract"):
                prepare_local_data(
                    FakeTokenizer(),
                    data,
                    tokenizer_path=tokenizer,
                    num_samples=1,
                    num_valid_samples=1,
                    max_seq_length=8,
                    seed=7,
                    tokenization="preformatted_chat",
                    text_dataset_factory=FakeTextDataset,
                )

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            for raw, message in (
                ('{"schema":"one","schema":"two"}\n', "duplicate JSON key"),
                ('{"value":NaN}\n', "non-finite JSON number"),
            ):
                with self.subTest(raw=raw):
                    path.write_text(raw)
                    with self.assertRaisesRegex(ValueError, message):
                        target_contract.load_json(path)


if __name__ == "__main__":
    unittest.main()

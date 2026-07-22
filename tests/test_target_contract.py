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
                {"format_version": 2, "tokenizer_files_sha256": tokenizer_hashes},
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


if __name__ == "__main__":
    unittest.main()

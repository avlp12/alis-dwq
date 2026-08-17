import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from alis_dwq.io_utils import artifact_file_manifest, directory_digest, move_no_replace


class TestMoveNoReplace(unittest.TestCase):
    def test_moves_file_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_source = root / "file-source"
            file_destination = root / "file-destination"
            file_source.write_text("content")
            move_no_replace(file_source, file_destination)
            self.assertEqual(file_destination.read_text(), "content")
            self.assertFalse(file_source.exists())

            dir_source = root / "dir-source"
            dir_destination = root / "dir-destination"
            dir_source.mkdir()
            (dir_source / "content").write_text("content")
            move_no_replace(dir_source, dir_destination)
            self.assertEqual((dir_destination / "content").read_text(), "content")
            self.assertFalse(dir_source.exists())

    def test_refuses_existing_file_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("file", "directory"):
                with self.subTest(kind=kind):
                    source = root / f"{kind}-source"
                    destination = root / f"{kind}-destination"
                    if kind == "file":
                        source.write_text("source")
                        destination.write_text("kept")
                    else:
                        source.mkdir()
                        destination.mkdir()
                        (source / "source").write_text("source")
                        (destination / "kept").write_text("kept")
                    with self.assertRaises(FileExistsError):
                        move_no_replace(source, destination)
                    self.assertTrue(source.exists())
                    self.assertTrue(destination.exists())
                    if kind == "file":
                        self.assertEqual(destination.read_text(), "kept")
                    else:
                        self.assertEqual((destination / "kept").read_text(), "kept")

    def test_directory_digest_uses_canonical_regular_file_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.bin").write_bytes(b"b")
            (root / "a.bin").write_bytes(b"aa")
            (root / "._ignored").write_bytes(b"metadata")
            files = artifact_file_manifest(root)
            self.assertEqual([row["path"] for row in files], ["a.bin", "b.bin"])
            expected = hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            self.assertEqual(directory_digest(root), expected)

    def test_directory_digest_rejects_symlinks_and_lfs_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_text("real")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(ValueError, "symlink"):
                directory_digest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pointer").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "a" * 64 + "\nsize 10\n"
            )
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                directory_digest(root)


if __name__ == "__main__":
    unittest.main()

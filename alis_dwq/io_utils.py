"""No-clobber filesystem helpers for immutable build outputs."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
from pathlib import Path

_AT_FDCWD = -2
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 1
_UNSUPPORTED = {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_file_manifest(root: Path) -> list[dict[str, object]]:
    """Return the canonical release file list, rejecting ambiguous artifacts."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"artifact contains a symlink: {relative}")
        if not path.is_file() or any(part.startswith("._") for part in relative.parts):
            continue
        with path.open("rb") as handle:
            if handle.read(len(_LFS_HEADER)) == _LFS_HEADER:
                raise ValueError(f"artifact contains a Git LFS pointer: {relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"artifact contains no regular files: {root}")
    return files


def directory_digest(root: Path) -> str:
    """Hash canonical JSON for every regular non-AppleDouble artifact file."""
    files = artifact_file_manifest(root)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _raise_rename_error(source: Path, destination: Path) -> None:
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), str(destination))
    raise OSError(error, os.strerror(error), f"{source} -> {destination}")


def move_no_replace(source: Path, destination: Path) -> None:
    """Move a file or directory without replacing a pre-existing path.

    APFS and Linux use their native atomic no-replace operation.  macOS ExFAT
    rejects that flag, so the fallback exclusively reserves the destination
    name before replacing only its own empty reservation.  That fallback has
    a brief placeholder visibility window but never clobbers a destination
    that existed when the operation began.
    """
    source = Path(source)
    destination = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()

    if system == "Darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL) == 0:
            return
        if ctypes.get_errno() not in _UNSUPPORTED:
            _raise_rename_error(source, destination)

    if system == "Linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if (
            rename(
                _AT_FDCWD,
                os.fsencode(source),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
            == 0
        ):
            return
        if ctypes.get_errno() not in _UNSUPPORTED:
            _raise_rename_error(source, destination)

    if source.is_dir():
        destination.mkdir()
        try:
            os.rename(source, destination)
        except BaseException:
            try:
                destination.rmdir()
            except OSError:
                pass
            raise
        return

    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        os.rename(source, destination)
    except BaseException:
        try:
            if destination.stat().st_size == 0:
                destination.unlink()
        except OSError:
            pass
        raise

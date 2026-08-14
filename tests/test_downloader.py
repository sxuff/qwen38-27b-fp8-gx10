#!/usr/bin/env python3
"""Bounded tests for downloader content identity, zero files, and locking."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "download.py"
SPEC = importlib.util.spec_from_file_location("public_downloader", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load downloader module")
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        downloader.configure(root, workers=1, reserve_gib=0)
        downloader.SNAPSHOT.mkdir(parents=True)

        expected = b"real!"
        target = downloader.SNAPSHOT / "config.json"
        target.write_bytes(b"junk!")
        row = {
            "name": "config.json",
            "bytes": len(expected),
            "lfs_sha256": None,
            "git_blob_sha1": git_blob(expected),
        }
        assert downloader.verify_existing(row) is None
        assert not target.exists()
        assert len(list(downloader.SNAPSHOT.glob("config.json.corrupt-*"))) == 1

        zero = downloader.SNAPSHOT / "safetensors-md5sum.txt"
        zero.write_bytes(b"must truncate")
        zero_row = {
            "name": "safetensors-md5sum.txt",
            "bytes": 0,
            "lfs_sha256": None,
            "git_blob_sha1": git_blob(b""),
        }
        verified = downloader.download_one(zero_row)
        assert zero.stat().st_size == 0
        assert verified["git_blob_sha1"] == git_blob(b"")

        downloader.acquire_process_lock()
        child = r'''
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("child_downloader", sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.configure(Path(sys.argv[2]),1,0)
try:
 module.acquire_process_lock()
except RuntimeError:
 raise SystemExit(0)
raise SystemExit(1)
'''
        result = subprocess.run(
            [sys.executable, "-c", child, str(MODULE_PATH), str(root)], check=False
        )
        assert result.returncode == 0
    print("DOWNLOADER_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

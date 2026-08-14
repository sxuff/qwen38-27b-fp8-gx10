#!/usr/bin/env python3
"""Verify a model directory byte-for-byte against the measured snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "environment" / "model-verification-report.json",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    report_path = args.reference_report.resolve()
    report = json.loads(report_path.read_text())
    if report.get("repo") != "Qwen/Qwen3.8-27B-FP8":
        raise RuntimeError("reference report has the wrong repository")
    if report.get("revision") != "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a":
        raise RuntimeError("reference report has the wrong revision")
    expected = {row["name"]: row for row in report.get("files") or []}
    if len(expected) != report.get("expected_files") or len(expected) != 81:
        raise RuntimeError("reference report has an invalid file set")
    actual_paths = {}
    for path in model_dir.rglob("*"):
        if path.is_symlink() or path.is_file():
            actual_paths[str(path.relative_to(model_dir))] = path
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise RuntimeError(f"snapshot file set mismatch: missing={missing}, extra={extra}")

    verified = []
    for index, name in enumerate(sorted(expected), 1):
        row = expected[name]
        path = actual_paths[name]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"snapshot entry is not a regular file: {name}")
        if path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"size mismatch for {name}")
        actual_sha = sha256_file(path)
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        verified.append({"name": name, "bytes": row["bytes"], "sha256": actual_sha})
        if index % 10 == 0 or index == len(expected):
            print(f"verified {index}/{len(expected)} files", file=sys.stderr, flush=True)

    canonical = hashlib.sha256(
        json.dumps(verified, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "repo": report["repo"],
        "revision": report["revision"],
        "verified_files": len(verified),
        "verified_bytes": sum(row["bytes"] for row in verified),
        "snapshot_canonical_sha256": canonical,
        "reference_report_sha256": sha256_file(report_path),
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

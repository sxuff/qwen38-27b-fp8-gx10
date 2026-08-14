#!/usr/bin/env python3
"""Verify the curated public receipts, frozen hashes, claims, and privacy boundary."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_frozen_sources() -> None:
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    for relative, expected in prereg["hashes"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing frozen artifact: {relative}")
        require(sha256(path) == expected, f"frozen artifact hash mismatch: {relative}")


def verify_public_rows() -> None:
    allowed_top = {
        "README.md",
        "summary.json",
        "posthoc-diagnostics.json",
        "telemetry-summary.json",
        "private-archive-manifest.json",
        "no-spec",
        "mtp",
    }
    require({path.name for path in (ROOT / "results").iterdir()} == allowed_top, "results allowlist mismatch")
    allowed_arm = {"serial.jsonl", "concurrency.jsonl", "concurrency-cells.json", "run-summary.json"}
    arm_rows = {}
    for arm in ("no-spec", "mtp"):
        arm_root = ROOT / "results" / arm
        require({path.name for path in arm_root.iterdir()} == allowed_arm, f"{arm}: public arm allowlist mismatch")
        serial = rows(arm_root / "serial.jsonl")
        concurrency = rows(arm_root / "concurrency.jsonl")
        measured = serial + concurrency
        require(len(serial) == 12, f"{arm}: expected 12 serial rows")
        require(len(concurrency) == 24, f"{arm}: expected 24 concurrency rows")
        require(len({row['request_id'] for row in measured}) == 36, f"{arm}: duplicate request IDs")
        for row in measured:
            request_id = row["request_id"]
            require(row.get("error") is None, f"{arm}/{request_id}: error row")
            require(row.get("completion_tokens") == 256, f"{arm}/{request_id}: wrong completion count")
            require("raw_sse" not in row, f"{arm}/{request_id}: private raw path leaked")
            require(re.fullmatch(r"[0-9a-f]{64}", row.get("raw_sse_sha256", "")) is not None, f"{arm}/{request_id}: missing raw commitment")
            require(isinstance(row.get("raw_sse_bytes"), int) and row["raw_sse_bytes"] > 0, f"{arm}/{request_id}: invalid raw byte count")
        arm_rows[arm] = {row["request_id"]: row for row in measured}
    require(set(arm_rows["no-spec"]) == set(arm_rows["mtp"]), "request ID sets differ across arms")
    for request_id in arm_rows["no-spec"]:
        left = arm_rows["no-spec"][request_id]
        right = arm_rows["mtp"][request_id]
        require(left["request_hash"] == right["request_hash"], f"request bytes differ: {request_id}")
        require(left["prompt_sha256"] == right["prompt_sha256"], f"prompt bytes differ: {request_id}")


def verify_private_archive_commitment() -> None:
    path = ROOT / "results" / "private-archive-manifest.json"
    value = json.loads(path.read_text())
    files = value.get("files") or []
    canonical = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(value.get("scope") == "private source archive before public curation", "archive scope mismatch")
    require(value.get("file_count") == len(files) == 96, "archive file count mismatch")
    require(value.get("total_bytes") == sum(row["bytes"] for row in files) == 4306709, "archive byte total mismatch")
    require(value.get("canonical_sha256") == canonical, "archive canonical hash mismatch")
    for row in files:
        require(re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None, "invalid archive SHA-256")
    source_verification = json.loads(
        (ROOT / "environment" / "source-archive-final-verification.json").read_text()
    )
    require(source_verification.get("pass") is True, "source archive verification did not pass")
    require(all(arm.get("raw_sse_complete") for arm in source_verification["arms"].values()), "source raw SSE was incomplete")


def verify_regenerated_outputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        shutil.copytree(ROOT / "results", target / "results")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "analyze.py"), str(target)],
            text=True,
            capture_output=True,
            check=False,
        )
        require(result.returncode == 0, f"analysis regeneration failed: {result.stderr}")
        regenerated = json.loads((target / "results" / "summary.json").read_text())
        published = json.loads((ROOT / "results" / "summary.json").read_text())
        require(regenerated == published, "published summary differs from regenerated rows")
        require((target / "REPORT.md").read_bytes() == (ROOT / "REPORT.md").read_bytes(), "published report differs from regenerated rows")


def verify_claim_surface() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    by_arm = {arm["arm"]: arm for arm in summary["arms"]}
    readme = (ROOT / "README.md").read_text()
    expected = [
        f"{by_arm['no-spec']['serial'][0]['median_decode_tokens_per_second']:.2f} tok/s",
        f"{by_arm['mtp']['serial'][0]['median_decode_tokens_per_second']:.2f} tok/s",
        f"{by_arm['no-spec']['concurrency'][2]['aggregate_completion_tokens_per_second']:.2f} tok/s",
        f"{by_arm['mtp']['concurrency'][2]['aggregate_completion_tokens_per_second']:.2f} tok/s",
        f"{summary['comparison']['median_decode_speed_ratio_mtp_over_no_spec']:.2f}x",
    ]
    for value in expected:
        require(value in readme, f"README is missing canonical value: {value}")
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if "://" in target or target.startswith("#"):
            continue
        relative = target.split("#", 1)[0]
        require((ROOT / relative).exists(), f"broken local README link: {target}")

    image = ROOT / "assets" / "qwen38-gx10-result-card.png"
    with image.open("rb") as handle:
        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", "invalid result-card PNG")
        require(handle.read(4) == b"\x00\x00\x00\r" and handle.read(4) == b"IHDR", "missing PNG IHDR")
        width, height = struct.unpack(">II", handle.read(8))
    require((width, height) == (1600, 900), "result card must be 1600x900")
    require(sha256(image) == "02b084f928bade5298e401437423526a00cc48bf879e534eb7324f408988a382", "result card hash mismatch")


def verify_privacy() -> None:
    forbidden = (
        b"/home/" + b"sxuf",
        b"100" + b".65.",
        b"github" + b"_pat_",
        b"gh" + b"p_",
        b"gx10" + b"-fe09",
    )
    forbidden_names = {"raw", "logs", "telemetry.jsonl", "smoke.json", "WATCHDOG_STOP"}
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        require(path.name not in forbidden_names, f"private artifact leaked: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix not in {".png", ".pyc"}:
            data = path.read_bytes()
            for marker in forbidden:
                require(marker not in data, f"private marker {marker!r} in {path.relative_to(ROOT)}")


def main() -> int:
    verify_frozen_sources()
    verify_public_rows()
    verify_private_archive_commitment()
    verify_regenerated_outputs()
    verify_claim_surface()
    verify_privacy()
    print(
        "PUBLICATION_VERIFY_OK: 72/72 curated rows, matched requests, regenerated aggregates, "
        "private-archive commitment, frozen hashes, privacy allowlist, 1600x900 card"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

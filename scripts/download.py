#!/usr/bin/env python3
"""Pinned, resumable, hash-verifying Hugging Face snapshot downloader."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
ROOT = Path.home() / "models" / "Qwen3.8-27B-FP8-017b9c7af6b5"
SNAPSHOT = ROOT / "snapshot"
MANIFEST = ROOT / "download-manifest.json"
REPORT = ROOT / "verification-report.json"
STATUS = ROOT / "download-status.json"
COMPLETE = ROOT / "DOWNLOAD_COMPLETE.json"
MAX_WORKERS = 8
RESERVE_BYTES = 20 * 2**30
USER_AGENT = "qwen38-27b-fp8-gx10/1.0"
LOCK = threading.Lock()
PROCESS_LOCK = None
START_MONO = time.monotonic()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=request_headers())
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def resolve_url(name: str) -> str:
    path = "/".join(urllib.parse.quote(part, safe="") for part in name.split("/"))
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}?download=true"


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_missing_size(name: str) -> int:
    request = urllib.request.Request(
        resolve_url(name), headers=request_headers(), method="HEAD"
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError(f"missing API size and HTTP Content-Length for {name}")
    return int(length)


def canonical_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest() -> dict:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO.split("/"))
    api = f"https://huggingface.co/api/models/{encoded_repo}/revision/{REVISION}?blobs=true"
    model = fetch_json(api)
    if model.get("sha") != REVISION:
        raise RuntimeError(f"revision mismatch: expected {REVISION}, got {model.get('sha')}")

    rows_by_name: dict[str, dict] = {}
    for item in model.get("siblings") or []:
        name = item.get("rfilename")
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe filename in model API: {name!r}")
        lfs = item.get("lfs") or {}
        size = item.get("size") or lfs.get("size")
        if size is None:
            size = fetch_missing_size(name)
        row = {
            "name": name,
            "bytes": int(size),
            "lfs_sha256": lfs.get("sha256"),
            "git_blob_sha1": item.get("blobId"),
        }
        if not row["lfs_sha256"] and not row["git_blob_sha1"]:
            raise RuntimeError(f"no enforceable upstream digest for {name}")
        previous = rows_by_name.get(name)
        if previous is not None and previous != row:
            raise RuntimeError(
                f"conflicting duplicate metadata for {name}: {previous!r} versus {row!r}"
            )
        rows_by_name[name] = row
    rows = sorted(rows_by_name.values(), key=lambda row: row["name"])
    if not rows:
        raise RuntimeError("empty repository manifest")
    weight_rows = [row for row in rows if row["name"].endswith((".safetensors", ".bin", ".gguf"))]
    if len(weight_rows) != 66:
        raise RuntimeError(f"expected 66 weight files, found {len(weight_rows)}")
    manifest = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "created_at": model.get("createdAt"),
        "last_modified": model.get("lastModified"),
        "generated_at": now(),
        "max_workers": MAX_WORKERS,
        "files": rows,
        "file_count": len(rows),
        "weight_file_count": len(weight_rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "weight_bytes": sum(row["bytes"] for row in weight_rows),
        "canonical_digest": canonical_digest(rows),
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def file_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def sha256_file(path: Path) -> str:
    return file_digests(path)[0]


def verify_file_digests(path: Path, row: dict) -> tuple[str, str]:
    actual_sha256, actual_git_blob = file_digests(path)
    if row.get("lfs_sha256"):
        if actual_sha256 != row["lfs_sha256"]:
            raise RuntimeError(
                f"{row['name']}: SHA-256 mismatch expected={row['lfs_sha256']} actual={actual_sha256}"
            )
    elif actual_git_blob != row.get("git_blob_sha1"):
        raise RuntimeError(
            f"{row['name']}: Git blob mismatch expected={row.get('git_blob_sha1')} actual={actual_git_blob}"
        )
    return actual_sha256, actual_git_blob


def encoded_resolve_url(name: str) -> str:
    return resolve_url(name)


def write_status(manifest: dict, verified: dict[str, dict], active: dict[str, int], phase: str, error: str | None = None) -> None:
    present = 0
    partial_files = 0
    for row in manifest["files"]:
        final = SNAPSHOT / row["name"]
        part = final.with_name(final.name + ".part")
        if final.exists():
            present += min(final.stat().st_size, row["bytes"])
        elif part.exists():
            present += min(part.stat().st_size, row["bytes"])
            partial_files += 1
    value = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "phase": phase,
        "updated_at": now(),
        "elapsed_seconds": round(time.monotonic() - START_MONO, 3),
        "total_bytes": manifest["total_bytes"],
        "bytes_present": present,
        "verified_bytes": sum(item["bytes"] for item in verified.values()),
        "verified_files": len(verified),
        "total_files": manifest["file_count"],
        "partial_files": partial_files,
        "active": sorted(active),
        "error": error,
    }
    atomic_json(STATUS, value)


def verify_existing(row: dict) -> dict | None:
    path = SNAPSHOT / row["name"]
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        return None
    try:
        actual, actual_git_blob = verify_file_digests(path, row)
    except RuntimeError:
        corrupt = path.with_name(path.name + f".corrupt-{int(time.time())}")
        path.rename(corrupt)
        return None
    return {
        "name": row["name"],
        "bytes": row["bytes"],
        "sha256": actual,
        "expected_lfs_sha256": row.get("lfs_sha256"),
        "git_blob_sha1": actual_git_blob,
        "expected_git_blob_sha1": row.get("git_blob_sha1"),
        "verified_at": now(),
    }


def promote_complete_part(part: Path, final: Path, expected_size: int) -> None:
    if part.is_file() and part.stat().st_size == expected_size:
        try:
            os.replace(part, final)
        except FileNotFoundError:
            pass
    if final.is_file() and final.stat().st_size == expected_size:
        return
    raise RuntimeError(
        f"promotion failed for {final.name}: part_exists={part.exists()} "
        f"final_exists={final.exists()} expected_size={expected_size}"
    )


def download_one(row: dict) -> dict:
    final = SNAPSHOT / row["name"]
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    expected_size = row["bytes"]
    url = encoded_resolve_url(row["name"])

    if expected_size == 0:
        with final.open("wb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if final.stat().st_size != 0:
            raise RuntimeError(f"{row['name']}: zero-byte truncation failed")
        actual, actual_git_blob = verify_file_digests(final, row)
        return {
            "name": row["name"],
            "bytes": 0,
            "sha256": actual,
            "expected_lfs_sha256": row.get("lfs_sha256"),
            "git_blob_sha1": actual_git_blob,
            "expected_git_blob_sha1": row.get("git_blob_sha1"),
            "verified_at": now(),
        }

    for attempt in range(1, 6):
        start = part.stat().st_size if part.exists() else 0
        if start > expected_size:
            part.rename(part.with_name(part.name + f".oversize-{int(time.time())}"))
            start = 0
        if start == expected_size:
            promote_complete_part(part, final, expected_size)
            break
        headers = request_headers()
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab" if start and status == 206 else "wb"
                if start and status != 206:
                    start = 0
                with part.open(mode, buffering=0) as handle:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                    os.fsync(handle.fileno())
            if part.is_file() and part.stat().st_size == expected_size:
                promote_complete_part(part, final, expected_size)
                break
            current_size = part.stat().st_size if part.is_file() else 0
            if current_size > expected_size:
                part.rename(part.with_name(part.name + f".oversize-{int(time.time())}"))
            raise OSError(f"incomplete file: {current_size}/{expected_size}")
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            if attempt == 5:
                raise RuntimeError(f"{row['name']}: {type(exc).__name__}: {exc}") from exc
            time.sleep(min(30, 2**attempt))

    if not final.is_file() or final.stat().st_size != expected_size:
        raise RuntimeError(f"{row['name']}: final size mismatch")
    actual, actual_git_blob = verify_file_digests(final, row)
    return {
        "name": row["name"],
        "bytes": expected_size,
        "sha256": actual,
        "expected_lfs_sha256": row.get("lfs_sha256"),
        "git_blob_sha1": actual_git_blob,
        "expected_git_blob_sha1": row.get("git_blob_sha1"),
        "verified_at": now(),
    }


def save_report(manifest: dict, verified: dict[str, dict], complete: bool) -> None:
    rows = [verified[name] for name in sorted(verified)]
    report = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "manifest_digest": manifest["canonical_digest"],
        "complete": complete,
        "verified_files": len(rows),
        "verified_bytes": sum(row["bytes"] for row in rows),
        "expected_files": manifest["file_count"],
        "expected_bytes": manifest["total_bytes"],
        "updated_at": now(),
        "files": rows,
    }
    atomic_json(REPORT, report)


def configure(root: Path, workers: int, reserve_gib: float) -> None:
    global ROOT, SNAPSHOT, MANIFEST, REPORT, STATUS, COMPLETE, MAX_WORKERS, RESERVE_BYTES
    ROOT = root.expanduser().resolve()
    SNAPSHOT = ROOT / "snapshot"
    MANIFEST = ROOT / "download-manifest.json"
    REPORT = ROOT / "verification-report.json"
    STATUS = ROOT / "download-status.json"
    COMPLETE = ROOT / "DOWNLOAD_COMPLETE.json"
    MAX_WORKERS = workers
    RESERVE_BYTES = int(reserve_gib * 2**30)


def acquire_process_lock() -> None:
    global PROCESS_LOCK
    lock_path = ROOT / ".download.lock"
    PROCESS_LOCK = lock_path.open("a+")
    try:
        fcntl.flock(PROCESS_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another downloader owns {lock_path}") from exc
    PROCESS_LOCK.seek(0)
    PROCESS_LOCK.truncate()
    PROCESS_LOCK.write(f"{os.getpid()}\n")
    PROCESS_LOCK.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Destination root. The verified files are written under ROOT/snapshot.",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Resolve and write the pinned manifest without downloading weights.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be non-negative")
    configure(args.root, args.workers, args.reserve_gib)

    ROOT.mkdir(parents=True, exist_ok=True)
    acquire_process_lock()
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    if args.manifest_only:
        print(json.dumps(manifest, sort_keys=True), flush=True)
        return 0
    if COMPLETE.exists():
        COMPLETE.unlink()

    verified: dict[str, dict] = {}
    for row in manifest["files"]:
        result = verify_existing(row)
        if result:
            verified[row["name"]] = result
    save_report(manifest, verified, complete=False)

    pending = [row for row in manifest["files"] if row["name"] not in verified]
    remaining_bytes = sum(row["bytes"] for row in pending)
    free = shutil.disk_usage(ROOT).free
    if free < remaining_bytes + RESERVE_BYTES:
        raise RuntimeError(
            f"disk gate failed: free={free}, required={remaining_bytes + RESERVE_BYTES}, "
            f"remaining_download={remaining_bytes}, reserve={RESERVE_BYTES}"
        )
    active: dict[str, int] = {}
    write_status(manifest, verified, active, "downloading")
    failure: Exception | None = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        iterator = iter(pending)
        futures: dict[concurrent.futures.Future, dict] = {}
        for _ in range(MAX_WORKERS):
            try:
                row = next(iterator)
            except StopIteration:
                break
            active[row["name"]] = row["bytes"]
            futures[pool.submit(download_one, row)] = row

        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                row = futures.pop(future)
                active.pop(row["name"], None)
                try:
                    result = future.result()
                except Exception as exc:
                    failure = exc
                    break
                with LOCK:
                    verified[result["name"]] = result
                    save_report(manifest, verified, complete=False)
                    print(
                        f"verified {len(verified)}/{manifest['file_count']} "
                        f"{sum(item['bytes'] for item in verified.values())}/{manifest['total_bytes']} "
                        f"{result['name']}",
                        flush=True,
                    )
                    write_status(manifest, verified, active, "downloading")
                try:
                    next_row = next(iterator)
                except StopIteration:
                    continue
                active[next_row["name"]] = next_row["bytes"]
                futures[pool.submit(download_one, next_row)] = next_row
            if failure:
                for future in futures:
                    future.cancel()
                break

    if failure:
        save_report(manifest, verified, complete=False)
        write_status(manifest, verified, active, "failed", str(failure))
        raise failure

    expected = {row["name"] for row in manifest["files"]}
    if set(verified) != expected:
        missing = sorted(expected - set(verified))
        raise RuntimeError(f"verification incomplete, missing: {missing}")
    if sum(item["bytes"] for item in verified.values()) != manifest["total_bytes"]:
        raise RuntimeError("verified byte total does not match manifest")

    save_report(manifest, verified, complete=True)
    completion = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "complete": True,
        "completed_at": now(),
        "file_count": len(verified),
        "verified_bytes": sum(item["bytes"] for item in verified.values()),
        "manifest_digest": manifest["canonical_digest"],
        "report_sha256": sha256_file(REPORT),
        "snapshot": str(SNAPSHOT),
    }
    atomic_json(COMPLETE, completion)
    write_status(manifest, verified, {}, "complete")
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

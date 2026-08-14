#!/usr/bin/env python3
"""Create and verify immutable launch manifests for the owned recipe container."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RECIPE_LABEL = "io.sxuff.qwen38.recipe"
RUN_LABEL = "io.sxuff.qwen38.run"
MODE_LABEL = "io.sxuff.qwen38.mode"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(container_id: str) -> dict | None:
    result = subprocess.run(
        ["docker", "inspect", container_id], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    rows = json.loads(result.stdout)
    if len(rows) != 1:
        raise RuntimeError("docker inspect returned an unexpected container count")
    return rows[0]


def create(args) -> dict:
    model_verification = Path(args.model_verification).resolve()
    model = json.loads(model_verification.read_text())
    value = inspect(args.container_id)
    if value is None:
        raise RuntimeError("created container is missing")
    labels = ((value.get("Config") or {}).get("Labels") or {})
    if labels.get(RECIPE_LABEL) != "true" or labels.get(RUN_LABEL) != args.run_id:
        raise RuntimeError("created container is missing ownership labels")
    if labels.get(MODE_LABEL) != args.mode:
        raise RuntimeError("created container mode label mismatch")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "container_id": args.container_id,
        "container_name": args.container_name,
        "image_reference": args.image_reference,
        "image_id": value.get("Image"),
        "host": args.host,
        "port": args.port,
        "server_args": args.server_arg,
        "model": model,
        "model_verification_sha256": sha256_file(model_verification),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def check(args) -> dict:
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    if args.expected_mode and manifest.get("mode") != args.expected_mode:
        raise RuntimeError(
            f"launch mode mismatch: expected={args.expected_mode}, actual={manifest.get('mode')}"
        )
    value = inspect(manifest["container_id"])
    if value is None:
        if args.allow_missing:
            return {**manifest, "container_status": "missing"}
        raise RuntimeError("launch-manifest container is missing")
    labels = ((value.get("Config") or {}).get("Labels") or {})
    if labels.get(RECIPE_LABEL) != "true":
        raise RuntimeError("container is not recipe-owned")
    if labels.get(RUN_LABEL) != manifest.get("run_id"):
        raise RuntimeError("container run label mismatch")
    if labels.get(MODE_LABEL) != manifest.get("mode"):
        raise RuntimeError("container mode label mismatch")
    if value.get("Image") != manifest.get("image_id"):
        raise RuntimeError("container image identity mismatch")
    running = bool((value.get("State") or {}).get("Running"))
    if not running and not args.allow_not_running:
        raise RuntimeError("owned container is not running")
    return {**manifest, "container_status": "running" if running else "stopped"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--output", required=True)
    make.add_argument("--mode", required=True, choices=["no-spec", "mtp"])
    make.add_argument("--run-id", required=True)
    make.add_argument("--container-id", required=True)
    make.add_argument("--container-name", required=True)
    make.add_argument("--image-reference", required=True)
    make.add_argument("--model-verification", required=True)
    make.add_argument("--host", required=True)
    make.add_argument("--port", required=True, type=int)
    make.add_argument("--server-arg", action="append", default=[])

    verify = sub.add_parser("check")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--expected-mode", choices=["no-spec", "mtp"])
    verify.add_argument("--allow-missing", action="store_true")
    verify.add_argument("--allow-not-running", action="store_true")
    verify.add_argument("--field")
    args = parser.parse_args()

    value = create(args) if args.command == "create" else check(args)
    if getattr(args, "field", None):
        field = args.field
        if field not in value:
            raise RuntimeError(f"manifest field not found: {field}")
        print(value[field])
    else:
        print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

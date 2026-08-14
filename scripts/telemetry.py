#!/usr/bin/env python3
"""Ownership-scoped GX10 telemetry and fail-closed memory watchdog."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RECIPE_LABEL = "io.sxuff.qwen38.recipe"
RUN_LABEL = "io.sxuff.qwen38.run"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0]) * 1024
    return values


def gpu_sample() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip() or 'empty output'}")
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected one GPU sample, found {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {lines[0]!r}")
    return {
        "temperature_c": float(parts[0]),
        "power_w": float(parts[1]),
        "utilization_percent": float(parts[2]),
    }


def inspect_container(container_id: str) -> dict | None:
    result = subprocess.run(
        ["docker", "inspect", container_id], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    values = json.loads(result.stdout)
    if len(values) != 1:
        raise RuntimeError("docker inspect returned an unexpected container count")
    return values[0]


def ownership_ok(value: dict, run_id: str) -> bool:
    labels = ((value.get("Config") or {}).get("Labels") or {})
    return labels.get(RECIPE_LABEL) == "true" and labels.get(RUN_LABEL) == run_id


def stop_owned(container_id: str, run_id: str) -> bool:
    value = inspect_container(container_id)
    if value is None or not ownership_ok(value, run_id):
        return False
    subprocess.run(
        ["docker", "rm", "-f", container_id], capture_output=True, check=False
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mem-floor-bytes", type=int, default=15 * 2**30)
    parser.add_argument("--max-swap-growth-bytes", type=int, default=512 * 2**20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    args = parser.parse_args()

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    samples_path = root / "telemetry.jsonl"
    incident_path = root / "safety-incident.json"
    stop_path = root / "WATCHDOG_STOP"
    base = meminfo()
    swap_baseline = base.get("SwapTotal", 0) - base.get("SwapFree", 0)
    baseline = {
        "timestamp": now(),
        "mem_available_bytes": base.get("MemAvailable"),
        "swap_used_bytes": swap_baseline,
        "mem_floor_bytes": args.mem_floor_bytes,
        "max_swap_growth_bytes": args.max_swap_growth_bytes,
        "container_id": args.container_id,
        "run_id": args.run_id,
    }
    atomic_json(root / "telemetry-baseline.json", baseline)
    started = time.monotonic()
    seen_running = False

    def incident(violation: str, sample: dict | None = None, detail: str | None = None) -> int:
        stopped = stop_owned(args.container_id, args.run_id)
        atomic_json(
            incident_path,
            {
                "violation": violation,
                "detail": detail,
                "sample": sample,
                "baseline": baseline,
                "action": "removed owned container" if stopped else "refused unowned or missing container",
            },
        )
        return 2

    try:
        with samples_path.open("x", encoding="utf-8", buffering=1) as handle:
            while True:
                if stop_path.exists():
                    return 0
                container = inspect_container(args.container_id)
                if container is None:
                    return incident("owned_container_disappeared")
                if not ownership_ok(container, args.run_id):
                    return incident("container_ownership_mismatch")
                running = bool((container.get("State") or {}).get("Running"))
                seen_running = seen_running or running
                if seen_running and not running:
                    return incident("owned_container_stopped_unexpectedly")
                info = meminfo()
                swap_used = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
                sample = {
                    "timestamp": now(),
                    "monotonic_seconds": time.monotonic(),
                    "container_running": running,
                    "mem_available_bytes": info.get("MemAvailable"),
                    "swap_used_bytes": swap_used,
                    "swap_growth_bytes": swap_used - swap_baseline,
                    "loadavg": list(os.getloadavg()),
                    **gpu_sample(),
                }
                handle.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
                if sample["mem_available_bytes"] < args.mem_floor_bytes:
                    return incident("mem_available_below_floor", sample)
                if sample["swap_growth_bytes"] > args.max_swap_growth_bytes:
                    return incident("swap_growth_above_limit", sample)
                if not seen_running and time.monotonic() - started > args.max_wait_seconds:
                    return incident("container_never_started", sample)
                time.sleep(args.interval)
    except Exception as exc:  # noqa: BLE001 - convert any watchdog failure into an incident
        return incident("watchdog_exception", detail=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

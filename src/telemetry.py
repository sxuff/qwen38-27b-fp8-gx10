#!/usr/bin/env python3
"""GX10 telemetry and fail-closed memory watchdog for one owned container."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0]) * 1024
    return values


def gpu_sample() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        return {"temperature_c": None, "power_w": None, "utilization_percent": None}
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    try:
        return {"temperature_c": float(parts[0]), "power_w": float(parts[1]), "utilization_percent": float(parts[2])}
    except Exception:
        return {"temperature_c": None, "power_w": None, "utilization_percent": None}


def container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--container", default="qwen38-bench-server")
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
        "container": args.container,
    }
    atomic_json(root / "telemetry-baseline.json", baseline)

    started = time.monotonic()
    seen_container = False
    with samples_path.open("a", encoding="utf-8", buffering=1) as handle:
        while True:
            if stop_path.exists():
                return 0
            running = container_running(args.container)
            seen_container = seen_container or running
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
            violation = None
            if sample["mem_available_bytes"] < args.mem_floor_bytes:
                violation = "mem_available_below_floor"
            elif sample["swap_growth_bytes"] > args.max_swap_growth_bytes:
                violation = "swap_growth_above_limit"
            if violation:
                incident = {"violation": violation, "sample": sample, "baseline": baseline, "action": f"stopped owned container {args.container}"}
                atomic_json(incident_path, incident)
                subprocess.run(["docker", "stop", "-t", "0", args.container], capture_output=True)
                return 2
            if seen_container and not running:
                return 0
            if not seen_container and time.monotonic() - started > args.max_wait_seconds:
                atomic_json(incident_path, {"violation": "container_never_started", "baseline": baseline})
                return 3
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

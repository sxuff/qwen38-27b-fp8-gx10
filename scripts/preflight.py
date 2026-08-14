#!/usr/bin/env python3
"""Fail-closed host and GPU preflight for the GX10 recipe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


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
            "--query-gpu=name,temperature.gpu,power.draw,utilization.gpu",
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
        raise RuntimeError(f"expected one GPU, found {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {lines[0]!r}")
    return {
        "gpu_name": parts[0],
        "temperature_c": float(parts[1]),
        "power_w": float(parts[2]),
        "utilization_percent": float(parts[3]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-floor-bytes", type=int, default=15 * 2**30)
    args = parser.parse_args()
    info = meminfo()
    available = info.get("MemAvailable")
    if available is None or available < args.mem_floor_bytes:
        raise RuntimeError(
            f"MemAvailable preflight failed: available={available}, floor={args.mem_floor_bytes}"
        )
    swap_used = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    output = {
        "mem_available_bytes": available,
        "swap_used_bytes": swap_used,
        "mem_floor_bytes": args.mem_floor_bytes,
        **gpu_sample(),
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

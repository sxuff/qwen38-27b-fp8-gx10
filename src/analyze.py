#!/usr/bin/env python3
"""Aggregate Qwen3.8 benchmark rows and telemetry into a verified report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def percentile(values, q):
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return None
    position = (len(finite) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    return finite[lower] * (upper - position) + finite[upper] * (position - lower)


def eligible(row: dict) -> bool:
    return (
        not row.get("error")
        and row.get("completion_tokens") == 256
        and row.get("ttft_seconds") is not None
        and row.get("e2e_seconds") is not None
        and row.get("decode_tokens_per_second") is not None
    )


def summarize_telemetry(path: Path) -> dict:
    rows = read_jsonl(path)
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "min_mem_available_bytes": min(row["mem_available_bytes"] for row in rows),
        "max_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows),
        "max_temperature_c": max((row["temperature_c"] for row in rows if row.get("temperature_c") is not None), default=None),
        "mean_power_w": statistics.fmean(row["power_w"] for row in rows if row.get("power_w") is not None),
        "max_power_w": max((row["power_w"] for row in rows if row.get("power_w") is not None), default=None),
        "max_utilization_percent": max((row["utilization_percent"] for row in rows if row.get("utilization_percent") is not None), default=None),
    }


def summarize_arm(root: Path, arm: str) -> dict | None:
    arm_root = root / "results" / arm
    serial = read_jsonl(arm_root / "serial.jsonl")
    concurrency = read_jsonl(arm_root / "concurrency.jsonl")
    if not serial and not concurrency:
        return None
    serial_groups = []
    for target in sorted({row["target_input_tokens"] for row in serial}):
        rows = [row for row in serial if row["target_input_tokens"] == target]
        ok = [row for row in rows if eligible(row)]
        serial_groups.append({
            "target_input_tokens": target,
            "planned": len(rows),
            "eligible": len(ok),
            "median_actual_prompt_tokens": median(row.get("prompt_tokens") for row in ok),
            "median_completion_tokens": median(row.get("completion_tokens") for row in ok),
            "median_ttft_seconds": median(row.get("ttft_seconds") for row in ok),
            "p95_ttft_seconds": percentile([row.get("ttft_seconds") for row in ok], 0.95),
            "median_e2e_seconds": median(row.get("e2e_seconds") for row in ok),
            "median_decode_tokens_per_second": median(row.get("decode_tokens_per_second") for row in ok),
        })
    concurrency_cells_path = arm_root / "concurrency-cells.json"
    cells_raw = json.loads(concurrency_cells_path.read_text()) if concurrency_cells_path.exists() else []
    concurrency_groups = []
    for cell in cells_raw:
        rows = [row for row in concurrency if row["concurrency"] == cell["concurrency"]]
        ok = [row for row in rows if eligible(row)]
        concurrency_groups.append({
            **cell,
            "eligible_requests": len(ok),
            "median_actual_prompt_tokens": median(row.get("prompt_tokens") for row in ok),
            "median_ttft_seconds": median(row.get("ttft_seconds") for row in ok),
            "p95_ttft_seconds": percentile([row.get("ttft_seconds") for row in ok], 0.95),
            "median_e2e_seconds": median(row.get("e2e_seconds") for row in ok),
            "median_decode_tokens_per_second": median(row.get("decode_tokens_per_second") for row in ok),
        })
    return {
        "arm": arm,
        "serial_rows": len(serial),
        "serial_eligible": sum(eligible(row) for row in serial),
        "concurrency_rows": len(concurrency),
        "concurrency_eligible": sum(eligible(row) for row in concurrency),
        "serial": serial_groups,
        "concurrency": concurrency_groups,
        "telemetry": summarize_telemetry(arm_root / "telemetry.jsonl"),
        "safety_incident": (arm_root / "safety-incident.json").exists(),
    }


def compare_arms(root: Path) -> dict | None:
    left = read_jsonl(root / "results" / "no-spec" / "serial.jsonl")
    right = read_jsonl(root / "results" / "mtp" / "serial.jsonl")
    if not left or not right:
        return None
    by_left = {row["request_id"]: row for row in left}
    by_right = {row["request_id"]: row for row in right}
    keys = sorted(set(by_left) & set(by_right))
    pairs = [(by_left[key], by_right[key]) for key in keys if eligible(by_left[key]) and eligible(by_right[key])]
    if not pairs:
        return {"paired_eligible": 0}
    speed_ratios = [r["decode_tokens_per_second"] / l["decode_tokens_per_second"] for l, r in pairs]
    e2e_ratios = [l["e2e_seconds"] / r["e2e_seconds"] for l, r in pairs]
    return {
        "paired_eligible": len(pairs),
        "output_hash_equal": sum(l["response_sha256"] == r["response_sha256"] for l, r in pairs),
        "median_decode_speed_ratio_mtp_over_no_spec": median(speed_ratios),
        "median_e2e_speed_ratio_mtp_over_no_spec": median(e2e_ratios),
    }


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    arms = [summary for arm in ["no-spec", "mtp"] if (summary := summarize_arm(root, arm))]
    comparison = compare_arms(root)
    output = {"schema_version": 1, "arms": arms, "comparison": comparison}
    (root / "results" / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")

    lines = ["# Qwen3.8-27B-FP8 on one GX10", "", "## Scope", "", "Official FP8 checkpoint, pinned SGLang image, text-only serving. Three measured serial repetitions per context target and eight requests per concurrency cell.", ""]
    for arm in arms:
        lines.extend([f"## {arm['arm']}", "", "### Serial context sweep", "", "| Target input | Complete | Actual prompt tokens | Median TTFT | Median decode | Median E2E |", "|---:|---:|---:|---:|---:|---:|"])
        for row in arm["serial"]:
            lines.append(f"| {row['target_input_tokens']:,} | {row['eligible']}/{row['planned']} | {fmt(row['median_actual_prompt_tokens'],0)} | {fmt(row['median_ttft_seconds'])} s | {fmt(row['median_decode_tokens_per_second'])} tok/s | {fmt(row['median_e2e_seconds'])} s |")
        lines.extend(["", "### Concurrency sweep", "", "| Concurrency | Complete | Aggregate output | Median TTFT | p95 TTFT | Median per-request decode |", "|---:|---:|---:|---:|---:|---:|"])
        for row in arm["concurrency"]:
            lines.append(f"| {row['concurrency']} | {row['eligible_requests']}/{row['planned_requests']} | {fmt(row['aggregate_completion_tokens_per_second'])} tok/s | {fmt(row['median_ttft_seconds'])} s | {fmt(row['p95_ttft_seconds'])} s | {fmt(row['median_decode_tokens_per_second'])} tok/s |")
        tele = arm["telemetry"]
        lines.extend(["", "### Telemetry", "", f"- Samples: {tele.get('samples',0)}", f"- Minimum MemAvailable: {fmt((tele.get('min_mem_available_bytes') or 0)/2**30)} GiB", f"- Maximum swap growth: {fmt((tele.get('max_swap_growth_bytes') or 0)/2**20)} MiB", f"- Maximum temperature: {fmt(tele.get('max_temperature_c'))} °C", f"- Mean / max sampled GPU power: {fmt(tele.get('mean_power_w'))} / {fmt(tele.get('max_power_w'))} W", f"- Safety incident: {arm['safety_incident']}", ""])
    if comparison:
        lines.extend(["## MTP matched comparison", "", f"- Paired eligible serial rows: {comparison.get('paired_eligible',0)}", f"- Byte-identical outputs: {comparison.get('output_hash_equal',0)}/{comparison.get('paired_eligible',0)}", f"- Median decode speed ratio, MTP/no-spec: {fmt(comparison.get('median_decode_speed_ratio_mtp_over_no_spec'))}x", f"- Median end-to-end speed ratio, MTP/no-spec: {fmt(comparison.get('median_e2e_speed_ratio_mtp_over_no_spec'))}x", ""])
    lines.extend(["## Evidence boundary", "", "These measurements describe the exact pinned checkpoint, runtime image, launch flags, and GX10 host. They are not model-quality results and do not generalize to BF16, GGUF, other GPUs, or other serving engines.", ""])
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

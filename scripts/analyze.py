#!/usr/bin/env python3
"""Aggregate validated Qwen3.8 benchmark rows with matched-arm identity checks."""

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


def eligible(row: dict, output_tokens: int) -> bool:
    return (
        row.get("error") is None
        and row.get("completion_tokens") == output_tokens
        and isinstance(row.get("usage"), dict)
        and isinstance(row.get("prompt_tokens"), int)
        and all(
            isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            for key in ("ttft_seconds", "e2e_seconds", "decode_tokens_per_second")
        )
    )


def summarize_telemetry(root: Path, arm: str) -> tuple[dict, bool]:
    rows = read_jsonl(root / "results" / arm / "telemetry.jsonl")
    incident = (root / "results" / arm / "safety-incident.json").exists()
    if rows:
        power = [float(row["power_w"]) for row in rows if row.get("power_w") is not None]
        telemetry = {
            "samples": len(rows),
            "min_mem_available_bytes": min(row["mem_available_bytes"] for row in rows),
            "max_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows),
            "max_temperature_c": max(
                (row["temperature_c"] for row in rows if row.get("temperature_c") is not None),
                default=None,
            ),
            "mean_power_w": statistics.fmean(power) if power else None,
            "max_power_w": max(power, default=None),
            "max_utilization_percent": max(
                (row["utilization_percent"] for row in rows if row.get("utilization_percent") is not None),
                default=None,
            ),
        }
        return telemetry, incident
    published = root / "results" / "telemetry-summary.json"
    if not published.exists():
        raise RuntimeError(f"missing telemetry receipts for {arm}")
    value = json.loads(published.read_text()).get(arm)
    if not isinstance(value, dict):
        raise RuntimeError(f"missing published telemetry summary for {arm}")
    return value["telemetry"], bool(value["safety_incident"])


def validate_run_summary(arm_root: Path, serial: list[dict], concurrency: list[dict]) -> dict:
    path = arm_root / "run-summary.json"
    if not path.exists():
        raise RuntimeError(f"missing run summary: {path}")
    run = json.loads(path.read_text())
    output_tokens = run.get("output_tokens")
    if not isinstance(output_tokens, int) or output_tokens <= 1:
        raise RuntimeError("invalid output token contract")
    planned_serial = len(run.get("serial_targets") or []) * int(run.get("repetitions") or 0)
    planned_concurrency = len(run.get("concurrency_levels") or []) * int(
        run.get("requests_per_concurrency") or 0
    )
    if len(serial) != planned_serial or run.get("serial_rows") != planned_serial:
        raise RuntimeError(f"serial denominator mismatch for {run.get('arm')}")
    if len(concurrency) != planned_concurrency or run.get("concurrency_rows") != planned_concurrency:
        raise RuntimeError(f"concurrency denominator mismatch for {run.get('arm')}")
    if sum(eligible(row, output_tokens) for row in serial) != planned_serial:
        raise RuntimeError(f"ineligible serial row for {run.get('arm')}")
    if sum(eligible(row, output_tokens) for row in concurrency) != planned_concurrency:
        raise RuntimeError(f"ineligible concurrency row for {run.get('arm')}")
    identity = run.get("source_identity") or run.get("launch_identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"missing source identity for {run.get('arm')}")
    return run


def summarize_arm(root: Path, arm: str) -> tuple[dict, dict]:
    arm_root = root / "results" / arm
    serial = read_jsonl(arm_root / "serial.jsonl")
    concurrency = read_jsonl(arm_root / "concurrency.jsonl")
    run = validate_run_summary(arm_root, serial, concurrency)
    output_tokens = run["output_tokens"]
    serial_groups = []
    for target in run["serial_targets"]:
        rows = [row for row in serial if row["target_input_tokens"] == target]
        ok = [row for row in rows if eligible(row, output_tokens)]
        serial_groups.append(
            {
                "target_input_tokens": target,
                "planned": len(rows),
                "eligible": len(ok),
                "median_actual_prompt_tokens": median(row.get("prompt_tokens") for row in ok),
                "median_completion_tokens": median(row.get("completion_tokens") for row in ok),
                "median_ttft_seconds": median(row.get("ttft_seconds") for row in ok),
                "p95_ttft_seconds": percentile([row.get("ttft_seconds") for row in ok], 0.95),
                "median_e2e_seconds": median(row.get("e2e_seconds") for row in ok),
                "median_decode_tokens_per_second": median(
                    row.get("decode_tokens_per_second") for row in ok
                ),
            }
        )
    cells_raw = json.loads((arm_root / "concurrency-cells.json").read_text())
    if [cell["concurrency"] for cell in cells_raw] != run["concurrency_levels"]:
        raise RuntimeError(f"concurrency cell set mismatch for {arm}")
    concurrency_groups = []
    for cell in cells_raw:
        rows = [row for row in concurrency if row["concurrency"] == cell["concurrency"]]
        ok = [row for row in rows if eligible(row, output_tokens)]
        concurrency_groups.append(
            {
                **cell,
                "eligible_requests": len(ok),
                "median_actual_prompt_tokens": median(row.get("prompt_tokens") for row in ok),
                "median_ttft_seconds": median(row.get("ttft_seconds") for row in ok),
                "p95_ttft_seconds": percentile([row.get("ttft_seconds") for row in ok], 0.95),
                "median_e2e_seconds": median(row.get("e2e_seconds") for row in ok),
                "median_decode_tokens_per_second": median(
                    row.get("decode_tokens_per_second") for row in ok
                ),
            }
        )
    telemetry, incident = summarize_telemetry(root, arm)
    return (
        {
            "arm": arm,
            "serial_rows": len(serial),
            "serial_eligible": len(serial),
            "concurrency_rows": len(concurrency),
            "concurrency_eligible": len(concurrency),
            "serial": serial_groups,
            "concurrency": concurrency_groups,
            "telemetry": telemetry,
            "safety_incident": incident,
        },
        run,
    )


def comparable_identity(run: dict) -> dict:
    identity = dict(run.get("source_identity") or run.get("launch_identity") or {})
    for key in ("mode", "run_id", "container_id"):
        identity.pop(key, None)
    return identity


def compare_arms(root: Path, runs: dict[str, dict]) -> dict:
    if comparable_identity(runs["no-spec"]) != comparable_identity(runs["mtp"]):
        raise RuntimeError("arm source identities do not match")
    left = read_jsonl(root / "results" / "no-spec" / "serial.jsonl")
    right = read_jsonl(root / "results" / "mtp" / "serial.jsonl")
    by_left = {row["request_id"]: row for row in left}
    by_right = {row["request_id"]: row for row in right}
    if len(by_left) != len(left) or len(by_right) != len(right) or set(by_left) != set(by_right):
        raise RuntimeError("serial request sets are not identical and unique")
    pairs = []
    for key in sorted(by_left):
        left_row, right_row = by_left[key], by_right[key]
        if left_row["request_hash"] != right_row["request_hash"]:
            raise RuntimeError(f"request hash mismatch: {key}")
        if left_row["prompt_sha256"] != right_row["prompt_sha256"]:
            raise RuntimeError(f"prompt hash mismatch: {key}")
        pairs.append((left_row, right_row))
    speed_ratios = [
        right_row["decode_tokens_per_second"] / left_row["decode_tokens_per_second"]
        for left_row, right_row in pairs
    ]
    e2e_ratios = [
        left_row["e2e_seconds"] / right_row["e2e_seconds"]
        for left_row, right_row in pairs
    ]
    return {
        "paired_eligible": len(pairs),
        "output_hash_equal": sum(
            left_row["response_sha256"] == right_row["response_sha256"]
            for left_row, right_row in pairs
        ),
        "median_decode_speed_ratio_mtp_over_no_spec": median(speed_ratios),
        "median_e2e_speed_ratio_mtp_over_no_spec": median(e2e_ratios),
    }


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def number_word(value: int, capitalized: bool = False) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 8: "eight"}
    text = words.get(value, str(value))
    return text.capitalize() if capitalized else text


def render_report(arms: list[dict], runs: dict[str, dict], comparison: dict) -> str:
    first = runs["no-spec"]
    lines = [
        "# Qwen3.8-27B-FP8 on one GX10",
        "",
        "## Scope",
        "",
        (
            "Official FP8 checkpoint, pinned SGLang image, text-only serving. "
            f"{number_word(first['repetitions'], True)} measured serial repetitions per context target "
            f"and {number_word(first['requests_per_concurrency'])} requests per concurrency cell."
        ),
        "",
    ]
    for arm in arms:
        lines.extend(
            [
                f"## {arm['arm']}",
                "",
                "### Serial context sweep",
                "",
                "| Target input | Complete | Actual prompt tokens | Median TTFT | Median decode | Median E2E |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in arm["serial"]:
            lines.append(
                f"| {row['target_input_tokens']:,} | {row['eligible']}/{row['planned']} | "
                f"{fmt(row['median_actual_prompt_tokens'],0)} | {fmt(row['median_ttft_seconds'])} s | "
                f"{fmt(row['median_decode_tokens_per_second'])} tok/s | {fmt(row['median_e2e_seconds'])} s |"
            )
        lines.extend(
            [
                "",
                "### Concurrency sweep",
                "",
                "| Concurrency | Complete | Aggregate output | Median TTFT | p95 TTFT | Median per-request decode |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in arm["concurrency"]:
            lines.append(
                f"| {row['concurrency']} | {row['eligible_requests']}/{row['planned_requests']} | "
                f"{fmt(row['aggregate_completion_tokens_per_second'])} tok/s | "
                f"{fmt(row['median_ttft_seconds'])} s | {fmt(row['p95_ttft_seconds'])} s | "
                f"{fmt(row['median_decode_tokens_per_second'])} tok/s |"
            )
        telemetry = arm["telemetry"]
        lines.extend(
            [
                "",
                "### Telemetry",
                "",
                f"- Samples: {telemetry.get('samples',0)}",
                f"- Minimum MemAvailable: {fmt((telemetry.get('min_mem_available_bytes') or 0)/2**30)} GiB",
                f"- Maximum swap growth: {fmt((telemetry.get('max_swap_growth_bytes') or 0)/2**20)} MiB",
                f"- Maximum temperature: {fmt(telemetry.get('max_temperature_c'))} °C",
                f"- Mean / max sampled GPU power: {fmt(telemetry.get('mean_power_w'))} / {fmt(telemetry.get('max_power_w'))} W",
                f"- Safety incident: {arm['safety_incident']}",
                "",
            ]
        )
    lines.extend(
        [
            "## MTP matched comparison",
            "",
            f"- Paired eligible serial rows: {comparison['paired_eligible']}",
            f"- Byte-identical outputs: {comparison['output_hash_equal']}/{comparison['paired_eligible']}",
            f"- Median decode speed ratio, MTP/no-spec: {fmt(comparison['median_decode_speed_ratio_mtp_over_no_spec'])}x",
            f"- Median end-to-end speed ratio, MTP/no-spec: {fmt(comparison['median_e2e_speed_ratio_mtp_over_no_spec'])}x",
            "",
            "## Evidence boundary",
            "",
            (
                "These measurements describe the exact pinned checkpoint, runtime image, launch flags, "
                "and GX10 host. They are not model-quality results and do not generalize to BF16, GGUF, "
                "other GPUs, or other serving engines."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    arms = []
    runs = {}
    for arm in ("no-spec", "mtp"):
        summary, run = summarize_arm(root, arm)
        arms.append(summary)
        runs[arm] = run
    shared = (
        "output_tokens",
        "repetitions",
        "serial_targets",
        "concurrency_levels",
        "requests_per_concurrency",
    )
    if any(runs["no-spec"].get(key) != runs["mtp"].get(key) for key in shared):
        raise RuntimeError("arm benchmark contracts do not match")
    comparison = compare_arms(root, runs)
    output = {"schema_version": 1, "arms": arms, "comparison": comparison}
    (root / "results" / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (root / "REPORT.md").write_text(render_report(arms, runs, comparison))
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

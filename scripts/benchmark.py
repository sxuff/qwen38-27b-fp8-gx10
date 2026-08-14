#!/usr/bin/env python3
"""Reproducible OpenAI-compatible throughput benchmark for Qwen3.8-27B."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def request_json(url: str, payload: dict | None = None, timeout: int = 3600):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_model(base_url: str) -> str:
    value = request_json(base_url + "/v1/models", timeout=30)
    models = value.get("data") or []
    if len(models) != 1 or not models[0].get("id"):
        raise RuntimeError(f"unexpected /v1/models payload: {value!r}")
    return models[0]["id"]


def smoke(base_url: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return only the integer obtained by multiplying 12 by 17."}],
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    start = time.monotonic()
    value = request_json(base_url + "/v1/chat/completions", payload)
    elapsed = time.monotonic() - start
    choice = (value.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if content.strip() != "204" or choice.get("finish_reason") != "stop":
        raise RuntimeError(f"smoke response was not the exact expected answer: {value!r}")
    return {
        "elapsed_seconds": elapsed,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": value.get("usage"),
        "response": value,
    }


def token_count(base_url: str, model: str, prompt: str) -> int | None:
    payload = {"model": model, "prompt": prompt}
    try:
        value = request_json(base_url + "/tokenize", payload, timeout=120)
    except Exception:  # noqa: BLE001 - tokenization fallback is deliberately broad
        return None
    if isinstance(value.get("count"), int):
        return value["count"]
    if isinstance(value.get("tokens"), list):
        return len(value["tokens"])
    if isinstance(value.get("token_ids"), list):
        return len(value["token_ids"])
    return None


def make_prompt(base_url: str, model: str, target_tokens: int, key: str) -> tuple[str, int | None]:
    prefix = f"Benchmark key {key}. Continue the deterministic sequence.\n"
    unit = " alpha"
    low, high = 0, max(1, target_tokens + 128)
    high_prompt = prefix + unit * high
    high_count = token_count(base_url, model, high_prompt)
    if high_count is None:
        return prefix + unit * target_tokens, None
    while high_count < target_tokens:
        high *= 2
        high_prompt = prefix + unit * high
        high_count = token_count(base_url, model, high_prompt)
        if high_count is None:
            return prefix + unit * target_tokens, None
    best_prompt, best_count = high_prompt, high_count
    while low <= high:
        middle = (low + high) // 2
        prompt = prefix + unit * middle
        count = token_count(base_url, model, prompt)
        if count is None:
            return prefix + unit * target_tokens, None
        if abs(count - target_tokens) < abs(best_count - target_tokens):
            best_prompt, best_count = prompt, count
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return prompt, count
    return best_prompt, best_count


def stream_completion(base_url: str, model: str, prompt: str, request_id: str, max_tokens: int, raw_path: Path) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    request_hash = hashlib.sha256(encoded).hexdigest()
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first_token_at = None
    chunks = []
    usage = None
    finish_reason = None
    raw_lines = []
    error = None
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                raw_lines.append(raw)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    text = choice.get("text") or ""
                    if text and first_token_at is None:
                        first_token_at = time.monotonic()
                    if text:
                        chunks.append(text)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
    except Exception as exc:  # noqa: BLE001 - preserve transport/parser failures as rows
        error = f"{type(exc).__name__}: {exc}"
    end = time.monotonic()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("wb") as handle:
        for line in raw_lines:
            handle.write(line)
    text = "".join(chunks)
    ttft = None if first_token_at is None else first_token_at - start
    e2e = end - start
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    decode_interval = None if ttft is None else max(0.0, e2e - ttft)
    decode_tps = None
    if isinstance(completion_tokens, int) and completion_tokens > 1 and decode_interval and decode_interval > 0:
        decode_tps = (completion_tokens - 1) / decode_interval
    return {
        "request_id": request_id,
        "request_hash": request_hash,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "e2e_seconds": e2e,
        "decode_interval_seconds": decode_interval,
        "decode_tokens_per_second": decode_tps,
        "finish_reason": finish_reason,
        "text_chars": len(text),
        "usage": usage,
        "error": error,
        "raw_sse": str(raw_path),
    }


def successful_row(row: dict, expected_completion_tokens: int) -> bool:
    usage = row.get("usage")
    raw = Path(row.get("raw_sse") or "")
    numeric = ("ttft_seconds", "e2e_seconds", "decode_tokens_per_second")
    return (
        row.get("error") is None
        and isinstance(usage, dict)
        and isinstance(row.get("prompt_tokens"), int)
        and row.get("completion_tokens") == expected_completion_tokens
        and row.get("finish_reason") is not None
        and all(
            isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
            for key in numeric
        )
        and raw.is_file()
        and raw.stat().st_size > 0
    )


def launch_identity(path: Path, arm: str) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("mode") != arm:
        raise RuntimeError(
            f"launch manifest mode mismatch: arm={arm}, manifest={manifest.get('mode')}"
        )
    model = manifest.get("model") or {}
    identity = {
        "mode": manifest.get("mode"),
        "run_id": manifest.get("run_id"),
        "container_id": manifest.get("container_id"),
        "image_reference": manifest.get("image_reference"),
        "image_id": manifest.get("image_id"),
        "model_repo": model.get("repo"),
        "model_revision": model.get("revision"),
        "snapshot_canonical_sha256": model.get("snapshot_canonical_sha256"),
        "model_verification_sha256": manifest.get("model_verification_sha256"),
    }
    if any(value in (None, "") for value in identity.values()):
        raise RuntimeError("launch manifest is missing identity fields")
    return identity, hashlib.sha256(raw).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def run(args) -> dict:
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    identity, manifest_sha256 = launch_identity(Path(args.launch_manifest), args.arm)
    model = get_model(base_url)
    smoke_result = smoke(base_url, model)
    atomic_json(root / "smoke.json", smoke_result)

    serial_rows = []
    for target in args.serial_inputs:
        warm_key = f"serial-{target}-warmup"
        warm_prompt, _ = make_prompt(base_url, model, target, warm_key)
        warm_row = stream_completion(base_url, model, warm_prompt, warm_key, args.output_tokens, root / "raw" / f"{warm_key}.sse")
        if not successful_row(warm_row, args.output_tokens):
            atomic_json(root / "warmup-failure.json", warm_row)
            raise RuntimeError(f"warm-up failed eligibility: {warm_key}")
        for repetition in range(args.repetitions):
            key = f"serial-{target}-r{repetition}"
            prompt, tokenizer_count = make_prompt(base_url, model, target, key)
            row = stream_completion(base_url, model, prompt, key, args.output_tokens, root / "raw" / f"{key}.sse")
            row.update(
                {
                    "family": "serial",
                    "target_input_tokens": target,
                    "tokenizer_count_before_request": tokenizer_count,
                    "repetition": repetition,
                    "concurrency": 1,
                }
            )
            serial_rows.append(row)
            write_jsonl(root / "serial.jsonl", serial_rows)

    planned_serial = len(args.serial_inputs) * args.repetitions
    serial_success = sum(successful_row(row, args.output_tokens) for row in serial_rows)
    if len(serial_rows) != planned_serial or serial_success != planned_serial:
        atomic_json(
            root / "run-failure.json",
            {
                "phase": "serial",
                "planned_rows": planned_serial,
                "actual_rows": len(serial_rows),
                "successful_rows": serial_success,
            },
        )
        raise RuntimeError("serial benchmark failed denominator or eligibility checks")

    concurrency_rows = []
    concurrency_cells = []
    for concurrency in args.concurrency:
        prompts = []
        for index in range(args.requests_per_concurrency):
            key = f"concurrency-{concurrency}-q{index}"
            prompt, tokenizer_count = make_prompt(base_url, model, args.concurrency_input_tokens, key)
            prompts.append((key, prompt, tokenizer_count))
        cell_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    stream_completion,
                    base_url,
                    model,
                    prompt,
                    key,
                    args.output_tokens,
                    root / "raw" / f"{key}.sse",
                ): (key, tokenizer_count)
                for key, prompt, tokenizer_count in prompts
            }
            cell_rows = []
            for future in concurrent.futures.as_completed(futures):
                key, tokenizer_count = futures[future]
                row = future.result()
                row.update(
                    {
                        "family": "concurrency",
                        "target_input_tokens": args.concurrency_input_tokens,
                        "tokenizer_count_before_request": tokenizer_count,
                        "concurrency": concurrency,
                    }
                )
                cell_rows.append(row)
        cell_end = time.monotonic()
        concurrency_rows.extend(sorted(cell_rows, key=lambda row: row["request_id"]))
        completion_total = sum(row.get("completion_tokens") or 0 for row in cell_rows if not row.get("error"))
        concurrency_cells.append({
            "concurrency": concurrency,
            "planned_requests": len(prompts),
            "successful_requests": sum(not row.get("error") for row in cell_rows),
            "wall_seconds": cell_end - cell_start,
            "completion_tokens": completion_total,
            "aggregate_completion_tokens_per_second": completion_total / (cell_end - cell_start),
        })
        write_jsonl(root / "concurrency.jsonl", concurrency_rows)
        atomic_json(root / "concurrency-cells.json", concurrency_cells)
        if any(not successful_row(row, args.output_tokens) for row in cell_rows):
            atomic_json(
                root / "run-failure.json",
                {
                    "phase": "concurrency",
                    "concurrency": concurrency,
                    "planned_rows": len(prompts),
                    "actual_rows": len(cell_rows),
                    "successful_rows": sum(successful_row(row, args.output_tokens) for row in cell_rows),
                },
            )
            raise RuntimeError(f"concurrency={concurrency} failed eligibility checks")

    planned_concurrency = len(args.concurrency) * args.requests_per_concurrency
    concurrency_success = sum(successful_row(row, args.output_tokens) for row in concurrency_rows)
    if len(concurrency_rows) != planned_concurrency or concurrency_success != planned_concurrency:
        raise RuntimeError("concurrency benchmark failed denominator checks")
    summary = {
        "schema_version": 1,
        "arm": args.arm,
        "model": model,
        "base_url": base_url,
        "output_tokens": args.output_tokens,
        "repetitions": args.repetitions,
        "serial_targets": args.serial_inputs,
        "concurrency_levels": args.concurrency,
        "requests_per_concurrency": args.requests_per_concurrency,
        "serial_rows": len(serial_rows),
        "serial_success": serial_success,
        "concurrency_rows": len(concurrency_rows),
        "concurrency_success": concurrency_success,
        "launch_identity": identity,
        "launch_manifest_sha256": manifest_sha256,
        "smoke": smoke_result,
        "completed_at_unix": time.time(),
    }
    atomic_json(root / "run-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch-manifest", required=True)
    parser.add_argument("--serial-inputs", nargs="+", type=int, default=[512, 4096, 16384, 32768])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--concurrency-input-tokens", type=int, default=1024)
    parser.add_argument("--requests-per-concurrency", type=int, default=8)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

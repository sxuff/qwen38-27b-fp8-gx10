#!/usr/bin/env python3
"""Verify the OpenAI-compatible endpoint with a deterministic answer."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    models = request_json(base + "/v1/models", timeout=30).get("data") or []
    if len(models) != 1 or not models[0].get("id"):
        raise RuntimeError(f"unexpected model list: {models!r}")
    model = models[0]["id"]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Return only the integer obtained by multiplying 12 by 17.",
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
    value = request_json(base + "/v1/chat/completions", payload)
    choice = (value.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if content.strip() != "204" or choice.get("finish_reason") != "stop":
        raise RuntimeError(f"smoke response was not the exact expected answer: {value!r}")
    print(json.dumps({"ok": True, "model": model, "content": content, "usage": value.get("usage")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthetic contract tests for the hardened public benchmark client."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_benchmark", ROOT / "scripts" / "benchmark.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load public benchmark module")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail_completion = False
    bad_smoke = False

    def log_message(self, format, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "mock-qwen"}]})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            self.send_json({"count": len(payload.get("prompt", "").split())})
            return
        if self.path == "/v1/chat/completions":
            content = "1204" if type(self).bad_smoke else "204"
            self.send_json(
                {
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                }
            )
            return
        if self.path == "/v1/completions":
            if type(self).fail_completion:
                self.send_json({"error": "forced"}, 500)
                return
            count = int(payload["max_tokens"])
            events = [
                {"choices": [{"text": "OK ", "finish_reason": None}]},
                {"choices": [{"text": "OK", "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 8, "completion_tokens": count},
                },
            ]
            body = (
                b"".join(
                    b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "not found"}, 404)


def launch_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "no-spec",
                "run_id": "test-run",
                "container_id": "test-container",
                "image_reference": "image@sha256:" + "1" * 64,
                "image_id": "sha256:" + "2" * 64,
                "model_verification_sha256": "3" * 64,
                "model": {
                    "repo": "Qwen/Qwen3.8-27B-FP8",
                    "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
                    "snapshot_canonical_sha256": "4" * 64,
                },
            }
        )
    )


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "launch.json"
            launch_manifest(manifest)
            args = SimpleNamespace(
                output=str(root / "results"),
                base_url=base,
                arm="no-spec",
                launch_manifest=str(manifest),
                serial_inputs=[8],
                output_tokens=4,
                repetitions=1,
                concurrency=[1],
                concurrency_input_tokens=8,
                requests_per_concurrency=1,
            )
            summary = benchmark.run(args)
            assert summary["serial_rows"] == summary["serial_success"] == 1
            assert summary["concurrency_rows"] == summary["concurrency_success"] == 1
            assert summary["launch_identity"]["mode"] == "no-spec"

            Handler.fail_completion = True
            failed = benchmark.stream_completion(
                base,
                "mock-qwen",
                "prompt",
                "forced-failure",
                4,
                root / "failed.sse",
            )
            assert not benchmark.successful_row(failed, 4)
            Handler.fail_completion = False

            Handler.bad_smoke = True
            try:
                benchmark.smoke(base, "mock-qwen")
            except RuntimeError:
                pass
            else:
                raise AssertionError("substring smoke response was incorrectly accepted")
            Handler.bad_smoke = False
    finally:
        server.shutdown()
        server.server_close()
    print("PUBLIC_HARNESS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

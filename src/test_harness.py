#!/usr/bin/env python3
"""Narrow synthetic contract test for benchmark.py."""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import benchmark


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
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
            self.send_json({"choices": [{"message": {"content": "204"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
            return
        if self.path == "/v1/completions":
            events = [
                {"choices": [{"text": "OK ", "finish_reason": None}]},
                {"choices": [{"text": "OK", "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": 32, "completion_tokens": 256}},
            ]
            body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "not found"}, 404)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        model = benchmark.get_model(base)
        assert model == "mock-qwen"
        smoke = benchmark.smoke(base, model)
        assert "204" in smoke["content"]
        assert benchmark.token_count(base, model, "one two three") == 3
        with tempfile.TemporaryDirectory() as directory:
            row = benchmark.stream_completion(base, model, "prompt", "case", 256, Path(directory) / "case.sse")
            assert row["completion_tokens"] == 256
            assert row["prompt_tokens"] == 32
            assert row["finish_reason"] == "length"
            assert row["decode_tokens_per_second"] is not None
            assert not row["error"]
    finally:
        server.shutdown()
        server.server_close()
    print("HARNESS_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

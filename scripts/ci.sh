#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m compileall -q src scripts tests
( cd src && python3 test_harness.py )
python3 tests/test_public_harness.py
python3 tests/test_downloader.py
python3 tests/test_safety_helpers.py
python3 scripts/verify_publication.py
bash -n scripts/serve.sh scripts/stop.sh scripts/run_benchmark.sh
echo "REPOSITORY_CHECKS_OK"

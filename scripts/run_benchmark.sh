#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-no-spec}"
PORT="${PORT:-8000}"
ACTIVE_MANIFEST="${LAUNCH_MANIFEST:-$ROOT/.state/active-launch.json}"
OUTPUT="${OUTPUT:-$ROOT/runs/repro/results/$MODE}"

if [[ "$MODE" != "mtp" && "$MODE" != "no-spec" ]]; then
  echo "MODE must be mtp or no-spec" >&2
  exit 2
fi
if [[ ! -f "$ACTIVE_MANIFEST" ]]; then
  echo "Missing active launch manifest. Start the server with scripts/serve.sh." >&2
  exit 2
fi
CONTAINER_ID="$(python3 "$ROOT/scripts/launch_manifest.py" check \
  --manifest "$ACTIVE_MANIFEST" --expected-mode "$MODE" --field container_id)"
RUN_ID="$(python3 "$ROOT/scripts/launch_manifest.py" check \
  --manifest "$ACTIVE_MANIFEST" --expected-mode "$MODE" --field run_id)"
if ! curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "No healthy server at http://127.0.0.1:$PORT" >&2
  exit 2
fi
if [[ -e "$OUTPUT/serial.jsonl" || -e "$OUTPUT/concurrency.jsonl" || -e "$OUTPUT/telemetry.jsonl" ]]; then
  echo "Refusing to mix a new run into existing receipts: $OUTPUT" >&2
  echo "Choose a fresh OUTPUT directory." >&2
  exit 2
fi
mkdir -p "$OUTPUT"
cp "$ACTIVE_MANIFEST" "$OUTPUT/launch-manifest.json"
rm -f "$OUTPUT/WATCHDOG_STOP" "$OUTPUT/safety-incident.json"

python3 "$ROOT/scripts/telemetry.py" \
  --output "$OUTPUT" \
  --container-id "$CONTAINER_ID" \
  --run-id "$RUN_ID" \
  --mem-floor-bytes "$((15 * 2**30))" \
  --max-swap-growth-bytes "$((512 * 2**20))" &
WATCHDOG_PID=$!
BENCHMARK_PID=""

# Invoked by the EXIT trap below.
# shellcheck disable=SC2329
cleanup() {
  if [[ -n "$BENCHMARK_PID" ]] && kill -0 "$BENCHMARK_PID" 2>/dev/null; then
    kill "$BENCHMARK_PID" 2>/dev/null || true
    wait "$BENCHMARK_PID" 2>/dev/null || true
  fi
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    touch "$OUTPUT/WATCHDOG_STOP"
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

SERIAL_INPUTS="${SERIAL_INPUTS:-512 4096 16384 32768}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
REPETITIONS="${REPETITIONS:-3}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4 8}"
CONCURRENCY_INPUT_TOKENS="${CONCURRENCY_INPUT_TOKENS:-1024}"
REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-8}"
read -r -a SERIAL_INPUTS_ARGS <<< "$SERIAL_INPUTS"
read -r -a CONCURRENCY_ARGS <<< "$CONCURRENCY_LEVELS"

python3 "$ROOT/scripts/benchmark.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --arm "$MODE" \
  --output "$OUTPUT" \
  --launch-manifest "$OUTPUT/launch-manifest.json" \
  --serial-inputs "${SERIAL_INPUTS_ARGS[@]}" \
  --output-tokens "$OUTPUT_TOKENS" \
  --repetitions "$REPETITIONS" \
  --concurrency "${CONCURRENCY_ARGS[@]}" \
  --concurrency-input-tokens "$CONCURRENCY_INPUT_TOKENS" \
  --requests-per-concurrency "$REQUESTS_PER_CONCURRENCY" \
  > "$OUTPUT/benchmark-console.json" &
BENCHMARK_PID=$!
WATCHDOG_DIED=0
while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    WATCHDOG_DIED=1
    kill "$BENCHMARK_PID" 2>/dev/null || true
    break
  fi
  sleep 1
done
BENCHMARK_STATUS=0
wait "$BENCHMARK_PID" || BENCHMARK_STATUS=$?
BENCHMARK_PID=""

if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
  touch "$OUTPUT/WATCHDOG_STOP"
fi
WATCHDOG_STATUS=0
wait "$WATCHDOG_PID" || WATCHDOG_STATUS=$?
trap - EXIT

if [[ "$WATCHDOG_DIED" == "1" || "$WATCHDOG_STATUS" != "0" ]]; then
  echo "Benchmark watchdog failed or exited early." >&2
  [[ -f "$OUTPUT/safety-incident.json" ]] && python3 -m json.tool "$OUTPUT/safety-incident.json" >&2
  exit 1
fi
if [[ "$BENCHMARK_STATUS" != "0" ]]; then
  echo "Benchmark client failed with status $BENCHMARK_STATUS." >&2
  exit "$BENCHMARK_STATUS"
fi
if [[ ! -s "$OUTPUT/telemetry.jsonl" || -f "$OUTPUT/safety-incident.json" ]]; then
  echo "Benchmark telemetry is incomplete or ineligible." >&2
  exit 1
fi
python3 - "$OUTPUT/run-summary.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit("missing run-summary.json")
x=json.loads(p.read_text())
if x.get("serial_rows") != x.get("serial_success"):
 raise SystemExit("serial success denominator mismatch")
if x.get("concurrency_rows") != x.get("concurrency_success"):
 raise SystemExit("concurrency success denominator mismatch")
if not x.get("launch_identity") or not x.get("launch_manifest_sha256"):
 raise SystemExit("run summary lacks launch identity")
print(json.dumps({"arm":x["arm"],"serial":x["serial_rows"],"concurrency":x["concurrency_rows"]},sort_keys=True))
PY

echo "Completed eligible $MODE benchmark: $OUTPUT"

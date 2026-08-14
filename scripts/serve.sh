#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-FP8-017b9c7af6b5/snapshot}"
MODE="${MODE:-no-spec}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-fp8-sglang}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
STATE_ROOT="$ROOT/.state"
ACTIVE_MANIFEST="$STATE_ROOT/active-launch.json"

if [[ "$MODE" != "mtp" && "$MODE" != "no-spec" ]]; then
  echo "MODE must be mtp or no-spec" >&2
  exit 2
fi
if [[ "$(uname -m)" != "aarch64" && "${ALLOW_UNSUPPORTED_ARCH:-0}" != "1" ]]; then
  echo "This recipe is validated on aarch64 GB10. Set ALLOW_UNSUPPORTED_ARCH=1 to override." >&2
  exit 2
fi
for command in docker python3 curl nvidia-smi; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 2; }
done
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model snapshot not found at $MODEL_DIR" >&2
  echo "Run: python3 scripts/download.py" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable to the current user." >&2
  exit 2
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  EXISTING_ID="$(docker inspect -f '{{.Id}}' "$CONTAINER_NAME")"
  EXISTING_OWNED="$(docker inspect -f '{{ index .Config.Labels "io.sxuff.qwen38.recipe" }}' "$EXISTING_ID")"
  EXISTING_RUNNING="$(docker inspect -f '{{.State.Running}}' "$EXISTING_ID")"
  if [[ "$EXISTING_OWNED" != "true" ]]; then
    echo "Refusing to touch unowned container named $CONTAINER_NAME" >&2
    exit 2
  fi
  if [[ "$EXISTING_RUNNING" == "true" ]]; then
    echo "Owned container $CONTAINER_NAME is already running. Use scripts/stop.sh first." >&2
    exit 2
  fi
  docker rm "$EXISTING_ID" >/dev/null
fi

mkdir -p "$STATE_ROOT"
rm -f "$ACTIVE_MANIFEST"
RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
STATE_DIR="$STATE_ROOT/startup-$RUN_ID"
mkdir -p "$STATE_DIR"
MODEL_VERIFICATION="$STATE_DIR/model-verification.json"
LAUNCH_MANIFEST="$STATE_DIR/launch-manifest.json"
CONTAINER_ID=""
WATCHDOG_PID=""
READY=0

# Invoked by the EXIT trap below.
# shellcheck disable=SC2329
cleanup() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    touch "$STATE_DIR/WATCHDOG_STOP"
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ "$READY" != "1" && -n "$CONTAINER_ID" ]]; then
    OWNED="$(docker inspect -f '{{ index .Config.Labels "io.sxuff.qwen38.recipe" }}' "$CONTAINER_ID" 2>/dev/null || true)"
    ACTUAL_RUN="$(docker inspect -f '{{ index .Config.Labels "io.sxuff.qwen38.run" }}' "$CONTAINER_ID" 2>/dev/null || true)"
    if [[ "$OWNED" == "true" && "$ACTUAL_RUN" == "$RUN_ID" ]]; then
      docker rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

python3 "$ROOT/scripts/preflight.py" \
  --mem-floor-bytes "$((15 * 2**30))" > "$STATE_DIR/preflight.json"
python3 "$ROOT/scripts/verify_model.py" \
  --model-dir "$MODEL_DIR" > "$MODEL_VERIFICATION"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker pull "$IMAGE"
fi

SERVER_ARGS=(
  python3 -m sglang.launch_server
  --trust-remote-code
  --model-path /model
  --served-model-name qwen38-27b-fp8
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --attention-backend flashinfer
  --chunked-prefill-size 8192
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --host "$HOST"
  --port "$PORT"
)
if [[ "$MODE" == "mtp" ]]; then
  SERVER_ARGS+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

CONTAINER_ID="$(docker create --rm \
  --name "$CONTAINER_NAME" \
  --label io.sxuff.qwen38.recipe=true \
  --label "io.sxuff.qwen38.run=$RUN_ID" \
  --label "io.sxuff.qwen38.mode=$MODE" \
  --gpus all \
  --ipc host \
  --network host \
  -v "$MODEL_DIR:/model:ro" \
  "$IMAGE" \
  "${SERVER_ARGS[@]}")"

MANIFEST_COMMAND=(
  python3 "$ROOT/scripts/launch_manifest.py" create
  --output "$LAUNCH_MANIFEST"
  --mode "$MODE"
  --run-id "$RUN_ID"
  --container-id "$CONTAINER_ID"
  --container-name "$CONTAINER_NAME"
  --image-reference "$IMAGE"
  --model-verification "$MODEL_VERIFICATION"
  --host "$HOST"
  --port "$PORT"
)
for argument in "${SERVER_ARGS[@]}"; do
  MANIFEST_COMMAND+=("--server-arg=$argument")
done
"${MANIFEST_COMMAND[@]}" >/dev/null

rm -f "$STATE_DIR/WATCHDOG_STOP" "$STATE_DIR/safety-incident.json"
python3 "$ROOT/scripts/telemetry.py" \
  --output "$STATE_DIR" \
  --container-id "$CONTAINER_ID" \
  --run-id "$RUN_ID" \
  --mem-floor-bytes "$((15 * 2**30))" \
  --max-swap-growth-bytes "$((512 * 2**20))" \
  --max-wait-seconds "$STARTUP_TIMEOUT" &
WATCHDOG_PID=$!

for _ in $(seq 1 100); do
  [[ -s "$STATE_DIR/telemetry-baseline.json" ]] && break
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    echo "Startup watchdog exited before container start." >&2
    exit 1
  fi
  sleep 0.1
done
if [[ ! -s "$STATE_DIR/telemetry-baseline.json" ]]; then
  echo "Startup watchdog did not establish a baseline." >&2
  exit 1
fi

docker start "$CONTAINER_ID" >/dev/null
DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < DEADLINE )); do
  if [[ -f "$STATE_DIR/safety-incident.json" ]]; then
    echo "The startup watchdog rejected the run:" >&2
    python3 -m json.tool "$STATE_DIR/safety-incident.json" >&2
    exit 1
  fi
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    echo "Startup watchdog exited unexpectedly." >&2
    exit 1
  fi
  if ! python3 "$ROOT/scripts/launch_manifest.py" check \
    --manifest "$LAUNCH_MANIFEST" --expected-mode "$MODE" >/dev/null 2>&1; then
    echo "Owned container identity or state changed before readiness." >&2
    docker logs --tail 200 "$CONTAINER_ID" 2>&1 >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    touch "$STATE_DIR/WATCHDOG_STOP"
    if ! wait "$WATCHDOG_PID"; then
      echo "Startup watchdog did not exit cleanly." >&2
      exit 1
    fi
    if [[ ! -s "$STATE_DIR/telemetry.jsonl" || -f "$STATE_DIR/safety-incident.json" ]]; then
      echo "Startup telemetry is incomplete or ineligible." >&2
      exit 1
    fi
    docker logs "$CONTAINER_ID" > "$STATE_DIR/server.log" 2>&1 || true
    cp "$LAUNCH_MANIFEST" "$ACTIVE_MANIFEST"
    READY=1
    echo "Ready: mode=$MODE base_url=http://127.0.0.1:$PORT/v1"
    echo "Launch manifest: $ACTIVE_MANIFEST"
    exit 0
  fi
  sleep 5
done

echo "Timed out after ${STARTUP_TIMEOUT}s waiting for readiness." >&2
docker logs --tail 200 "$CONTAINER_ID" 2>&1 >&2 || true
exit 1

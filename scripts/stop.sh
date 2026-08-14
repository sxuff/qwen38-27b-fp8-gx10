#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${LAUNCH_MANIFEST:-$ROOT/.state/active-launch.json}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "No active recipe launch manifest is present."
  exit 0
fi

STATUS="$(python3 "$ROOT/scripts/launch_manifest.py" check \
  --manifest "$MANIFEST" --allow-not-running --allow-missing --field container_status)"
if [[ "$STATUS" == "missing" ]]; then
  echo "The recorded recipe container is already absent."
  exit 0
fi
CONTAINER_ID="$(python3 "$ROOT/scripts/launch_manifest.py" check \
  --manifest "$MANIFEST" --allow-not-running --field container_id)"
if [[ "$STATUS" == "running" ]]; then
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_ID" >/dev/null
fi
for _ in $(seq 1 100); do
  if ! docker inspect "$CONTAINER_ID" >/dev/null 2>&1; then
    echo "Stopped owned container $CONTAINER_ID"
    exit 0
  fi
  sleep 0.1
done
echo "Owned container stopped but was not removed within 10 seconds." >&2
exit 1

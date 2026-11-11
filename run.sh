#!/usr/bin/env bash
# run.sh - one-command setup and launch for this recipe on one NVIDIA GB10.
#
#   git clone https://github.com/sxuff/qwen38-27b-fp8-gx10 && cd qwen38-27b-fp8-gx10 && ./run.sh
#
# Every pin and flag below is copied from this repo's README, manifests or
# receipts; the comment next to each value names its source. Nothing here
# tunes, "improves" or changes a measured setting.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

# ===================== recipe config (per repo) =============================
RECIPE_NAME="qwen38-27b-fp8-gx10"
PORT="${PORT:-8000}"                             # scripts/serve.sh:9 (PORT default)
MODE="${MODE:-no-spec}"                          # scripts/serve.sh:7; README.md:110: MODE=mtp ./run.sh opts into native MTP (not output-lossless, README.md:107)
MODELS_DIR="${MODELS_DIR:-$HOME/models}"         # scripts/download.py:24 default root lives under $HOME/models
SRC_DIR="${SRC_DIR:-$HOME/src}"                  # unused: the engine is a pulled Docker image, not a source build
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-27b-fp8-gx10}"
NEEDS_TOOLS="git curl sha256sum docker python3 nvidia-smi"  # scripts/serve.sh:24 required commands
NEED_MEM_BYTES=16106127360                       # preregistration.json:37 safety.mem_available_floor_bytes (= serve.sh:81 15*2^30)
ENGINE_DISK_BYTES=38619182431                    # runtime-manifest.json:8 image.size_bytes (container image, ~38.6 GB; README.md:58)
ENGINE_KIND=docker
CONTAINER_NAME="qwen38-fp8-sglang"               # scripts/serve.sh:10 (CONTAINER_NAME default)
READY_URL="http://127.0.0.1:${PORT}/v1/models"   # scripts/serve.sh:181 readiness probe
READY_TIMEOUT=900                                # scripts/serve.sh:12 STARTUP_TIMEOUT (serve.sh itself also waits this long)
EXPECTED_LINE="7.86 tok/s (serial decode at 512 input tokens, default no-spec mode; 20.11 tok/s with MODE=mtp ./run.sh, MTP is not output-lossless; README.md:13)"
RECEIPT_PATH="results/summary.json"

# Pinned engine: the stock SGLang image, by digest. No build, no overlay.
BASE_IMAGE="lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"  # scripts/serve.sh:5; preregistration.json:18

# Pinned weights: Qwen/Qwen3.8-27B-FP8 @ 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a,
# 81 files, 30,890,049,597 bytes (preregistration.json:12-16), fetched and
# SHA-256-verified by the repo's own scripts/download.py (README.md:66,71).
MODEL_ROOT="${MODEL_ROOT:-$MODELS_DIR/Qwen3.8-27B-FP8-017b9c7af6b5}"  # scripts/download.py:24 (default ROOT)
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/snapshot}"                        # scripts/download.py:25; scripts/serve.sh:6

# Disk: the weights are fetched by the repo's own scripts/download.py, which the shared disk step
# does not see. Count the remaining weight bytes and the base image here so --check reports a shortfall.
_present() { if [ -e "$1" ]; then du -sb "$1" 2>/dev/null | cut -f1; else echo 0; fi; }
_want=30890049597   # preregistration.json:12-16 (81 files)
_have=$(( $(_present "$MODEL_ROOT") ))
ENGINE_DISK_BYTES=$(( _want > _have ? _want - _have : 0 ))
ENGINE_DISK_BYTES=$(( ENGINE_DISK_BYTES + 21474836480 ))   # + the free-disk reserve scripts/download.py keeps (its --reserve-gib default, 20 GiB)
# Base image 38,619,182,431 bytes (runtime-manifest.json:8); not counted once pulled.
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || ENGINE_DISK_BYTES=$(( ENGINE_DISK_BYTES + 38619182431 ))

WEIGHTS="
"

engine_step() {
  image_pin "$BASE_IMAGE"   # scripts/serve.sh:5 (serve.sh would also pull it, serve.sh:85-87)
}

weights_step() { # repo downloader: pinned revision, resume, LFS-sha256 + git-blob verification (README.md:71)
  if [ -f "$MODEL_ROOT/DOWNLOAD_COMPLETE.json" ]; then
    ok "verified earlier: $MODEL_DIR (scripts/download.py completion receipt; serve.sh re-verifies all 81 files before launch, serve.sh:82-83)"
  elif [ "$CHECK" = 1 ]; then
    plan "python3 scripts/download.py --root $MODEL_ROOT  (pinned revision, 81 files, 30,890,049,597 bytes; 20 GiB disk reserve, scripts/download.py:372)"
  else
    python3 "$REPO_DIR/scripts/download.py" --root "$MODEL_ROOT"  # README.md:66 (quick start) with the script's own pinned defaults
    ok "downloaded + verified: $MODEL_DIR"
  fi
}

serve_step() {   # scripts/serve.sh does everything: preflight, 81-file model verification,
                 # launch manifest, startup watchdog (15 GiB MemAvailable floor, 512 MiB swap
                 # guard), detached container, readiness wait (serve.sh:80-199). It returns
                 # only after http://127.0.0.1:$PORT/v1/models is ready.
  if [ "$CHECK" = 1 ]; then
    plan "MODE=$MODE MODEL_DIR=$MODEL_DIR PORT=$PORT scripts/serve.sh  (detached container $CONTAINER_NAME; no-spec default, serve.sh:7; MTP adds only the four flags in serve.sh:102-108)"
    return
  fi
  env MODE="$MODE" MODEL_DIR="$MODEL_DIR" PORT="$PORT" "$REPO_DIR/scripts/serve.sh"
}

stop_override() {  # the repo's own manifest-scoped stopper
  doit "$REPO_DIR/scripts/stop.sh"   # scripts/stop.sh (stops only the recipe-owned container)
  ok "stopped (or nothing to stop)"
}

curl_test() {      # the repo's own endpoint check (README.md:68); request fields per PROTOCOL.md:63-72
  cat <<EOF
python3 scripts/smoke_test.py          # expects the exact answer 204 (scripts/smoke_test.py)
# or, by hand (model name: scripts/serve.sh:93):
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"qwen38-27b-fp8","messages":[{"role":"user","content":"Return only the integer obtained by multiplying 12 by 17."}],"max_tokens":64,"temperature":0,"top_p":1,"seed":42,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":false}}'
EOF
}
# ===================== end of recipe config =================================

# ---------------------------------------------------------------------------
# Shared machinery (identical in every sxuff recipe repo).
# ---------------------------------------------------------------------------
CHECK=0; ACTION=run
usage() {
  cat <<EOF
Usage: ./run.sh [--check] [--stop] [--help]

Sets up and launches ${RECIPE_NAME} on one NVIDIA GB10 (SM121).

  (no flag)  check the machine, build or pull the pinned engine, download the
             pinned weights, verify SHA-256, start the server, print a test.
             Safe to re-run: finished steps are skipped.
  --check    print what would happen; change nothing.
  --stop     stop the server this script started.
  --help     show this text.

Environment:
  HF_TOKEN     Hugging Face token, read from the environment only, never saved.
  MODELS_DIR   weights        (default: ${MODELS_DIR})
  SRC_DIR      engine source  (default: ${SRC_DIR})
  STATE_DIR    pid and log    (default: ${STATE_DIR})
  PORT         server port    (default: ${PORT})

Expected: ${EXPECTED_LINE}
Receipt:  ${RECEIPT_PATH}
EOF
}
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --stop)  ACTION=stop ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok    %s\n' "$*"; }
plan() { printf '    would %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
CHECK_FAILED=0   # under --check, a failed pre-flight check is reported and the plan continues
fail() { if [ "$CHECK" = 1 ]; then printf '    FAIL  %s\n' "$*"; CHECK_FAILED=1; else die "$*"; fi; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
doit() { if [ "$CHECK" = 1 ]; then plan "$*"; else "$@"; fi; }   # mutate, or describe under --check

PIDFILE="${STATE_DIR}/server.pid"
LOGFILE="${STATE_DIR}/server.log"
STOP_CMD="cd ${REPO_DIR} && ./run.sh --stop"

server_running() {
  if [ "$ENGINE_KIND" = docker ]; then
    [ -n "$(docker ps -q --filter "name=^${CONTAINER_NAME}\$" 2>/dev/null)" ]
  else
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
  fi
}

stop_step() {   # may be overridden in the config block (e.g. to call scripts/stop.sh)
  if ! server_running; then ok "nothing started by this script is running"; return; fi
  if [ "$ENGINE_KIND" = docker ]; then doit docker stop "$CONTAINER_NAME"
  else
    local pid; pid="$(cat "$PIDFILE")"; doit kill "$pid"
    if [ "$CHECK" = 0 ]; then
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; rm -f "$PIDFILE"
    fi
  fi
  ok "stopped"
}
declare -f stop_override >/dev/null && stop_step() { stop_override; }

if [ "$ACTION" = stop ]; then say "Stopping ${RECIPE_NAME}"; stop_step; exit 0; fi
[ "$CHECK" = 1 ] && say "--check: nothing will be changed"

# git checkout of an exact commit, idempotent. Usage: git_pin <url> <sha> <dir>
git_pin() {
  local url="$1" sha="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then ok "source at ${sha:0:12}: $dir"; return; fi
  if [ ! -d "$dir/.git" ]; then doit git clone --quiet "$url" "$dir"; fi
  if [ "$CHECK" = 1 ]; then plan "check out $sha in $dir"; return; fi
  [ -z "$(git -C "$dir" status --porcelain)" ] || die "$dir has local changes; refusing to move it to $sha"
  git -C "$dir" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet --detach "$sha"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "could not check out $sha in $dir"
  ok "source at ${sha:0:12}: $dir"
}

# docker image by digest, idempotent. Usage: image_pin <repo@sha256:...>
image_pin() {
  if docker image inspect "$1" >/dev/null 2>&1; then ok "image present: $1"; else doit docker pull "$1"; fi
}

# --- 1. machine ------------------------------------------------------------
say "Machine"
[ "$(uname -m)" = aarch64 ] || die "expected aarch64 (GB10), got $(uname -m)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
case "$GPU" in *GB10*", 12.1") ok "GPU: $GPU" ;; *) die "measured on an NVIDIA GB10 (SM121); found: ${GPU:-none}" ;; esac
MISSING=""; for t in $NEEDS_TOOLS; do command -v "$t" >/dev/null || MISSING="$MISSING $t"; done
if [ -n "$MISSING" ]; then fail "missing tool(s):$MISSING"; else ok "tools: $NEEDS_TOOLS"; fi
ALREADY=0; server_running && ALREADY=1
MEM_AVAIL=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
if [ "$ALREADY" = 1 ]; then ok "server already running; memory check skipped"
elif [ "$MEM_AVAIL" -lt "$NEED_MEM_BYTES" ]; then
  fail "need $(human "$NEED_MEM_BYTES") available memory, have $(human "$MEM_AVAIL"). Stop other models first (nvidia-smi, docker ps)."
else ok "memory: $(human "$MEM_AVAIL") available, need $(human "$NEED_MEM_BYTES")"; fi
if [ "$ALREADY" = 1 ]; then ok "port ${PORT}: served by this recipe"
else
  # The server binds 127.0.0.1 (or the host network). A listener on 127.0.0.1, a wildcard or ::1
  # blocks it; one bound to another interface address does not, and is only reported.
  PORT_ALL="$(ss -ltnH "sport = :${PORT}" 2>/dev/null || true)"
  PORT_CLASH="$(printf '%s\n' "$PORT_ALL" | awk '{print $4}' | grep -E '^(127\.0\.0\.1|0\.0\.0\.0|\*|\[::\]|\[::1\]|\[::ffff:127\.0\.0\.1\]):' || true)"
  PORT_DOCKER="$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "(127\.0\.0\.1|0\.0\.0\.0|\[::\]|:::?):${PORT}->" || true)"
  if [ -n "$PORT_CLASH$PORT_DOCKER" ]; then
    printf '    port %s is taken:\n%s\n%s\n' "$PORT" "$PORT_CLASH" "$PORT_DOCKER"
    fail "free port ${PORT} or run: PORT=<other> ./run.sh"
  elif [ -n "$PORT_ALL" ]; then
    ok "port ${PORT}: free on 127.0.0.1 (also bound elsewhere: $(printf '%s' "$PORT_ALL" | awk '{print $4}' | tr '\n' ' '))"
  else ok "port ${PORT}: free"; fi
fi

# --- 2. disk ---------------------------------------------------------------
say "Disk"
nearest_dir() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; echo "$d"; }
NEED_DISK=$ENGINE_DISK_BYTES
while read -r sha size repo rev rpath dest; do
  [ -z "${sha:-}" ] && continue
  have=0; [ -f "$MODELS_DIR/$dest" ] && have=$(stat -c %s "$MODELS_DIR/$dest")
  [ "$have" -lt "$size" ] && NEED_DISK=$(( NEED_DISK + size - have ))
done <<< "$WEIGHTS"
FREE_DISK=$(df -B1 --output=avail "$(nearest_dir "$MODELS_DIR")" | tail -1 | tr -d ' ')
if [ "$FREE_DISK" -ge "$NEED_DISK" ]; then ok "need $(human "$NEED_DISK") more, have $(human "$FREE_DISK")"
else fail "need $(human "$NEED_DISK") more disk under $MODELS_DIR, have $(human "$FREE_DISK"). Nothing was deleted."; fi

# --- 3. engine -------------------------------------------------------------
say "Engine"
engine_step

# --- 4. weights: pinned revision + SHA-256 -----------------------------------
hf_fetch() {   # repo rev path dest. The token reaches curl on stdin: never in argv or a file.
  { [ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; true; } |
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - --config - \
         --output "$4.part" "https://huggingface.co/$1/resolve/$2/$3"
  mv "$4.part" "$4"
}
default_weights_step() {
  while read -r sha size repo rev rpath dest; do
    [ -z "${sha:-}" ] && continue
    local f="$MODELS_DIR/$dest" mark="$MODELS_DIR/$dest.sha256-ok"
    if [ -f "$mark" ] && [ "$(cat "$mark")" = "$sha $size" ] && [ "$(stat -c %s "$f" 2>/dev/null)" = "$size" ]; then
      ok "verified earlier: $dest"; continue
    fi
    if [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" != "$size" ]; then
      if [ "$CHECK" = 1 ]; then plan "download $repo@${rev:0:12} $rpath ($(human "$size"))"; continue; fi
      mkdir -p "$(dirname "$f")"; printf '    get   %s (%s)\n' "$rpath" "$(human "$size")"
      hf_fetch "$repo" "$rev" "$rpath" "$f"
    elif [ "$CHECK" = 1 ]; then plan "verify SHA-256 of $dest"; continue; fi
    local got; got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$sha" ]; then mv "$f" "$f.sha256-mismatch"; die "SHA-256 mismatch for $dest: expected $sha, got $got (kept as $f.sha256-mismatch)"; fi
    echo "$sha $size" > "$mark"; ok "sha256 ok: $dest"
  done <<< "$WEIGHTS"
}
declare -f weights_step >/dev/null || weights_step() { default_weights_step; }
say "Weights"
weights_step

# --- 5. serve --------------------------------------------------------------
say "Server"
if [ "$ALREADY" = 1 ]; then ok "already running"
elif [ "$CHECK" = 1 ]; then serve_step
else
  mkdir -p "$STATE_DIR"; serve_step
  printf '    wait  for %s (up to %ss; log: %s)\n' "$READY_URL" "$READY_TIMEOUT" "$LOGFILE"
  t=0
  until curl -fsS -o /dev/null "$READY_URL" 2>/dev/null; do
    server_running || die "server exited during start-up; see $LOGFILE"
    [ "$t" -ge "$READY_TIMEOUT" ] && die "not ready after ${READY_TIMEOUT}s; see $LOGFILE"
    sleep 5; t=$((t + 5))
  done
  ok "ready after ${t}s"
fi

# --- 6. done ---------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  if [ "$CHECK_FAILED" = 1 ]; then say "Check finished with FAIL lines above; nothing was changed. ./run.sh would stop at the first one."; exit 1; fi
  say "Check finished; nothing was changed. Run ./run.sh to do it."; exit 0
fi
say "Test it"
curl_test
say "Stop it"
echo "$STOP_CMD"
echo
echo "Expected: ${EXPECTED_LINE} - receipt: ${RECEIPT_PATH}"

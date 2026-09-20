#!/bin/zsh

# Start ONE on macOS while keeping camera localization on the host-side
# MPS/CUDA worker. Docker Desktop cannot expose Apple's Metal runtime to the
# Linux vision-worker container, so that container is kept only as Compose's
# healthy dependency; the API is explicitly routed to host.docker.internal.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT_DIR="${SCRIPT_DIR:h}"
RUNTIME_DIR="$ROOT_DIR/camera_positioning_lab/.runtime"
GPU_PID_FILE="$RUNTIME_DIR/geometry-gpu.pid"
GPU_LOG_FILE="$RUNTIME_DIR/geometry-gpu.log"
GEOMETRY_PYTHON="$ROOT_DIR/.geometry-venv/bin/python"

die() {
  print -u2 -- "start_mac_gpu.sh: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

health_payload() {
  curl -sS --max-time 5 "${1%/}/health" 2>/dev/null || true
}

health_is_gpu_ready() {
  "$GEOMETRY_PYTHON" -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
runtime = payload.get("runtime") or {}
device = runtime.get("device")
ready = payload.get("status") == "ready" and runtime.get("framework") == "pytorch-ultralytics"
if ready and device in {"mps", "cuda"}:
    print(f"GPU geometry worker ready: device={device}, mode={payload.get("mode", "unknown")}")
    raise SystemExit(0)
print(
    "GPU geometry worker is not ready: "
    f"status={payload.get("status")!r}, device={device!r}, "
    f"framework={runtime.get("framework")!r}"
)
raise SystemExit(1)
' <<<"$1"
}

wait_for_gpu_worker() {
  local payload
  local attempt
  for ((attempt = 1; attempt <= 180; attempt++)); do
    payload="$(health_payload "http://127.0.0.1:8090")"
    if [[ -n "$payload" ]] && health_is_gpu_ready "$payload" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  print -u2 -- "Timed out waiting for the host GPU geometry worker."
  print -u2 -- "Last worker log output:"
  tail -n 80 "$GPU_LOG_FILE" 2>/dev/null || true
  return 1
}

wait_for_api() {
  local attempt
  for ((attempt = 1; attempt <= 180; attempt++)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:8000/api/v1/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  print -u2 -- "Timed out waiting for the Docker API at http://127.0.0.1:8000."
  docker compose -f "$ROOT_DIR/docker-compose.yml" ps || true
  docker compose -f "$ROOT_DIR/docker-compose.yml" logs --tail 80 api vision-worker || true
  return 1
}

require_command docker
require_command curl

[[ -x "$GEOMETRY_PYTHON" ]] || die "missing $GEOMETRY_PYTHON; create the geometry virtualenv first"
[[ -f "$ROOT_DIR/.env" ]] || die "missing $ROOT_DIR/.env; copy .env.example and set the local model paths"

cd "$ROOT_DIR"
set -a
source "$ROOT_DIR/.env"
set +a

: "${ONE_GEOMETRY_MODEL_PATH:?ONE_GEOMETRY_MODEL_PATH is required in .env}"
: "${ONE_GEOMETRY_MODEL_CONFIG:?ONE_GEOMETRY_MODEL_CONFIG is required in .env}"
[[ -f "$ONE_GEOMETRY_MODEL_PATH" ]] || die "model checkpoint not found: $ONE_GEOMETRY_MODEL_PATH"
[[ -f "$ONE_GEOMETRY_MODEL_CONFIG" ]] || die "model configuration not found: $ONE_GEOMETRY_MODEL_CONFIG"

# These are deliberate launcher overrides. The Compose worker stays CPU-only,
# but the API must never select it for camera localization.
export ONE_GEOMETRY_SERVICE_URL="http://host.docker.internal:8090"
export ONE_GEOMETRY_REQUIRE_GPU=true
export ONE_GEOMETRY_DOCKER_DEVICE=cpu
export ONE_GEOMETRY_DOCKER_ALLOW_CPU=true
export ONE_GEOMETRY_MODE=model
export ONE_GEOMETRY_ALLOW_CPU=false
export ONE_GEOMETRY_DEVICE="${ONE_GEOMETRY_DEVICE:-auto}"
export ONE_GEOMETRY_SOLVER_DEVICE="${ONE_GEOMETRY_SOLVER_DEVICE:-auto}"

mkdir -p "$RUNTIME_DIR"

existing_health="$(health_payload "http://127.0.0.1:8090")"
if [[ -n "$existing_health" ]]; then
  if health_is_gpu_ready "$existing_health"; then
    print -- "Reusing the GPU geometry worker already listening on 127.0.0.1:8090."
  else
    die "port 8090 is occupied by a worker that is not GPU-ready; stop it or configure it for MPS/CUDA first"
  fi
else
  print -- "Starting host geometry worker with device=${ONE_GEOMETRY_DEVICE}, solver=${ONE_GEOMETRY_SOLVER_DEVICE}."
  env \
    ONE_GEOMETRY_HOST="${ONE_GEOMETRY_HOST:-0.0.0.0}" \
    ONE_GEOMETRY_PORT=8090 \
    ONE_GEOMETRY_MODE=model \
    ONE_GEOMETRY_MODEL_PATH="$ONE_GEOMETRY_MODEL_PATH" \
    ONE_GEOMETRY_MODEL_CONFIG="$ONE_GEOMETRY_MODEL_CONFIG" \
    ONE_GEOMETRY_DEVICE="$ONE_GEOMETRY_DEVICE" \
    ONE_GEOMETRY_SOLVER_DEVICE="$ONE_GEOMETRY_SOLVER_DEVICE" \
    ONE_GEOMETRY_ALLOW_CPU=false \
    ONE_POSITIONING_WORKERS="${ONE_POSITIONING_WORKERS:-3}" \
    nohup "$GEOMETRY_PYTHON" -m geometry_service \
      >"$GPU_LOG_FILE" 2>&1 &
  print $! >"$GPU_PID_FILE"
  wait_for_gpu_worker
fi

print -- "Starting Docker services with the API routed to host.docker.internal:8090."
docker compose -f "$ROOT_DIR/docker-compose.yml" up -d --build
wait_for_api

# Verify the route from inside the API container, not only from the Mac host.
docker compose -f "$ROOT_DIR/docker-compose.yml" exec -T api python -c '
import json
import os
import urllib.request

base = os.environ["ONE_GEOMETRY_SERVICE_URL"].rstrip("/")
payload = json.load(urllib.request.urlopen(base + "/health", timeout=8))
runtime = payload.get("runtime") or {}
device = runtime.get("device")
if payload.get("status") != "ready" or device not in {"mps", "cuda"}:
    raise SystemExit(f"Docker API reached a non-GPU geometry worker: {payload}")
print(f"Docker API verified GPU geometry route: device={device}, endpoint={base}")
'

print -- "ONE is running with GPU-backed geometry/localization."
print -- "Frontend: http://127.0.0.1:${ONE_FRONTEND_PORT:-4175}"
print -- "API:      http://127.0.0.1:8000"
print -- "Worker:   ${ONE_GEOMETRY_SERVICE_URL}"
print -- "Worker log: $GPU_LOG_FILE"

#!/bin/zsh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="$ROOT/camera_positioning_lab/.runtime"
mkdir -p "$RUNTIME"
OWN_GEOMETRY_PID=""
LAB_PID=""

cleanup() {
  if [[ -n "$LAB_PID" ]] && kill -0 "$LAB_PID" 2>/dev/null; then
    kill "$LAB_PID" 2>/dev/null || true
  fi
  if [[ -n "$OWN_GEOMETRY_PID" ]] && kill -0 "$OWN_GEOMETRY_PID" 2>/dev/null; then
    kill "$OWN_GEOMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
  echo "Starting ONE geometry worker on 127.0.0.1:8090..."
  cd "$ROOT"
  set -a
  source .env
  set +a
  env ONE_GEOMETRY_HOST=127.0.0.1 \
    "$ROOT/.geometry-venv/bin/python" -m geometry_service \
    >"$RUNTIME/geometry.log" 2>&1 &
  OWN_GEOMETRY_PID=$!
  echo "$OWN_GEOMETRY_PID" > "$RUNTIME/geometry.pid"
fi

for _ in {1..45}; do
  if curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then break; fi
  sleep 1
done

if ! curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
  echo "Geometry worker did not become ready. See $RUNTIME/geometry.log"
  exit 1
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
  echo "ONE API is not reachable at 127.0.0.1:8000. Start Docker first: docker compose up -d"
  exit 1
fi

if curl -fsS --max-time 2 http://127.0.0.1:8765/api/status >/dev/null 2>&1; then
  echo "Lab already running."
  open http://127.0.0.1:8765/
  echo "Camera Positioning Lab: http://127.0.0.1:8765/"
  exit 0
else
  echo "Starting Camera Positioning Lab on 127.0.0.1:8765..."
  cd "$ROOT"
  "$ROOT/.venv/bin/python" -m uvicorn camera_positioning_lab.lab_server:app --host 127.0.0.1 --port 8765 \
    >"$RUNTIME/lab.log" 2>&1 &
  LAB_PID=$!
  echo "$LAB_PID" > "$RUNTIME/lab.pid"
fi

for _ in {1..20}; do
  if curl -fsS --max-time 2 http://127.0.0.1:8765/api/status >/dev/null 2>&1; then break; fi
  sleep 0.25
done
if ! curl -fsS --max-time 2 http://127.0.0.1:8765/api/status >/dev/null 2>&1; then
  echo "Camera Positioning Lab did not become ready. See $RUNTIME/lab.log"
  exit 1
fi

open http://127.0.0.1:8765/
echo "Camera Positioning Lab: http://127.0.0.1:8765/"
echo "Logs: $RUNTIME/geometry.log and $RUNTIME/lab.log"
echo "Keep this terminal open while testing. Press Ctrl-C to stop the lab."
wait "$LAB_PID"

# Android integration environment

The default Compose stack starts only the services needed by the Android API:
PostgreSQL, Redis, MinIO, LiveKit and the API. Optional services use profiles so
an absent `one-demo-service` checkout does not block Android development.

## Core stack

1. Copy `.env.example` to `.env` and replace every development password and
   secret. Generate a distinct biometric encryption key when face enrollment
   is required.
2. Run `docker compose up --build -d`.
3. Verify `docker compose ps` and
   `curl.exe http://127.0.0.1:8000/api/v1/health`.

The Android emulator reaches the host API at `http://10.0.2.2:8000/api/v1`.
For a USB-connected phone, keep the loopback binding and run
`adb reverse tcp:8000 tcp:8000`; the app may then use
`http://127.0.0.1:8000/api/v1`.

## Face recognition worker

Run `powershell -ExecutionPolicy Bypass -File scripts/download_face_models.ps1`.
Set `ONE_FACE_MODELS_PATH` and `ONE_GEOMETRY_MODEL_PATH` in `.env`, then set
`ONE_GEOMETRY_SERVICE_URL=http://vision-worker:8090` and start:

`docker compose --profile vision up --build -d`

The YOLO-World checkpoint is intentionally not stored in Git. The path in
`ONE_GEOMETRY_MODEL_PATH` must point to `yolov8s-worldv2.pt` and must be shared
with Docker Desktop on Windows.

## Optional web and demo services

- Website and HTTPS proxy: `docker compose --profile web up --build -d`
- Isolated demo service: clone `one-demo-service` beside this repository, then
  run `docker compose --profile demo up --build -d`

To test from a physical phone over Wi-Fi, set `ONE_API_BIND=0.0.0.0` only on a
trusted network and allow the selected port through the firewall. Prefer the
HTTPS Caddy profile for any shared or persistent environment.

<p align="center">
  <img src="branding/one-logo.png" alt="ONE logo" width="128" />
</p>

# ONE local-first backend

This is the runnable FastAPI foundation for ONE. It is intentionally a modular monolith for the local-LAN MVP: Docker uses PostgreSQL as the authoritative database, while SQLite remains an explicit zero-setup development/test fallback. Redis, MinIO, LiveKit, and Caddy are supplied in `docker-compose.yml` as local services.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
test -f .env || cp .env.example .env
ONE_DATABASE_URL=sqlite:///./one.db uvicorn app.main:app --reload --port 8000
ONE_DATABASE_URL=sqlite:///./one.db pytest
```

OpenAPI is at `/api/v1/openapi.json`. Start pairing with `POST /api/v1/pairing/start`, complete the one-time six-digit code, then send `Authorization: Bearer <access_token>`.

The pinned client contract is committed at `contracts/openapi.json`. Regenerate it
after route or schema changes with `python scripts/generate_openapi.py`; CI rejects
contract drift.

By default the adapter targets local LM Studio at `http://127.0.0.1:1234/v1`
with model `qwen3.6-35b-a3b`. Existing `ONE_LM_STUDIO_*` settings remain
supported. To use any OpenAI-compatible endpoint, set `ONE_LLM_BASE_URL`,
`ONE_LLM_MODEL`, and (when required) `ONE_LLM_API_KEY`; `ONE_LLM_PROVIDER` is
an optional label for deployment metadata. `ONE_LLM_TIMEOUT_SECONDS` controls
the request timeout (default 15 seconds). Secrets are sent only as bearer
headers and are never logged or returned by the API. Set `ONE_LLM_ENABLED=false`
to skip all LLM requests and use the deterministic fallback paths.

## Contracts and safety

All product endpoints are versioned under `/api/v1`. Object observations are approximate and include uncertainty; live video is not persisted by this API, except for the single encrypted image explicitly attached to a confirmed safety event. Events are derived metadata with a 30-day expiry, and clip records are designed for seven-day expiry. The LM Studio adapter uses `qwen3.6-35b-a3b` and falls back explicitly to a deterministic, non-medical summary when the local endpoint is unavailable.

## Camera-derived room geometry

Camera setup can submit a short guided RGB sweep to the local room geometry
service. The API persists only the derived relative 2D polygons, walls, camera
pose, confidence, and model metadata; frame bytes are processed in memory and
discarded. On Apple Silicon, run the host-side service from
`geometry_service/` on port `8090` with Metal/MPS selected. Docker reaches it
through `host.docker.internal` using `ONE_GEOMETRY_SERVICE_URL`.

The host worker accepts independent camera-localization and visual-landmark
requests concurrently. `ONE_POSITIONING_WORKERS` defaults to `3` (bounded to
`1..8`): CPU ORB/SIFT/PnP work overlaps for cameras in the same room or on
different home maps, while shared YOLO/MPS/CUDA calls stay serialized and
bounded for stability. The service's `/health` response reports the positioning
and detector queue limits.

The service is required for live camera map generation. If it is unavailable or
the model confidence is too low, the API returns a rescan/unavailable state and
does not create a fabricated map. RGB maps are relative and never report fake
metric accuracy. A 3D scene is accepted only through the native RoomPlan/LiDAR
contract; Safari camera capture cannot create one.

SQLite remains the zero-setup local/test backend. The API also supports PostgreSQL
through the optional `psycopg` 3 adapter (`pip install -e '.[postgres]'`). Set
`ONE_DATABASE_URL=postgresql://user:password@host:5432/database`; startup applies
the tracked migrations in `migrations/` transactionally, and `/api/v1/health`
reports the selected backend and connectivity status. The Docker image installs
the PostgreSQL extra and Compose waits for the database health check before
starting the API.

## Media/vision slice

`app/vision.py` provides bounded temporal stabilization and calibrated projection for detections returned by the same local YOLO-World worker used by camera mapping. `POST /api/v1/homes/{home_id}/vision/frames` accepts bounded base64 frame bytes and processes them in memory; it does not persist ordinary raw frames, while a confirmed safety event may persist its single encrypted evidence snapshot. When the camera has an active RoomPlan registration, stable detections are projected into the metric `roomplan-local` frame, associated with the RoomPlan room zone, and saved as derived observations/events. Without a usable registration they remain an explicit approximate fallback rather than being presented as metric 3D evidence. `app/media.py` supplies a bounded ring buffer and AES-GCM encrypted local clip bytes. The clip store is an internal worker primitive: authorization must be checked by the API before retrieval, and expiry must be enforced by the retention job.

LiveKit webhook tokens are verified with HS256 issuer/expiry checks and an optional body digest claim when API credentials are configured. The official `livekit-server-sdk` is not bundled, so pin and prefer it for a production deployment matching the server version.

`app/fall.py` adds the first safety-signal preparation layer. It observes only
stable person tracks, requires an upright-to-low posture transition and several
confirming frames, then creates a `fall_suspected` event with
`status=needs_review`. The event is surfaced through the normal Events API and
SSE stream for the iOS and web caregiver views. At that moment only, the
current JPEG/PNG frame may be stored as one AES-GCM encrypted event snapshot;
the event list exposes metadata and an authorized private/no-store image route,
with the same bounded retention and privacy deletion path as the event. It is
intentionally a bounded heuristic, not an emergency detector or diagnosis; a
later pose/temporal model can replace the tracker behind the same event
contract.

## Family mode (bounded MVP)

Family mode is an explicit, synthetic-demo-only caregiver view. After the
actor records `family_mode` consent, an admin or caregiver can list household
members and issue a single-use, hashed invitation for a resident or another
caregiver
(`GET /api/v1/homes/{home_id}/family/members`,
`POST /api/v1/homes/{home_id}/family/invites`). The invite accept route returns
a device/session token for the new demo account; it does not establish legal
representation or consent.

Medication organization is purpose-gated by `medication_management` consent
for the subject. Plans contain a name, dose, human-entered schedule rule,
instructions, active flag, optimistic version, and optional same-home assigned
caregiver. Legacy `08:00,20:00` means daily; rules such as
`Mon,Wed,Fri @ 08:00`, `weekdays 08:00`, `weekends 20:00`, and
`2026-09-12 @ 08:00` are filtered by the requested reminder day. Check-ins are
deterministic administrative states (`pending`, `taken`, `skipped`, `missed`),
and reminder slots are parsed only from explicit `HH:MM` values. The API never
gives dosing advice or makes an automated medical decision. A caregiver can own
their own plan or administer a plan for another same-household member;
`created_by`, `assigned_caregiver_id`, and `marked_by` preserve that
least-privilege ownership context.

`POST /api/v1/homes/{home_id}/check-ins` records the short, caregiver-led daily
check-in result as a bounded summary and a `daily_check_in` event. `GET
/api/v1/homes/{home_id}/analytics` returns a 7–90 day window of daily check-in
completion and heuristic fall-safety signals for caregiver review.

`POST /api/v1/homes/{home_id}/family-assistant` sends only the selected
subject's active medication records, recent daily check-in summaries, and
bounded fall-safety analytics to the configured local Qwen endpoint. It never
sends raw camera frames, face templates, event snapshot bytes, or an unbounded
household stream. If LM Studio is unavailable, the endpoint returns a clearly
labelled deterministic administrative summary. All safety values are review
signals, not diagnoses or emergency decisions. Before real resident data, the
controller must approve the DPIA, lawful basis, representative process,
notices, retention, and rights workflows in `docs/privacy/`.

## Docker

`docker compose up --build` starts the API plus the local YOLO-World vision worker, PostgreSQL, Redis, MinIO, a self-hosted LiveKit development server, and Caddy. The API waits for both PostgreSQL and the vision worker to become healthy before it starts, and the worker uses `restart: unless-stopped` so it comes back automatically with the stack. `ONE_GEOMETRY_MODEL_PATH` must point to the local YOLO-World checkpoint; Compose bind-mounts that file into the worker container. Docker Desktop cannot expose Apple Metal/MPS to Linux containers, so the Compose-managed worker defaults to CPU. You can still override `ONE_GEOMETRY_SERVICE_URL` with `http://host.docker.internal:8090` when intentionally running the faster host-side MPS worker yourself. Compose applies the numbered migrations before the API starts. The Compose defaults use LiveKit's local `devkey`/`secret` placeholders, so no LiveKit Cloud subscription is involved. Replace every example password/secret before sharing the LAN, set `ONE_LIVEKIT_URL` to a host-reachable `ws://` or trusted `wss://` endpoint for phones, add a real `.env`, and provision the local Caddy CA on client devices before using this on a LAN.

For the complete Apple Silicon startup path, use `./scripts/start_mac_gpu.sh` from
this repository after configuring `.env` and creating `.geometry-venv`. The
script starts or reuses the host-side geometry worker with CPU fallback disabled,
starts the Compose stack, forces the API route to
`http://host.docker.internal:8090`, and verifies that the API container sees an
`mps` or `cuda` worker. The Compose `vision-worker` still starts because it is a
healthy dependency of the default stack, but it is not the solver selected by
the API in this mode. The script writes the host-worker log to
`camera_positioning_lab/.runtime/geometry-gpu.log` and fails rather than silently
falling back to CPU.

The frontend binds to `127.0.0.1` by default. For a private phone browser, keep
that setting and run `tailscale serve --bg http://127.0.0.1:${ONE_FRONTEND_PORT:-4175}`;
open the HTTPS URL shown by `tailscale serve status`. For a trusted same-Wi-Fi
test only, set `ONE_FRONTEND_BIND=0.0.0.0` and open
`http://<this-Mac-LAN-IP>:<ONE_FRONTEND_PORT>`; browser camera permissions still
require a secure origin, so Tailscale HTTPS is preferred.

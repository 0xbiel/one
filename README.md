# ONE local-first backend

This is the runnable FastAPI foundation for ONE. It is intentionally a modular monolith for the local-LAN MVP: SQLite is the zero-setup development/test backend, while PostgreSQL/Redis/MinIO/LiveKit/Caddy are supplied in `docker-compose.yml` for deployment wiring.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload --port 8000
pytest
```

OpenAPI is at `/api/v1/openapi.json`. Start pairing with `POST /api/v1/pairing/start`, complete the one-time six-digit code, then send `Authorization: Bearer <access_token>`.

The pinned client contract is committed at `contracts/openapi.json`. Regenerate it
after route or schema changes with `python scripts/generate_openapi.py`; CI rejects
contract drift.

If LM Studio authentication is enabled, set `ONE_LM_STUDIO_API_KEY` or the existing `LLM_API_KEY` alias (the adapter sends `Authorization: Bearer ...`). Leave it blank only when the local LM Studio server has authentication disabled.

## Contracts and safety

All product endpoints are versioned under `/api/v1`. Object observations are approximate and include uncertainty; video is not persisted by this API. Events are derived metadata with a 30-day expiry, and clip records are designed for seven-day expiry. The LM Studio adapter uses `qwen3.6-35b-a3b` and falls back explicitly to a deterministic, non-medical summary when the local endpoint is unavailable.

The current SQLite adapter is complete for local development. `ONE_DATABASE_URL` values for PostgreSQL are accepted as configuration intent, but a production PostgreSQL adapter/migrations still needs to be added before deployment; do not point this build at PostgreSQL yet.

## Media/vision slice

`app/vision.py` defines a bounded detector contract, an OWLv2-compatible injected adapter (no model download), a deterministic no-download demo detector, temporal hit tracking, and calibrated approximate projection with an explicit zone fallback. `POST /api/v1/homes/{home_id}/vision/frames` accepts bounded base64 frame bytes, processes them in memory, and returns stable detections; it does not persist raw frames. `app/media.py` supplies a bounded ring buffer and AES-GCM encrypted local clip bytes. The clip store is an internal worker primitive: authorization must be checked by the API before retrieval, and expiry must be enforced by the retention job.

LiveKit webhook tokens are verified with HS256 issuer/expiry checks and an optional body digest claim when API credentials are configured. The official `livekit-server-sdk` is not bundled, so pin and prefer it for a production deployment matching the server version.

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

`POST /api/v1/homes/{home_id}/family-assistant` sends only the selected
subject's active medication plans and bounded check-in rows to the configured
local Qwen endpoint. It never sends camera frames, transcripts, events, or a
full household stream. If LM Studio is unavailable, the endpoint returns a
clearly labelled deterministic administrative summary. Before real resident
data, the controller must approve the DPIA, lawful basis, representative
process, notices, retention, and rights workflows in `docs/privacy/`.

## Docker

`docker compose up --build` starts the API plus Postgres, Redis, MinIO, a self-hosted LiveKit development server, and Caddy. The Compose defaults use LiveKit's local `devkey`/`secret` placeholders, so no LiveKit Cloud subscription is involved. Replace every example password/secret before sharing the LAN, set `ONE_LIVEKIT_URL` to a host-reachable `ws://` or trusted `wss://` endpoint for phones, add a real `.env`, and provision the local Caddy CA on client devices before using this on a LAN.

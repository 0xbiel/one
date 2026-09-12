# Data inventory and flow

Status: draft for controller review. Scope: local-LAN MVP and synthetic hackathon household only.

## Inventory

| Data | Current location | Purpose | Sensitivity / minimisation |
|---|---|---|---|
| Account display name, optional email, role | SQLite `users`, `memberships` | Sign-in and authorization | Keep only fields needed for the demo; email is optional. |
| Pairing code and session token hashes | SQLite `pairing_codes`, `sessions` | One-time device pairing and authenticated API calls | Only SHA-256 hashes are stored; raw codes/tokens are returned once. |
| Consent records | SQLite `consents` | Purpose-specific authorization and withdrawal | Version, purpose, timestamps, and actor; no free-text evidence. |
| Camera, room, map, calibration metadata | SQLite; map JSON artifact in object store | Render a room and interpret camera observations | Geometry and pose can reveal a home layout; no continuous video. |
| Object observations and events | SQLite `observations`, `events` | Approximate last-seen memory and caregiver review | Derived, uncertain, non-diagnostic metadata. |
| Event clip bytes | Encrypted local clip store | Short evidence review | Optional; AES-256-GCM envelope encryption; seven-day default. |
| Check-in transcript and summary | SQLite `summaries` | Requested assistant response | User-provided text and derived explanation; no medical conclusion. |
| Security/audit records | SQLite `audit_log`, `deletion_requests` | Accountability and operation | Minimise metadata; never store raw frames, prompts, or bearer tokens. |

## Current flow

1. An admin starts a pairing session. The API hashes the six-digit code and stores the expiry.
2. A browser or device completes pairing and receives a bearer token. The API stores only its hash.
3. After purpose-specific `video_capture` consent, a publisher sends bounded frames to the API. The vision path processes them in memory; this endpoint returns detections and does not persist frame bytes.
4. Stable observations may be written as approximate coordinates or a room/camera zone. Clips are separately registered and, when uploaded, encrypted before being written under the local object-store root.
5. A requested check-in sends recent derived events and the transcript to the configured local LM Studio endpoint. Continuous video is never sent to the language model. A deterministic fallback is used when the model is unavailable.
6. Rights requests create an export or deletion response. Deletion removes database household rows through foreign-key cascades and removes map/clip artifacts; a minimal deletion request and completion audit proof remain.

## Boundaries to verify before production

- Replace the SQLite development adapter with the reviewed PostgreSQL migrations/adapter.
- Confirm Redis, MinIO, LiveKit, reverse proxy, logs, backups, and host swap/page files do not retain untracked media.
- Document every enabled subprocessor and transfer location before enabling it.
- Confirm the resident information, capacity, representation, and withdrawal process with the controller.

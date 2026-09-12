# ONE privacy and GDPR readiness

## Purpose and minimisation

ONE supports a resident's daily check-in and a caregiver-facing explanation of recent household observations. It is not a medical device and must not diagnose, infer dementia, identify people, or claim clinical deterioration. Camera streams are intended to remain ephemeral; only event clips (default seven days) and derived event/observation metadata (default thirty days) are retained. RoomPlan maps and camera calibration are configuration data retained until replacement or deletion.

Frame processing is bounded and in-memory. The demo detector performs no model download, and the OWLv2 integration is an injected interface rather than a claim of verified streaming inference. Stored local clip bytes use AES-GCM authenticated encryption; encryption keys must be supplied by deployment secret management and are not written beside the media.

## Data inventory

The service stores account and home membership data, consent records, camera/map/calibration configuration, approximate object observations, event metadata, optional event clips, check-in text, derived summaries, and minimal security audit records. LM Studio, LiveKit, Redis, PostgreSQL, and MinIO are self-hosted in the deployment model; inference input should not leave the local host.

## Rights and controls

Consent is versioned and revocable. Authenticated users can request a machine-readable export. An admin can request home deletion; the workflow removes database rows, local map artifacts, and pending application state, while backup expiry must be handled by the operational runbook. Access, role changes, clip views, exports, and deletion are audited without recording prompts, video, or raw frames.

Before real deployment, complete a controller/processor assessment, privacy notice, records of processing, DPIA, threat model, retention schedule, capacity/representative-consent process, breach-response runbook, encrypted-backup/restore test, and access review. A caregiver is not automatically authorised to consent on behalf of a resident.

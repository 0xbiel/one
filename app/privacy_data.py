"""Household privacy export. Callers must authorize the current home admin.

Only explicitly listed retained fields are exported. Future schema columns are
excluded until reviewed; exports never dereference media or biometric artifacts.
"""
import json
import re
from datetime import datetime, timezone


# Static SQL field allowlist: this is intentionally not discovered at runtime.
EXPORT_COLUMNS = {
    'homes': 'id name created_at care_setting support_focus',
    'users': 'id display_name email created_at',
    'memberships': 'home_id user_id role',
    'care_recipients': 'id home_id display_name relationship room_label created_at',
    'home_runtime': 'home_id paused updated_at',
    'consents': 'id home_id subject_user_id purpose policy_version granted_at revoked_at care_recipient_id',
    'cameras': 'id home_id name room_id enabled created_at resolution_width resolution_height metadata_json',
    'rooms': 'id home_id name created_at',
    'room_maps': 'id home_id room_id revision coordinate_frame map_json created_at source dimension approximate localization_status metadata_json',
    'calibrations': 'id home_id camera_id map_id intrinsics_json extrinsics_json accuracy_m created_at resolution_width resolution_height camera_metadata_json metrics_json source status invalidated_at invalidation_reason',
    'camera_localization_references': 'home_id camera_id map_id x z source created_at updated_at',
    'camera_map_generation_jobs': 'id home_id camera_id room_id room_label orientation status frame_count resolution_width resolution_height map_id error_code error_message metrics_json model_version created_at updated_at completed_at',
    'objects': 'id home_id label display_name enabled created_at care_recipient_id',
    'observations': 'id home_id object_id camera_id map_id x y z uncertainty_m confidence detector_version observed_at care_recipient_id identity_confidence',
    'events': 'id home_id event_type status explanation confidence evidence_json first_seen_at last_seen_at expires_at care_recipient_id event_group_key observation_count',
    'face_profiles': 'id home_id care_recipient_id model_version sample_count status created_at updated_at',
    'clips': 'id home_id event_id starts_at ends_at expires_at',
    'event_snapshots': 'event_id home_id content_type created_at expires_at',
    'summaries': 'id home_id subject_user_id care_recipient_id status trend explanation evidence_json limitations model_version created_at expires_at',
    'audit_log': 'id home_id user_id action target_type target_id metadata_json created_at',
    'deletion_requests': 'id home_id requested_by status created_at completed_at',
    'family_invites': 'id home_id invited_by email display_name role expires_at accepted_at created_at',
    'medication_plans': 'id home_id subject_user_id name dose schedule instructions active version created_by assigned_caregiver_id created_at updated_at care_recipient_id',
    'medication_check_ins': 'id home_id plan_id subject_user_id scheduled_for status note marked_by created_at updated_at care_recipient_id',
    'tracking_devices': 'id home_id care_recipient_id label platform status registered_by last_seen_at created_at updated_at',
    'location_points': 'id home_id care_recipient_id device_id client_sample_id latitude longitude accuracy_m speed_mps bearing_deg battery_percent captured_at received_at street_name dwell_duration_millis',
    'location_clear_watermarks': 'home_id care_recipient_id cleared_at',
    'safe_places': 'id home_id care_recipient_id name latitude longitude radius_m created_at updated_at kind revision',
    'care_entries': 'id home_id care_recipient_id kind title body location starts_at ends_at timezone_name reminder_minutes version created_by created_at updated_at',
}

EXCLUSIONS = {
    "authentication": "Sessions, pairing codes, reconnect tokens, email verifications, credentials and invite/code hashes are excluded.",
    "biometrics": "Recognition templates, embeddings, landmark descriptors and enrollment samples are excluded; profile status and timestamps remain.",
    "media": "Media bytes, raw frames/audio/transcripts and storage artifact locators are excluded; retained clip/snapshot metadata remains.",
    "retention": "This contains currently retained records only; expired or previously deleted content cannot be reconstructed.",
    "scope": "Only this household and its current member profiles; other household memberships and records are excluded.",
}


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(word in normalized for word in (
        "password", "passwd", "secret", "credential", "token", "codehash",
        "pairingcode", "invitecode", "authorization", "apikey", "privatekey",
        "artifactkey", "objectkey", "embedding", "descriptor", "template",
        "biometric", "base64", "rawframe", "framebytes", "imagebytes",
        "audiobytes", "mediabytes", "transcript", "enrollmentsample",
        "signedurl", "downloadurl", "storageurl", "dataurl",
    )) or normalized.endswith("hash") or normalized in {"code", "hash", "bytes", "frame", "frames", "audio", "image", "images", "samples", "visual_landmarks".replace("_", "")}


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if not _sensitive_key(str(key))}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    # Binary values are never serialized into the export.
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    return value


def export_home_data(db, home_id: str) -> dict:
    """Read reviewed fields with a household predicate on every table.

    This helper does not grant access. The ordinary API route must verify home
    membership and the admin role before calling it. JSON columns retain their
    existing string representation for compatibility, with sensitive keys removed.
    Malformed JSON is omitted rather than returned as unreviewed opaque content.
    """
    result = {}
    for table, fields in EXPORT_COLUMNS.items():
        columns = fields.split()
        if table == "users":
            selected = ", ".join("u." + field for field in columns)
            query = f"SELECT {selected} FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=?"
        else:
            predicate = "id" if table == "homes" else "home_id"
            query = f"SELECT {', '.join(columns)} FROM {table} WHERE {predicate}=?"
        records = db.many(query, (home_id,))
        for record in records:
            for field in columns:
                if field.endswith("_json"):
                    try:
                        decoded = json.loads(record[field]) if isinstance(record[field], str) else record[field]
                        record[field] = json.dumps(_sanitize(decoded), ensure_ascii=False, allow_nan=False)
                    except (TypeError, ValueError):
                        record[field] = "{}"
                else:
                    record[field] = _sanitize(record[field])
        result[table] = records
    return result


def build_privacy_export(db, home_id: str, *, exported_at: str | None = None) -> dict:
    """Envelope for the admin-authorized ordinary household export route."""
    return {
        "home_id": home_id,
        "exported_at": exported_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schema_version": "1",
        "scope": {"home_id": home_id, "authorization": "household_admin", "content": "currently_retained_user_visible_records", "includes_media_bytes": False, "includes_recognition_templates": False},
        "exclusions": dict(EXCLUSIONS),
        "data": export_home_data(db, home_id),
    }


def validate_deletion_confirmation(home_id: str, confirmed: bool, confirmation_home_id: str | None) -> None:
    """Validate explicit target confirmation before any deletion-side mutation."""
    if confirmed is not True or confirmation_home_id != home_id:
        raise ValueError("Explicit confirmation and matching confirmation_home_id are required")

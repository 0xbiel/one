import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app


def client(tmp_path: Path):
    app = make_app(Settings(database_url="sqlite:///:memory:", object_store_path=tmp_path / "objects", bootstrap_secret="test", env="test", lm_studio_url="http://127.0.0.1:9/v1"))
    return TestClient(app)


def auth(c):
    started = c.post("/api/v1/pairing/start", json={"display_name": "Resident", "home_name": "Test Home"})
    assert started.status_code == 200
    completed = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    assert completed.status_code == 200
    return completed.json()["access_token"], completed.json()["home_id"]


def test_health_and_pairing(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/v1/health").json()["status"] == "ok"
    started = c.post("/api/v1/pairing/start", json={"display_name": "Admin", "home_name": "Test Home", "role": "caregiver"})
    assert started.status_code == 200 and started.json()["role"] == "caregiver"
    completed = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    assert completed.status_code == 200
    token, home = completed.json()["access_token"], completed.json()["home_id"]
    assert home
    me = c.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["actor"]["role"] == "caregiver" and me.json()["home"]["id"] == home
    assert c.get("/api/v1/homes/invalid/cameras", headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_errors_use_safe_replayable_envelope(tmp_path):
    c = client(tmp_path)
    request_id = "9f4d9f1f-f1a3-49b5-a2d8-8f4b5ecf20a1"
    response = c.post("/api/v1/pairing/complete", json={"code": "not-a-code"}, headers={"X-Request-ID": request_id})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["retryable"] is False
    assert payload["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id


def test_publisher_pairing_token_scopes_and_video_consent(tmp_path):
    c = client(tmp_path); admin_token, home = auth(c); admin_headers = {"Authorization": f"Bearer {admin_token}"}
    settings = c.app.state.settings
    settings.livekit_api_key, settings.livekit_api_secret = "lk-key", "lk-secret"
    started = c.post(f"/api/v1/homes/{home}/pairing/start", headers=admin_headers, json={"label": "Hall iPhone"})
    assert started.status_code == 200 and started.json()["home_id"] == home
    publisher = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    assert publisher.status_code == 200
    publisher_headers = {"Authorization": f"Bearer {publisher.json()['access_token']}"}
    assert c.post(f"/api/v1/homes/{home}/consents", headers=publisher_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True}).status_code == 403
    denied = c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={})
    assert denied.status_code == 403
    consent = c.post(f"/api/v1/homes/{home}/consents", headers=admin_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True})
    assert consent.status_code == 200 and consent.json()["paused"] is False
    token_response = c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={})
    assert token_response.status_code == 200 and token_response.json()["mode"] == "publish"
    claims = json.loads(base64.urlsafe_b64decode(token_response.json()["token"].split(".")[1] + "=="))
    assert claims["video"]["canPublish"] is True and claims["video"]["canSubscribe"] is False
    assert c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={"mode": "subscribe"}).status_code == 403
    paused = c.post(f"/api/v1/homes/{home}/consents", headers=admin_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": False})
    assert paused.status_code == 200 and paused.json()["paused"] is True
    assert c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={}).status_code == 403


def test_camera_map_observation_and_sse_schema(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Kitchen"}).json()
    room = c.post(f"/api/v1/homes/{home}/rooms", headers=h, json={"name": "Kitchen"}).json()
    room_map = c.post(f"/api/v1/homes/{home}/maps", headers=h, json={"room_id": room["id"], "map_data": {"objects": []}}).json()
    c.post(f"/api/v1/homes/{home}/calibrations", headers=h, json={"camera_id": camera["id"], "map_id": room_map["id"], "intrinsics": {}, "extrinsics": {}, "accuracy_m": 0.5})
    obj = c.post(f"/api/v1/homes/{home}/objects", headers=h, json={"label": "keys"}).json()
    observed = c.post(f"/api/v1/homes/{home}/observations", headers=h, json={"object_id": obj["id"], "camera_id": camera["id"], "map_id": room_map["id"], "x": 1, "y": 2, "z": 0.5, "uncertainty_m": 0.8, "confidence": 0.9})
    assert observed.status_code == 200 and observed.json()["approximate_location"]["uncertainty_m"] == 0.8
    events = c.get(f"/api/v1/homes/{home}/events", headers=h).json()["data"]
    assert events[0]["event_type"] == "object_observed"
    assert c.get(f"/api/v1/homes/{home}/cameras", headers=h).json()["data"][0]["id"] == camera["id"]
    assert c.get(f"/api/v1/homes/{home}/maps", headers=h).json()["data"][0]["id"] == room_map["id"]
    assert c.get(f"/api/v1/homes/{home}/maps/current", headers=h).json()["id"] == room_map["id"]
    scene = c.get(f"/api/v1/homes/{home}/scene", headers=h)
    assert scene.status_code == 200 and scene.json()["sceneId"] == room_map["id"]
    objects = c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h)
    assert objects.status_code == 200 and objects.json()["data"][0]["lastSeenAt"]
    with c.stream("GET", f"/api/v1/homes/{home}/events/stream?once=true", headers=h) as response:
        assert response.status_code == 200
        assert next(response.iter_lines()).startswith(": connected")


def test_assistant_degraded_and_consent_export_delete(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    consent = c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": "camera", "policy_version": "2026-01"})
    assert consent.status_code == 200
    summary = c.post(f"/api/v1/homes/{home}/check-ins", headers=h, json={"transcript": "Hello"})
    assert summary.status_code == 200 and summary.json()["degraded"] is True and summary.json()["inference_status"] in {"connection_error", "http_401", "timeout", "invalid_model_response"}
    export = c.post(f"/api/v1/homes/{home}/privacy/export", headers=h)
    assert export.status_code == 200 and "consents" in export.json()["data"]
    assert export.json()["data"]["audit_log"]
    deletion = c.post(f"/api/v1/homes/{home}/privacy/delete", headers=h)
    assert deletion.status_code == 200 and deletion.json()["status"] == "completed"
    request = c.app.state.db.one("SELECT status, completed_at FROM deletion_requests WHERE id=?", (deletion.json()["request_id"],))
    proof = c.app.state.db.one("SELECT action FROM audit_log WHERE action='privacy.delete.completed' AND target_id=?", (home,))
    assert request["status"] == "completed" and request["completed_at"] and proof["action"] == "privacy.delete.completed"
    assert c.post(f"/api/v1/homes/{home}/privacy/export", headers=h).status_code == 403


def test_bounded_vision_ingestion_requires_camera_and_stabilizes(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Hall"}).json()
    payload = {"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True}
    assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json=payload).status_code == 200
    frame = base64.b64encode(b"demo-frame").decode()
    payload = {"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480, "candidate_labels": ["keys"]}
    assert c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload).json()["data"] == []
    assert c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload).json()["data"] == []
    stable = c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload).json()
    assert stable["detector_version"] == "demo-deterministic-v1" and stable["data"][0]["projection"]["quality"] == "zone-fallback"
    prohibited = {**payload, "candidate_labels": ["person identity"]}
    assert c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=prohibited).status_code == 422


def test_clip_content_is_encrypted_at_rest_and_requires_home_authorization(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    observation = c.post(f"/api/v1/homes/{home}/observations", headers=h, json={"confidence": 0.7}).json()
    clip = c.post(f"/api/v1/homes/{home}/events/{observation['event_id']}/clips", headers=h, json={"object_key": "event/source.mp4", "starts_at": "2026-01-01T00:00:00+00:00", "ends_at": "2026-01-01T00:00:05+00:00"}).json()
    content = base64.b64encode(b"fake-video-bytes").decode()
    uploaded = c.post(f"/api/v1/homes/{home}/clips/{clip['id']}/content", headers=h, json={"content_base64": content})
    assert uploaded.status_code == 200 and uploaded.json()["encrypted"] is True
    downloaded = c.get(f"/api/v1/clips/{clip['id']}/content", headers=h)
    assert downloaded.status_code == 200 and downloaded.content == b"fake-video-bytes"
    stored = list((tmp_path / "objects" / "encrypted-clips" / "clips" / home).glob("*.bin"))[0]
    assert b"fake-video-bytes" not in stored.read_bytes()


def test_family_invite_is_hashed_single_use_and_role_safe(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": "family_mode", "policy_version": "2026-09-01"}).status_code == 200
    invite = c.post(f"/api/v1/homes/{home}/family/invites", headers=h, json={"display_name": "Caregiver Two", "role": "caregiver"})
    assert invite.status_code == 200 and invite.json()["synthetic_demo"] is True
    invite_row = c.app.state.db.one("SELECT code_hash, accepted_at FROM family_invites WHERE id=?", (invite.json()["id"],))
    assert invite_row["code_hash"] != invite.json()["code"] and invite_row["accepted_at"] is None
    accepted = c.post("/api/v1/family/invites/accept", json={"code": invite.json()["code"]})
    assert accepted.status_code == 200 and accepted.json()["role"] == "caregiver"
    assert c.post("/api/v1/family/invites/accept", json={"code": invite.json()["code"]}).status_code == 400
    caregiver_id = accepted.json()["user_id"]
    for purpose in ("family_mode", "medication_management"):
        assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": purpose, "policy_version": "2026-09-01", "subject_user_id": caregiver_id}).status_code == 200
    members = c.get(f"/api/v1/homes/{home}/family/members", headers=h)
    assert members.status_code == 200 and {row["role"] for row in members.json()["data"]} == {"admin", "caregiver"}
    caregiver_h = {"Authorization": f"Bearer {accepted.json()['access_token']}"}
    assert c.get(f"/api/v1/homes/{home}/family/members", headers=caregiver_h).status_code == 200
    assert c.post(f"/api/v1/homes/{home}/family/invites", headers=caregiver_h, json={"display_name": "Escalation", "role": "admin"}).status_code == 422
    self_plan = c.post(f"/api/v1/homes/{home}/medication-plans", headers=caregiver_h, json={"subject_user_id": caregiver_id, "name": "Caregiver plan", "dose": "1 tablet", "schedule": "09:00"})
    assert self_plan.status_code == 200 and self_plan.json()["created_by"] == caregiver_id


def test_medication_plan_reminders_checkins_and_bounded_assistant(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    for purpose in ("family_mode", "medication_management", "family_assistant"):
        assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": purpose, "policy_version": "2026-09-01"}).status_code == 200
    plan = c.post(f"/api/v1/homes/{home}/medication-plans", headers=h, json={"subject_user_id": c.app.state.db.one("SELECT user_id FROM memberships WHERE home_id=?", (home,))["user_id"], "name": "Morning plan", "dose": "1 tablet", "schedule": "08:00,20:00", "instructions": "Use the labelled pack."})
    assert plan.status_code == 200 and plan.json()["version"] == 1 and plan.json()["medical_advice"] is False
    plan_id = plan.json()["id"]
    reminders = c.get(f"/api/v1/homes/{home}/medication-reminders?day=2026-09-12", headers=h)
    assert reminders.status_code == 200 and len(reminders.json()["data"]) == 2 and reminders.json()["data"][0]["status"] == "pending"
    checkin = c.post(f"/api/v1/homes/{home}/medication-plans/{plan_id}/check-ins", headers=h, json={"scheduled_for": "2026-09-12T08:00:00+00:00", "status": "taken"})
    assert checkin.status_code == 200 and checkin.json()["data"]["status"] == "taken"
    summary = c.post(f"/api/v1/homes/{home}/family-assistant", headers=h, json={"message": "What is pending today?"})
    assert summary.status_code == 200 and summary.json()["context_scope"] == "medication plans and check-ins only" and summary.json()["degraded"] is True
    assert "medical advice" in summary.json()["data"]["limitations"]


def test_medication_schedule_respects_weekdays_and_caregiver_assignment(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    for purpose in ("family_mode", "medication_management"):
        assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": purpose, "policy_version": "2026-09-01"}).status_code == 200
    invite = c.post(f"/api/v1/homes/{home}/family/invites", headers=h, json={"display_name": "Marta", "role": "caregiver"}).json()
    caregiver = c.post("/api/v1/family/invites/accept", json={"code": invite["code"]}).json()
    admin_id = c.app.state.db.one("SELECT user_id FROM memberships WHERE home_id=? AND role='admin'", (home,))["user_id"]
    plan = c.post(f"/api/v1/homes/{home}/medication-plans", headers=h, json={
        "subject_user_id": admin_id,
        "name": "Variable weekly plan",
        "dose": "1 tablet",
        "schedule": "Mon,Wed,Fri @ 08:00; Tue @ 20:00",
        "assigned_caregiver_id": caregiver["user_id"],
    })
    assert plan.status_code == 200 and plan.json()["assigned_caregiver_id"] == caregiver["user_id"]
    monday = c.get(f"/api/v1/homes/{home}/medication-reminders?day=2026-09-14&subject_user_id={admin_id}", headers=h).json()["data"]
    tuesday = c.get(f"/api/v1/homes/{home}/medication-reminders?day=2026-09-15&subject_user_id={admin_id}", headers=h).json()["data"]
    sunday = c.get(f"/api/v1/homes/{home}/medication-reminders?day=2026-09-13&subject_user_id={admin_id}", headers=h).json()["data"]
    assert [row["scheduled_for"] for row in monday] == ["2026-09-14T08:00:00+00:00"]
    assert [row["scheduled_for"] for row in tuesday] == ["2026-09-15T20:00:00+00:00"]
    assert sunday == []
    assert monday[0]["assigned_caregiver_name"] == "Marta"

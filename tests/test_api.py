import base64
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app
from app.vision import Detection, DeterministicDemoDetector


def client(tmp_path: Path, vision_detector=None):
    app = make_app(Settings(database_url="sqlite:///:memory:", object_store_path=tmp_path / "objects", bootstrap_secret="test", env="test", lm_studio_url="http://127.0.0.1:9/v1"), vision_detector=vision_detector or DeterministicDemoDetector())
    return TestClient(app)


class TwoPersonDetector:
    model_version = "two-person-test-v1"

    def detect(self, frame, candidate_labels):
        return [
            Detection("person", 0.92, (80, 70, 220, 430), frame.captured_at),
            Detection("person", 0.90, (300, 65, 450, 425), frame.captured_at),
        ]


class SinglePersonDetector:
    model_version = "single-person-test-v1"

    def detect(self, frame, candidate_labels):
        return [Detection("person", 0.93, (180, 70, 330, 430), frame.captured_at)]


class FallSequenceDetector:
    model_version = "fall-sequence-test-v1"

    def detect(self, frame, candidate_labels):
        step = frame.data[-1]
        bbox = (220, 80, 420, 460) if step < 3 else (150, 250, 490, 390)
        return [Detection("person", 0.92, bbox, frame.captured_at)]


class FacePersonDetector:
    model_version = "face-person-test-v1"

    def detect(self, frame, candidate_labels, *, include_faces=False):
        faces = ()
        if include_faces:
            faces = ({
                "bbox": [205.0, 75.0, 305.0, 185.0],
                "confidence": 0.99,
                "landmarks": [220.0, 110.0, 290.0, 110.0, 255.0, 140.0, 230.0, 165.0, 280.0, 165.0],
                "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },)
        return [Detection("person", 0.95, (180, 70, 330, 430), frame.captured_at, face_observations=faces)]


class FaceEnrollmentGeometry:
    def enroll_faces(self, *, frames):
        return {
            "status": "ready",
            "model_version": "test-face-v1",
            "embeddings": [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }


def auth(c):
    started = c.post("/api/v1/pairing/start", json={"display_name": "Resident", "home_name": "Test Home"})
    assert started.status_code == 200
    completed = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    assert completed.status_code == 200
    return completed.json()["access_token"], completed.json()["home_id"]


def face_client(tmp_path: Path, vision_detector=None):
    settings = Settings(
        database_url="sqlite:///:memory:",
        object_store_path=tmp_path / "objects",
        bootstrap_secret="test",
        env="test",
        lm_studio_url="http://127.0.0.1:9/v1",
        biometric_encryption_key_b64=base64.b64encode(b"b" * 32).decode(),
    )
    return TestClient(make_app(settings, geometry_service=FaceEnrollmentGeometry(), vision_detector=vision_detector))


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


def test_native_arkit_video_scan_creates_metric_approximate_3d_usdz(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "surfaces": [
            {
                "id": "floor-1", "kind": "floor", "alignment": "horizontal", "confidence": 0.9,
                "vertices": [
                    {"x": -2, "y": 0, "z": -2}, {"x": 2, "y": 0, "z": -2},
                    {"x": 2, "y": 0, "z": 2}, {"x": -2, "y": 0, "z": 2},
                ],
            },
            {
                "id": "wall-1", "kind": "wall", "alignment": "vertical", "confidence": 0.8,
                "vertices": [
                    {"x": -2, "y": 0, "z": -2}, {"x": 2, "y": 0, "z": -2},
                    {"x": 2, "y": 2.5, "z": -2}, {"x": -2, "y": 2.5, "z": -2},
                ],
            },
            {
                "id": "wall-2", "kind": "wall", "alignment": "vertical", "confidence": 0.8,
                "vertices": [
                    {"x": 2, "y": 0, "z": -2}, {"x": 2, "y": 0, "z": 2},
                    {"x": 2, "y": 2.5, "z": 2}, {"x": 2, "y": 2.5, "z": -2},
                ],
            },
        ],
        "diagnostics": {
            "frame_sample_count": 9,
            "normal_tracking_samples": 9,
            "plane_count": 3,
            "tracking_state": "normal",
        },
    }

    created = c.post(f"/api/v1/homes/{home}/maps/arkit-video", headers=headers, json=payload)
    assert created.status_code == 200, created.text
    result = created.json()
    assert result["source"] == "arkit-video-3d"
    assert result["dimension"] == "3d"
    assert result["coordinate_frame"] == "arkit-world"
    assert result["approximate"] is True
    assert result["metric_scale_known"] is True
    assert result["usdz"]["available"] is True

    scene = c.get(f"/api/v1/homes/{home}/scene", headers=headers).json()
    assert scene["source"] == "arkit-video-3d"
    assert scene["dimension"] == "3d"
    assert scene["approximate"] is True
    assert len(scene["geometry"]["surfaces"]) == 3
    assert scene["cameraRegistration"] is None

    model = c.get(f"/api/v1/homes/{home}/maps/{result['id']}/usdz", headers=headers)
    assert model.status_code == 200
    assert model.headers["content-type"].startswith("model/vnd.usdz+zip")
    with zipfile.ZipFile(io.BytesIO(model.content)) as archive:
        assert "room.usda" in archive.namelist()
        assert any(name.startswith("assets/Model/Floors/") for name in archive.namelist())
        assert any(name.startswith("assets/Model/Walls/") for name in archive.namelist())

    unstable = json.loads(json.dumps(payload))
    unstable["diagnostics"]["tracking_state"] = "limited"
    assert c.post(f"/api/v1/homes/{home}/maps/arkit-video", headers=headers, json=unstable).status_code == 422


def test_email_identity_survives_device_change_and_verifies_once(tmp_path):
    c = client(tmp_path)
    created = c.post("/api/v1/auth/email/request", json={
        "purpose": "create",
        "email": " Caregiver@Example.COM ",
        "display_name": "Caregiver",
        "home_name": "Persistent Home",
    })
    assert created.status_code == 200
    payload = created.json()
    assert payload["email"] == "caregiver@example.com"
    assert payload["delivery"] == "development_outbox"
    duplicate = c.post("/api/v1/auth/email/request", json={
        "purpose": "create",
        "email": "  CAREGIVER@example.com",
        "display_name": "Duplicate",
    })
    assert duplicate.status_code == 409
    verified = c.post("/api/v1/auth/email/verify", json={"email": "CAREGIVER@example.com", "code": payload["dev_code"]})
    assert verified.status_code == 200
    token = verified.json()["access_token"]
    assert c.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).json()["home"]["name"] == "Persistent Home"
    assert c.post("/api/v1/auth/email/verify", json={"email": "caregiver@example.com", "code": payload["dev_code"]}).status_code == 400
    login = c.post("/api/v1/auth/email/request", json={"purpose": "login", "email": "caregiver@example.com"})
    assert login.status_code == 200
    assert c.post("/api/v1/auth/email/verify", json={"email": "caregiver@example.com", "code": login.json()["dev_code"]}).status_code == 200


def test_account_can_create_list_and_switch_care_spaces(tmp_path):
    c = client(tmp_path)
    created = c.post("/api/v1/auth/email/request", json={
        "purpose": "create",
        "email": "multi-home@example.com",
        "display_name": "Caregiver",
        "home_name": "Family Home",
    }).json()
    first_session = c.post(
        "/api/v1/auth/email/verify",
        json={"email": "multi-home@example.com", "code": created["dev_code"]},
    ).json()
    first_home = first_session["home_id"]
    first_headers = {"Authorization": f"Bearer {first_session['access_token']}"}

    initial = c.get("/api/v1/account/homes", headers=first_headers)
    assert initial.status_code == 200
    assert [(item["name"], item["active"]) for item in initial.json()["data"]] == [("Family Home", True)]

    second_session = c.post(
        "/api/v1/account/homes",
        headers=first_headers,
        json={"name": "Grandparents Residence", "care_setting": "residence", "support_focus": "mci"},
    )
    assert second_session.status_code == 200
    second_headers = {"Authorization": f"Bearer {second_session.json()['access_token']}"}
    second_home = second_session.json()["home_id"]
    assert second_home != first_home
    assert c.get("/api/v1/me", headers=second_headers).json()["home"]["name"] == "Grandparents Residence"

    spaces = c.get("/api/v1/account/homes", headers=second_headers).json()["data"]
    assert {item["name"] for item in spaces} == {"Family Home", "Grandparents Residence"}
    assert next(item for item in spaces if item["id"] == second_home)["active"] is True

    switched = c.post(f"/api/v1/account/homes/{first_home}/activate", headers=second_headers)
    assert switched.status_code == 200
    switched_headers = {"Authorization": f"Bearer {switched.json()['access_token']}"}
    assert c.get("/api/v1/me", headers=switched_headers).json()["home"]["name"] == "Family Home"
    assert c.post("/api/v1/account/homes/missing/activate", headers=switched_headers).status_code == 404


def test_care_recipient_crud_is_separate_from_home_membership(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}

    empty = c.get(f"/api/v1/homes/{home}/care-recipients", headers=headers)
    assert empty.status_code == 200 and empty.json() == {"data": []}

    created = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=headers,
        json={"display_name": "  María García  ", "relationship": "  Partner  ", "room_label": "  Room 12  "},
    )
    assert created.status_code == 201
    recipient = created.json()["data"]
    assert recipient["display_name"] == "María García"
    assert recipient["relationship"] == "Partner"
    assert recipient["room_label"] == "Room 12"
    assert recipient["id"] and recipient["created_at"]
    assert c.app.state.db.one("SELECT 1 FROM memberships WHERE user_id=?", (recipient["id"],)) is None

    second = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=headers,
        json={"display_name": "Joan", "relationship": "Resident"},
    )
    assert second.status_code == 201
    listed = c.get(f"/api/v1/homes/{home}/care-recipients", headers=headers).json()["data"]
    assert {item["display_name"] for item in listed} == {"María García", "Joan"}

    updated = c.patch(
        f"/api/v1/homes/{home}/care-recipients/{recipient['id']}",
        headers=headers,
        json={"display_name": "Maria", "relationship": None, "room_label": "Suite A"},
    )
    assert updated.status_code == 200
    assert updated.json()["data"] | {"created_at": recipient["created_at"]} == {
        "id": recipient["id"], "display_name": "Maria", "relationship": None,
        "room_label": "Suite A", "medication_reminders_enabled": False,
        "face_recognition_status": "not_enrolled", "face_profile_updated_at": None,
        "created_at": recipient["created_at"],
    }
    assert c.patch(f"/api/v1/homes/{home}/care-recipients/{recipient['id']}", headers=headers, json={}).status_code == 422
    assert c.post(f"/api/v1/homes/{home}/care-recipients", headers=headers, json={"display_name": "   "}).status_code == 422

    removed = c.delete(f"/api/v1/homes/{home}/care-recipients/{recipient['id']}", headers=headers)
    assert removed.status_code == 200
    assert removed.json()["data"]["id"] == recipient["id"]
    assert removed.json()["data"]["display_name"] == "Maria"
    assert c.get(f"/api/v1/homes/{home}/care-recipients", headers=headers).json()["data"][0]["display_name"] == "Joan"
    assert c.get("/api/v1/homes/not-this-home/care-recipients", headers=headers).status_code == 403


def test_outside_location_tracking_is_consent_gated_idempotent_and_bounded(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    recipient = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=headers,
        json={"display_name": "María"},
    ).json()["data"]
    base = f"/api/v1/homes/{home}/care-recipients/{recipient['id']}"

    assert c.post(f"{base}/tracking-devices", headers=headers, json={"label": "María phone"}).status_code == 403
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=headers,
        json={"purpose": "outside_location", "policy_version": "2026-09", "granted": True, "care_recipient_id": recipient["id"]},
    ).status_code == 200

    registered = c.post(f"{base}/tracking-devices", headers=headers, json={"label": "María phone"})
    assert registered.status_code == 201
    device = registered.json()["data"]
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    batch = {
        "device_id": device["id"],
        "points": [{
            "client_sample_id": "sample-1", "latitude": 38.3452, "longitude": -0.4810,
            "accuracy_m": 12.5, "speed_mps": 1.2, "bearing_deg": 90,
            "battery_percent": 78, "captured_at": captured_at,
        }],
    }
    first = c.post(f"{base}/location-points", headers=headers, json=batch)
    second = c.post(f"{base}/location-points", headers=headers, json=batch)
    assert first.status_code == 200 and first.json()["accepted"] == 1
    assert second.status_code == 200 and second.json() == {"accepted": 0, "duplicates": 1, "retention_days": 7}
    latest = c.get(f"{base}/locations/latest", headers=headers)
    assert latest.status_code == 200 and latest.json()["data"]["battery_percent"] == 78
    history = c.get(f"{base}/locations", headers=headers)
    assert history.status_code == 200 and len(history.json()["data"]) == 1

    place = c.post(
        f"{base}/safe-places", headers=headers,
        json={"name": "Home", "latitude": 38.3452, "longitude": -0.4810, "radius_m": 120},
    )
    assert place.status_code == 201
    place_id = place.json()["data"]["id"]
    assert c.patch(f"{base}/safe-places/{place_id}", headers=headers, json={"radius_m": 180}).json()["data"]["radius_m"] == 180
    assert c.delete(f"{base}/safe-places/{place_id}", headers=headers).status_code == 200

    paused = c.patch(f"{base}/tracking-devices/{device['id']}", headers=headers, json={"status": "paused"})
    assert paused.status_code == 200
    assert c.post(f"{base}/location-points", headers=headers, json={**batch, "points": [{**batch["points"][0], "client_sample_id": "sample-2"}]}).status_code == 409
    assert c.get(f"/api/v1/homes/not-this-home/care-recipients/{recipient['id']}/locations/latest", headers=headers).status_code == 403


def test_face_profile_is_consent_gated_encrypted_and_revocable(tmp_path):
    c = face_client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    recipient = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=headers,
        json={"display_name": "María"},
    ).json()["data"]
    recipient_id = recipient["id"]

    profile_path = c.get(
        f"/api/v1/homes/{home}/care-recipients/{recipient_id}/face-profile",
        headers=headers,
    )
    assert profile_path.status_code == 200
    assert profile_path.json()["status"] == "not_enrolled"

    frames = [
        {"frame_base64": f"transient-frame-{index}", "width": 640, "height": 480, "camera_position": "front"}
        for index in range(3)
    ]
    blocked = c.post(
        f"/api/v1/homes/{home}/care-recipients/{recipient_id}/face-profile/enroll",
        headers=headers,
        json={"frames": frames},
    )
    assert blocked.status_code == 403

    consent = c.post(
        f"/api/v1/homes/{home}/consents",
        headers=headers,
        json={"purpose": "face_recognition", "policy_version": "2026-09", "granted": True, "care_recipient_id": recipient_id},
    )
    assert consent.status_code == 200

    enrolled = c.post(
        f"/api/v1/homes/{home}/care-recipients/{recipient_id}/face-profile/enroll",
        headers=headers,
        json={"frames": frames},
    )
    assert enrolled.status_code == 200, enrolled.text
    assert enrolled.json()["status"] == "ready"
    assert enrolled.json()["sample_count"] == 3
    artifact = next((tmp_path / "objects" / "face-profiles" / home).glob("*.bin"))
    assert artifact.read_bytes().startswith(b"ONETPL1")
    assert b"transient-frame" not in artifact.read_bytes()
    listed = c.get(f"/api/v1/homes/{home}/care-recipients", headers=headers).json()["data"]
    assert listed[0]["face_recognition_status"] == "ready"

    revoked = c.delete(
        f"/api/v1/homes/{home}/care-recipients/{recipient_id}/face-profile",
        headers=headers,
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"
    assert not artifact.exists()
    face_profile_row = c.app.state.db.one(
        "SELECT status FROM face_profiles WHERE home_id=? AND care_recipient_id=?", (home, recipient_id)
    )
    assert face_profile_row["status"] == "revoked"


def test_live_identity_redacts_embeddings_and_internal_profile_ids(tmp_path):
    c = face_client(tmp_path, vision_detector=FacePersonDetector())
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    recipient = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=headers,
        json={"display_name": "María"},
    ).json()["data"]
    recipient_id = recipient["id"]
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=headers,
        json={"purpose": "face_recognition", "policy_version": "2026-09", "granted": True, "care_recipient_id": recipient_id},
    ).status_code == 200
    frames = [
        {"frame_base64": f"transient-frame-{index}", "width": 640, "height": 480, "camera_position": "front"}
        for index in range(3)
    ]
    assert c.post(
        f"/api/v1/homes/{home}/care-recipients/{recipient_id}/face-profile/enroll",
        headers=headers,
        json={"frames": frames},
    ).status_code == 200
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=headers, json={"name": "Hall"}).json()
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=headers,
        json={"purpose": "video_capture", "policy_version": "2026-09", "granted": True},
    ).status_code == 200
    frame = base64.b64encode(b"face-frame").decode()
    payload = {"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480}
    for _ in range(4):
        response = c.post(f"/api/v1/homes/{home}/vision/frames", headers=headers, json=payload)
    response = c.post(f"/api/v1/homes/{home}/vision/frames", headers=headers, json=payload)
    assert response.status_code == 200
    person = response.json()["data"][0]
    assert person["identity"]["status"] == "matched"
    assert person["identity"]["care_recipient_id"] == recipient_id
    assert "profile_id" not in person["identity"]
    assert "embedding" not in person["faces"][0]


def test_resident_account_can_view_but_not_manage_care_recipients(tmp_path):
    c = client(tmp_path)
    started = c.post("/api/v1/pairing/start", json={"display_name": "Resident account", "home_name": "Shared Home", "role": "resident"}).json()
    session = c.post("/api/v1/pairing/complete", json={"code": started["pairing_code"]}).json()
    headers = {"Authorization": f"Bearer {session['access_token']}"}
    home = session["home_id"]
    assert c.get(f"/api/v1/homes/{home}/care-recipients", headers=headers).status_code == 200
    assert c.post(f"/api/v1/homes/{home}/care-recipients", headers=headers, json={"display_name": "Partner"}).status_code == 403


def test_email_invitation_requires_matching_existing_identity(tmp_path):
    c = client(tmp_path)
    owner = c.post("/api/v1/auth/email/request", json={"purpose": "create", "email": "owner@example.com", "display_name": "Owner"}).json()
    owner_session = c.post("/api/v1/auth/email/verify", json={"email": "owner@example.com", "code": owner["dev_code"]}).json()
    headers = {"Authorization": f"Bearer {owner_session['access_token']}"}
    home = owner_session["home_id"]
    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "family_mode", "policy_version": "2026-09-01"}).status_code == 200
    invite = c.post(f"/api/v1/homes/{home}/family/invites", headers=headers, json={"email": "sibling@example.com", "display_name": "Sibling", "role": "caregiver"}).json()
    assert c.post("/api/v1/family/invites/accept", json={"code": invite["code"], "email": "sibling@example.com"}).status_code == 404
    account = c.post("/api/v1/auth/email/request", json={"purpose": "create", "email": "sibling@example.com", "display_name": "Sibling"}).json()
    c.post("/api/v1/auth/email/verify", json={"email": "sibling@example.com", "code": account["dev_code"]})
    accepted = c.post("/api/v1/family/invites/accept", json={"code": invite["code"], "email": "SIBLING@example.com"})
    assert accepted.status_code == 200 and accepted.json()["home_id"] == home
    assert c.post("/api/v1/family/invites/accept", json={"code": invite["code"], "email": "sibling@example.com"}).status_code == 400


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
    settings.livekit_url = "wss://one-test.ts.net:8444"
    settings.livekit_lan_url = "wss://192.168.1.128:8080"
    started = c.post(f"/api/v1/homes/{home}/pairing/start", headers=admin_headers, json={"label": "Hall iPhone"})
    assert started.status_code == 200 and started.json()["home_id"] == home
    pairing_id = started.json()["pairing_id"]
    pending = c.get(f"/api/v1/homes/{home}/pairing/{pairing_id}/status", headers=admin_headers)
    assert pending.status_code == 200 and pending.json()["status"] == "pending"
    assert "pairing_code" not in pending.json() and "code" not in pending.json()
    publisher = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    assert publisher.status_code == 200
    reconnect_token = publisher.json()["reconnect_token"]
    assert reconnect_token
    cameras = c.get(f"/api/v1/homes/{home}/cameras", headers=admin_headers)
    assert cameras.status_code == 200 and cameras.json()["data"][0]["id"] == pairing_id
    saved_camera = cameras.json()["data"][0]
    assert saved_camera["status"] == "online"
    connected = c.get(f"/api/v1/homes/{home}/pairing/{pairing_id}/status", headers=admin_headers)
    assert connected.status_code == 200 and connected.json()["status"] == "connected" and connected.json()["connected_at"]
    publisher_headers = {"Authorization": f"Bearer {publisher.json()['access_token']}"}
    denied = c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={})
    assert denied.status_code == 403
    assert c.post(f"/api/v1/homes/{home}/consents", headers=publisher_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True}).status_code == 200
    consent = c.post(f"/api/v1/homes/{home}/consents", headers=admin_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True})
    assert consent.status_code == 200 and consent.json()["paused"] is False
    token_response = c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={})
    assert token_response.status_code == 200 and token_response.json()["mode"] == "publish"
    assert token_response.json()["url"] == "wss://one-test.ts.net:8444"
    claims = json.loads(base64.urlsafe_b64decode(token_response.json()["token"].split(".")[1] + "=="))
    assert claims["video"]["canPublish"] is True and claims["video"]["canSubscribe"] is False
    lan_token = c.post(
        f"/api/v1/homes/{home}/livekit/token",
        headers={**publisher_headers, "host": "192.168.1.128:8443"},
        json={},
    )
    assert lan_token.status_code == 200
    assert lan_token.json()["url"] == "wss://192.168.1.128:8080"
    sslip_lan_token = c.post(
        f"/api/v1/homes/{home}/livekit/token",
        headers={**publisher_headers, "host": "one.192-168-1-128.sslip.io"},
        json={},
    )
    assert sslip_lan_token.status_code == 200
    assert sslip_lan_token.json()["url"] == "wss://192.168.1.128:8080"
    assert c.post(f"/api/v1/homes/{home}/livekit/token", headers=publisher_headers, json={"mode": "subscribe"}).status_code == 403
    resumed = c.post("/api/v1/camera/reconnect", json={"camera_id": pairing_id, "reconnect_token": reconnect_token})
    assert resumed.status_code == 200
    assert resumed.json()["user_id"] == pairing_id and resumed.json()["home_id"] == home
    resumed_headers = {"Authorization": f"Bearer {resumed.json()['access_token']}"}
    assert c.get("/api/v1/me", headers=resumed_headers).json()["actor"]["role"] == "publisher"
    refreshed_link = c.post("/api/v1/camera/reconnect-link", headers=resumed_headers)
    assert refreshed_link.status_code == 200 and refreshed_link.json()["camera_id"] == pairing_id
    assert c.post("/api/v1/camera/reconnect", json={"camera_id": pairing_id, "reconnect_token": reconnect_token}).status_code == 401
    assert c.post("/api/v1/camera/reconnect", json={"camera_id": pairing_id, "reconnect_token": refreshed_link.json()["reconnect_token"]}).status_code == 200
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


def test_caregiver_can_remove_camera_without_erasing_history(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=headers, json={"name": "Hallway phone"}).json()

    removed = c.delete(f"/api/v1/homes/{home}/cameras/{camera['id']}", headers=headers)
    assert removed.status_code == 200
    assert removed.json()["status"] == "deleted"
    assert c.get(f"/api/v1/homes/{home}/cameras", headers=headers).json()["data"] == []
    assert c.get("/api/v1/me", headers=headers).json()["device"] is None
    tombstone = c.app.state.db.one("SELECT enabled FROM cameras WHERE id=?", (camera["id"],))
    audit = c.app.state.db.one("SELECT action, target_id FROM audit_log WHERE action='camera.delete' AND target_id=?", (camera["id"],))
    assert tombstone["enabled"] == 0
    assert audit["action"] == "camera.delete"

    # DELETE remains idempotent for a camera that is already disabled.
    assert c.delete(f"/api/v1/homes/{home}/cameras/{camera['id']}", headers=headers).status_code == 200


def test_room_lifecycle_renames_and_deletes_without_erasing_camera_or_map(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}

    created = c.post(f"/api/v1/homes/{home}/rooms", headers=headers, json={"name": "  Room 1  "})
    assert created.status_code == 200
    room = created.json()
    assert room["home_id"] == home
    assert room["name"] == "Room 1"
    assert room["created_at"]

    camera = c.post(
        f"/api/v1/homes/{home}/cameras",
        headers=headers,
        json={"name": "Room camera", "room_id": room["id"]},
    ).json()
    room_map = c.post(
        f"/api/v1/homes/{home}/maps",
        headers=headers,
        json={"room_id": room["id"], "map_data": {"objects": []}},
    ).json()

    renamed = c.patch(
        f"/api/v1/homes/{home}/rooms/{room['id']}",
        headers=headers,
        json={"name": "Kitchen"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Kitchen"
    assert c.get(f"/api/v1/homes/{home}/rooms", headers=headers).json()["data"][0]["name"] == "Kitchen"

    deleted = c.delete(f"/api/v1/homes/{home}/rooms/{room['id']}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert deleted.json()["cameras_unassigned"] == 1
    assert deleted.json()["maps_unassigned"] == 1
    assert c.get(f"/api/v1/homes/{home}/rooms", headers=headers).json()["data"] == []

    camera_row = c.app.state.db.one("SELECT enabled, room_id FROM cameras WHERE id=?", (camera["id"],))
    map_row = c.app.state.db.one("SELECT room_id FROM room_maps WHERE id=?", (room_map["id"],))
    assert camera_row and camera_row["enabled"] == 1 and camera_row["room_id"] is None
    assert map_row and map_row["room_id"] is None


def test_deleted_paired_camera_revokes_reconnect_link(tmp_path):
    c = client(tmp_path)
    admin_token, home = auth(c)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    started = c.post(f"/api/v1/homes/{home}/pairing/start", headers=admin_headers, json={"label": "Kitchen tablet"})
    pairing_id = started.json()["pairing_id"]
    publisher = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    reconnect_token = publisher.json()["reconnect_token"]

    assert c.post("/api/v1/camera/reconnect", json={"camera_id": pairing_id, "reconnect_token": reconnect_token}).status_code == 200
    assert c.delete(f"/api/v1/homes/{home}/cameras/{pairing_id}", headers=admin_headers).status_code == 200
    assert c.post("/api/v1/camera/reconnect", json={"camera_id": pairing_id, "reconnect_token": reconnect_token}).status_code == 401


def test_camera_provisional_roomplan_revision_and_calibration_invalidation(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Hall", "resolution_width": 640, "resolution_height": 480}).json()
    provisional = c.post(f"/api/v1/homes/{home}/maps/provisional", headers=h, json={"camera_id": camera["id"], "resolution_width": 640, "resolution_height": 480, "zones": [{"id": "hall", "confidence": 0.4}]})
    assert provisional.status_code == 200 and provisional.json()["approximate"] is True and provisional.json()["localization_status"] == "rescan-required" and provisional.json()["source"] == "legacy-2d"
    provisional_detail = c.get(f"/api/v1/homes/{home}/maps/{provisional.json()['id']}", headers=h)
    zone = provisional_detail.json()["map_data"]["zones"][0]
    assert provisional_detail.status_code == 200 and not {"x", "y", "width", "height"}.intersection(zone)
    rejected_roomplan = c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json={"normalized_scan": {"rooms": [{"id": "hall"}]}, "scan_metadata": {"device": "iPhone"}})
    assert rejected_roomplan.status_code == 422
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    roomplan = c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json=roomplan_payload)
    assert roomplan.status_code == 200 and roomplan.json()["source"] == "roomplan-lidar-3d" and roomplan.json()["dimension"] == "3d"
    assert roomplan.json()["coordinate_frame"] == "roomplan-local"
    non_lidar_payload = json.loads(json.dumps(roomplan_payload))
    non_lidar_payload["scan_metadata"]["lidar"] = False
    assert c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json=non_lidar_payload).status_code == 422
    scene = c.get(f"/api/v1/homes/{home}/scene", headers=h).json()
    assert scene["source"] == "roomplan-lidar-3d" and scene["dimension"] == "3d"
    assert scene["geometry"]["surfaces"][0]["vertices"]
    assert scene["geometry"]["walls"][0]["start"]["z"] == 0
    calibration = c.post(f"/api/v1/homes/{home}/calibrations", headers=h, json={"camera_id": camera["id"], "map_id": roomplan.json()["id"], "intrinsics": {}, "extrinsics": {}, "resolution_width": 640, "resolution_height": 480})
    assert calibration.status_code == 200 and calibration.json()["status"] == "active"
    changed = c.patch(f"/api/v1/homes/{home}/cameras/{camera['id']}", headers=h, json={"resolution_width": 1280, "resolution_height": 720})
    assert changed.status_code == 200 and changed.json()["calibrations_invalidated"] is True
    assert c.get(f"/api/v1/homes/{home}/calibrations", headers=h).json()["data"][0]["status"] == "invalidated"


def test_roomplan_camera_registration_tracks_only_active_metric_map(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Hall camera"}).json()
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    first_map = c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json=roomplan_payload).json()
    identity_pose = [
        [1.0, 0.0, 0.0, 1.25],
        [0.0, 1.0, 0.0, 1.55],
        [0.0, 0.0, 1.0, -0.75],
        [0.0, 0.0, 0.0, 1.0],
    ]

    before = c.get(f"/api/v1/homes/{home}/scene", headers=h).json()
    assert before["camera"] is None
    assert before["cameraRegistration"]["status"] == "unavailable"

    registered = c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": first_map["id"], "camera_to_world": identity_pose, "confidence": 0.94, "tracking_state": "normal"},
    )
    assert registered.status_code == 200 and registered.json()["status"] == "positioned"
    positioned = c.get(f"/api/v1/homes/{home}/scene", headers=h).json()["cameraRegistration"]
    assert positioned["cameraId"] == camera["id"]
    assert positioned["cameraToWorld"] == identity_pose

    invalid_matrix = c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": first_map["id"], "camera_to_world": [[1.0, 0.0], [0.0, 1.0]]},
    )
    assert invalid_matrix.status_code == 422

    nonfinite_payload = {"camera_id": camera["id"], "map_id": first_map["id"], "camera_to_world": [row[:] for row in identity_pose]}
    nonfinite_payload["camera_to_world"][0][0] = float("nan")
    nonfinite = c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers={**h, "Content-Type": "application/json"},
        content=json.dumps(nonfinite_payload),
    )
    assert nonfinite.status_code == 422

    needs_rescan = c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": first_map["id"], "camera_to_world": identity_pose, "confidence": 0.4, "tracking_state": "limited"},
    )
    assert needs_rescan.status_code == 200 and needs_rescan.json()["status"] == "needs_rescan"
    assert c.get(f"/api/v1/homes/{home}/scene", headers=h).json()["cameraRegistration"]["cameraToWorld"] is None

    second_map = c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json=roomplan_payload).json()
    current = c.get(f"/api/v1/homes/{home}/scene", headers=h).json()
    assert current["mapId"] == second_map["id"]
    assert current["cameraRegistration"]["status"] == "unavailable"
    assert c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": first_map["id"], "camera_to_world": identity_pose},
    ).status_code == 409

    c.app.state.db.execute("UPDATE cameras SET enabled=0 WHERE id=? AND home_id=?", (camera["id"], home))
    assert c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": second_map["id"], "camera_to_world": identity_pose},
    ).status_code == 404

    other_camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Kitchen camera"}).json()
    legacy_map = c.post(f"/api/v1/homes/{home}/maps", headers=h, json={"map_data": {"zones": []}}).json()
    assert c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": other_camera["id"], "map_id": legacy_map["id"], "camera_to_world": identity_pose},
    ).status_code == 422


def test_assistant_degraded_and_consent_export_delete(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    consent = c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": "camera", "policy_version": "2026-01"})
    assert consent.status_code == 200
    summary = c.post(f"/api/v1/homes/{home}/check-ins", headers=h, json={"transcript": "Hello"})
    assert summary.status_code == 200 and summary.json()["degraded"] is True and summary.json()["event_id"] and summary.json()["inference_status"] in {"disabled", "connection_error", "http_401", "timeout", "invalid_model_response"}
    events = c.get(f"/api/v1/homes/{home}/events", headers=h)
    assert events.status_code == 200 and events.json()["data"][0]["event_type"] == "daily_check_in"
    analytics = c.get(f"/api/v1/homes/{home}/analytics", headers=h)
    assert analytics.status_code == 200 and analytics.json()["data"]["daily_check_in"]["completed_today"] == 1
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


def test_bounded_vision_persists_multiple_people_as_distinct_live_objects(tmp_path):
    c = client(tmp_path, TwoPersonDetector()); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Living room"}).json()
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=h,
        json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True},
    ).status_code == 200
    frame = base64.b64encode(b"two-people-frame").decode()
    payload = {"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480}
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    stable = c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    assert stable.status_code == 200
    assert len(stable.json()["data"]) == 2
    assert len(stable.json()["observations"]) == 2
    people = [item for item in c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h).json()["data"] if item["label"] == "Person"]
    assert len(people) == 2
    assert len({item["id"] for item in people}) == 2
    assert all(item["presenceState"] == "current" for item in people)


def test_fall_pattern_creates_a_reviewable_event(tmp_path):
    c = client(tmp_path, FallSequenceDetector()); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Living room"}).json()
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=h,
        json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True},
    ).status_code == 200

    responses = []
    for step in range(6):
        frame = base64.b64encode(b"\xff\xd8\xff" + bytes([step])).decode()
        response = c.post(
            f"/api/v1/homes/{home}/vision/frames",
            headers=h,
            json={"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480},
        )
        assert response.status_code == 200
        responses.append(response.json())

    safety_events = [event for body in responses for event in body["safety_events"]]
    assert len(safety_events) == 1
    assert safety_events[0]["type"] == "fall_suspected"
    assert safety_events[0]["status"] == "needs_review"
    assert safety_events[0]["confidence"] <= 0.88
    assert safety_events[0]["snapshot_path"]
    assert safety_events[0]["snapshot_content_type"] == "image/jpeg"

    events = c.get(f"/api/v1/homes/{home}/events", headers=h).json()["data"]
    fall_event = next(event for event in events if event["event_type"] == "fall_suspected")
    assert fall_event["status"] == "needs_review"
    assert "not a diagnosis" in fall_event["explanation"]
    assert fall_event["evidence_json"] != "[]"
    assert fall_event["snapshot_path"]
    assert fall_event["snapshot_content_type"] == "image/jpeg"
    snapshot = c.get(fall_event["snapshot_path"], headers=h)
    assert snapshot.status_code == 200
    assert snapshot.headers["content-type"] == "image/jpeg"
    assert snapshot.content.startswith(b"\xff\xd8\xff")
    stored = list((tmp_path / "objects" / "encrypted-clips" / "snapshots" / home).glob("*.bin"))
    assert len(stored) == 1
    assert b"\xff\xd8\xff" not in stored[0].read_bytes()
    analytics = c.get(f"/api/v1/homes/{home}/analytics", headers=h)
    assert analytics.status_code == 200
    assert analytics.json()["data"]["fall"]["total_signals"] == 1
    assert analytics.json()["data"]["fall"]["needs_review"] == 1
    assert "raw_frames" in analytics.json()["data"]["assistant_context"]["excludes"]


def test_anonymous_person_handoff_moves_latest_presence_to_the_new_camera(tmp_path):
    c = client(tmp_path, SinglePersonDetector()); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera_a = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Room A"}).json()
    camera_b = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Room B"}).json()
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=h,
        json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True},
    ).status_code == 200
    frame = base64.b64encode(b"person-handoff-frame").decode()

    def publish(camera_id):
        payload = {"camera_id": camera_id, "frame_base64": frame, "width": 640, "height": 480}
        c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
        c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
        return c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)

    assert publish(camera_a["id"]).status_code == 200
    first = [item for item in c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h).json()["data"] if item["label"] == "Person"]
    assert len(first) == 1
    person_id = first[0]["id"]
    assert first[0]["cameraId"] == camera_a["id"]

    tracks = c.app.state.vision_person_objects
    for key, (object_id, _last_seen, map_id, x, z) in list(tracks.items()):
        if key[0] == home and key[1] == camera_a["id"]:
            tracks[key] = (object_id, datetime.now(timezone.utc) - timedelta(seconds=2), map_id, x, z)

    assert publish(camera_b["id"]).status_code == 200
    after = [item for item in c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h).json()["data"] if item["label"] == "Person"]
    assert len(after) == 1
    assert after[0]["id"] == person_id
    assert after[0]["cameraId"] == camera_b["id"]
    assert after[0]["presenceState"] == "current"


def test_pose_only_roomplan_registration_persists_without_inventing_world_ray(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    camera = c.post(f"/api/v1/homes/{home}/cameras", headers=h, json={"name": "Living room camera"}).json()
    assert c.post(
        f"/api/v1/homes/{home}/consents",
        headers=h,
        json={"purpose": "video_capture", "policy_version": "2026-09-01", "granted": True},
    ).status_code == 200

    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    roomplan = {
        "room_id": None,
        "normalized_scan": {
            "schema_version": "roomplan-normalized.v1",
            "producer": "native-ios",
            "framework": "RoomPlan",
            "units": "m",
            "up_axis": "Y",
            "coordinate_frame": "roomplan-local",
            "geometry_type": "3d",
            "room_id": "living",
            "walls": [],
            "floors": [{
                "id": "floor-living",
                "category": "floor",
                "confidence": "high",
                "center": {"x": 0.0, "y": 0.0, "z": -10.0},
                "dimensions": {"x": 40.0, "y": 0.1, "z": 40.0},
                "transform": identity,
                "vertices": [
                    {"x": -20.0, "y": 0.0, "z": -30.0},
                    {"x": 20.0, "y": 0.0, "z": -30.0},
                    {"x": 20.0, "y": 0.0, "z": 10.0},
                    {"x": -20.0, "y": 0.0, "z": 10.0},
                ],
                "attributes": ["room-boundary"],
            }],
            "openings": [],
            "doors": [],
            "windows": [],
            "objects": [],
            "sections": [{"id": "living", "label": "Living room", "center": {"x": 0.0, "y": 0.0, "z": -10.0}, "story": 0}],
        },
        "scan_metadata": {
            "provenance": "native-roomplan",
            "device_model": "iPhone15,4",
            "lidar": True,
            "roomplan_version": "1.0",
            "units": "m",
            "up_axis": "Y",
            "geometry_type": "3d",
        },
    }
    room_map = c.post(f"/api/v1/homes/{home}/maps/roomplan", headers=h, json=roomplan).json()
    camera_to_world = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    registration = c.post(
        f"/api/v1/homes/{home}/camera-registrations/roomplan",
        headers=h,
        json={"camera_id": camera["id"], "map_id": room_map["id"], "camera_to_world": camera_to_world, "confidence": 0.95, "tracking_state": "normal"},
    )
    assert registration.status_code == 200 and registration.json()["status"] == "positioned"

    frame = base64.b64encode(b"stable-roomplan-frame").decode()
    payload = {"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480, "candidate_labels": ["keys"]}
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    stable = c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=payload)
    assert stable.status_code == 200
    body = stable.json()
    # A pose-only RoomPlan registration has no camera intrinsics. Do not
    # invent a 60 degree FOV just to produce a precise-looking ray; until a
    # visual localization/calibration supplies intrinsics, projection is
    # intentionally zone-level.
    assert body["data"][0]["projection"]["quality"] == "zone-fallback"
    assert body["data"][0]["projection"]["world_xyz"] is None
    assert body["data"][0]["projection"].get("room_zone") is None
    assert body["observations"][0]["zone"] is None

    objects = c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h).json()["data"]
    detected = next(item for item in objects if item["label"] == "Keys")
    assert detected["observation"]["map_id"] == room_map["id"]
    assert detected["zone"] is None
    assert detected["worldPoint"] is None
    assert detected["observation"]["x"] is None
    assert detected["observation"]["y"] is None
    assert detected["observation"]["z"] is None

    # Publisher frames leave candidate_labels empty. Person detection must be
    # part of that default set so calibrated presence can appear on the map.
    person_payload = {"camera_id": camera["id"], "frame_base64": frame, "width": 640, "height": 480}
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=person_payload)
    c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=person_payload)
    person_stable = c.post(f"/api/v1/homes/{home}/vision/frames", headers=h, json=person_payload)
    assert person_stable.status_code == 200
    assert person_stable.json()["data"][0]["label"] == "person"
    people = c.get(f"/api/v1/homes/{home}/objects/last-seen", headers=h).json()["data"]
    person = next(item for item in people if item["label"] == "Person")
    assert person["mapId"] == room_map["id"]
    assert person["worldPoint"] is None


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


def test_clip_upload_keeps_only_latest_video_per_home(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    observation = c.post(f"/api/v1/homes/{home}/observations", headers=h, json={"confidence": 0.7}).json()

    first = c.post(
        f"/api/v1/homes/{home}/events/{observation['event_id']}/clips",
        headers=h,
        json={"object_key": "event/first.mp4", "starts_at": "2026-01-01T00:00:00+00:00", "ends_at": "2026-01-01T00:00:05+00:00"},
    ).json()
    assert c.post(
        f"/api/v1/homes/{home}/clips/{first['id']}/content",
        headers=h,
        json={"content_base64": base64.b64encode(b"first-video").decode()},
    ).status_code == 200

    second = c.post(
        f"/api/v1/homes/{home}/events/{observation['event_id']}/clips",
        headers=h,
        json={"object_key": "event/second.mp4", "starts_at": "2026-01-01T00:01:00+00:00", "ends_at": "2026-01-01T00:01:05+00:00"},
    ).json()
    assert c.post(
        f"/api/v1/homes/{home}/clips/{second['id']}/content",
        headers=h,
        json={"content_base64": base64.b64encode(b"second-video").decode()},
    ).status_code == 200

    clips = c.get(f"/api/v1/homes/{home}/clips", headers=h).json()["data"]
    assert [item["id"] for item in clips] == [second["id"]]
    stored = list((tmp_path / "objects" / "encrypted-clips" / "clips" / home).glob("*.bin"))
    assert [path.stem for path in stored] == [second["id"]]
    assert c.app.state.db.one("SELECT id FROM clips WHERE id=?", (first["id"],)) is None


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


def test_family_access_can_be_edited_and_revoked(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    assert c.post(f"/api/v1/homes/{home}/consents", headers=h, json={"purpose": "family_mode", "policy_version": "2026-09-01"}).status_code == 200
    invite = c.post(f"/api/v1/homes/{home}/family/invites", headers=h, json={"display_name": "Caregiver", "role": "caregiver"}).json()
    joined = c.post("/api/v1/family/invites/accept", json={"code": invite["code"]}).json()
    member_id = joined["user_id"]
    changed = c.patch(f"/api/v1/homes/{home}/family/members/{member_id}", headers=h, json={"role": "resident"})
    assert changed.status_code == 200 and changed.json()["data"]["role"] == "resident"
    assert c.get(f"/api/v1/homes/{home}/family/members", headers={"Authorization": f"Bearer {joined['access_token']}"}).status_code == 401
    removed = c.delete(f"/api/v1/homes/{home}/family/members/{member_id}", headers=h)
    assert removed.status_code == 200 and removed.json()["data"]["id"] == member_id
    assert c.get(f"/api/v1/homes/{home}/family/members", headers=h).status_code == 200
    assert c.delete(f"/api/v1/homes/{home}/family/members/{member_id}", headers=h).status_code == 404


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
    assert summary.status_code == 200 and summary.json()["context_scope"] == "medication plans, medication check-ins, daily check-ins, and bounded fall-safety analytics" and summary.json()["degraded"] is True
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


def test_care_recipient_medication_plan_checkin_and_attribution(tmp_path):
    c = client(tmp_path); token, home = auth(c); h = {"Authorization": f"Bearer {token}"}
    actor_id = c.get("/api/v1/me", headers=h).json()["actor"]["id"]
    recipient = c.post(
        f"/api/v1/homes/{home}/care-recipients",
        headers=h,
        json={"display_name": "María", "relationship": "Mother"},
    ).json()["data"]
    blocked = c.post(
        f"/api/v1/homes/{home}/medication-plans",
        headers=h,
        json={"care_recipient_id": recipient["id"], "name": "Morning", "dose": "1 tablet", "schedule": "Mon,Wed,Fri @ 08:00"},
    )
    assert blocked.status_code == 403
    consent = c.post(
        f"/api/v1/homes/{home}/consents",
        headers=h,
        json={"purpose": "medication_management", "policy_version": "2026-09-01", "care_recipient_id": recipient["id"]},
    )
    assert consent.status_code == 200 and consent.json()["care_recipient_id"] == recipient["id"]
    listed_recipient = c.get(f"/api/v1/homes/{home}/care-recipients", headers=h).json()["data"][0]
    assert listed_recipient["medication_reminders_enabled"] is True

    plan = c.post(
        f"/api/v1/homes/{home}/medication-plans",
        headers=h,
        json={"care_recipient_id": recipient["id"].upper(), "assigned_caregiver_id": actor_id.upper(), "name": "Morning", "dose": "1 tablet", "schedule": "Mon,Wed,Fri @ 08:00"},
    )
    assert plan.status_code == 200 and plan.json()["care_recipient_id"] == recipient["id"]
    assert plan.json()["assigned_caregiver_id"] == actor_id
    plan_id = plan.json()["id"]
    reminders = c.get(
        f"/api/v1/homes/{home}/medication-reminders?day=2026-09-14&care_recipient_id={recipient['id']}",
        headers=h,
    )
    assert reminders.status_code == 200 and len(reminders.json()["data"]) == 1
    scheduled = reminders.json()["data"][0]["scheduled_for"]
    checkin = c.post(
        f"/api/v1/homes/{home}/medication-plans/{plan_id}/check-ins",
        headers=h,
        json={"scheduled_for": scheduled, "status": "taken"},
    )
    assert checkin.status_code == 200 and checkin.json()["data"]["marked_by_name"]
    marker_name = checkin.json()["data"]["marked_by_name"]
    refreshed = c.get(
        f"/api/v1/homes/{home}/medication-reminders?day=2026-09-14&care_recipient_id={recipient['id']}",
        headers=h,
    ).json()["data"][0]
    assert refreshed["status"] == "taken"
    assert refreshed["marked_by_name"] == marker_name
    assert refreshed["updated_at"] is not None

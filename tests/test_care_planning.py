from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app


def test_care_entries_are_consented_shared_and_versioned(tmp_path):
    app = make_app(Settings(database_url="sqlite:///:memory:", object_store_path=tmp_path / "objects", bootstrap_secret="test", env="test", lm_studio_url="http://127.0.0.1:9/v1"))
    c = TestClient(app)
    started = c.post("/api/v1/pairing/start", json={"display_name": "Admin", "home_name": "Test Home"}).json()
    session = c.post("/api/v1/pairing/complete", json={"code": started["pairing_code"]}).json()
    home = session["home_id"]
    headers = {"Authorization": f"Bearer {session['access_token']}"}
    recipient = c.post(f"/api/v1/homes/{home}/care-recipients", headers=headers, json={"display_name": "María"}).json()["data"]
    url = f"/api/v1/homes/{home}/care-entries"
    note = {"care_recipient_id": recipient["id"], "kind": "note", "title": "Good morning", "body": "Went for a walk"}

    assert c.get(url, headers=headers, params={"care_recipient_id": recipient["id"]}).status_code == 403
    assert c.post(url, headers=headers, json=note).status_code == 403
    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "care_planning", "policy_version": "2026-09", "care_recipient_id": recipient["id"]}).status_code == 200

    created = c.post(url, headers=headers, json=note)
    assert created.status_code == 201
    row = created.json()
    assert row["body"] == "Went for a walk" and row["version"] == 1
    assert [item["id"] for item in c.get(url, headers=headers, params={"care_recipient_id": recipient["id"]}).json()["data"]] == [row["id"]]
    assert c.post(url, headers=headers, json={**note, "body": " "}).status_code == 422
    assert c.post(url, headers=headers, json={**note, "title": " "}).status_code == 422
    assert c.post(url, headers=headers, json={**note, "kind": "appointment"}).status_code == 422
    appointment = {"care_recipient_id": recipient["id"], "kind": "appointment", "title": "Clinic", "body": "Bring documents", "location": "Central Clinic", "starts_at": "2026-10-01T10:00:00+02:00", "ends_at": "2026-10-01T11:00:00+02:00", "timezone_name": "Europe/Madrid", "reminder_minutes": 30}
    booked = c.post(url, headers=headers, json=appointment)
    assert booked.status_code == 201
    assert booked.json()["starts_at"] == "2026-10-01T08:00:00+00:00"
    assert c.post(url, headers=headers, json={**appointment, "starts_at": "2026-10-01T10:00:00"}).status_code == 422

    update = {"version": 1, "title": "Doctor", "body": "Bring documents", "location": "Central Clinic", "starts_at": "2026-10-01T10:00:00+02:00", "ends_at": "2026-10-01T11:00:00+02:00", "timezone_name": "Europe/Madrid", "reminder_minutes": 5}
    assert c.patch(f"{url}/{booked.json()['id']}", headers=headers, json=update).json()["version"] == 2
    assert c.patch(f"{url}/{booked.json()['id']}", headers=headers, json=update).status_code == 409
    assert c.delete(f"{url}/{booked.json()['id']}", headers=headers, params={"version": 1}).status_code == 409
    assert c.delete(f"{url}/{booked.json()['id']}", headers=headers, params={"version": 2}).status_code == 200
    assert len(c.get(url, headers=headers, params={"care_recipient_id": recipient["id"]}).json()["data"]) == 1
    assert c.app.state.db.one("SELECT id FROM care_entries WHERE id=?", (booked.json()["id"],)) is None
    assert c.app.state.db.one("SELECT COUNT(*) AS count FROM audit_log WHERE target_id=? AND action='care_entry.remove'", (booked.json()["id"],))["count"] == 1
    exported = c.post(f"/api/v1/homes/{home}/privacy/export", headers=headers)
    assert exported.status_code == 200
    assert row["id"] in [entry["id"] for entry in exported.json()["data"]["care_entries"]]

    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "family_mode", "policy_version": "2026-09"}).status_code == 200
    invite = c.post(f"/api/v1/homes/{home}/family/invites", headers=headers, json={"display_name": "Caregiver", "role": "caregiver"}).json()
    caregiver = c.post("/api/v1/family/invites/accept", json={"code": invite["code"]}).json()
    caregiver_headers = {"Authorization": f"Bearer {caregiver['access_token']}"}
    assert c.get(url, headers=caregiver_headers, params={"care_recipient_id": recipient["id"]}).status_code == 200
    assert c.patch(f"{url}/{row['id']}", headers=caregiver_headers, json={"version": 1, "title": "Changed", "body": "Text"}).status_code == 403
    assert c.delete(f"{url}/{row['id']}", headers=caregiver_headers, params={"version": 1}).status_code == 403

    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "care_planning", "policy_version": "2026-09", "care_recipient_id": recipient["id"], "granted": False}).status_code == 200
    assert c.get(url, headers=headers, params={"care_recipient_id": recipient["id"]}).status_code == 403
    assert c.get(f"/api/v1/homes/00000000-0000-0000-0000-000000000000/care-entries", headers=headers, params={"care_recipient_id": recipient["id"]}).status_code == 403

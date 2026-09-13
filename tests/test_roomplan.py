import hashlib
import io
import zipfile
from pathlib import Path

from app.config import Settings
from app.main import make_app
from fastapi.testclient import TestClient


def client(tmp_path: Path) -> TestClient:
    return TestClient(make_app(Settings(
        database_url="sqlite:///:memory:",
        object_store_path=tmp_path / "objects",
        bootstrap_secret="test",
        env="test",
        lm_studio_url="http://127.0.0.1:9/v1",
    )))


def auth(c: TestClient) -> tuple[str, str]:
    started = c.post("/api/v1/pairing/start", json={"display_name": "Admin", "role": "admin"})
    completed = c.post("/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]})
    return completed.json()["access_token"], completed.json()["home_id"]


def scan_payload() -> dict:
    return {
        "schema_version": "roomplan-normalized.v1",
        "producer": "native-ios",
        "framework": "RoomPlan",
        "units": "m",
        "up_axis": "Y",
        "coordinate_frame": "roomplan-local",
        "geometry_type": "3d",
        "walls": [{
            "id": "wall-1",
            "category": "wall",
            "confidence": "high",
            "center": {"x": 1, "y": 1.4, "z": 0},
            "dimensions": {"x": 2, "y": 2.8, "z": 0.1},
            "transform": [[1, 0, 0, 1], [0, 1, 0, 1.4], [0, 0, 1, 0], [0, 0, 0, 1]],
            "vertices": [
                {"x": 0, "y": 0, "z": 0}, {"x": 2, "y": 0, "z": 0},
                {"x": 2, "y": 2.8, "z": 0}, {"x": 0, "y": 2.8, "z": 0},
            ],
        }],
        "floors": [{
            "id": "floor-1",
            "category": "floor",
            "confidence": "medium",
            "center": {"x": 1, "y": 0, "z": 1},
            "dimensions": {"x": 2, "y": 0.02, "z": 2},
            "transform": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        }],
        "openings": [],
        "doors": [],
        "windows": [],
        "objects": [],
        "sections": [{"id": "section-1", "label": "Kitchen", "center": {"x": 1, "y": 0, "z": 1}, "story": 0}],
    }


def metadata() -> dict:
    return {
        "provenance": "native-roomplan",
        "device_model": "iPhone17,1",
        "lidar": True,
        "roomplan_version": "17",
        "units": "m",
        "up_axis": "Y",
        "geometry_type": "3d",
    }


def usdz_fixture() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("Payload/model.usdc", b"synthetic-test-model")
    return output.getvalue()


def test_roomplan_attachment_is_persisted_downloadable_and_deleted(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    uploaded = c.post(
        f"/api/v1/homes/{home}/maps/roomplan",
        headers=headers,
        json={"normalized_scan": scan_payload(), "scan_metadata": metadata()},
    )
    assert uploaded.status_code == 200
    map_id = uploaded.json()["id"]
    model = usdz_fixture()
    attached = c.put(
        f"/api/v1/homes/{home}/maps/{map_id}/usdz",
        headers={**headers, "Content-Type": "model/vnd.usdz+zip"},
        content=model,
    )
    assert attached.status_code == 200
    assert attached.json()["usdz"]["bytes"] == len(model)
    assert attached.json()["usdz"]["sha256"] == hashlib.sha256(model).hexdigest()
    detail = c.get(f"/api/v1/homes/{home}/maps/{map_id}", headers=headers).json()
    assert detail["usdz"]["available"] is True
    downloaded = c.get(f"/api/v1/homes/{home}/maps/{map_id}/usdz", headers=headers)
    assert downloaded.status_code == 200
    assert downloaded.content == model
    assert downloaded.headers["content-type"] == "model/vnd.usdz+zip"
    scene = c.get(f"/api/v1/homes/{home}/scene", headers=headers).json()
    assert scene["source"] == "roomplan-lidar-3d" and scene["dimension"] == "3d"
    assert scene["geometry"]["surfaces"]
    assert scene["geometry"]["surfaces"][0]["kind"] == "wall"
    assert scene["usdz"]["download_path"].endswith(f"/{map_id}/usdz")

    deleted = c.post(f"/api/v1/homes/{home}/privacy/delete", headers=headers)
    assert deleted.status_code == 200
    assert not (tmp_path / "objects" / "maps" / home / f"{map_id}.usdz").exists()


def test_generic_map_with_3d_looking_json_cannot_unlock_3d(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    generic = c.post(
        f"/api/v1/homes/{home}/maps",
        headers=headers,
        json={"map_data": scan_payload()},
    )
    assert generic.status_code == 200
    assert generic.json()["source"] == "legacy-2d"
    assert generic.json()["dimension"] == "2d"
    scene = c.get(f"/api/v1/homes/{home}/scene", headers=headers).json()
    assert scene["source"] == "legacy-2d" and scene["dimension"] == "2d"
    assert c.put(
        f"/api/v1/homes/{home}/maps/{generic.json()['id']}/usdz",
        headers={**headers, "Content-Type": "model/vnd.usdz+zip"},
        content=usdz_fixture(),
    ).status_code == 422


def test_roomplan_rejects_malformed_geometry_and_usdz(tmp_path):
    c = client(tmp_path)
    token, home = auth(c)
    headers = {"Authorization": f"Bearer {token}"}
    malformed = scan_payload()
    malformed["walls"][0]["dimensions"]["z"] = 0
    assert c.post(
        f"/api/v1/homes/{home}/maps/roomplan",
        headers=headers,
        json={"normalized_scan": malformed, "scan_metadata": metadata()},
    ).status_code == 422
    uploaded = c.post(
        f"/api/v1/homes/{home}/maps/roomplan",
        headers=headers,
        json={"normalized_scan": scan_payload(), "scan_metadata": metadata()},
    )
    map_id = uploaded.json()["id"]
    assert c.put(
        f"/api/v1/homes/{home}/maps/{map_id}/usdz",
        headers={**headers, "Content-Type": "text/plain"},
        content=usdz_fixture(),
    ).status_code == 415
    assert c.put(
        f"/api/v1/homes/{home}/maps/{map_id}/usdz",
        headers={**headers, "Content-Type": "model/vnd.usdz+zip"},
        content=b"not-a-zip",
    ).status_code == 422

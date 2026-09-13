import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app


class FakeRoomLayoutService:
    def __init__(self, *, confidence: float = 0.86, unavailable: bool = False):
        self.confidence = confidence
        self.unavailable = unavailable
        self.calls: list[dict] = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if self.unavailable:
            return {
                "status": "unavailable",
                "source": "camera-cv-2d",
                "dimension": "2d",
                "metric_scale_known": False,
                "model_version": "fake-unavailable-v1",
            }
        return {
            "status": "ready",
            "source": "camera-cv-2d",
            "dimension": "2d",
            "metric_scale_known": False,
            "model_version": "fake-camera-room-v1",
            "confidence": self.confidence,
            "geometry": {
                "coordinate_frame": "camera-relative-image",
                "polygons": [{
                    "id": "room-1",
                    "label": kwargs["room_label"],
                    "points": [{"x": 0.10, "y": 0.12}, {"x": 0.90, "y": 0.12}, {"x": 0.90, "y": 0.88}, {"x": 0.10, "y": 0.88}],
                    "confidence": self.confidence,
                }],
                "walls": [{
                    "id": "wall-1",
                    "start": {"x": 0.10, "y": 0.12},
                    "end": {"x": 0.90, "y": 0.12},
                    "confidence": self.confidence,
                }],
                "camera_pose": {
                    "coordinate_frame": "camera-relative",
                    "position": {"x": 0.5, "y": 0.5, "z": 0.0},
                    "rotation_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                    "confidence": self.confidence,
                },
                "intrinsics": {"source": "fake"},
                "metrics": {
                    "confidence": self.confidence,
                    "reprojection_error_px": 5.0,
                    "homography_inlier_ratio": 0.82,
                },
            },
            "diagnostics": {"raw_frames_persisted": False},
        }


def make_client(tmp_path: Path, service: FakeRoomLayoutService) -> TestClient:
    settings = Settings(
        database_url="sqlite:///:memory:",
        object_store_path=tmp_path / "objects",
        bootstrap_secret="test",
        env="test",
        lm_studio_url="http://127.0.0.1:9/v1",
    )
    return TestClient(make_app(settings, geometry_service=service))


def make_admin_and_publisher(client: TestClient) -> tuple[dict, dict, str, str]:
    owner_start = client.post("/api/v1/pairing/start", json={"display_name": "Admin", "role": "admin"}).json()
    owner = client.post("/api/v1/pairing/complete", json={"code": owner_start["pairing_code"]}).json()
    admin_headers = {"Authorization": f"Bearer {owner['access_token']}"}
    publisher_start = client.post(
        f"/api/v1/homes/{owner['home_id']}/pairing/start",
        headers=admin_headers,
        json={"label": "Hall iPhone"},
    ).json()
    publisher = client.post("/api/v1/pairing/complete", json={"code": publisher_start["pairing_code"]}).json()
    publisher_headers = {"Authorization": f"Bearer {publisher['access_token']}"}
    return admin_headers, publisher_headers, owner["home_id"], publisher_start["pairing_id"]


def sweep_frames(width: int = 640, height: int = 480) -> list[dict]:
    jpeg = base64.b64encode(b"\xff\xd8camera\xff\xd9").decode()
    return [{"frame_base64": jpeg, "width": width, "height": height, "captured_at": "2026-09-13T00:00:00Z"} for _ in range(16)]


def test_camera_sweep_persists_derived_geometry_and_passes_only_device_context(tmp_path):
    service = FakeRoomLayoutService()
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    consent = client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=publisher_headers,
        json={"purpose": "video_capture", "policy_version": "2026-09-01"},
    )
    assert consent.status_code == 200

    started = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation",
        headers=publisher_headers,
        json={"room_label": "Hallway", "orientation": "landscape", "resolution_width": 640, "resolution_height": 480},
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    submitted = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames",
        headers=publisher_headers,
        json={"frames": sweep_frames()},
    )
    assert submitted.status_code == 202

    job = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}",
        headers=admin_headers,
    ).json()
    assert job["status"] == "ready"
    assert job["geometry_status"] == "ready"
    assert job["dimension"] == "2d" and job["source"] == "camera-cv-2d"
    assert job["map_id"]
    assert service.calls[0]["room_label"] == "Hallway"
    assert service.calls[0]["orientation"] == "landscape"
    assert len(service.calls[0]["frames"]) == 16

    current = client.get(f"/api/v1/homes/{home_id}/maps/current", headers=admin_headers).json()
    assert current["source"] == "camera-cv-2d"
    assert current["dimension"] == "2d"
    assert current["metric_scale_known"] is False
    assert current["map_data"]["geometry"]["polygons"][0]["points"][0] == {"x": 0.1, "y": 0.12}
    assert current["map_data"]["geometry"]["walls"]
    assert current["map_data"]["confidence"] == 0.86
    assert not any("camera" in key and "base64" in key for key in current["map_data"])

    calibration = client.get(f"/api/v1/homes/{home_id}/calibrations", headers=admin_headers).json()["data"][0]
    assert calibration["source"] == "camera-cv"
    assert calibration["accuracy_m"] is None
    assert calibration["metrics"]["reprojection_error_px"] == 5.0


def test_camera_sweep_confidence_failure_does_not_save_map(tmp_path):
    service = FakeRoomLayoutService(confidence=0.35)
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    client.post(f"/api/v1/homes/{home_id}/consents", headers=publisher_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01"})
    started = client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation", headers=publisher_headers, json={"resolution_width": 640, "resolution_height": 480}).json()
    job_id = started["job_id"]
    client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames", headers=publisher_headers, json={"frames": sweep_frames()})
    job = client.get(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}", headers=admin_headers).json()
    assert job["status"] == "needs_rescan"
    assert job["map_id"] is None
    assert client.get(f"/api/v1/homes/{home_id}/maps", headers=admin_headers).json()["data"] == []


def test_camera_sweep_rejects_wrong_publisher_and_service_unavailability(tmp_path):
    service = FakeRoomLayoutService(unavailable=True)
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    client.post(f"/api/v1/homes/{home_id}/consents", headers=publisher_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01"})
    started = client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation", headers=publisher_headers, json={"resolution_width": 640, "resolution_height": 480}).json()
    job_id = started["job_id"]
    assert client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames", headers=admin_headers, json={"frames": sweep_frames()}).status_code == 403
    client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames", headers=publisher_headers, json={"frames": sweep_frames()})
    job = client.get(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}", headers=admin_headers).json()
    assert job["status"] == "unavailable"
    assert job["error_code"] == "geometry_service_unavailable"
    assert client.get(f"/api/v1/homes/{home_id}/maps", headers=admin_headers).json()["data"] == []

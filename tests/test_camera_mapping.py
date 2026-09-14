import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import make_app


class FakeRoomLayoutService:
    def __init__(self, *, confidence: float = 0.86, reprojection_error_px: float = 5.0, homography_inlier_ratio: float = 0.82, unavailable: bool = False):
        self.confidence = confidence
        self.reprojection_error_px = reprojection_error_px
        self.homography_inlier_ratio = homography_inlier_ratio
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
                "furniture": [{
                    "id": "bed-1",
                    "label": "Bed",
                    "center": {"x": 0.72, "y": 0.30},
                    "size": {"x": 0.25, "y": 0.16},
                    "confidence": self.confidence,
                }],
                "openings": [{
                    "id": "window-1",
                    "kind": "window",
                    "start": {"x": 0.20, "y": 0.12},
                    "end": {"x": 0.39, "y": 0.12},
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
                    "reprojection_error_px": self.reprojection_error_px,
                    "homography_inlier_ratio": self.homography_inlier_ratio,
                },
            },
            "diagnostics": {"raw_frames_persisted": False},
        }

    def build_visual_landmarks(self, **kwargs):
        self.calls.append({"visual_landmarks": kwargs})
        return {
            "status": "ready",
            "detector": "opencv-orb",
            "landmarks": [
                {"point": [float(index), 1.0, -2.0], "descriptor_base64": base64.b64encode(bytes([index]) * 32).decode(), "response": 1.0}
                for index in range(8)
            ],
            "diagnostics": {"raw_frames_persisted": False},
        }

    def localize_camera(self, **kwargs):
        self.calls.append({"camera_localization": kwargs})
        return {
            "status": "positioned",
            "camera_to_world": [
                [1.0, 0.0, 0.0, 1.25],
                [0.0, 1.0, 0.0, 1.55],
                [0.0, 0.0, 1.0, -0.75],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "confidence": 0.92,
            "inlier_count": 19,
            "match_count": 24,
            "reprojection_error_px": 1.8,
            "intrinsics_source": "estimated-fov",
            "intrinsics": [[554.3, 0.0, 320.0], [0.0, 554.3, 240.0], [0.0, 0.0, 1.0]],
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


def file_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'restart.db'}",
        object_store_path=tmp_path / "objects",
        bootstrap_secret="test",
        env="test",
        lm_studio_url="http://127.0.0.1:9/v1",
    )


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
    assert current["map_data"]["geometry"]["furniture"][0]["label"] == "Bed"
    assert current["map_data"]["geometry"]["openings"][0]["kind"] == "window"
    assert current["map_data"]["confidence"] == 0.86
    assert not any("camera" in key and "base64" in key for key in current["map_data"])

    measured = client.post(
        f"/api/v1/homes/{home_id}/maps/{job['map_id']}/scale",
        headers=admin_headers,
        json={"start": {"x": 0.10, "y": 0.12}, "end": {"x": 0.90, "y": 0.12}, "length_m": 8.0, "label": "North wall"},
    )
    assert measured.status_code == 200
    assert measured.json()["scale"]["status"] == "measured_reference"
    assert measured.json()["scale"]["reference_label"] == "North wall"
    assert measured.json()["scale"]["meters_per_normalized_unit"] == 10.0
    assert measured.json()["map_data"]["geometry"]["scale"]["reference_length_m"] == 8.0

    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers)
    assert scene.status_code == 200
    assert scene.json()["scale"]["method"] == "caregiver_reference"

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


def test_camera_sweep_rejects_live_like_unstable_geometry(tmp_path):
    service = FakeRoomLayoutService(confidence=0.91, reprojection_error_px=27.54, homography_inlier_ratio=0.539)
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    client.post(f"/api/v1/homes/{home_id}/consents", headers=publisher_headers, json={"purpose": "video_capture", "policy_version": "2026-09-01"})
    started = client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation", headers=publisher_headers, json={"resolution_width": 640, "resolution_height": 480}).json()
    job_id = started["job_id"]
    client.post(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames", headers=publisher_headers, json={"frames": sweep_frames()})

    job = client.get(f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}", headers=admin_headers).json()
    assert job["status"] == "needs_rescan"
    assert job["map_id"] is None
    assert job["metrics"]["reprojection_error_px"] == 27.54
    assert job["metrics"]["homography_inlier_ratio"] == 0.539
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


def test_api_restart_makes_interrupted_walkthrough_retryable_without_losing_camera(tmp_path):
    service = FakeRoomLayoutService()
    settings = file_settings(tmp_path)
    client = TestClient(make_app(settings, geometry_service=service))
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=publisher_headers,
        json={"purpose": "video_capture", "policy_version": "2026-09-01"},
    )
    started = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation",
        headers=publisher_headers,
        json={"resolution_width": 640, "resolution_height": 480},
    ).json()
    client.app.state.db.execute(
        "UPDATE camera_map_generation_jobs SET status='processing' WHERE id=?",
        (started["job_id"],),
    )
    client.close()

    restarted = TestClient(make_app(settings, geometry_service=service))
    job = restarted.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{started['job_id']}",
        headers=admin_headers,
    ).json()
    assert job["status"] == "failed"
    assert job["error_code"] == "generation_interrupted"
    assert "camera remains saved" in job["error_message"].lower()
    cameras = restarted.get(f"/api/v1/homes/{home_id}/cameras", headers=admin_headers).json()["data"]
    assert any(camera["id"] == camera_id for camera in cameras)
    restarted.close()


def test_roomplan_visual_landmarks_localize_separate_publisher_camera(tmp_path):
    service = FakeRoomLayoutService()
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    room_map = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=roomplan_payload).json()

    assert client.get(f"/api/v1/homes/{home_id}/maps/current", headers=publisher_headers).status_code == 403
    readiness = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-readiness",
        headers=publisher_headers,
    )
    assert readiness.status_code == 200
    assert readiness.json() == {
        "camera_id": camera_id,
        "map_id": room_map["id"],
        "source": "roomplan-lidar-3d",
        "dimension": "3d",
        "visual_landmarks_ready": False,
        "ready": False,
    }

    other_start = client.post(
        f"/api/v1/homes/{home_id}/pairing/start",
        headers=admin_headers,
        json={"label": "Other camera"},
    ).json()
    other_publisher = client.post("/api/v1/pairing/complete", json={"code": other_start["pairing_code"]}).json()
    other_headers = {"Authorization": f"Bearer {other_publisher['access_token']}"}
    assert client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-readiness",
        headers=other_headers,
    ).status_code == 403

    visual_frame = {
        "frame_base64": base64.b64encode(b"jpeg").decode(),
        "width": 640,
        "height": 480,
        "intrinsics": {"values": [[554.3, 0.0, 320.0], [0.0, 554.3, 240.0], [0.0, 0.0, 1.0]]},
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "captured_at": "2026-09-13T12:00:00Z",
    }
    built = client.post(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/visual-landmarks",
        headers=admin_headers,
        json={"frames": [visual_frame]},
    )
    assert built.status_code == 200
    assert built.json()["status"] == "ready" and built.json()["landmark_count"] == 8
    assert len(service.calls[-1]["visual_landmarks"]["frames"]) == 1
    assert service.calls[-1]["visual_landmarks"]["frames"][0].get("depth_base64") is None
    assert service.calls[-1]["visual_landmarks"]["frames"][0]["camera_to_world"] == {
        "values": visual_frame["camera_to_world"]
    }
    readiness = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-readiness",
        headers=publisher_headers,
    ).json()
    assert readiness["ready"] is True
    assert readiness["visual_landmarks_ready"] is True
    assert "map_data" not in readiness

    client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=publisher_headers,
        json={"purpose": "video_capture", "policy_version": "2026-09-01"},
    )
    localized = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan",
        headers=publisher_headers,
        json={"frames": [{"frame_base64": base64.b64encode(b"fixed-camera").decode(), "width": 640, "height": 480}], "fov_degrees": 60.0},
    )
    assert localized.status_code == 200
    assert localized.json()["status"] == "positioned"
    assert localized.json()["source"] == "visual-roomplan-registration"
    assert localized.json()["inlier_count"] == 19

    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert scene["mapId"] == room_map["id"]
    assert scene["cameraRegistrations"][0]["cameraId"] == camera_id
    assert scene["cameraRegistrations"][0]["cameraToWorld"][0][3] == 1.25
    assert service.calls[-1]["camera_localization"]["landmarks"]


def test_roomplan_visual_landmarks_accumulate_across_incremental_scan_frames(tmp_path):
    class IncrementalVisualService(FakeRoomLayoutService):
        def build_visual_landmarks(self, **kwargs):
            batch = sum(1 for call in self.calls if "visual_landmarks" in call)
            self.calls.append({"visual_landmarks": kwargs})
            return {
                "status": "ready",
                "detector": "opencv-orb",
                "landmarks": [
                    {
                        "point": [float(index), 1.0, -2.0],
                        "descriptor_base64": base64.b64encode(bytes([index + batch + 1]) * 32).decode(),
                        "response": 1.0,
                    }
                    for index in range(8)
                ],
                "diagnostics": {"source_frame_count": len(kwargs["frames"]), "raw_frames_persisted": False},
            }

    service = IncrementalVisualService()
    client = make_client(tmp_path, service)
    admin_headers, _, home_id, _ = make_admin_and_publisher(client)
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    room_map = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=roomplan_payload).json()
    visual_frame = {
        "frame_base64": base64.b64encode(b"jpeg").decode(),
        "width": 640,
        "height": 480,
        "intrinsics": {"values": [[554.3, 0.0, 320.0], [0.0, 554.3, 240.0], [0.0, 0.0, 1.0]]},
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "captured_at": "2026-09-14T11:00:00Z",
    }

    first = client.post(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/visual-landmarks",
        headers=admin_headers,
        json={"frames": [visual_frame]},
    )
    second = client.post(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/visual-landmarks",
        headers=admin_headers,
        json={"frames": [{**visual_frame, "captured_at": "2026-09-14T11:00:01Z"}]},
    )

    assert first.status_code == 200 and first.json()["landmark_count"] == 8
    assert second.status_code == 200 and second.json()["landmark_count"] == 16
    assert second.json()["diagnostics"]["source_frame_count"] == 2
    assert second.json()["diagnostics"]["incremental_batch_count"] == 2
    assert second.json()["diagnostics"]["view_count"] == 2

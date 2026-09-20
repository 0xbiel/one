import base64
import io
import json
import math
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import (
    _camera_localization_search_prior,
    _roomplan_calibration_timeout_message,
    _semantic_candidate_search_support,
    _stabilize_camera_localization_diagnostics,
    make_app,
)


class FakeRoomLayoutService:
    def __init__(self, *, confidence: float = 0.86, reprojection_error_px: float = 5.0, homography_inlier_ratio: float = 0.82, unavailable: bool = False):
        self.confidence = confidence
        self.reprojection_error_px = reprojection_error_px
        self.homography_inlier_ratio = homography_inlier_ratio
        self.unavailable = unavailable
        self.detect_person_visible: bool | None = None
        self.calls: list[dict] = []

    def health(self):
        if self.unavailable:
            return {"status": "unavailable", "runtime": {"device": None}, "model": {"model_version": None}}
        return {"status": "ready", "runtime": {"device": "mps"}, "model": {"model_version": "fake-camera-room-v1"}}

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

    def detect(self, **kwargs):
        self.calls.append({"detect": kwargs})
        if self.detect_person_visible is None:
            return {"status": "unavailable", "detections": []}
        return {
            "status": "ready",
            "model_version": "fake-person-detector",
            "detections": ([{
                "label": "person",
                "confidence": 0.91,
                "bbox": [220.0, 100.0, 360.0, 470.0],
            }] if self.detect_person_visible else []),
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


def test_roomplan_calibration_timeout_message_preserves_solver_context():
    message = _roomplan_calibration_timeout_message(42, "Matching RoomPlan landmarks", 180.0)
    assert "42%" in message
    assert "Matching RoomPlan landmarks" in message
    assert "180 seconds" in message
    assert "camera position was not changed" in message


def test_camera_localization_search_prior_requires_recurring_tight_cluster():
    def row(calibration_id: str, summaries: list[dict]) -> dict:
        return {
            "id": calibration_id,
            "metrics_json": json.dumps({"diagnostics": {"candidate_summaries": summaries}}),
        }

    stable = [
        row("a", [
            {"scene_plausible": True, "landmark_view_id": "scan-view-4", "camera_center": [-0.3904, -0.8438, -1.0177], "selected_fov_degrees": 96.0},
            {"scene_plausible": True, "landmark_view_id": "scan-view-4", "camera_center": [-0.3920, -0.8420, -1.0180], "selected_fov_degrees": 96.0},
        ]),
        row("b", [{"scene_plausible": True, "landmark_view_id": "scan-view-4", "camera_center": [-0.3958, -0.8427, -1.0234], "selected_fov_degrees": 96.0}]),
        row("c", [{"scene_plausible": True, "landmark_view_id": "scan-view-4", "camera_center": [-0.3975, -0.8415, -1.0203], "selected_fov_degrees": 96.0}]),
        row("noise", [{"scene_plausible": True, "landmark_view_id": "scan-view-2", "camera_center": [3.19, 0.39, -0.61], "selected_fov_degrees": 60.0}]),
    ]

    prior = _camera_localization_search_prior(stable)

    assert prior is not None
    assert prior["source"] == "visual-pnp"
    assert prior["support_count"] == 3
    assert prior["landmark_view_id"] == "scan-view-4"
    assert prior["fov_degrees"] == 96.0
    assert prior["mean_residual_m"] < 0.01
    assert math.dist(prior["center"], [-0.3958, -0.8427, -1.0203]) < 0.01
    assert _camera_localization_search_prior(stable[:2]) is None


def test_camera_localization_search_prior_prefers_repeated_semantic_basin():
    def row(calibration_id: str, center: list[float], *, minimum_iou: float = 0.72) -> dict:
        return {
            "id": calibration_id,
            "metrics_json": json.dumps(
                {
                    "diagnostics": {
                        "candidate_summaries": [
                            {
                                "scene_plausible": True,
                                "landmark_view_id": "scan-view-4",
                                "camera_center": [-0.39, -0.84, -1.02],
                                "selected_fov_degrees": 96.0,
                            }
                        ],
                        "semantic_cuboid_candidates": [
                            {
                                "camera_center": center,
                                "selected_fov_degrees": 74.0,
                                "contradictory_object_count": 0,
                                "contradictory_group_count": 0,
                                "contradictions": [],
                                "matched_object_count": 2,
                                "semantic_group_count": 2,
                                "mean_iou": 0.82,
                                "minimum_iou": minimum_iou,
                            }
                        ],
                    }
                }
            ),
        }

    rows = [
        row("a", [1.16, -0.42, 1.80]),
        row("b", [1.14, -0.44, 1.78]),
        row("c", [1.17, -0.43, 1.81]),
        # A weak extra assignment must not become semantic prior evidence.
        row("weak", [0.2, -0.4, -2.0], minimum_iou=0.01),
    ]

    prior = _camera_localization_search_prior(rows)

    assert prior is not None
    assert prior["source"] == "semantic-cuboid"
    assert prior["landmark_view_id"] is None
    assert prior["support_count"] == 3
    assert prior["fov_degrees"] == 74.0
    assert math.dist(prior["center"], [1.16, -0.43, 1.80]) < 0.03


def test_semantic_search_support_tolerates_one_open_vocabulary_outlier():
    candidate = {
        "contradictory_object_count": 0,
        "matched_object_count": 3,
        "semantic_group_count": 2,
        "mean_iou": 0.496,
        "minimum_iou": 0.015,
        "matches": [
            {"label": "chair", "iou": 0.94, "positive_depth_ratio": 1.0},
            {"label": "storage", "iou": 0.53, "positive_depth_ratio": 0.5},
            {"label": "storage", "iou": 0.015, "positive_depth_ratio": 1.0},
        ],
    }

    assert _semantic_candidate_search_support(candidate) is True


def test_semantic_search_support_rejects_explicit_depth_contradiction():
    candidate = {
        "contradictory_object_count": 0,
        "supported_object_count": 2,
        "supported_group_count": 2,
        "supported_mean_iou": 0.85,
        "supported_minimum_iou": 0.74,
        "contradictory_object_count": 1,
        "contradictory_group_count": 1,
        "contradictions": [
            {
                "label": "chair",
                "detection_confidence": 0.73,
                "positive_depth_count": 0,
                "reason": "visible-detection-fully-behind-camera",
            }
        ],
    }

    assert _semantic_candidate_search_support(candidate) is False


def test_semantic_search_support_rejects_legacy_fully_behind_visible_object():
    candidate = {
        "supported_object_count": 2,
        "supported_group_count": 2,
        "supported_mean_iou": 0.85,
        "supported_minimum_iou": 0.74,
        "attempts": [
            {
                "label": "chair",
                "detection_confidence": 0.73,
                "detection_bbox": [600.0, 292.0, 921.0, 720.0],
                "positive_depth_count": 0,
                "rejection_reason": "insufficient-positive-depth-corners",
            }
        ],
    }

    assert _semantic_candidate_search_support(candidate) is False


def test_semantic_search_support_does_not_reject_partial_near_plane_object():
    candidate = {
        "contradictory_object_count": 0,
        "supported_object_count": 2,
        "supported_group_count": 2,
        "supported_mean_iou": 0.75,
        "supported_minimum_iou": 0.55,
        "attempts": [
            {
                "label": "storage",
                "detection_confidence": 0.82,
                "detection_bbox": [820.0, 180.0, 1270.0, 720.0],
                "positive_depth_count": 3,
                "rejection_reason": "insufficient-positive-depth-corners",
            }
        ],
    }

    assert _semantic_candidate_search_support(candidate) is True


def test_semantic_search_support_rejects_legacy_candidate_without_contradiction_marker():
    candidate = {
        "supported_object_count": 2,
        "supported_group_count": 2,
        "supported_mean_iou": 0.88,
        "supported_minimum_iou": 0.75,
    }

    assert _semantic_candidate_search_support(candidate) is False


def test_camera_localization_diagnostics_hold_stable_semantic_prior_during_occlusion():
    diagnostics = {
        "selected_camera_center": [-0.98, -1.15, -0.98],
        "selected_estimate_source": "visual-pnp",
    }
    prior = {
        "center": [0.81, -0.67, 0.82],
        "support_count": 5,
        "mean_residual_m": 0.056,
        "source": "semantic-cuboid",
        "landmark_view_id": None,
        "fov_degrees": 96.0,
    }

    stabilized = _stabilize_camera_localization_diagnostics(diagnostics, prior, positioned=False)

    assert stabilized["selected_estimate_source"] == "temporal-prior"
    assert stabilized["selected_camera_center"] == [0.81, -0.67, 0.82]
    assert stabilized["unstabilized_selected_camera_center"] == [-0.98, -1.15, -0.98]
    assert stabilized["selected_estimate_stabilized"] is True


def test_camera_localization_diagnostics_never_replace_current_semantic_or_positioned_pose():
    prior = {
        "center": [0.81, -0.67, 0.82],
        "support_count": 5,
        "mean_residual_m": 0.056,
        "source": "semantic-cuboid",
    }
    semantic = {
        "selected_camera_center": [0.84, -0.68, 0.79],
        "selected_estimate_source": "semantic-cuboid",
    }
    positioned = {
        "selected_camera_center": [1.4, 1.5, -0.4],
        "selected_estimate_source": "visual-pnp",
    }

    assert _stabilize_camera_localization_diagnostics(semantic, prior, positioned=False) == semantic
    assert _stabilize_camera_localization_diagnostics(positioned, prior, positioned=True) == positioned


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
    object_transform = [
        [1.0, 0.0, 0.0, -0.88],
        [0.0, 1.0, 0.0, 0.30],
        [0.0, 0.0, 1.0, -0.88],
        [0.0, 0.0, 0.0, 1.0],
    ]
    roomplan_payload["normalized_scan"]["objects"] = [{
        "id": "bed-legacy-transform",
        "category": "bed",
        "confidence": "high",
        "center": {"x": -0.88, "y": 0.30, "z": -0.88},
        "dimensions": {"x": 1.20, "y": 0.60, "z": 1.20},
        "transform": object_transform,
        "vertices": [],
        "attributes": [],
    }]
    room_map = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=roomplan_payload).json()

    # Historical RoomPlan rows can have geometry objects without transforms
    # even though normalized_scan still contains the original RoomPlan matrix.
    stored_map = client.app.state.db.one("SELECT map_json FROM room_maps WHERE id=?", (room_map["id"],))
    stored_map_json = json.loads(stored_map["map_json"])
    assert stored_map_json["geometry"]["objects"][0].pop("transform") == object_transform
    client.app.state.db.execute(
        "UPDATE room_maps SET map_json=? WHERE id=?",
        (json.dumps(stored_map_json), room_map["id"]),
    )

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
        "alignment_status": "aligned",
        "home_frame_id": room_map["id"],
        "localization_worker": {
            "status": "ready",
            "ready": True,
            "device": "mps",
            "model_version": "fake-camera-room-v1",
        },
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
    review = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan",
        headers=publisher_headers,
        json={
            "frames": [{"frame_base64": base64.b64encode(b"fixed-camera-review").decode(), "width": 640, "height": 480}],
            "fov_degrees": 60.0,
            "review_only": True,
            "person_anchors": [{"frame_index": 0, "x": 0.5, "y": 0.0, "z": -0.5}],
        },
    )
    assert review.status_code == 200
    assert review.json()["status"] == "positioned"
    assert review.json()["review_required"] is True
    assert review.json()["camera_to_world"][0][3] == 1.25
    direct_progress = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan/progress",
        headers=publisher_headers,
    )
    assert direct_progress.status_code == 200
    assert direct_progress.json()["status"] == "complete"
    assert direct_progress.json()["progress"] == 100
    assert service.calls[-1]["camera_localization"]["person_anchors"] == []
    # Review-only/iPhone-guided calibration keeps RoomPlan object semantics so
    # the progress-job worker can run the same object-detection-assisted pose
    # path as ordinary localization.
    assert service.calls[-1]["camera_localization"]["room_objects"][0]["label"] == "bed"
    scene_before_confirmation = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert scene_before_confirmation["cameraRegistration"]["status"] == "unavailable"
    assert scene_before_confirmation["cameraRegistration"]["cameraToWorld"] is None
    review_row = client.app.state.db.one(
        "SELECT status FROM calibrations WHERE id=?",
        (review.json()["id"],),
    )
    assert review_row["status"] == "needs_review"

    confirmed = client.post(
        f"/api/v1/homes/{home_id}/camera-registrations/roomplan",
        headers=publisher_headers,
        json={
            "camera_id": camera_id,
            "map_id": room_map["id"],
            "camera_to_world": review.json()["camera_to_world"],
            "confidence": review.json()["confidence"],
            "tracking_state": "normal",
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "positioned"
    assert client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()["cameraRegistration"]["cameraToWorld"][0][3] == 1.25
    assert client.post(
        f"/api/v1/homes/{home_id}/camera-registrations/roomplan",
        headers=other_headers,
        json={
            "camera_id": camera_id,
            "map_id": room_map["id"],
            "camera_to_world": review.json()["camera_to_world"],
            "tracking_state": "normal",
        },
    ).status_code == 403

    localized = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan",
        headers=publisher_headers,
        json={"frames": [{"frame_base64": base64.b64encode(b"fixed-camera").decode(), "width": 640, "height": 480}], "fov_degrees": 60.0},
    )
    assert localized.status_code == 200
    assert localized.json()["status"] == "positioned"
    assert localized.json()["source"] == "visual-roomplan-registration"
    assert localized.json()["inlier_count"] == 19

    history = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localization-history",
        headers=publisher_headers,
    )
    assert history.status_code == 200
    history_body = history.json()
    assert history_body["map_id"] == room_map["id"]
    assert history_body["reference"]["camera_center"] == [1.25, 1.55, -0.75]
    assert history_body["attempts"][-1]["status"] == "positioned"
    assert history_body["attempts"][-1]["selected_distance_to_reference_m"] == 0.0

    saved_reference = client.put(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localization-reference",
        headers=admin_headers,
        json={"x": 1.0, "z": -0.5, "source": "test-floor-reference"},
    )
    assert saved_reference.status_code == 200
    history_with_ground_truth = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localization-history",
        headers=admin_headers,
    ).json()
    assert history_with_ground_truth["reference"]["kind"] == "ground-truth-floor"
    assert history_with_ground_truth["reference"]["floor_position"] == [1.0, -0.5]
    assert history_with_ground_truth["distance_metric"] == "horizontal-floor"
    assert history_with_ground_truth["attempts"][-1]["selected_distance_to_reference_m"] == 0.3536
    assert client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localization-history",
        headers=other_headers,
    ).status_code == 403

    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert scene["mapId"] == room_map["id"]
    assert scene["cameraRegistrations"][0]["cameraId"] == camera_id
    assert scene["cameraRegistrations"][0]["cameraToWorld"][0][3] == 1.25
    assert service.calls[-1]["camera_localization"]["landmarks"]


def test_remote_roomplan_calibration_session_uses_publisher_scene_references_and_requires_review(tmp_path):
    service = FakeRoomLayoutService()
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    roomplan_payload["normalized_scan"]["floors"] = [{
        "id": "floor-1",
        "category": "floor",
        "confidence": "high",
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "dimensions": {"x": 4.0, "y": 0.1, "z": 4.0},
        "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "vertices": [
            {"x": -2.0, "y": 0.0, "z": -2.0},
            {"x": 2.0, "y": 0.0, "z": -2.0},
            {"x": 2.0, "y": 0.0, "z": 2.0},
            {"x": -2.0, "y": 0.0, "z": 2.0},
        ],
        "attributes": ["room-floor"],
    }]
    room_map = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=roomplan_payload).json()
    visual_frame = {
        "frame_base64": base64.b64encode(b"jpeg").decode(),
        "width": 640,
        "height": 480,
        "intrinsics": {"values": [[554.3, 0.0, 320.0], [0.0, 554.3, 240.0], [0.0, 0.0, 1.0]]},
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    }
    assert client.post(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/visual-landmarks",
        headers=admin_headers,
        json={"frames": [visual_frame]},
    ).status_code == 200
    assert client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=publisher_headers,
        json={"purpose": "video_capture", "policy_version": "2026-09-01"},
    ).status_code == 200

    started = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session",
        headers=admin_headers,
    )
    assert started.status_code == 200
    session = started.json()
    assert session["mode"] == "scene_reference"
    assert session["status"] == "waiting_for_scene"
    assert session["capture_round_count"] == 3
    assert session["targets"] == []
    assert session["raw_frames_persisted"] is False

    # One reviewed still may be retained intentionally after confirmation; all
    # solve frames remain transient. No person/standing-point anchors are sent.
    fixed_frame = {
        "frame_base64": base64.b64encode(b"\xff\xd8reference-view\xff\xd9").decode(),
        "width": 640,
        "height": 480,
    }
    for capture_round in range(3):
        requested = client.post(
            f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session/request-capture",
            headers=admin_headers,
            json={"target_index": capture_round},
        )
        assert requested.status_code == 200
        assert requested.json()["status"] == "capture_requested"
        submitted = client.post(
            f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session/frames",
            headers=publisher_headers,
            json={"target_index": capture_round, "frames": [fixed_frame, fixed_frame]},
        )
        assert submitted.status_code == 200
        session = submitted.json()

    assert session["status"] == "solving"
    assert session["solve_progress"] >= 1
    assert session["solve_stage"]
    for _ in range(100):
        session = client.get(
            f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session",
            headers=admin_headers,
        ).json()
        if session["status"] != "solving":
            break
        time.sleep(0.01)
    assert session["status"] == "review"
    assert session["solve_progress"] == 100
    assert session["solve_stage"] == "Camera pose solved"
    assert session["captured_target_count"] == 3
    assert session["reference_snapshot_pending"] is True
    assert session["proposal"]["camera_to_world"][0][3] == 1.25
    assert service.calls[-1]["camera_localization"]["person_anchors"] == []
    assert len(service.calls[-1]["camera_localization"]["frames"]) == 6

    camera_during_review = client.get(f"/api/v1/homes/{home_id}/cameras", headers=admin_headers).json()["data"][0]
    assert camera_during_review["calibration_needed"] is True
    assert camera_during_review["roomplan_registration_status"] == "needs_review"

    proposal = session["proposal"]
    confirmed = client.post(
        f"/api/v1/homes/{home_id}/camera-registrations/roomplan",
        headers=admin_headers,
        json={
            "camera_id": camera_id,
            "map_id": room_map["id"],
            "camera_to_world": proposal["camera_to_world"],
            "confidence": proposal["confidence"],
            "tracking_state": "normal",
        },
    )
    assert confirmed.status_code == 200
    committed = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session/commit-reference",
        headers=admin_headers,
    )
    assert committed.status_code == 200
    assert committed.json()["map_id"] == room_map["id"]
    reference = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot",
        headers=admin_headers,
    )
    assert reference.status_code == 200
    assert reference.content == b"\xff\xd8reference-view\xff\xd9"

    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    registration = next(item for item in scene["cameraRegistrations"] if item["cameraId"] == camera_id)
    assert registration["referenceSnapshot"]["mapId"] == room_map["id"]
    camera_after = client.get(f"/api/v1/homes/{home_id}/cameras", headers=admin_headers).json()["data"][0]
    assert camera_after["calibration_needed"] is False
    assert camera_after["roomplan_registration_status"] == "positioned"

    requested_reference = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot/request-capture",
        headers=admin_headers,
    )
    assert requested_reference.status_code == 200
    assert requested_reference.json()["status"] == "capture_requested"
    publisher_request = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot/capture-request",
        headers=publisher_headers,
    )
    assert publisher_request.status_code == 200
    assert publisher_request.json()["request_id"] == requested_reference.json()["request_id"]
    refreshed_frame = {
        "frame_base64": base64.b64encode(b"\xff\xd8refreshed-view\xff\xd9").decode(),
        "width": 1280,
        "height": 720,
    }
    refreshed = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot",
        headers=publisher_headers,
        json=refreshed_frame,
    )
    assert refreshed.status_code == 200
    completed_request = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot/capture-request",
        headers=admin_headers,
    ).json()
    assert completed_request["status"] == "captured"
    assert completed_request["captured_at"]
    refreshed_reference = client.get(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/reference-snapshot",
        headers=admin_headers,
    )
    assert refreshed_reference.content == b"\xff\xd8refreshed-view\xff\xd9"


def test_roomplan_placement_preview_is_scoped_and_serves_usdz(tmp_path):
    service = FakeRoomLayoutService()
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    preview_path = f"/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-placement-preview"

    assert client.get(preview_path, headers=publisher_headers).status_code == 409

    assert client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=admin_headers,
        json={"purpose": "family_mode", "policy_version": "2026-09-01"},
    ).status_code == 200
    caregiver_invite = client.post(
        f"/api/v1/homes/{home_id}/family/invites",
        headers=admin_headers,
        json={"display_name": "Placement reviewer", "role": "caregiver"},
    ).json()
    caregiver = client.post("/api/v1/family/invites/accept", json={"code": caregiver_invite["code"]}).json()
    caregiver_headers = {"Authorization": f"Bearer {caregiver['access_token']}"}

    other_start = client.post(
        f"/api/v1/homes/{home_id}/pairing/start",
        headers=admin_headers,
        json={"label": "Other camera"},
    ).json()
    other_publisher = client.post("/api/v1/pairing/complete", json={"code": other_start["pairing_code"]}).json()
    other_headers = {"Authorization": f"Bearer {other_publisher['access_token']}"}

    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    room_map = client.post(
        f"/api/v1/homes/{home_id}/maps/roomplan",
        headers=admin_headers,
        json=roomplan_payload,
    ).json()

    for headers in (publisher_headers, admin_headers, caregiver_headers):
        preview = client.get(preview_path, headers=headers)
        assert preview.status_code == 200
        assert preview.json()["mapId"] == room_map["id"]
        assert preview.json()["source"] == "roomplan-lidar-3d"
    assert client.get(preview_path, headers=other_headers).status_code == 403
    assert client.get(f"/api/v1/homes/{home_id}/scene", headers=publisher_headers).status_code == 403

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("Payload/model.usdc", b"placement-preview-test-model")
    model = output.getvalue()
    attached = client.put(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/usdz",
        headers={**admin_headers, "Content-Type": "model/vnd.usdz+zip"},
        content=model,
    )
    assert attached.status_code == 200

    usdz_path = f"{preview_path}/usdz"
    for headers in (publisher_headers, admin_headers, caregiver_headers):
        downloaded = client.get(usdz_path, headers=headers)
        assert downloaded.status_code == 200
        assert downloaded.content == model
        assert downloaded.headers["content-type"].startswith("model/vnd.usdz+zip")
    assert client.get(usdz_path, headers=other_headers).status_code == 403
    assert client.get(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/usdz",
        headers=publisher_headers,
    ).status_code == 403


def test_roomplan_visual_localization_rejects_pose_below_scanned_floor(tmp_path):
    class BelowFloorService(FakeRoomLayoutService):
        def localize_camera(self, **kwargs):
            result = super().localize_camera(**kwargs)
            result["camera_to_world"][1][3] = -0.45
            return result

    service = BelowFloorService()
    client = make_client(tmp_path, service)
    admin_headers, publisher_headers, home_id, camera_id = make_admin_and_publisher(client)
    roomplan_payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    roomplan_payload["normalized_scan"]["floors"] = [{
        "id": "floor-main",
        "category": "floor",
        "confidence": "high",
        "center": {"x": 1.0, "y": 0.0, "z": 0.0},
        "dimensions": {"x": 4.0, "y": 0.001, "z": 4.0},
        "transform": identity,
        "vertices": [
            {"x": -1.0, "y": 0.0, "z": -2.0},
            {"x": 3.0, "y": 0.0, "z": -2.0},
            {"x": 3.0, "y": 0.0, "z": 2.0},
            {"x": -1.0, "y": 0.0, "z": 2.0},
        ],
        "attributes": [],
    }]
    room_map = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=roomplan_payload).json()

    visual_frame = {
        "frame_base64": base64.b64encode(b"jpeg").decode(),
        "width": 640,
        "height": 480,
        "intrinsics": {"values": [[554.3, 0.0, 320.0], [0.0, 554.3, 240.0], [0.0, 0.0, 1.0]]},
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "captured_at": "2026-09-13T12:00:00Z",
    }
    assert client.post(
        f"/api/v1/homes/{home_id}/maps/{room_map['id']}/visual-landmarks",
        headers=admin_headers,
        json={"frames": [visual_frame]},
    ).status_code == 200
    assert client.post(
        f"/api/v1/homes/{home_id}/consents",
        headers=publisher_headers,
        json={"purpose": "video_capture", "policy_version": "2026-09-01"},
    ).status_code == 200

    localized = client.post(
        f"/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan",
        headers=publisher_headers,
        json={"frames": [{"frame_base64": base64.b64encode(b"fixed-camera").decode(), "width": 640, "height": 480}]},
    )
    assert localized.status_code == 200
    assert localized.json()["status"] == "needs_rescan"
    assert localized.json()["camera_to_world"] is None
    scene_validation = localized.json()["diagnostics"]["scene_validation"]
    assert scene_validation["accepted"] is False
    assert scene_validation["reason"] == "pose_outside_roomplan_bounds"
    assert scene_validation["camera_height_above_floor_m"] == -0.45

    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert scene["cameraRegistration"]["status"] == "needs_rescan"
    assert scene["cameraRegistration"]["cameraToWorld"] is None

    # Historical versions could persist this same impossible pose as active.
    # The scene contract must suppress it immediately, even before a fresh
    # localization attempt replaces the old calibration row.
    calibration_id = localized.json()["id"]
    client.app.state.db.execute(
        "UPDATE calibrations SET status='active', extrinsics_json=? WHERE id=?",
        (json.dumps({"camera_to_world": [
            [1.0, 0.0, 0.0, 1.25],
            [0.0, 1.0, 0.0, -0.45],
            [0.0, 0.0, 1.0, -0.75],
            [0.0, 0.0, 0.0, 1.0],
        ]}), calibration_id),
    )
    historical_scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert historical_scene["cameraRegistration"]["status"] == "needs_rescan"
    assert historical_scene["cameraRegistration"]["cameraToWorld"] is None


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


def test_aligned_room_fragment_keeps_home_frame_and_existing_camera_calibration(tmp_path):
    service = FakeRoomLayoutService()
    client = make_client(tmp_path, service)
    admin_headers, _, home_id, camera_id = make_admin_and_publisher(client)
    payload = json.loads((Path(__file__).parent / "fixtures" / "roomplan-lidar-valid.json").read_text())
    canonical = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=payload).json()
    camera_to_world = [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.0, -0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    registered = client.post(
        f"/api/v1/homes/{home_id}/camera-registrations/roomplan",
        headers=admin_headers,
        json={
            "camera_id": camera_id,
            "map_id": canonical["id"],
            "camera_to_world": camera_to_world,
            "tracking_state": "normal",
        },
    )
    assert registered.status_code == 200

    fragment_payload = {
        **payload,
        "room_id": "bedroom-2",
        "fragment_to_home": {
            "values": [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
        "alignment_status": "aligned",
    }
    fragment = client.post(f"/api/v1/homes/{home_id}/maps/roomplan", headers=admin_headers, json=fragment_payload)
    assert fragment.status_code == 200
    assert fragment.json()["coordinate_frame"] == "home-world"
    assert fragment.json()["metadata"]["map_scope"] == "fragment"
    assert fragment.json()["metadata"]["canonical_map_id"] == canonical["id"]

    current = client.get(f"/api/v1/homes/{home_id}/maps/current", headers=admin_headers).json()
    assert current["id"] == canonical["id"]
    calibration = client.app.state.db.one("SELECT status FROM calibrations WHERE id=?", (registered.json()["id"],))
    assert calibration["status"] == "active"
    scene = client.get(f"/api/v1/homes/{home_id}/scene", headers=admin_headers).json()
    assert scene["cameraRegistrations"][0]["cameraId"] == camera_id
    assert scene["roomPlanFragments"][0]["mapId"] == fragment.json()["id"]
    assert scene["roomPlanFragments"][0]["alignmentStatus"] == "aligned"

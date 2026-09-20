import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from geometry_service.app import create_app
from geometry_service.config import ServiceSettings


class _FakeRuntime:
    ready = True
    reason = None
    model_version = "fake-runtime"
    device_label = "cpu"

    def __init__(self, _settings: ServiceSettings) -> None:
        pass

    def health(self) -> dict:
        return {
            "status": "ready",
            "mode": "model",
            "model_version": self.model_version,
            "runtime": {"device": self.device_label},
        }


def _settings(workers: int = 2) -> ServiceSettings:
    return ServiceSettings(
        mode="model",
        checkpoint_path=None,
        config_path=Path("geometry_service/model_config.yolo-world.json"),
        device_preference="cpu",
        allow_cpu=True,
        max_frames=20,
        max_frame_bytes=3_000_000,
        max_total_frame_bytes=18_000_000,
        max_request_bytes=26_000_000,
        minimum_confidence=0.6,
        positioning_workers=workers,
    )


def _payload() -> dict:
    landmarks = [
        {
            "point": [float(index), 0.0, -2.0],
            "descriptor_base64": "AA==",
            "response": 0.1,
            "view_id": f"view-{index % 2}",
        }
        for index in range(6)
    ]
    return {
        "schema_version": "roomplan-camera-localization.v1",
        "landmarks": landmarks,
        "frames": [{"frame_base64": "AA==", "width": 10, "height": 10}],
    }


class PositioningConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_localization_solves_can_enter_two_positioning_workers(self) -> None:
        barrier = threading.Barrier(2, timeout=1.0)

        def fake_localize(_payload) -> dict:
            barrier.wait()
            return {
                "status": "needs_rescan",
                "coordinate_frame": "roomplan-local",
                "camera_to_world": None,
                "confidence": None,
                "inlier_count": 0,
                "match_count": 0,
                "reprojection_error_px": None,
                "intrinsics_source": "estimated-fov-sweep",
                "intrinsics": None,
                "diagnostics": {"test": "parallel"},
            }

        with patch("geometry_service.app.RoomLayoutRuntime", _FakeRuntime), patch("geometry_service.app.localize_camera", fake_localize):
            app = create_app(_settings(2))
            transport = httpx.ASGITransport(app=app)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    first, second = await asyncio.gather(
                        client.post("/v1/camera-localization", json=_payload()),
                        client.post("/v1/camera-localization", json=_payload()),
                    )
                    health = await client.get("/health")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["positioning"]["workers"], 2)
        self.assertEqual(health.json()["positioning"]["detector_workers"], 1)

    async def test_camera_localization_job_exposes_solver_progress(self) -> None:
        first_progress = threading.Event()
        finish = threading.Event()

        def fake_localize(_payload, *, progress_callback=None) -> dict:
            self.assertIsNotNone(progress_callback)
            progress_callback(24, "Matching RoomPlan landmarks")
            first_progress.set()
            finish.wait(timeout=1.0)
            progress_callback(73, "Checking pose hypotheses")
            return {
                "status": "needs_rescan",
                "coordinate_frame": "roomplan-local",
                "camera_to_world": None,
                "confidence": 0.0,
                "inlier_count": 0,
                "match_count": 0,
                "reprojection_error_px": None,
                "intrinsics_source": "estimated-fov-sweep",
                "intrinsics": None,
                "diagnostics": {"test": "progress"},
            }

        with patch("geometry_service.app.RoomLayoutRuntime", _FakeRuntime), patch("geometry_service.app.localize_camera", fake_localize):
            app = create_app(_settings(1))
            transport = httpx.ASGITransport(app=app)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    started = await client.post("/v1/camera-localization/jobs", json=_payload())
                    self.assertEqual(started.status_code, 200)
                    job_id = started.json()["job_id"]
                    await asyncio.get_running_loop().run_in_executor(None, first_progress.wait, 1.0)
                    running = await client.get(f"/v1/camera-localization/jobs/{job_id}")
                    self.assertEqual(running.status_code, 200)
                    self.assertEqual(running.json()["status"], "running")
                    self.assertEqual(running.json()["progress"], 24)
                    self.assertEqual(running.json()["stage"], "Matching RoomPlan landmarks")
                    finish.set()
                    for _ in range(50):
                        completed = await client.get(f"/v1/camera-localization/jobs/{job_id}")
                        if completed.json()["status"] == "complete":
                            break
                        await asyncio.sleep(0.01)

        self.assertEqual(completed.json()["status"], "complete")
        self.assertEqual(completed.json()["progress"], 100)
        self.assertEqual(completed.json()["result"]["status"], "needs_rescan")

    async def test_camera_localization_job_masks_reflective_regions_before_solver(self) -> None:
        class WindowRuntime(_FakeRuntime):
            def detect_jpeg(
                self,
                _jpeg: bytes,
                _width: int,
                _height: int,
                candidate_labels: list[str],
                *,
                minimum_confidence: float | None = None,
            ) -> list[dict]:
                self_test.assertIn("window", candidate_labels)
                self_test.assertIn("person", candidate_labels)
                self_test.assertIn("bed", candidate_labels)
                self_test.assertEqual(minimum_confidence, 0.10)
                return [
                    {
                        "label": "window",
                        "confidence": 0.91,
                        "bbox": [0.0, 0.0, 6.0, 10.0],
                    },
                    {
                        "label": "bed",
                        "confidence": 0.88,
                        "bbox": [2.0, 2.0, 9.0, 9.0],
                    },
                ]

        self_test = self
        saw_window = threading.Event()

        def fake_localize(payload, *, progress_callback=None) -> dict:
            self.assertIsNotNone(progress_callback)
            self.assertTrue(any(item.label == "window" for item in payload.object_detections))
            self.assertTrue(any(item.label == "bed" for item in payload.object_detections))
            self.assertEqual([item.label for item in payload.room_objects], ["bed"])
            saw_window.set()
            return {
                "status": "needs_rescan",
                "coordinate_frame": "roomplan-local",
                "camera_to_world": None,
                "confidence": 0.0,
                "inlier_count": 0,
                "match_count": 0,
                "reprojection_error_px": None,
                "intrinsics_source": "estimated-fov-sweep",
                "intrinsics": None,
                "diagnostics": {"test": "reflective-mask"},
            }

        with patch("geometry_service.app.RoomLayoutRuntime", WindowRuntime), patch(
            "geometry_service.app.localize_camera", fake_localize
        ):
            app = create_app(_settings(1))
            transport = httpx.ASGITransport(app=app)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    request_payload = _payload()
                    request_payload["room_objects"] = [
                        {
                            "id": "bed-1",
                            "label": "bed",
                            "center": {"x": 0.0, "y": 0.4, "z": -2.0},
                            "dimensions": {"x": 1.6, "y": 0.8, "z": 2.0},
                            "confidence": 0.95,
                        }
                    ]
                    started = await client.post("/v1/camera-localization/jobs", json=request_payload)
                    self.assertEqual(started.status_code, 200)
                    job_id = started.json()["job_id"]
                    for _ in range(50):
                        completed = await client.get(f"/v1/camera-localization/jobs/{job_id}")
                        if completed.json()["status"] == "complete":
                            break
                        await asyncio.sleep(0.01)

        self.assertTrue(saw_window.is_set())
        result = completed.json()["result"]
        self.assertEqual(result["diagnostics"]["semantic_object_detection"]["detected_labels"], ["bed", "window"])
        self.assertEqual(result["diagnostics"]["semantic_object_detection"]["room_object_count"], 1)


if __name__ == "__main__":
    unittest.main()

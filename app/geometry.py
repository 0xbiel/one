"""Local room-layout service integration.

The API deliberately keeps the geometry model outside the backend process. A
host-side service can use PyTorch/MPS (or another local implementation) while
the API owns authorization, bounded request handling, persistence, and the
privacy boundary around temporary RGB frames.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .config import Settings


class RoomLayoutServiceError(RuntimeError):
    """A non-retryable or malformed room-layout service response."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RoomLayoutServiceUnavailable(RoomLayoutServiceError):
    """The configured local service cannot be reached or is not configured."""


class RoomLayoutService(Protocol):
    def infer(
        self,
        *,
        camera_id: str,
        resolution_width: int,
        resolution_height: int,
        frames: Sequence[dict],
        room_label: str = "Room",
        orientation: str = "portrait",
    ) -> object:
        """Infer derived room geometry from temporary RGB frames."""

    def detect(self, *, frame_base64: str, width: int, height: int, candidate_labels: Sequence[str]) -> object:
        """Run the configured real local detector on one transient frame."""

    def build_visual_landmarks(self, *, map_id: str, frames: Sequence[dict]) -> object:
        """Build derived RoomPlan visual landmarks from transient RGB/depth samples."""

    def localize_camera(
        self,
        *,
        landmarks: Sequence[dict],
        frames: Sequence[dict],
        intrinsics: list[list[float]] | None,
        fov_degrees: float,
        room_zones: Sequence[dict] = (),
        search_prior: dict | None = None,
        room_objects: Sequence[dict] = (),
    ) -> object:
        """Estimate a fixed camera pose in RoomPlan coordinates."""


class HttpRoomLayoutService:
    """Call the internal ``POST /v1/room-layout`` service over HTTP.

    The service URL is intentionally configurable so Docker can point at a
    host-side MPS process (for example ``host.docker.internal``) without
    changing API code. Frame bytes are included only in this request and are
    never written by this adapter.
    """

    _MAX_RESPONSE_BYTES = 4_000_000

    def __init__(self, settings: Settings):
        self.settings = settings

    def _endpoint(self, path: str = "room-layout") -> str:
        base = (self.settings.geometry_service_url or "").strip().rstrip("/")
        if not base:
            raise RoomLayoutServiceUnavailable("not_configured")
        if base.endswith("/v1/room-layout"):
            base = base.removesuffix("/room-layout")
        if base.endswith("/v1"):
            return f"{base}/{path}"
        return f"{base}/v1/{path}"

    def _post(self, path: str, payload: dict) -> object:
        request = urllib.request.Request(
            self._endpoint(path),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.geometry_timeout_seconds) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
            if len(raw) > self._MAX_RESPONSE_BYTES:
                raise RoomLayoutServiceError("response_too_large")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RoomLayoutServiceError("invalid_response")
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in {408, 425, 429, 503, 504}:
                raise RoomLayoutServiceUnavailable(f"http_{exc.code}") from exc
            raise RoomLayoutServiceError(f"http_{exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise RoomLayoutServiceUnavailable("connection_error") from exc
        except json.JSONDecodeError as exc:
            raise RoomLayoutServiceError("invalid_response") from exc

    def infer(
        self,
        *,
        camera_id: str,
        resolution_width: int,
        resolution_height: int,
        frames: Sequence[dict],
        room_label: str = "Room",
        orientation: str = "portrait",
    ) -> object:
        payload = {
            "schema_version": "room-layout-request.v1",
            "camera_id": camera_id,
            "resolution": {
                "width": resolution_width,
                "height": resolution_height,
            },
            "room_label": room_label,
            "orientation": orientation,
            "output": {
                "dimension": "2d",
                "coordinate_frame": "camera-relative-image",
                "require_gpu": self.settings.geometry_require_gpu,
            },
            "frames": [
                {
                    "frame_base64": frame["frame_base64"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "captured_at": frame.get("captured_at"),
                }
                for frame in frames
            ],
        }
        return self._post("room-layout", payload)

    def detect(self, *, frame_base64: str, width: int, height: int, candidate_labels: Sequence[str]) -> object:
        return self._post(
            "vision/detect",
            {
                "frame_base64": frame_base64,
                "width": width,
                "height": height,
                "candidate_labels": list(candidate_labels),
            },
        )

    def build_visual_landmarks(self, *, map_id: str, frames: Sequence[dict]) -> object:
        return self._post(
            "visual-landmarks",
            {
                "schema_version": "roomplan-visual-landmarks.v1",
                "map_id": map_id,
                "frames": list(frames),
            },
        )

    def localize_camera(
        self,
        *,
        landmarks: Sequence[dict],
        frames: Sequence[dict],
        intrinsics: list[list[float]] | None,
        fov_degrees: float,
        room_zones: Sequence[dict] = (),
        search_prior: dict | None = None,
        room_objects: Sequence[dict] = (),
    ) -> object:
        payload: dict = {
            "schema_version": "roomplan-camera-localization.v1",
            "landmarks": list(landmarks),
            "frames": list(frames),
            "fov_degrees": fov_degrees,
            "room_zones": list(room_zones),
            "room_objects": list(room_objects),
        }
        if intrinsics is not None:
            payload["intrinsics"] = {"values": intrinsics}
        if search_prior is not None:
            payload["search_prior"] = search_prior
        return self._post("camera-localization", payload)

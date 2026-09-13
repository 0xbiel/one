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


class HttpRoomLayoutService:
    """Call the internal ``POST /v1/room-layout`` service over HTTP.

    The service URL is intentionally configurable so Docker can point at a
    host-side MPS process (for example ``host.docker.internal``) without
    changing API code. Frame bytes are included only in this request and are
    never written by this adapter.
    """

    _MAX_RESPONSE_BYTES = 2_000_000

    def __init__(self, settings: Settings):
        self.settings = settings

    def _endpoint(self) -> str:
        base = (self.settings.geometry_service_url or "").strip().rstrip("/")
        if not base:
            raise RoomLayoutServiceUnavailable("not_configured")
        if base.endswith("/v1/room-layout"):
            return base
        if base.endswith("/v1"):
            return f"{base}/room-layout"
        return f"{base}/v1/room-layout"

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
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.geometry_timeout_seconds
            ) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
            if len(raw) > self._MAX_RESPONSE_BYTES:
                raise RoomLayoutServiceError("response_too_large")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RoomLayoutServiceError("invalid_response")
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in {408, 425, 429, 500, 502, 503, 504}:
                raise RoomLayoutServiceUnavailable(f"http_{exc.code}") from exc
            raise RoomLayoutServiceError(f"http_{exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise RoomLayoutServiceUnavailable("connection_error") from exc
        except json.JSONDecodeError as exc:
            raise RoomLayoutServiceError("invalid_response") from exc

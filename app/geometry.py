"""Local room-layout service integration.

The API deliberately keeps the geometry model outside the backend process. A
host-side service can use PyTorch/MPS (or another local implementation) while
the API owns authorization, bounded request handling, persistence, and the
privacy boundary around temporary RGB frames.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator, Protocol, Sequence

from .config import Settings


LocalizationProgressCallback = Callable[[int, str], None]
_localization_progress_callback: ContextVar[LocalizationProgressCallback | None] = ContextVar(
    "one_localization_progress_callback",
    default=None,
)


@contextmanager
def localization_progress(callback: LocalizationProgressCallback) -> Iterator[None]:
    """Expose progress from a calibration solve without changing public route signatures."""

    token = _localization_progress_callback.set(callback)
    try:
        yield
    finally:
        _localization_progress_callback.reset(token)


def current_localization_progress_callback() -> LocalizationProgressCallback | None:
    """Return the progress callback active in this request, if any."""

    return _localization_progress_callback.get()


class RoomLayoutServiceError(RuntimeError):
    """A non-retryable or malformed room-layout service response."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RoomLayoutServiceUnavailable(RoomLayoutServiceError):
    """The configured local service cannot be reached or is not configured."""


class RoomLayoutService(Protocol):
    def health(self) -> object:
        """Return the local worker health payload without running inference."""

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

    def detect(
        self,
        *,
        frame_base64: str,
        width: int,
        height: int,
        candidate_labels: Sequence[str],
        include_faces: bool = False,
    ) -> object:
        """Run the configured real local detector on one transient frame."""

    def enroll_faces(self, *, frames: Sequence[dict]) -> object:
        """Extract transient face templates for a consented enrollment."""

    def build_visual_landmarks(self, *, map_id: str, frames: Sequence[dict]) -> object:
        """Build derived RoomPlan visual landmarks from transient RGB/depth samples."""

    def localize_camera(
        self,
        *,
        landmarks: Sequence[dict],
        frames: Sequence[dict],
        intrinsics: list[list[float]] | None,
        fov_degrees: float | None,
        room_zones: Sequence[dict] = (),
        search_prior: dict | None = None,
        room_objects: Sequence[dict] = (),
        person_anchors: Sequence[dict] = (),
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
    _LOCALIZATION_RETRYABLE_CODES = frozenset({
        "connection_error",
        "http_408",
        "http_425",
        "http_429",
        "http_503",
        "http_504",
    })
    _LOCALIZATION_RETRY_DELAYS_SECONDS = (0.5, 1.5)

    def __init__(self, settings: Settings):
        self.settings = settings

    def _base_url(self) -> str:
        base = (self.settings.geometry_service_url or "").strip().rstrip("/")
        if not base:
            raise RoomLayoutServiceUnavailable("not_configured")
        if base.endswith("/v1/room-layout"):
            base = base.removesuffix("/v1/room-layout")
        elif base.endswith("/v1"):
            base = base.removesuffix("/v1")
        return base

    def _endpoint(self, path: str = "room-layout") -> str:
        return f"{self._base_url()}/v1/{path}"

    def health(self) -> object:
        request = urllib.request.Request(
            f"{self._base_url()}/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(3.0, self.settings.geometry_timeout_seconds)) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
            if len(raw) > self._MAX_RESPONSE_BYTES:
                raise RoomLayoutServiceError("response_too_large")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RoomLayoutServiceError("invalid_response")
            return result
        except urllib.error.HTTPError as exc:
            raise RoomLayoutServiceUnavailable(f"http_{exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise RoomLayoutServiceUnavailable("connection_error") from exc
        except json.JSONDecodeError as exc:
            raise RoomLayoutServiceError("invalid_response") from exc

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
        except TimeoutError as exc:
            raise RoomLayoutServiceUnavailable("timeout") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            code = "timeout" if isinstance(reason, TimeoutError) else "connection_error"
            raise RoomLayoutServiceUnavailable(code) from exc
        except OSError as exc:
            raise RoomLayoutServiceUnavailable("connection_error") from exc
        except json.JSONDecodeError as exc:
            raise RoomLayoutServiceError("invalid_response") from exc

    def _get(self, path: str, *, timeout: float | None = None) -> object:
        request = urllib.request.Request(
            self._endpoint(path),
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout if timeout is not None else self.settings.geometry_timeout_seconds,
            ) as response:
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
        except TimeoutError as exc:
            raise RoomLayoutServiceUnavailable("timeout") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            code = "timeout" if isinstance(reason, TimeoutError) else "connection_error"
            raise RoomLayoutServiceUnavailable(code) from exc
        except OSError as exc:
            raise RoomLayoutServiceUnavailable("connection_error") from exc
        except json.JSONDecodeError as exc:
            raise RoomLayoutServiceError("invalid_response") from exc

    def _localize_camera_job(self, payload: dict, callback: LocalizationProgressCallback) -> object:
        started: object | None = None
        for attempt, delay in enumerate((0.0, *self._LOCALIZATION_RETRY_DELAYS_SECONDS)):
            if delay:
                time.sleep(delay)
            try:
                started = self._post("camera-localization/jobs", payload)
                break
            except RoomLayoutServiceUnavailable as exc:
                if exc.code not in self._LOCALIZATION_RETRYABLE_CODES or attempt == len(self._LOCALIZATION_RETRY_DELAYS_SECONDS):
                    raise
        if started is None:  # pragma: no cover - the loop either returns or raises
            raise RoomLayoutServiceUnavailable("connection_error")
        if not isinstance(started, dict) or not isinstance(started.get("job_id"), str):
            raise RoomLayoutServiceError("invalid_response")
        job_id = started["job_id"]
        last_progress = -1
        last_growth_at = time.monotonic()
        poll_timeout = min(5.0, self.settings.geometry_timeout_seconds)
        stall_timeout = self.settings.geometry_localization_stall_timeout_seconds

        while True:
            try:
                state = self._get(f"camera-localization/jobs/{job_id}", timeout=poll_timeout)
            except RoomLayoutServiceUnavailable as exc:
                if time.monotonic() - last_growth_at >= stall_timeout:
                    raise RoomLayoutServiceUnavailable("timeout") from exc
                time.sleep(0.5)
                continue

            progress = state.get("progress")
            if isinstance(progress, (int, float)):
                progress_value = max(0, min(100, int(progress)))
                if progress_value > last_progress:
                    last_progress = progress_value
                    last_growth_at = time.monotonic()
                    callback(progress_value, str(state.get("stage") or "Solving camera pose"))

            status = state.get("status")
            if status == "complete":
                result = state.get("result")
                if not isinstance(result, dict):
                    raise RoomLayoutServiceError("invalid_response")
                if last_progress < 100:
                    callback(100, "Camera pose solved")
                return result
            if status == "failed":
                code = str(state.get("error_code") or "solver_failed")
                if code == "input_error":
                    raise RoomLayoutServiceError("http_422")
                raise RoomLayoutServiceError(code)
            if time.monotonic() - last_growth_at >= stall_timeout:
                raise RoomLayoutServiceUnavailable("timeout")
            time.sleep(0.5)

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

    def detect(
        self,
        *,
        frame_base64: str,
        width: int,
        height: int,
        candidate_labels: Sequence[str],
        include_faces: bool = False,
    ) -> object:
        return self._post(
            "vision/detect",
            {
                "frame_base64": frame_base64,
                "width": width,
                "height": height,
                "candidate_labels": list(candidate_labels),
                "include_faces": include_faces,
            },
        )

    def enroll_faces(self, *, frames: Sequence[dict]) -> object:
        return self._post("face/enroll", {"frames": list(frames)})

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
        fov_degrees: float | None,
        room_zones: Sequence[dict] = (),
        search_prior: dict | None = None,
        room_objects: Sequence[dict] = (),
        person_anchors: Sequence[dict] = (),
    ) -> object:
        payload: dict = {
            "schema_version": "roomplan-camera-localization.v1",
            "landmarks": list(landmarks),
            "frames": list(frames),
            "room_zones": list(room_zones),
            "room_objects": list(room_objects),
            "person_anchors": list(person_anchors),
        }
        if fov_degrees is not None:
            payload["fov_degrees"] = fov_degrees
        if intrinsics is not None:
            payload["intrinsics"] = {"values": intrinsics}
        if search_prior is not None:
            payload["search_prior"] = search_prior
        progress_callback = _localization_progress_callback.get()
        if progress_callback is not None:
            return self._localize_camera_job(payload, progress_callback)
        for attempt, delay in enumerate((0.0, *self._LOCALIZATION_RETRY_DELAYS_SECONDS)):
            if delay:
                time.sleep(delay)
            try:
                return self._post("camera-localization", payload)
            except RoomLayoutServiceUnavailable as exc:
                if exc.code not in self._LOCALIZATION_RETRYABLE_CODES or attempt == len(self._LOCALIZATION_RETRY_DELAYS_SECONDS):
                    raise
        raise AssertionError("camera localization retry loop did not return or raise")

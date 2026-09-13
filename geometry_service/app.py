"""FastAPI entrypoint for the local camera-room geometry service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ServiceSettings
from .contracts import RoomLayoutRequest, RoomLayoutResponse
from .frames import FrameInputError, decode_request_frames
from .geometry import ConfidenceBelowThreshold, ModelOutputError, normalize_model_output
from .mock import deterministic_room_layout
from .model_input import DependencyUnavailable, build_torch_batch
from .runtime import RuntimeInferenceError, RuntimeUnavailable, RoomLayoutRuntime


def _response_payload(
    runtime: RoomLayoutRuntime,
    *,
    status: str,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    payload = RoomLayoutResponse(
        status=status,  # type: ignore[arg-type]
        model_version=runtime.model_version,
        confidence=confidence,
        reason=reason,
        diagnostics={
            "reason": reason,
            "raw_frames_persisted": False,
            **(diagnostics or {}),
        },
    )
    return payload.model_dump(mode="json")


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    service_settings = settings or ServiceSettings.from_env()
    runtime = RoomLayoutRuntime(service_settings)
    api = FastAPI(
        title="ONE local geometry service",
        version="0.1.0",
        description="In-memory camera-room-2d inference; no raw frame persistence.",
    )

    @api.middleware("http")
    async def enforce_request_bound(request: Request, call_next: Any) -> Any:
        if request.url.path == "/v1/room-layout":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    oversized = int(content_length) > service_settings.max_request_bytes
                except ValueError:
                    oversized = True
                if oversized:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "status": "failed",
                            "code": "request_too_large",
                            "message": "room-layout request exceeds the configured in-memory bound",
                            "raw_frames_persisted": False,
                        },
                    )
        return await call_next(request)

    @api.get("/health")
    async def health() -> JSONResponse:
        health_payload = runtime.health()
        return JSONResponse(
            status_code=200 if runtime.ready else 503,
            content=health_payload,
        )

    @api.post("/v1/room-layout", response_model=RoomLayoutResponse)
    async def room_layout(payload: RoomLayoutRequest) -> Any:
        if not runtime.ready:
            return JSONResponse(
                status_code=503,
                content=_response_payload(
                    runtime,
                    status="unavailable",
                    reason=runtime.reason or "geometry runtime is unavailable",
                    diagnostics={
                        "device": runtime.device_label,
                        "mode": runtime.mode,
                    },
                ),
            )

        if (
            payload.output
            and payload.output.require_gpu
            and runtime.mode == "model"
            and runtime.device_label not in {"mps", "cuda"}
        ):
            return JSONResponse(
                status_code=503,
                content=_response_payload(
                    runtime,
                    status="unavailable",
                    reason="request_requires_gpu_but_no GPU runtime is active",
                    diagnostics={"device": runtime.device_label, "mode": runtime.mode},
                ),
            )

        samples = []
        batch = None
        try:
            samples = decode_request_frames(payload, service_settings)
            base_diagnostics: dict[str, Any] = {
                "mode": runtime.mode,
                "device": runtime.device_label,
                "camera_id": payload.camera_id,
                "received_frame_count": len(samples),
                "frame_dimensions": [
                    {"width": sample.width, "height": sample.height} for sample in samples
                ],
                "orientation": payload.orientation,
                "raw_frames_persisted": False,
            }

            if runtime.mode == "mock":
                raw_output = deterministic_room_layout(
                    [sample.jpeg_bytes for sample in samples],
                    payload.room_label,
                    payload.orientation,
                )
            else:
                batch, preprocess_diagnostics = build_torch_batch(
                    samples,
                    runtime.model_config,
                    runtime.device_label,
                )
                base_diagnostics.update(preprocess_diagnostics)
                raw_output = runtime.predict(batch)

            return normalize_model_output(
                raw_output,
                room_label=payload.room_label,
                model_version=runtime.model_version,
                minimum_confidence=runtime.minimum_confidence,
                base_diagnostics=base_diagnostics,
            )
        except FrameInputError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "failed",
                    "code": exc.code,
                    "message": exc.message,
                    "raw_frames_persisted": False,
                },
            )
        except ConfidenceBelowThreshold as exc:
            return JSONResponse(
                status_code=422,
                content=_response_payload(
                    runtime,
                    status="needs_rescan",
                    reason="confidence_below_threshold",
                    confidence=round(exc.confidence, 6),
                    diagnostics={"minimum_confidence": exc.threshold},
                ),
            )
        except (DependencyUnavailable, RuntimeUnavailable) as exc:
            return JSONResponse(
                status_code=503,
                content=_response_payload(
                    runtime,
                    status="unavailable",
                    reason=str(exc),
                    diagnostics={"device": runtime.device_label, "mode": runtime.mode},
                ),
            )
        except ModelOutputError as exc:
            return JSONResponse(
                status_code=502,
                content=_response_payload(
                    runtime,
                    status="failed",
                    reason="model_output_invalid",
                    diagnostics={"detail": str(exc)},
                ),
            )
        except RuntimeInferenceError as exc:
            return JSONResponse(
                status_code=500,
                content=_response_payload(
                    runtime,
                    status="failed",
                    reason="model_inference_failed",
                    diagnostics={"detail": str(exc)},
                ),
            )
        finally:
            # Explicitly drop references so request bytes cannot outlive the call.
            batch = None
            samples.clear()

    return api


app = create_app()

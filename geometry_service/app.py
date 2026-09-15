"""FastAPI entrypoint for the local camera-room geometry service."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ServiceSettings
from .contracts import (
    CameraLocalizationRequest,
    CameraLocalizationObjectDetection,
    CameraLocalizationResponse,
    RoomLayoutRequest,
    RoomLayoutResponse,
    VisionFrameRequest,
    VisionFrameResponse,
    VisualLandmarkBuildRequest,
    VisualLandmarkBuildResponse,
)
from .frames import FrameInputError, decode_jpeg, decode_request_frames
from .geometry import ConfidenceBelowThreshold, ModelOutputError, normalize_model_output
from .localization import LocalizationInputError, build_visual_landmarks, localize_camera
from .model_input import DependencyUnavailable, build_torch_batch
from .runtime import RuntimeInferenceError, RuntimeNeedsRescan, RuntimeUnavailable, RoomLayoutRuntime


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
        if request.url.path in {"/v1/room-layout", "/v1/vision/detect", "/v1/visual-landmarks", "/v1/camera-localization"}:
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
                status_code=200,
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
        except RuntimeNeedsRescan as exc:
            return JSONResponse(
                status_code=200,
                content=_response_payload(
                    runtime,
                    status="needs_rescan",
                    reason="insufficient_room_structure",
                    diagnostics={
                        "detail": str(exc),
                        "device": runtime.device_label,
                        "mode": runtime.mode,
                    },
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

    @api.post("/v1/vision/detect", response_model=VisionFrameResponse)
    async def vision_detect(payload: VisionFrameRequest) -> Any:
        if not runtime.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "model_version": runtime.model_version,
                    "detections": [],
                    "diagnostics": {"reason": runtime.reason or "vision runtime is unavailable", "raw_frames_persisted": False},
                },
            )
        try:
            jpeg = decode_jpeg(payload.frame_base64, service_settings)
            detections = runtime.detect_jpeg(jpeg, payload.width, payload.height, payload.candidate_labels)
            return {
                "status": "ready",
                "model_version": runtime.model_version,
                "detections": detections,
                "diagnostics": {"device": runtime.device_label, "raw_frames_persisted": False},
            }
        except FrameInputError as exc:
            return JSONResponse(status_code=exc.status_code, content={"status": "failed", "model_version": runtime.model_version, "detections": [], "diagnostics": {"reason": exc.code, "raw_frames_persisted": False}})
        except (RuntimeUnavailable, RuntimeInferenceError) as exc:
            return JSONResponse(status_code=503, content={"status": "unavailable", "model_version": runtime.model_version, "detections": [], "diagnostics": {"reason": str(exc), "raw_frames_persisted": False}})

    @api.post("/v1/visual-landmarks", response_model=VisualLandmarkBuildResponse)
    async def visual_landmarks(payload: VisualLandmarkBuildRequest) -> Any:
        try:
            return build_visual_landmarks(payload)
        except LocalizationInputError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "failed",
                    "schema_version": "roomplan-visual-landmarks.v1",
                    "detector": "opencv-orb",
                    "landmarks": [],
                    "diagnostics": {"reason": str(exc), "raw_frames_persisted": False},
                },
            )
        except Exception as exc:  # pragma: no cover - defensive service boundary
            return JSONResponse(
                status_code=500,
                content={
                    "status": "failed",
                    "schema_version": "roomplan-visual-landmarks.v1",
                    "detector": "opencv-orb",
                    "landmarks": [],
                    "diagnostics": {"reason": "landmark_build_failed", "detail": str(exc)[:200], "raw_frames_persisted": False},
                },
            )

    @api.post("/v1/camera-localization", response_model=CameraLocalizationResponse)
    async def camera_localization(payload: CameraLocalizationRequest) -> Any:
        try:
            object_detection_diagnostics: dict[str, Any] = {
                "status": "skipped",
                "room_object_count": len(payload.room_objects),
                "detected_count": 0,
                "detected_labels": [],
                "detections": [],
            }
            if payload.room_objects or payload.person_anchors:
                prompt_aliases = {
                    "bed": ["bed"],
                    "chair": ["chair"],
                    "table": ["table", "desk", "dining table"],
                    "storage": ["cabinet", "shelf", "bookcase", "wardrobe", "dresser", "nightstand"],
                    "sofa": ["sofa", "couch"],
                }
                candidate_labels: list[str] = []
                for room_object in payload.room_objects:
                    category = room_object.label.strip().lower()
                    candidate_labels.extend(prompt_aliases.get(category, [category]))
                # People are transient occluders during ordinary fixed-camera
                # localization. In guided calibration the same local detection
                # becomes a deliberate floor-point correspondence.
                candidate_labels = list(dict.fromkeys(["person", *(label for label in candidate_labels if label)]))[:32]
                if runtime.ready and candidate_labels:
                    object_detections: list[CameraLocalizationObjectDetection] = []
                    frame_diagnostics: list[dict[str, Any]] = []
                    detection_error: str | None = None
                    for frame_index, frame in enumerate(payload.frames):
                        encoded = frame.frame_base64.strip()
                        if encoded.startswith("data:image/jpeg;base64,"):
                            encoded = encoded.partition(",")[2]
                        jpeg = base64.b64decode(encoded, validate=True)
                        try:
                            raw_detections = runtime.detect_jpeg(jpeg, frame.width, frame.height, candidate_labels)
                        except (RuntimeInferenceError, RuntimeUnavailable) as exc:
                            detection_error = str(exc) or "object detection failed"
                            frame_diagnostics.append(
                                {
                                    "frame_index": frame_index,
                                    "status": "unavailable",
                                    "reason": detection_error,
                                }
                            )
                            continue
                        frame_items = [
                            CameraLocalizationObjectDetection(
                                frame_index=frame_index,
                                label=str(item["label"]),
                                confidence=float(item["confidence"]),
                                bbox=[float(value) for value in item["bbox"]],
                            )
                            for item in raw_detections
                            if isinstance(item, dict)
                            and isinstance(item.get("label"), str)
                            and isinstance(item.get("confidence"), (int, float))
                            and isinstance(item.get("bbox"), list)
                            and len(item["bbox"]) == 4
                        ]
                        object_detections.extend(frame_items)
                        frame_diagnostics.append(
                            {
                                "frame_index": frame_index,
                                "status": "ready",
                                "detected_count": len(frame_items),
                                "detected_labels": sorted({item.label for item in frame_items}),
                            }
                        )
                    payload = payload.model_copy(update={"object_detections": object_detections})
                    object_detection_diagnostics = {
                        "status": "ready" if object_detections or detection_error is None else "unavailable",
                        "reason": detection_error if detection_error and not object_detections else None,
                        "room_object_count": len(payload.room_objects),
                        "detected_count": len(object_detections),
                        "detected_labels": sorted({item.label for item in object_detections}),
                        "detections": [
                            {
                                "frame_index": item.frame_index,
                                "label": item.label,
                                "confidence": round(float(item.confidence), 6),
                                "bbox": [round(float(value), 3) for value in item.bbox],
                            }
                            for item in object_detections
                        ],
                        "frames": frame_diagnostics,
                        "candidate_labels": candidate_labels,
                    }
                elif not runtime.ready:
                    object_detection_diagnostics["status"] = "unavailable"
                    object_detection_diagnostics["reason"] = runtime.reason or "vision runtime is unavailable"
                else:
                    object_detection_diagnostics["status"] = "skipped"
                    object_detection_diagnostics["reason"] = "roomplan_object_labels_are_not_supported_by_the_detector"
            result = localize_camera(payload)
            if isinstance(result, dict):
                diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
                result["diagnostics"] = {
                    **diagnostics,
                    "semantic_object_detection": object_detection_diagnostics,
                }
            return result
        except (binascii.Error, ValueError) as exc:
            # The localization path will report malformed frame input below;
            # this branch keeps the worker boundary explicit if object-seed
            # preparation rejects the transient JPEG first.
            return JSONResponse(
                status_code=422,
                content={
                    "status": "failed",
                    "coordinate_frame": "roomplan-local",
                    "camera_to_world": None,
                    "confidence": None,
                    "inlier_count": 0,
                    "match_count": 0,
                    "reprojection_error_px": None,
                    "intrinsics_source": "estimated-fov",
                    "intrinsics": None,
                    "diagnostics": {"reason": str(exc), "raw_frames_persisted": False},
                },
            )
        except LocalizationInputError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "failed",
                    "coordinate_frame": "roomplan-local",
                    "camera_to_world": None,
                    "confidence": None,
                    "inlier_count": 0,
                    "match_count": 0,
                    "reprojection_error_px": None,
                    "intrinsics_source": "estimated-fov",
                    "intrinsics": None,
                    "diagnostics": {"reason": str(exc), "raw_frames_persisted": False},
                },
            )
        except Exception as exc:  # pragma: no cover - defensive service boundary
            return JSONResponse(
                status_code=500,
                content={
                    "status": "failed",
                    "coordinate_frame": "roomplan-local",
                    "camera_to_world": None,
                    "confidence": None,
                    "inlier_count": 0,
                    "match_count": 0,
                    "reprojection_error_px": None,
                    "intrinsics_source": "estimated-fov",
                    "intrinsics": None,
                    "diagnostics": {"reason": "camera_localization_failed", "detail": str(exc)[:200], "raw_frames_persisted": False},
                },
            )

    return api


app = create_app()

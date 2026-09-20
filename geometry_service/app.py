"""FastAPI entrypoint for the local camera-room geometry service."""

from __future__ import annotations

import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
import threading
import time
from typing import Any
import uuid

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
    positioning_executor = ThreadPoolExecutor(
        max_workers=service_settings.positioning_workers,
        thread_name_prefix="one-positioning",
    )
    detector_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="one-detector",
    )
    # Keep a small bounded admission window in front of the executor. This lets
    # multiple fixed cameras solve concurrently without allowing an arbitrary
    # number of large, in-memory localization requests to queue behind them.
    positioning_slots = asyncio.Semaphore(service_settings.positioning_workers * 2)
    localization_jobs: dict[str, dict[str, Any]] = {}
    localization_jobs_lock = threading.Lock()
    localization_tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            for task in list(localization_tasks):
                task.cancel()
            positioning_executor.shutdown(wait=False, cancel_futures=True)
            detector_executor.shutdown(wait=False, cancel_futures=True)

    api = FastAPI(
        title="ONE local geometry service",
        version="0.1.0",
        description="In-memory camera-room-2d inference; no raw frame persistence.",
        lifespan=lifespan,
    )
    api.state.positioning_workers = service_settings.positioning_workers

    async def run_positioning_work(function: Any, *args: Any, **kwargs: Any) -> Any:
        async with positioning_slots:
            loop = asyncio.get_running_loop()
            call = partial(function, *args, **kwargs)
            return await loop.run_in_executor(positioning_executor, call)

    async def run_detector_work(function: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        call = partial(function, *args, **kwargs)
        return await loop.run_in_executor(detector_executor, call)

    @api.middleware("http")
    async def enforce_request_bound(request: Request, call_next: Any) -> Any:
        if request.url.path in {"/v1/room-layout", "/v1/vision/detect", "/v1/visual-landmarks", "/v1/camera-localization"} or request.url.path.startswith("/v1/camera-localization/jobs"):
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

    def localization_job_view(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "result": job.get("result"),
            "error_code": job.get("error_code"),
            "error": job.get("error"),
            "raw_frames_persisted": False,
        }

    def update_localization_job_progress(job_id: str, progress: int, stage: str) -> None:
        with localization_jobs_lock:
            job = localization_jobs.get(job_id)
            if not job or job.get("status") != "running":
                return
            next_progress = max(int(job.get("progress") or 0), min(99, int(progress)))
            if next_progress > int(job.get("progress") or 0):
                job["progress"] = next_progress
                job["stage"] = stage
                job["updated_at"] = time.monotonic()

    async def prepare_localization_feature_masks(
        payload: CameraLocalizationRequest,
        *,
        progress_callback: Any | None = None,
    ) -> tuple[CameraLocalizationRequest, dict[str, Any]]:
        """Detect semantic, transient, and reflective regions before localization.

        Guided calibration uses the progress-job endpoint, so run the same kind
        of local semantic detection used by the synchronous localization path.
        RoomPlan furniture detections can seed the pose, while people and
        reflective openings are retained for feature masking.
        """
        detect_jpeg = getattr(runtime, "detect_jpeg", None)
        if not runtime.ready or not callable(detect_jpeg):
            return payload, {
                "status": "unavailable",
                "reason": runtime.reason or "vision runtime is unavailable",
                "detected_count": 0,
                "detected_labels": [],
                "frames": [],
            }

        prompt_aliases = {
            "bed": ["bed"],
            "chair": ["chair"],
            "table": ["table", "desk", "dining table"],
            "storage": ["cabinet", "shelf", "bookcase", "wardrobe", "dresser", "nightstand"],
            "sofa": ["sofa", "couch"],
        }
        semantic_labels: list[str] = []
        for room_object in payload.room_objects:
            category = room_object.label.strip().lower()
            semantic_labels.extend(prompt_aliases.get(category, [category]))
        labels = list(
            dict.fromkeys(
                [
                    "person",
                    "window",
                    "mirror",
                    "glass door",
                    "sliding glass door",
                    *(label for label in semantic_labels if label),
                ]
            )
        )[:32]
        detections: list[CameraLocalizationObjectDetection] = []
        frame_diagnostics: list[dict[str, Any]] = []
        frame_count = max(1, len(payload.frames))
        for frame_index, frame in enumerate(payload.frames):
            if progress_callback is not None:
                progress_callback(
                    2 + int(5 * (frame_index + 1) / frame_count),
                    f"Masking people and reflective regions in reference frame {frame_index + 1} of {frame_count}",
                )
            encoded = frame.frame_base64.strip()
            if encoded.startswith("data:image/jpeg;base64,"):
                encoded = encoded.partition(",")[2]
            try:
                jpeg = base64.b64decode(encoded, validate=True)
                raw = await run_detector_work(
                    detect_jpeg,
                    jpeg,
                    frame.width,
                    frame.height,
                    labels,
                    minimum_confidence=0.10,
                )
            except (binascii.Error, ValueError, RuntimeInferenceError, RuntimeUnavailable) as exc:
                frame_diagnostics.append(
                    {
                        "frame_index": frame_index,
                        "status": "unavailable",
                        "reason": str(exc)[:200],
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
                for item in raw
                if isinstance(item, dict)
                and isinstance(item.get("label"), str)
                and isinstance(item.get("confidence"), (int, float))
                and isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4
            ]
            detections.extend(frame_items)
            frame_diagnostics.append(
                {
                    "frame_index": frame_index,
                    "status": "ready",
                    "detected_count": len(frame_items),
                    "detected_labels": sorted({item.label.strip().lower() for item in frame_items}),
                }
            )

        merged = [*payload.object_detections, *detections]
        prepared = payload.model_copy(update={"object_detections": merged})
        diagnostics = {
            "status": "ready",
            "room_object_count": len(payload.room_objects),
            "candidate_labels": labels,
            "detected_count": len(detections),
            "detected_labels": sorted({item.label.strip().lower() for item in detections}),
            "detections": [
                {
                    "frame_index": item.frame_index,
                    "label": item.label,
                    "confidence": round(float(item.confidence), 6),
                    "bbox": [round(float(value), 3) for value in item.bbox],
                }
                for item in detections
            ],
            "frames": frame_diagnostics,
        }
        return prepared, diagnostics

    async def run_localization_job(job_id: str, payload: CameraLocalizationRequest) -> None:
        try:
            payload, mask_detection_diagnostics = await prepare_localization_feature_masks(
                payload,
                progress_callback=lambda progress, stage: update_localization_job_progress(job_id, progress, stage),
            )
            result = await run_positioning_work(
                localize_camera,
                payload,
                progress_callback=lambda progress, stage: update_localization_job_progress(job_id, progress, stage),
            )
        except LocalizationInputError as exc:
            with localization_jobs_lock:
                job = localization_jobs.get(job_id)
                if job:
                    job.update(
                        status="failed",
                        stage="Localization input was rejected",
                        error_code="input_error",
                        error=str(exc)[:240],
                        finished_at=time.monotonic(),
                    )
            return
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            with localization_jobs_lock:
                job = localization_jobs.get(job_id)
                if job:
                    job.update(
                        status="failed",
                        stage="Localization solver failed",
                        error_code="solver_failed",
                        error=str(exc)[:240],
                        finished_at=time.monotonic(),
                    )
            return
        if isinstance(result, dict):
            diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
            result["diagnostics"] = {
                **diagnostics,
                "semantic_object_detection": mask_detection_diagnostics,
            }
        with localization_jobs_lock:
            job = localization_jobs.get(job_id)
            if job:
                job.update(
                    status="complete",
                    progress=100,
                    stage="Camera pose solved" if result.get("status") == "positioned" else "Localization finished without a confident pose",
                    result=result,
                    finished_at=time.monotonic(),
                )

    @api.post("/v1/camera-localization/jobs")
    async def start_camera_localization_job(payload: CameraLocalizationRequest) -> Any:
        if service_settings.mode == "model" and not service_settings.allow_cpu and (
            not runtime.ready or runtime.device_label not in {"mps", "cuda"}
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "code": "gpu_runtime_unavailable",
                    "message": "Camera localization requires the configured MPS/CUDA runtime.",
                    "diagnostics": {"device": runtime.device_label, "mode": runtime.mode},
                    "raw_frames_persisted": False,
                },
            )
        # The progress job is the iPhone-guided path, so it must accept the same
        # RoomPlan semantic objects as synchronous localization. Legacy guided
        # person anchors remain unsupported here because fixed-camera
        # calibration treats people as transient occluders.
        if payload.person_anchors:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "failed",
                    "code": "person_anchors_not_supported_for_progress_job",
                    "message": "progress jobs do not accept guided person anchors",
                    "raw_frames_persisted": False,
                },
            )
        now = time.monotonic()
        with localization_jobs_lock:
            stale = [
                job_id
                for job_id, job in localization_jobs.items()
                if isinstance(job.get("finished_at"), (int, float)) and now - float(job["finished_at"]) > 600.0
            ]
            for stale_job_id in stale:
                localization_jobs.pop(stale_job_id, None)
            job_id = str(uuid.uuid4())
            localization_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "progress": 1,
                "stage": "Preparing fixed-camera reference frames",
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error_code": None,
                "error": None,
            }
            response = localization_job_view(localization_jobs[job_id])
        task = asyncio.create_task(run_localization_job(job_id, payload))
        localization_tasks.add(task)
        task.add_done_callback(localization_tasks.discard)
        return response

    @api.get("/v1/camera-localization/jobs/{job_id}")
    async def camera_localization_job(job_id: str) -> Any:
        with localization_jobs_lock:
            job = localization_jobs.get(job_id)
            if not job:
                return JSONResponse(status_code=404, content={"status": "missing", "raw_frames_persisted": False})
            return localization_job_view(job)

    @api.get("/health")
    async def health() -> JSONResponse:
        health_payload = runtime.health()
        health_payload["positioning"] = {
            "executor": "thread-pool",
            "workers": service_settings.positioning_workers,
            "queue_bound": service_settings.positioning_workers * 2,
            "detector_workers": 1,
            "detector_serialized": True,
        }
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
        if service_settings.mode == "model" and not service_settings.allow_cpu and (
            not runtime.ready or runtime.device_label not in {"mps", "cuda"}
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "code": "gpu_runtime_unavailable",
                    "message": "Camera localization requires the configured MPS/CUDA runtime.",
                    "diagnostics": {"device": runtime.device_label, "mode": runtime.mode},
                    "raw_frames_persisted": False,
                },
            )
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
                candidate_labels = list(
                    dict.fromkeys(
                        [
                            "person",
                            "window",
                            "mirror",
                            "glass door",
                            "sliding glass door",
                            *(label for label in candidate_labels if label),
                        ]
                    )
                )[:32]
                if runtime.ready and candidate_labels:
                    object_detections: list[CameraLocalizationObjectDetection] = []
                    frame_diagnostics: list[dict[str, Any]] = []
                    detection_error: str | None = None
                    dedicated_person_mode = False
                    dedicated_person_frame_count = 0
                    for frame_index, frame in enumerate(payload.frames):
                        encoded = frame.frame_base64.strip()
                        if encoded.startswith("data:image/jpeg;base64,"):
                            encoded = encoded.partition(",")[2]
                        jpeg = base64.b64decode(encoded, validate=True)
                        try:
                            # A transient person is an occluder, not scene
                            # geometry. Use a lower detection floor during
                            # localization so a partially visible person is
                            # more likely to be masked before ORB matching.
                            # Furniture still has its own >=0.20 assignment
                            # gate in localization.py.
                            raw_detections = await run_detector_work(
                                runtime.detect_jpeg,
                                jpeg,
                                frame.width,
                                frame.height,
                                candidate_labels,
                                minimum_confidence=0.10,
                            )
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
                        main_person_detections = [
                            item
                            for item in raw_detections
                            if isinstance(item, dict)
                            and str(item.get("label") or "").strip().lower() == "person"
                            and isinstance(item.get("confidence"), (int, float))
                            and isinstance(item.get("bbox"), list)
                            and len(item["bbox"]) == 4
                        ]
                        main_has_person = bool(main_person_detections)
                        suspicious_broad_person = any(
                            float(item["confidence"]) < 0.65
                            and max(0.0, float(item["bbox"][2]) - float(item["bbox"][0]))
                            * max(0.0, float(item["bbox"][3]) - float(item["bbox"][1]))
                            >= 0.55 * float(frame.width * frame.height)
                            for item in main_person_detections
                        )
                        person_fallback_error: str | None = None
                        # YOLO-World can lose the broad ``person`` prompt when
                        # it competes with several furniture prompts in the
                        # same open-vocabulary pass. Probe the first burst frame
                        # once with a person-only prompt. If that finds an
                        # occluder, keep the dedicated pass enabled for the
                        # rest of this short fixed-camera burst so ORB never
                        # learns features from the moving people.
                        if dedicated_person_mode or suspicious_broad_person or (frame_index == 0 and not main_has_person):
                            try:
                                person_only = await run_detector_work(
                                    runtime.detect_jpeg,
                                    jpeg,
                                    frame.width,
                                    frame.height,
                                    ["person"],
                                    minimum_confidence=0.05,
                                )
                            except (RuntimeInferenceError, RuntimeUnavailable) as exc:
                                person_fallback_error = str(exc) or "person detection failed"
                                person_only = []
                            person_only = [
                                item
                                for item in person_only
                                if isinstance(item, dict)
                                and str(item.get("label") or "").strip().lower() == "person"
                                and isinstance(item.get("confidence"), (int, float))
                                and float(item["confidence"]) >= 0.05
                            ]
                            if frame_index == 0 and person_only:
                                dedicated_person_mode = True
                            if person_only:
                                dedicated_person_frame_count += 1
                                raw_detections = [
                                    item
                                    for item in raw_detections
                                    if not (
                                        isinstance(item, dict)
                                        and str(item.get("label") or "").strip().lower() == "person"
                                    )
                                ] + person_only
                            elif suspicious_broad_person:
                                # A broad low-confidence open-vocabulary
                                # person box is not strong enough evidence to
                                # erase most of the image when the dedicated
                                # person prompt cannot reproduce it.
                                raw_detections = [
                                    item
                                    for item in raw_detections
                                    if not (
                                        isinstance(item, dict)
                                        and str(item.get("label") or "").strip().lower() == "person"
                                    )
                                ]
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
                                "person_detection_mode": (
                                    "dedicated-fallback"
                                    if any(item.label.strip().lower() == "person" for item in frame_items) and dedicated_person_mode
                                    else (
                                        "dedicated-verification"
                                        if suspicious_broad_person
                                        else "candidate-labels"
                                    )
                                ),
                                **({"person_detection_error": person_fallback_error} if person_fallback_error else {}),
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
                        "dedicated_person_detection": dedicated_person_mode,
                        "dedicated_person_frame_count": dedicated_person_frame_count,
                    }
                elif not runtime.ready:
                    object_detection_diagnostics["status"] = "unavailable"
                    object_detection_diagnostics["reason"] = runtime.reason or "vision runtime is unavailable"
                else:
                    object_detection_diagnostics["status"] = "skipped"
                    object_detection_diagnostics["reason"] = "roomplan_object_labels_are_not_supported_by_the_detector"
            # PnP/FOV search is CPU-heavy OpenCV work. Run it off the ASGI
            # event loop so independent camera positioning requests can make
            # progress in parallel. The shared YOLO model remains protected by
            # RoomLayoutRuntime's model lock, while each pose solve is isolated
            # to request-local arrays and transient frames.
            result = await run_positioning_work(localize_camera, payload)
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

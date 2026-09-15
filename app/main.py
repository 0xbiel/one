import asyncio
import base64
import hashlib
import ipaddress
import json
import math
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Sequence
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .config import Settings, get_settings
from .db import Database, now_iso
from .events import EventBus, sse
from .geometry import (
    HttpRoomLayoutService,
    RoomLayoutService,
    RoomLayoutServiceError,
    RoomLayoutServiceUnavailable,
)
from .integrations import LMStudioAdapter, livekit_jwt, verify_livekit_webhook
from .media import EncryptedLocalClipStore
from .roomplan import (
    ARVideoMapIn,
    RoomPlanMapIn,
    arkit_video_geometry,
    build_arkit_video_usdz,
    roomplan_geometry,
    roomplan_usdz_metadata,
    validate_roomplan_usdz,
)
from .security import expired, hash_secret, new_pairing_code, new_token, iso_after, normalize_email
from .storage import LocalObjectStore
from .vision import Calibration, CameraVisionPipeline, Detector, Frame, LocalServiceDetector


class PairStart(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    email: str | None = Field(default=None, max_length=254)
    home_name: str = Field(default="ONE Home", min_length=1, max_length=120)
    care_setting: str = Field(default="home", pattern="^(home|residence)$")
    support_focus: str = Field(default="general", pattern="^(general|mci)$")
    role: str = Field(default="admin", pattern="^(admin|resident|caregiver)$")


class PairStartResponse(BaseModel):
    pairing_code: str
    expires_in_seconds: int
    home_id: str
    user_id: str
    role: str


class PairingStatusResponse(BaseModel):
    pairing_id: str
    home_id: str
    status: str = Field(pattern="^(pending|connected|expired)$")
    expires_at: str
    connected_at: str | None = None
    device: dict

class DevicePairingStart(BaseModel):
    # `display_name` is accepted as a compatibility alias for the web client;
    # this endpoint always creates a publisher membership regardless of it.
    label: str | None = Field(default=None, min_length=1, max_length=120)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    expires_in_seconds: int = Field(default=600, ge=60, le=900)

class PairComplete(BaseModel): code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class CameraReconnectIn(BaseModel):
    camera_id: str = Field(min_length=1, max_length=120)
    reconnect_token: str = Field(min_length=24, max_length=512)


MAP_JOB_STATUSES = ("collecting", "processing", "ready", "needs_rescan", "unavailable", "failed")
MAP_FRAME_MAX_BYTES = 3_000_000
MAP_BATCH_MAX_BYTES = 18_000_000
MAP_FRAME_MAX_COUNT = 20
MAP_COLLECTING_STALE_AFTER = timedelta(minutes=15)
ROOMPLAN_CALIBRATION_SESSION_TTL = timedelta(minutes=10)
ROOMPLAN_USDZ_MAX_BYTES = 50 * 1024 * 1024
ROOMPLAN_USDZ_CONTENT_TYPES = {
    "model/vnd.usdz+zip",
    "application/zip",
    "application/octet-stream",
}


class CameraMapGenerationStartIn(BaseModel):
    room_id: str | None = None
    room_label: str | None = Field(default=None, min_length=1, max_length=120)
    orientation: str = Field(default="portrait", min_length=1, max_length=32)
    resolution_width: int = Field(gt=0, le=7680)
    resolution_height: int = Field(gt=0, le=4320)


class CameraMapFrameIn(BaseModel):
    frame_base64: str = Field(min_length=1, max_length=4_000_000)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    captured_at: datetime | None = None


class CameraMapFramesIn(BaseModel):
    frames: list[CameraMapFrameIn] = Field(min_length=3, max_length=MAP_FRAME_MAX_COUNT)


def recover_interrupted_map_jobs(db: Database) -> None:
    """Close jobs that cannot be resumed after an API process restart.

    Walkthrough frames are deliberately transient. A processing job therefore
    has no safe way to resume after its in-memory task disappears. Collecting
    jobs remain reusable briefly for request retries, then expire so a changed
    camera resolution or orientation cannot trap future walkthroughs.
    """
    now = datetime.now(timezone.utc)
    for row in db.many(
        "SELECT id,status,updated_at FROM camera_map_generation_jobs WHERE status IN ('collecting','processing')"
    ):
        interrupted = row["status"] == "processing"
        if not interrupted:
            try:
                updated_at = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                interrupted = updated_at <= now - MAP_COLLECTING_STALE_AFTER
            except (TypeError, ValueError):
                interrupted = True
        if not interrupted:
            continue
        completed = now_iso()
        error_code = "generation_interrupted" if row["status"] == "processing" else "walkthrough_expired"
        db.execute(
            """UPDATE camera_map_generation_jobs
                  SET status='failed', error_code=?, error_message=?, updated_at=?, completed_at=?
                WHERE id=? AND status=?""",
            (
                error_code,
                "The room walkthrough was interrupted. The camera remains saved; start a new walkthrough when convenient.",
                completed,
                completed,
                row["id"],
                row["status"],
            ),
        )


class EmailAuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    purpose: str = Field(default="login", pattern="^(create|login)$")
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    home_name: str = Field(default="ONE Home", min_length=1, max_length=120)
    care_setting: str = Field(default="home", pattern="^(home|residence)$")
    support_focus: str = Field(default="general", pattern="^(general|mci)$")
    role: str = Field(default="admin", pattern="^(admin|resident|caregiver)$")


class EmailAuthVerify(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class EmailAuthRequestResponse(BaseModel):
    verification_id: str
    expires_in_seconds: int
    delivery: str
    # Development/test only. Production integrations must deliver this via a
    # configured mail provider and never expose it in an HTTP response.
    dev_code: str | None = None
    email: str
    purpose: str
    home_id: str
    user_id: str
    role: str


class CareSpaceCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    care_setting: str = Field(default="home", pattern="^(home|residence)$")
    support_focus: str = Field(default="general", pattern="^(general|mci)$")


class ConsentIn(BaseModel):
    purpose: str = Field(min_length=1, max_length=120)
    policy_version: str = Field(min_length=1, max_length=40)
    granted: bool = True
    # A caregiver may record a resident's explicit decision only when the
    # representation process is separately documented; this field never
    # infers authority from a caregiver role.
    subject_user_id: str | None = None
    care_recipient_id: str | None = None
class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    room_id: str | None = None
    resolution_width: int | None = Field(default=None, gt=0, le=7680)
    resolution_height: int | None = Field(default=None, gt=0, le=4320)
    metadata: dict = Field(default_factory=dict)
class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    room_id: str | None = None
    resolution_width: int | None = Field(default=None, gt=0, le=7680)
    resolution_height: int | None = Field(default=None, gt=0, le=4320)
    metadata: dict = Field(default_factory=dict)
class RoomIn(BaseModel): name: str = Field(min_length=1, max_length=120)
class MapIn(BaseModel): room_id: str | None = None; coordinate_frame: str = Field(default="manual-2d", max_length=80); map_data: dict
class ProvisionalMapIn(BaseModel):
    camera_id: str
    room_id: str | None = None
    resolution_width: int = Field(gt=0, le=7680)
    resolution_height: int = Field(gt=0, le=4320)
    zones: list[dict] = Field(default_factory=list, max_length=100)
class ImagePoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ImageSize(BaseModel):
    x: float = Field(gt=0, le=1)
    y: float = Field(gt=0, le=1)


class RoomLayoutPolygon(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    points: list[ImagePoint] = Field(min_length=3, max_length=128)
    confidence: float = Field(ge=0, le=1)


class RoomLayoutWall(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    start: ImagePoint
    end: ImagePoint
    confidence: float = Field(ge=0, le=1)


class RoomLayoutFurniture(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    center: ImagePoint
    size: ImageSize
    rotation_degrees: float = Field(default=0, ge=-180, le=180)
    confidence: float = Field(ge=0, le=1)


class RoomLayoutOpening(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    kind: Literal["door", "window"]
    start: ImagePoint
    end: ImagePoint
    confidence: float = Field(ge=0, le=1)


class RoomLayoutCameraPose(BaseModel):
    coordinate_frame: Literal["camera-relative"]
    position: dict
    rotation_degrees: dict
    confidence: float = Field(ge=0, le=1)


class RoomLayoutMetrics(BaseModel):
    confidence: float = Field(ge=0, le=1)
    reprojection_error_px: float = Field(ge=0, le=100_000)
    homography_inlier_ratio: float = Field(ge=0, le=1)


class RoomLayoutGeometry(BaseModel):
    coordinate_frame: Literal["camera-relative-image"]
    polygons: list[RoomLayoutPolygon] = Field(min_length=1, max_length=100)
    walls: list[RoomLayoutWall] = Field(default_factory=list, max_length=256)
    furniture: list[RoomLayoutFurniture] = Field(default_factory=list, max_length=100)
    openings: list[RoomLayoutOpening] = Field(default_factory=list, max_length=100)
    camera_pose: RoomLayoutCameraPose
    intrinsics: dict = Field(default_factory=dict)
    metrics: RoomLayoutMetrics


class RoomLayoutResult(BaseModel):
    status: Literal["ready", "needs_rescan", "unavailable", "failed"]
    source: Literal["camera-cv-2d"] = "camera-cv-2d"
    dimension: Literal["2d"] = "2d"
    geometry: RoomLayoutGeometry | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metric_scale_known: Literal[False] = False
    model_version: str = Field(min_length=1, max_length=120)
    diagnostics: dict = Field(default_factory=dict)


class MapScaleReferenceIn(BaseModel):
    """One caregiver-supplied real-world reference for an RGB 2D map."""

    start: ImagePoint
    end: ImagePoint
    length_m: float = Field(gt=0.05, le=100)
    label: str = Field(default="Measured reference", min_length=1, max_length=120)

    @field_validator("label", mode="before")
    @classmethod
    def trim_label(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value
class CalibrationIn(BaseModel):
    camera_id: str
    map_id: str
    intrinsics: dict
    extrinsics: dict
    accuracy_m: float | None = Field(default=None, ge=0, le=100)
    resolution_width: int | None = Field(default=None, gt=0, le=7680)
    resolution_height: int | None = Field(default=None, gt=0, le=4320)
    camera_metadata: dict = Field(default_factory=dict)
    source: str = Field(default="manual", max_length=40)


class RoomPlanCameraRegistrationIn(BaseModel):
    camera_id: str
    map_id: str
    camera_to_world: list[list[float]]
    confidence: float | None = Field(default=None, ge=0, le=1)
    tracking_state: Literal["normal", "limited", "unavailable"] = "normal"

    @field_validator("camera_to_world")
    @classmethod
    def validate_camera_to_world(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("camera_to_world must be an exact 4x4 matrix")
        if not all(math.isfinite(component) for row in value for component in row):
            raise ValueError("camera_to_world must contain only finite values")
        return value


class Matrix3x3In(BaseModel):
    values: list[list[float]]

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("matrix must be an exact 3x3 matrix")
        if not all(math.isfinite(component) for row in value for component in row):
            raise ValueError("matrix must contain only finite values")
        return value


class RoomPlanVisualFrameIn(BaseModel):
    frame_base64: str = Field(min_length=1, max_length=4_000_000)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    depth_base64: str | None = Field(default=None, min_length=1, max_length=2_000_000)
    depth_width: int | None = Field(default=None, gt=0, le=2048)
    depth_height: int | None = Field(default=None, gt=0, le=2048)
    intrinsics: Matrix3x3In
    camera_to_world: list[list[float]]
    captured_at: datetime | None = None

    @model_validator(mode="after")
    def validate_optional_depth_bundle(self) -> "RoomPlanVisualFrameIn":
        depth_values = (self.depth_base64, self.depth_width, self.depth_height)
        if any(value is not None for value in depth_values) and not all(value is not None for value in depth_values):
            raise ValueError("depth_base64, depth_width, and depth_height must be supplied together")
        return self

    @field_validator("camera_to_world")
    @classmethod
    def validate_camera_to_world(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("camera_to_world must be an exact 4x4 matrix")
        if not all(math.isfinite(component) for row in value for component in row):
            raise ValueError("camera_to_world must contain only finite values")
        return value


class RoomPlanVisualLandmarksIn(BaseModel):
    frames: list[RoomPlanVisualFrameIn] = Field(min_length=1, max_length=12)


class CameraLocalizationFrameIn(BaseModel):
    frame_base64: str = Field(min_length=1, max_length=4_000_000)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)


class CameraLocalizationPersonAnchorIn(BaseModel):
    """Known RoomPlan floor point occupied by a person in one calibration frame."""

    frame_index: int = Field(ge=0, le=7)
    x: float
    y: float
    z: float

    @model_validator(mode="after")
    def validate_finite_position(self) -> "CameraLocalizationPersonAnchorIn":
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("guided calibration anchor must be finite")
        return self


class CameraLocalizationIn(BaseModel):
    frames: list[CameraLocalizationFrameIn] = Field(min_length=1, max_length=8)
    intrinsics: Matrix3x3In | None = None
    fov_degrees: float = Field(default=60.0, ge=30.0, le=120.0)
    review_only: bool = False
    person_anchors: list[CameraLocalizationPersonAnchorIn] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_person_anchor_frames(self) -> "CameraLocalizationIn":
        if any(anchor.frame_index >= len(self.frames) for anchor in self.person_anchors):
            raise ValueError("guided calibration anchor references a missing frame")
        return self


class RoomPlanCalibrationCaptureRequestIn(BaseModel):
    target_index: int = Field(ge=0, le=3)


class RoomPlanCalibrationFramesIn(BaseModel):
    target_index: int = Field(ge=0, le=3)
    frames: list[CameraLocalizationFrameIn] = Field(min_length=1, max_length=2)


class CameraLocalizationReferenceIn(BaseModel):
    """Physical top-down camera location in the active RoomPlan coordinate frame."""

    x: float
    z: float
    source: str = Field(default="manual-floor-reference", min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_finite_position(self) -> "CameraLocalizationReferenceIn":
        if not math.isfinite(self.x) or not math.isfinite(self.z):
            raise ValueError("camera localization reference must be finite")
        return self


class ObjectIn(BaseModel): label: str = Field(min_length=1, max_length=80); display_name: str | None = Field(default=None, max_length=120)
class ObservationIn(BaseModel): object_id: str | None = None; camera_id: str | None = None; map_id: str | None = None; x: float | None = None; y: float | None = None; z: float | None = None; uncertainty_m: float | None = Field(default=None, ge=0, le=100); confidence: float = Field(default=0.0, ge=0, le=1); detector_version: str = "local-cv-v1"
class CheckInIn(BaseModel): subject_user_id: str | None = None; transcript: str = Field(default="", max_length=4000)
class VisionIn(BaseModel):
    camera_id: str
    frame_base64: str = Field(min_length=1, max_length=4_000_000)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    candidate_labels: list[str] = Field(default_factory=list, max_length=20)
    captured_at: datetime | None = None
    depth_m: float | None = Field(default=None, gt=0, le=100)
class ClipIn(BaseModel):
    object_key: str = Field(min_length=1, max_length=500, pattern=r"^[A-Za-z0-9_./-]+$")
    starts_at: str
    ends_at: str

    @field_validator("object_key")
    @classmethod
    def no_parent_paths(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("object_key cannot contain parent traversal")
        return value

class ClipBytesIn(BaseModel):
    content_base64: str = Field(min_length=1, max_length=12_000_000)
class LiveKitTokenIn(BaseModel):
    mode: str = Field(default="auto", pattern="^(auto|publish|subscribe)$")


class FamilyInviteIn(BaseModel):
    email: str | None = Field(default=None, max_length=254)
    display_name: str = Field(min_length=1, max_length=120)
    role: str = Field(default="caregiver", pattern="^(resident|caregiver)$")
    expires_in_seconds: int = Field(default=86_400, ge=300, le=604_800)


class FamilyInviteAcceptIn(BaseModel):
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    display_name: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, min_length=3, max_length=254)


class FamilyMemberUpdateIn(BaseModel):
    """Editable access for an existing, non-device household member."""
    role: str = Field(pattern="^(resident|caregiver)$")


class FamilyMemberMutationResponse(BaseModel):
    data: dict
    invalidated_sessions: int = 0


class CareRecipientCreateIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    relationship: str | None = Field(default=None, max_length=120)
    room_label: str | None = Field(default=None, max_length=120)

    @field_validator("display_name", mode="before")
    @classmethod
    def trim_display_name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("display_name is required")
        return value.strip()

    @field_validator("relationship", "room_label", mode="before")
    @classmethod
    def trim_optional_text(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None


class CareRecipientUpdateIn(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    relationship: str | None = Field(default=None, max_length=120)
    room_label: str | None = Field(default=None, max_length=120)

    @field_validator("display_name", mode="before")
    @classmethod
    def trim_optional_display_name(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("display_name cannot be empty")
        return value.strip()

    @field_validator("relationship", "room_label", mode="before")
    @classmethod
    def trim_optional_text(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None


class CareRecipientOut(BaseModel):
    id: str
    display_name: str
    relationship: str | None = None
    room_label: str | None = None
    medication_reminders_enabled: bool = False
    created_at: str


class CareRecipientMutationResponse(BaseModel):
    data: CareRecipientOut


class CareRecipientListResponse(BaseModel):
    data: list[CareRecipientOut]


class MedicationPlanIn(BaseModel):
    subject_user_id: str | None = None
    care_recipient_id: str | None = None
    name: str = Field(min_length=1, max_length=160)
    dose: str = Field(min_length=1, max_length=120)
    schedule: str = Field(min_length=1, max_length=500)
    instructions: str = Field(default="", max_length=1000)
    active: bool = True
    assigned_caregiver_id: str | None = None


class MedicationPlanUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    dose: str | None = Field(default=None, min_length=1, max_length=120)
    schedule: str | None = Field(default=None, min_length=1, max_length=500)
    instructions: str | None = Field(default=None, max_length=1000)
    active: bool | None = None
    assigned_caregiver_id: str | None = None
    version: int | None = Field(default=None, ge=1)


class MedicationCheckInIn(BaseModel):
    scheduled_for: datetime
    status: str = Field(pattern="^(pending|taken|skipped|missed)$")
    note: str = Field(default="", max_length=500)


class FamilyAssistantIn(BaseModel):
    message: str = Field(default="", max_length=1000)
    subject_user_id: str | None = None
    care_recipient_id: str | None = None


def _camera_localization_search_prior(rows: Sequence[dict]) -> dict | None:
    """Find a tight recurring rejected-pose cluster to steer the next solve.

    This history is deliberately search-only: it never contributes to
    localization consensus or activation.  Each calibration can contribute at
    most one sample to a cluster so duplicate fixed-camera frames cannot turn
    into independent evidence.
    """
    samples: list[dict] = []
    for row in rows:
        try:
            metrics = json.loads(row.get("metrics_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        diagnostics = metrics.get("diagnostics") if isinstance(metrics, dict) else None
        if not isinstance(diagnostics, dict):
            continue
        summaries = diagnostics.get("candidate_summaries")
        if isinstance(summaries, list):
            for summary in summaries:
                if not isinstance(summary, dict) or summary.get("scene_plausible") is not True:
                    continue
                view_id = summary.get("landmark_view_id")
                center = summary.get("camera_center")
                if not isinstance(view_id, str) or not view_id or view_id == "all-views":
                    continue
                if not isinstance(center, list) or len(center) != 3:
                    continue
                try:
                    values = tuple(float(value) for value in center)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in values):
                    continue
                fov = summary.get("selected_fov_degrees")
                try:
                    fov_value = float(fov) if fov is not None else None
                except (TypeError, ValueError):
                    fov_value = None
                samples.append(
                    {
                        "calibration_id": str(row.get("id") or ""),
                        "source": "visual-pnp",
                        "view_id": view_id,
                        "center": values,
                        "fov_degrees": fov_value,
                    }
                )

        # A repeated high-quality semantic cuboid fit can also establish a
        # useful fixed-camera search basin. It remains search-only and never
        # contributes to localization consensus or activation.
        semantic_candidates = diagnostics.get("semantic_cuboid_candidates")
        if isinstance(semantic_candidates, list):
            for candidate in semantic_candidates:
                if not isinstance(candidate, dict):
                    continue
                if (
                    int(candidate.get("matched_object_count") or 0) < 2
                    or int(candidate.get("semantic_group_count") or 0) < 2
                    or float(candidate.get("minimum_iou") or 0.0) < 0.30
                    or float(candidate.get("mean_iou") or 0.0) < 0.55
                ):
                    continue
                center = candidate.get("camera_center")
                if not isinstance(center, list) or len(center) != 3:
                    continue
                try:
                    values = tuple(float(value) for value in center)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in values):
                    continue
                fov = candidate.get("selected_fov_degrees")
                try:
                    fov_value = float(fov) if fov is not None else None
                except (TypeError, ValueError):
                    fov_value = None
                samples.append(
                    {
                        "calibration_id": str(row.get("id") or ""),
                        "source": "semantic-cuboid",
                        "view_id": None,
                        "center": values,
                        "fov_degrees": fov_value,
                    }
                )
                # Diagnostics are already ranked; retain at most one semantic
                # sample from each calibration so one frame cannot self-vote.
                break

    best_by_source: dict[str, tuple[tuple[int, float], dict]] = {}
    for anchor in samples:
        nearby = [
            sample
            for sample in samples
            if sample["source"] == anchor["source"]
            and sample["view_id"] == anchor["view_id"]
            and math.dist(sample["center"], anchor["center"]) <= 0.35
        ]
        nearest_by_calibration: dict[str, dict] = {}
        for sample in nearby:
            calibration_id = sample["calibration_id"]
            current = nearest_by_calibration.get(calibration_id)
            if current is None or math.dist(sample["center"], anchor["center"]) < math.dist(current["center"], anchor["center"]):
                nearest_by_calibration[calibration_id] = sample
        # Rows arrive newest-first. Cap the retained support so a camera that
        # stays installed for a long time cannot overflow the wire contract,
        # while still letting several rapid retries coexist with the last
        # tight recurring cluster.
        chosen = list(nearest_by_calibration.values())[:30]
        if len(chosen) < 3:
            continue

        center = tuple(
            sorted(sample["center"][axis] for sample in chosen)[len(chosen) // 2]
            for axis in range(3)
        )
        residuals = [math.dist(sample["center"], center) for sample in chosen]
        mean_residual = sum(residuals) / len(residuals)
        # A loose historical cloud is more likely to encode repeated PnP
        # ambiguity than a stable physical camera location.
        if mean_residual > 0.12:
            continue

        fovs = [sample["fov_degrees"] for sample in chosen if sample["fov_degrees"] is not None]
        stable_fov = None
        if len(fovs) >= 3 and max(fovs) - min(fovs) <= 8.0:
            stable_fov = sorted(fovs)[len(fovs) // 2]
        prior = {
            "center": [round(value, 4) for value in center],
            "support_count": len(chosen),
            "mean_residual_m": round(mean_residual, 4),
            "source": anchor["source"],
            "landmark_view_id": anchor["view_id"],
            "fov_degrees": stable_fov,
        }
        score = (len(chosen), -mean_residual)
        current = best_by_source.get(anchor["source"])
        if current is None or score > current[0]:
            best_by_source[anchor["source"]] = (score, prior)

    # Prefer a robust semantic basin once it has repeated across calibrations:
    # it is tied to RoomPlan object geometry and remains only a search hint.
    # Fall back to the existing visual-view cluster when semantic support is
    # not yet stable enough.
    semantic_best = best_by_source.get("semantic-cuboid")
    if semantic_best is not None:
        return semantic_best[1]
    visual_best = best_by_source.get("visual-pnp")
    return None if visual_best is None else visual_best[1]


def _stabilize_camera_localization_diagnostics(
    diagnostics: dict,
    search_prior: dict | None,
    *,
    positioned: bool,
) -> dict:
    """Keep rejected fixed-camera diagnostics stable through occlusion.

    This only affects the numeric trace shown while the fresh attempt is still
    ``needs_rescan``.  It never supplies ``camera_to_world`` and therefore
    cannot activate a camera registration.
    """
    if positioned or not isinstance(search_prior, dict) or search_prior.get("source") != "semantic-cuboid":
        return diagnostics
    if int(search_prior.get("support_count") or 0) < 3 or float(search_prior.get("mean_residual_m") or 1.0) > 0.12:
        return diagnostics
    if diagnostics.get("selected_estimate_source") == "semantic-cuboid":
        return diagnostics

    prior_center = search_prior.get("center")
    selected_center = diagnostics.get("selected_camera_center")
    if not (
        isinstance(prior_center, list)
        and len(prior_center) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in prior_center)
    ):
        return diagnostics
    selected_is_valid = (
        isinstance(selected_center, list)
        and len(selected_center) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in selected_center)
    )
    if selected_is_valid and math.dist(
        [float(value) for value in selected_center],
        [float(value) for value in prior_center],
    ) <= 0.80:
        return diagnostics

    return {
        **diagnostics,
        "unstabilized_selected_camera_center": selected_center if selected_is_valid else None,
        "unstabilized_selected_estimate_source": diagnostics.get("selected_estimate_source"),
        "selected_camera_center": [round(float(value), 4) for value in prior_center],
        "selected_estimate_source": "temporal-prior",
        "selected_estimate_stabilized": True,
    }


def make_app(
    settings: Settings | None = None,
    geometry_service: RoomLayoutService | None = None,
    vision_detector: Detector | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings)
    recover_interrupted_map_jobs(db)
    store = LocalObjectStore(settings.object_store_path)
    clip_key = None
    if settings.clip_encryption_key_b64:
        try:
            clip_key = base64.b64decode(settings.clip_encryption_key_b64, validate=True)
        except ValueError as exc:
            raise RuntimeError("ONE_CLIP_ENCRYPTION_KEY_B64 must be valid base64") from exc
    clip_store = EncryptedLocalClipStore(settings.object_store_path / "encrypted-clips", clip_key)
    bus = EventBus()
    lm = LMStudioAdapter(settings)
    geometry = geometry_service or HttpRoomLayoutService(settings)
    vision = CameraVisionPipeline(vision_detector or LocalServiceDetector(geometry))
    app = FastAPI(title="ONE API", version="0.1.0", openapi_url="/api/v1/openapi.json")
    roomplan_calibration_sessions: dict[tuple[str, str], dict] = {}
    roomplan_calibration_lock = threading.RLock()

    def request_id(request: Request) -> str:
        """Return a bounded correlation id without reflecting arbitrary input."""
        candidate = request.headers.get("x-request-id")
        try:
            return str(uuid.UUID(candidate)) if candidate else str(uuid.uuid4())
        except (ValueError, AttributeError):
            return str(uuid.uuid4())

    def canonical_uuid(value: str, field_name: str) -> str:
        """Normalize UUID text before comparing it with canonical database ids."""
        try:
            return str(uuid.UUID(value))
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(422, f"{field_name} must be a UUID")

    def error_code(http_status: int) -> str:
        return {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            410: "gone",
            413: "payload_too_large",
            422: "validation_error",
            429: "rate_limited",
            500: "internal_error",
            503: "service_unavailable",
        }.get(http_status, "request_failed")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        status_code = int(exc.status_code)
        detail = exc.detail if isinstance(exc.detail, str) else "Request could not be completed"
        correlation_id = request_id(request)
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": error_code(status_code),
                    "message": detail,
                    "details": {},
                    "retryable": status_code == 429 or status_code >= 500,
                },
                "request_id": correlation_id,
                "api_version": "v1",
            },
        )
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        response.headers["X-Request-ID"] = correlation_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = []
        for item in exc.errors():
            fields.append({"loc": [str(part) for part in item.get("loc", [])], "msg": str(item.get("msg", "Invalid value")), "type": str(item.get("type", "value_error"))})
        correlation_id = request_id(request)
        response = JSONResponse(
            status_code=422,
            content={
                "error": {"code": "validation_error", "message": "Request validation failed", "details": {"fields": fields}, "retryable": False},
                "request_id": correlation_id,
                "api_version": "v1",
            },
        )
        response.headers["X-Request-ID"] = correlation_id
        return response

    app.state.db, app.state.store, app.state.clip_store, app.state.bus, app.state.settings, app.state.vision = db, store, clip_store, bus, settings, vision
    app.state.vision_person_objects = {}
    app.state.geometry_service = geometry
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=True, allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-Bootstrap-Secret"])

    def auth(request: Request) -> dict:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "): raise HTTPException(401, "Bearer session required")
        row = db.one("SELECT s.*, u.display_name FROM sessions s JOIN users u ON u.id=s.user_id WHERE token_hash=?", (hash_secret(header[7:]),))
        if not row or expired(row["expires_at"]): raise HTTPException(401, "Session expired")
        member = db.one("SELECT role FROM memberships WHERE home_id=? AND user_id=?", (row["home_id"], row["user_id"]))
        if not member: raise HTTPException(403, "Membership revoked")
        return {**row, **member}

    Current = Annotated[dict, Depends(auth)]

    def audit(actor: dict | None, action: str, target_type: str | None = None, target_id: str | None = None, home_id: str | None = None):
        db.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), home_id or (actor or {}).get("home_id"), (actor or {}).get("user_id"), action, target_type, target_id, "{}", now_iso()))

    def home_check(actor: dict, home_id: str):
        if actor["home_id"] != home_id: raise HTTPException(403, "Home access denied")

    def publisher_block(actor: dict):
        # A paired camera is a data publisher, not a home administrator. Keep
        # its bearer useful for media publishing while preventing it from
        # changing consent, metadata, or privacy controls.
        if actor["role"] == "publisher":
            raise HTTPException(403, "Publisher devices cannot access home controls")

    def is_paused(home_id: str) -> bool:
        row = db.one("SELECT paused FROM home_runtime WHERE home_id=?", (home_id,))
        return bool(row and row["paused"])

    def active_video_consent(home_id: str) -> bool:
        if is_paused(home_id):
            return False
        row = db.one("SELECT revoked_at FROM consents WHERE home_id=? AND purpose='video_capture' ORDER BY granted_at DESC LIMIT 1", (home_id,))
        return bool(row and row["revoked_at"] is None)

    def active_consent(home_id: str, subject_user_id: str, purpose: str) -> bool:
        row = db.one(
            "SELECT revoked_at FROM consents WHERE home_id=? AND subject_user_id=? AND purpose=? ORDER BY granted_at DESC LIMIT 1",
            (home_id, subject_user_id, purpose),
        )
        return bool(row and row["revoked_at"] is None)

    def require_consent(home_id: str, subject_user_id: str, purpose: str):
        if not active_consent(home_id, subject_user_id, purpose):
            raise HTTPException(403, f"Active {purpose} consent is required")

    def active_care_recipient_consent(home_id: str, care_recipient_id: str, purpose: str) -> bool:
        row = db.one(
            "SELECT revoked_at FROM consents WHERE home_id=? AND care_recipient_id=? AND purpose=? ORDER BY granted_at DESC LIMIT 1",
            (home_id, care_recipient_id, purpose),
        )
        return bool(row and row["revoked_at"] is None)

    def require_care_recipient_consent(home_id: str, care_recipient_id: str, purpose: str):
        if not active_care_recipient_consent(home_id, care_recipient_id, purpose):
            raise HTTPException(403, f"Active {purpose} consent is required for this cared-for person")

    def medication_care_recipient(home_id: str, actor: dict, care_recipient_id: str, purpose: str = "medication_management") -> dict:
        family_actor(actor)
        care_recipient_id = canonical_uuid(care_recipient_id, "care_recipient_id")
        row = db.one("SELECT * FROM care_recipients WHERE id=? AND home_id=?", (care_recipient_id, home_id))
        if not row:
            raise HTTPException(404, "Care recipient not found")
        require_care_recipient_consent(home_id, care_recipient_id, purpose)
        return row

    def member(home_id: str, user_id: str) -> dict:
        row = db.one(
            "SELECT u.id, u.display_name, u.email, u.created_at, m.role FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=? AND u.id=?",
            (home_id, user_id),
        )
        if not row or row["role"] == "publisher":
            raise HTTPException(404, "Family member not found")
        return row

    def family_actor(actor: dict):
        publisher_block(actor)
        if actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Caregiver permission required")

    def family_subject(home_id: str, actor: dict, subject_user_id: str | None, purpose: str) -> dict:
        subject_id = subject_user_id or actor["user_id"]
        subject = member(home_id, subject_id)
        if subject_id != actor["user_id"]:
            family_actor(actor)
        require_consent(home_id, subject_id, purpose)
        return subject

    def assigned_caregiver(home_id: str, caregiver_id: str | None, fallback_actor: dict | None = None) -> dict | None:
        """Validate a named same-home caregiver without granting extra rights."""
        selected_id = canonical_uuid(caregiver_id, "assigned_caregiver_id") if caregiver_id else (fallback_actor["user_id"] if fallback_actor and fallback_actor["role"] in {"admin", "caregiver"} else None)
        if not selected_id:
            return None
        selected = member(home_id, selected_id)
        if selected["role"] not in {"admin", "caregiver"}:
            raise HTTPException(422, "assigned_caregiver_id must reference a caregiver in this home")
        return selected

    def require_video_capture(home_id: str):
        if not active_video_consent(home_id):
            raise HTTPException(403, "Active video_capture consent is required")

    def camera_view(row: dict) -> dict:
        metadata = json.loads(row.get("metadata_json") or "{}")
        latest_session = db.one(
            "SELECT created_at, expires_at FROM sessions WHERE home_id=? AND user_id=? ORDER BY created_at DESC LIMIT 1",
            (row["home_id"], row["id"]),
        )
        connected = bool(row["enabled"] and latest_session and not expired(latest_session["expires_at"]))
        view = {
            **row,
            "metadata": metadata,
            "label": row["name"],
            "platform": "browser",
            "status": "paused" if connected and is_paused(row["home_id"]) else ("online" if connected else "offline"),
            "lastSeenAt": latest_session["created_at"] if latest_session else row["created_at"],
        }
        view.update(camera_roomplan_calibration_state(row["home_id"], row["id"]))
        view.pop("metadata_json", None)
        return view

    @app.get("/api/v1/health")
    def health():
        database = db.health()
        return {
            "status": "ok" if database["status"] == "ok" else "degraded",
            "database": database["backend"],
            "database_status": database["status"],
            "local_inference_model": settings.effective_llm_model,
            "geometry_service_configured": bool(settings.geometry_service_url),
            "geometry_service_requires_gpu": settings.geometry_require_gpu,
        }

    @app.get("/api/v1/me")
    def me(actor: Current):
        home = db.one("SELECT id, name, care_setting, support_focus FROM homes WHERE id=?", (actor["home_id"],))
        resident = db.one("SELECT display_name FROM care_recipients WHERE home_id=? ORDER BY created_at LIMIT 1", (actor["home_id"],)) or db.one("SELECT u.display_name FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=? AND m.role='resident' ORDER BY u.created_at LIMIT 1", (actor["home_id"],))
        device = db.one("SELECT id, home_id, name, room_id, enabled, created_at FROM cameras WHERE home_id=? AND enabled=1 ORDER BY created_at DESC LIMIT 1", (actor["home_id"],))
        return {"actor": {"id": actor["user_id"], "role": actor["role"], "name": actor["display_name"]}, "home": {"id": home["id"], "name": home["name"], "residentName": resident["display_name"] if resident else "Resident", "careSetting": home.get("care_setting") or "home", "supportFocus": home.get("support_focus") or "general"}, "device": camera_view(device) if device else None, "paused": is_paused(actor["home_id"])}

    def care_space_view(home: dict, role: str, active_home_id: str) -> dict:
        recipients = db.many("SELECT display_name FROM care_recipients WHERE home_id=? ORDER BY created_at, display_name", (home["id"],))
        resident = (recipients[0] if recipients else None) or db.one(
            "SELECT u.display_name FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=? AND m.role='resident' ORDER BY u.created_at LIMIT 1",
            (home["id"],),
        )
        return {
            "id": home["id"],
            "name": home["name"],
            "careSetting": home.get("care_setting") or "home",
            "supportFocus": home.get("support_focus") or "general",
            "residentName": resident["display_name"] if resident else "Resident",
            "recipientNames": [recipient["display_name"] for recipient in recipients],
            "recipientCount": len(recipients),
            "role": role,
            "active": home["id"] == active_home_id,
        }

    def issue_care_space_session(user_id: str, home_id: str) -> dict:
        membership = db.one(
            "SELECT role FROM memberships WHERE home_id=? AND user_id=? AND role != 'publisher'",
            (home_id, user_id),
        )
        if not membership:
            raise HTTPException(404, "Care space membership not found")
        token = new_token()
        db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?)",
            (hash_secret(token), user_id, home_id, iso_after(settings.session_ttl_minutes), now_iso()),
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": settings.session_ttl_minutes * 60,
            "home_id": home_id,
            "user_id": user_id,
            "role": membership["role"],
        }

    @app.get("/api/v1/account/homes")
    def account_homes(actor: Current):
        publisher_block(actor)
        rows = db.many(
            """SELECT h.id, h.name, h.care_setting, h.support_focus, m.role
                 FROM memberships m
                 JOIN homes h ON h.id=m.home_id
                WHERE m.user_id=? AND m.role != 'publisher'
                ORDER BY h.created_at, h.name""",
            (actor["user_id"],),
        )
        return {"data": [care_space_view(row, row["role"], actor["home_id"]) for row in rows]}

    @app.post("/api/v1/account/homes")
    def account_home_create(body: CareSpaceCreateIn, actor: Current):
        publisher_block(actor)
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "Care space name is required")
        home_id, created = str(uuid.uuid4()), now_iso()
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO homes(id,name,created_at,care_setting,support_focus) VALUES (?,?,?,?,?)",
                (home_id, name, created, body.care_setting, body.support_focus),
            )
            conn.execute("INSERT INTO memberships VALUES (?,?,?)", (home_id, actor["user_id"], "admin"))
            conn.execute("INSERT INTO home_runtime VALUES (?,?,?)", (home_id, 0, created))
        audit(actor, "home.create", "home", home_id, home_id)
        return issue_care_space_session(actor["user_id"], home_id)

    @app.post("/api/v1/account/homes/{home_id}/activate")
    def account_home_activate(home_id: str, actor: Current):
        publisher_block(actor)
        result = issue_care_space_session(actor["user_id"], home_id)
        audit(actor, "session.home.switch", "home", home_id, home_id)
        return result

    @app.post("/api/v1/pairing/start", response_model=PairStartResponse)
    def pairing_start(body: PairStart, x_bootstrap_secret: str | None = Header(default=None)):
        if settings.env == "production" and x_bootstrap_secret != settings.bootstrap_secret: raise HTTPException(403, "Bootstrap authorization required")
        if settings.env != "production" and x_bootstrap_secret not in (None, settings.bootstrap_secret): raise HTTPException(403, "Invalid bootstrap secret")
        email = None
        if body.email:
            try:
                email = normalize_email(body.email)
            except ValueError as exc:
                raise HTTPException(422, "A valid email address is required") from exc
            if db.one("SELECT id FROM users WHERE lower(trim(email))=?", (email,)):
                raise HTTPException(409, "An account already exists for this email")
        home_id, user_id, code = str(uuid.uuid4()), str(uuid.uuid4()), new_pairing_code()
        db.execute("INSERT INTO homes(id,name,created_at,care_setting,support_focus) VALUES (?,?,?,?,?)", (home_id, body.home_name, now_iso(), body.care_setting, body.support_focus))
        db.execute("INSERT INTO users VALUES (?,?,?,?)", (user_id, body.display_name, email, now_iso()))
        db.execute("INSERT INTO memberships VALUES (?,?,?)", (home_id, user_id, body.role))
        db.execute("INSERT INTO home_runtime VALUES (?,?,?)", (home_id, 0, now_iso()))
        db.execute("INSERT INTO pairing_codes VALUES (?,?,?,?,?,NULL)", (hash_secret(code), home_id, user_id, body.role, iso_after(10)))
        return {
            "pairing_code": code,
            "expires_in_seconds": 600,
            "home_id": home_id,
            "user_id": user_id,
            "role": body.role,
        }

    @app.post("/api/v1/auth/email/request", response_model=EmailAuthRequestResponse)
    def email_auth_request(body: EmailAuthRequest):
        """Create a short-lived passwordless sign-in challenge.

        The development outbox returns the code once so a local Docker setup
        works without an email subscription. A production mail adapter should
        consume the same event and keep ``dev_code`` absent.
        """
        try:
            email = normalize_email(body.email)
        except ValueError as exc:
            raise HTTPException(422, "A valid email address is required") from exc

        existing = db.one("SELECT * FROM users WHERE lower(trim(email))=? ORDER BY created_at LIMIT 1", (email,))
        if body.purpose == "create":
            if existing:
                raise HTTPException(409, "An account already exists for this email")
            if not body.display_name:
                raise HTTPException(422, "Display name is required to create an account")
            home_id, user_id, created = str(uuid.uuid4()), str(uuid.uuid4()), now_iso()
            with db.transaction() as conn:
                conn.execute("INSERT INTO homes(id,name,created_at,care_setting,support_focus) VALUES (?,?,?,?,?)", (home_id, body.home_name, created, body.care_setting, body.support_focus))
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (user_id, body.display_name, email, created))
                conn.execute("INSERT INTO memberships VALUES (?,?,?)", (home_id, user_id, body.role))
                conn.execute("INSERT INTO home_runtime VALUES (?,?,?)", (home_id, 0, created))
        else:
            if not existing:
                raise HTTPException(404, "No ONE account exists for this email")
            user_id = existing["id"]
            membership = db.one("SELECT home_id, role FROM memberships WHERE user_id=? AND role != 'publisher' ORDER BY home_id LIMIT 1", (user_id,))
            if not membership:
                raise HTTPException(403, "This account has no caregiver or resident household")
            home_id = membership["home_id"]

        membership = db.one("SELECT role FROM memberships WHERE home_id=? AND user_id=?", (home_id, user_id))
        role = membership["role"] if membership else body.role
        code, verification_id, created = new_pairing_code(), str(uuid.uuid4()), now_iso()
        db.execute("INSERT INTO email_verifications VALUES (?,?,?,?,?,?,?,?,?)", (verification_id, email, user_id, home_id, body.purpose, hash_secret(code), iso_after(10), None, created))
        return {
            "verification_id": verification_id,
            "expires_in_seconds": 600,
            "delivery": "development_outbox" if settings.env != "production" else "email_provider_required",
            "dev_code": code if settings.env != "production" else None,
            "email": email,
            "purpose": body.purpose,
            "home_id": home_id,
            "user_id": user_id,
            "role": role,
        }

    @app.post("/api/v1/auth/email/verify")
    def email_auth_verify(body: EmailAuthVerify):
        try:
            email = normalize_email(body.email)
        except ValueError as exc:
            raise HTTPException(422, "A valid email address is required") from exc
        verification = db.one("SELECT * FROM email_verifications WHERE email=? AND code_hash=? AND used_at IS NULL ORDER BY created_at DESC LIMIT 1", (email, hash_secret(body.code)))
        if not verification or expired(verification["expires_at"]):
            raise HTTPException(400, "Invalid or expired email verification code")
        user = db.one("SELECT id FROM users WHERE id=? AND lower(trim(email))=?", (verification["user_id"], email))
        membership = db.one("SELECT role FROM memberships WHERE home_id=? AND user_id=?", (verification["home_id"], verification["user_id"]))
        if not user or not membership or membership["role"] == "publisher":
            raise HTTPException(403, "Email account is not allowed in this household")
        now = now_iso()
        token = new_token()
        with db.transaction() as conn:
            conn.execute("UPDATE email_verifications SET used_at=? WHERE id=?", (now, verification["id"]))
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (hash_secret(token), user["id"], verification["home_id"], iso_after(settings.session_ttl_minutes), now))
            conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), verification["home_id"], user["id"], "email.auth.verify", "user", user["id"], "{}", now))
        return {"access_token": token, "token_type": "bearer", "expires_in": settings.session_ttl_minutes * 60, "home_id": verification["home_id"], "user_id": user["id"], "role": membership["role"], "email": email}

    def ensure_publisher_camera(home_id: str, user_id: str, name: str, created_at: str | None = None):
        """Backfill the camera read model for a completed publisher pairing."""
        db.execute(
            """INSERT INTO cameras(id,home_id,name,room_id,enabled,created_at,resolution_width,resolution_height,metadata_json)
               VALUES (?,?,?,NULL,1,?,NULL,NULL,'{}')
               ON CONFLICT(id) DO NOTHING""",
            (user_id, home_id, name, created_at or now_iso()),
        )

    def issue_camera_reconnect(home_id: str, camera_id: str, timestamp: str | None = None) -> str:
        issued_at = timestamp or now_iso()
        reconnect_token = new_token()
        db.execute(
            "UPDATE camera_reconnect_tokens SET revoked_at=? WHERE home_id=? AND camera_id=? AND revoked_at IS NULL",
            (issued_at, home_id, camera_id),
        )
        db.execute(
            "INSERT INTO camera_reconnect_tokens(token_hash,home_id,camera_id,created_at,last_used_at,revoked_at) VALUES (?,?,?,?,NULL,NULL)",
            (hash_secret(reconnect_token), home_id, camera_id, issued_at),
        )
        return reconnect_token

    @app.post("/api/v1/homes/{home_id}/pairing/start")
    def device_pairing_start(home_id: str, body: DevicePairingStart, actor: Current):
        home_check(actor, home_id)
        if actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Only an admin or caregiver can pair a publisher")
        label = body.label or body.display_name
        if not label:
            raise HTTPException(422, "A publisher label is required")
        user_id, code = str(uuid.uuid4()), new_pairing_code()
        db.execute("INSERT INTO users VALUES (?,?,?,?)", (user_id, label, None, now_iso()))
        db.execute("INSERT INTO memberships VALUES (?,?,?)", (home_id, user_id, "publisher"))
        db.execute("INSERT INTO pairing_codes VALUES (?,?,?,?,?,NULL)", (hash_secret(code), home_id, user_id, "publisher", iso_after(body.expires_in_seconds / 60)))
        audit(actor, "pairing.publisher.start", "user", user_id, home_id)
        return {"pairing_id": user_id, "pairing_code": code, "code": code, "expires_in_seconds": body.expires_in_seconds, "home_id": home_id, "user_id": user_id}

    @app.get("/api/v1/homes/{home_id}/pairing/{pairing_id}/status", response_model=PairingStatusResponse)
    def device_pairing_status(home_id: str, pairing_id: str, actor: Current):
        """Return publisher setup state without returning the pairing code."""
        home_check(actor, home_id)
        if actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Only an admin or caregiver can view pairing state")
        row = db.one(
            """SELECT pc.expires_at, pc.used_at, u.id AS user_id, u.display_name,
                      pc.role
               FROM pairing_codes pc
               JOIN users u ON u.id=pc.user_id
              WHERE pc.home_id=? AND pc.user_id=? AND pc.role='publisher'
              ORDER BY pc.expires_at DESC LIMIT 1""",
            (home_id, pairing_id),
        )
        if not row:
            raise HTTPException(404, "Pairing session not found")
        is_connected = row["used_at"] is not None
        is_expired = not is_connected and expired(row["expires_at"])
        if is_connected:
            ensure_publisher_camera(home_id, row["user_id"], row["display_name"], row["used_at"])
        return {
            "pairing_id": pairing_id,
            "home_id": home_id,
            "status": "connected" if is_connected else ("expired" if is_expired else "pending"),
            "expires_at": row["expires_at"],
            "connected_at": row["used_at"] if is_connected else None,
            "device": {"id": row["user_id"], "label": row["display_name"], "role": row["role"]},
        }

    @app.post("/api/v1/pairing/complete")
    def pairing_complete(body: PairComplete):
        row = db.one("SELECT * FROM pairing_codes WHERE code_hash=? AND used_at IS NULL", (hash_secret(body.code),))
        if not row or expired(row["expires_at"]): raise HTTPException(400, "Invalid or expired pairing code")
        token = new_token(); timestamp = now_iso()
        db.execute("UPDATE pairing_codes SET used_at=? WHERE code_hash=?", (timestamp, row["code_hash"]))
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (hash_secret(token), row["user_id"], row["home_id"], iso_after(settings.session_ttl_minutes), timestamp))
        if row["role"] == "publisher":
            user = db.one("SELECT display_name FROM users WHERE id=?", (row["user_id"],))
            # Keep the camera id equal to the publisher identity so the
            # caregiver status response and metadata endpoint always refer to
            # one device.
            ensure_publisher_camera(row["home_id"], row["user_id"], user["display_name"] if user else "Paired camera", timestamp)
            reconnect_token = issue_camera_reconnect(row["home_id"], row["user_id"], timestamp)
        else:
            reconnect_token = None
        audit({"user_id": row["user_id"], "home_id": row["home_id"]}, "pairing.complete", "user", row["user_id"])
        return {"access_token": token, "token_type": "bearer", "expires_in": settings.session_ttl_minutes * 60, "home_id": row["home_id"], "user_id": row["user_id"], "reconnect_token": reconnect_token}

    @app.post("/api/v1/camera/reconnect")
    def camera_reconnect(body: CameraReconnectIn):
        token_hash = hash_secret(body.reconnect_token)
        row = db.one(
            """SELECT crt.home_id, crt.camera_id, c.enabled
                 FROM camera_reconnect_tokens crt
                 JOIN cameras c ON c.id=crt.camera_id AND c.home_id=crt.home_id
                WHERE crt.token_hash=? AND crt.camera_id=? AND crt.revoked_at IS NULL""",
            (token_hash, body.camera_id),
        )
        if not row or not row["enabled"]:
            raise HTTPException(401, "Camera reconnect link is invalid or revoked")
        membership = db.one(
            "SELECT role FROM memberships WHERE home_id=? AND user_id=?",
            (row["home_id"], row["camera_id"]),
        )
        if not membership or membership["role"] != "publisher":
            raise HTTPException(401, "Camera publisher is unavailable")
        timestamp = now_iso()
        access_token = new_token()
        db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?)",
            (hash_secret(access_token), row["camera_id"], row["home_id"], iso_after(settings.session_ttl_minutes), timestamp),
        )
        db.execute("UPDATE camera_reconnect_tokens SET last_used_at=? WHERE token_hash=?", (timestamp, token_hash))
        audit({"user_id": row["camera_id"], "home_id": row["home_id"]}, "camera.reconnect", "camera", row["camera_id"], row["home_id"])
        return {"access_token": access_token, "token_type": "bearer", "expires_in": settings.session_ttl_minutes * 60, "home_id": row["home_id"], "user_id": row["camera_id"]}

    @app.post("/api/v1/camera/reconnect-link")
    def camera_reconnect_link(actor: Current):
        if actor["role"] != "publisher":
            raise HTTPException(403, "Only a paired camera can create its reconnect link")
        camera = db.one(
            "SELECT id, enabled FROM cameras WHERE id=? AND home_id=?",
            (actor["user_id"], actor["home_id"]),
        )
        if not camera or not camera["enabled"]:
            raise HTTPException(404, "Camera is unavailable")
        reconnect_token = issue_camera_reconnect(actor["home_id"], actor["user_id"])
        audit(actor, "camera.reconnect_link.create", "camera", actor["user_id"], actor["home_id"])
        return {"camera_id": actor["user_id"], "reconnect_token": reconnect_token}

    @app.delete("/api/v1/sessions/current")
    def logout(request: Request, actor: Current):
        db.execute("DELETE FROM sessions WHERE token_hash=?", (hash_secret(request.headers["authorization"][7:]),)); audit(actor, "session.logout"); return {"ok": True}

    @app.post("/api/v1/homes/{home_id}/consents")
    def consent(home_id: str, body: ConsentIn, actor: Current):
        home_check(actor, home_id)
        # A paired camera may record only its own camera/microphone decision.
        # It cannot use this endpoint to change household, family, or privacy
        # controls. This keeps the permission prompt on the camera device
        # while still making the consent gate durable for map generation.
        if actor["role"] == "publisher":
            if body.purpose not in {"video_capture", "audio_capture"} or body.subject_user_id not in (None, actor["user_id"]):
                publisher_block(actor)
        else:
            publisher_block(actor)
        if body.subject_user_id and body.care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        subject_user_id = body.subject_user_id or actor["user_id"]
        if actor["role"] != "publisher":
            member(home_id, subject_user_id)
        if subject_user_id != actor["user_id"] and actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Only a caregiver or admin can record a represented subject decision")
        care_recipient_id = canonical_uuid(body.care_recipient_id, "care_recipient_id") if body.care_recipient_id else None
        if care_recipient_id:
            family_actor(actor)
            if not db.one("SELECT id FROM care_recipients WHERE id=? AND home_id=?", (care_recipient_id, home_id)):
                raise HTTPException(404, "Care recipient not found")
        cid = str(uuid.uuid4()); timestamp = now_iso(); db.execute(
            "INSERT INTO consents(id,home_id,subject_user_id,purpose,policy_version,granted_at,revoked_at,care_recipient_id) VALUES (?,?,?,?,?,?,?,?)",
            (cid, home_id, subject_user_id, body.purpose, body.policy_version, timestamp, None if body.granted else timestamp, care_recipient_id),
        )
        if body.purpose == "video_capture":
            db.execute("INSERT INTO home_runtime(home_id,paused,updated_at) VALUES (?,?,?) ON CONFLICT(home_id) DO UPDATE SET paused=excluded.paused, updated_at=excluded.updated_at", (home_id, 0 if body.granted else 1, timestamp))
        audit(actor, "consent.grant" if body.granted else "consent.revoke", "consent", cid, home_id); return {"id": cid, "granted": body.granted, "subject_user_id": subject_user_id, "care_recipient_id": care_recipient_id, "paused": is_paused(home_id)}

    @app.get("/api/v1/homes/{home_id}/consents")
    def consent_list(home_id: str, actor: Current): home_check(actor, home_id); publisher_block(actor); return {"data": db.many("SELECT * FROM consents WHERE home_id=? ORDER BY granted_at DESC", (home_id,))}

    @app.get("/api/v1/homes/{home_id}/runtime")
    def runtime(home_id: str, actor: Current):
        home_check(actor, home_id); return {"home_id": home_id, "paused": is_paused(home_id), "video_capture_consented": active_video_consent(home_id)}

    @app.post("/api/v1/homes/{home_id}/cameras")
    def camera(home_id: str, body: CameraIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); cid = str(uuid.uuid4()); db.execute("INSERT INTO cameras(id,home_id,name,room_id,enabled,created_at,resolution_width,resolution_height,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)", (cid, home_id, body.name, body.room_id, 1, now_iso(), body.resolution_width, body.resolution_height, json.dumps(body.metadata))); return {"id": cid, **body.model_dump(), "enabled": True}

    @app.patch("/api/v1/homes/{home_id}/cameras/{camera_id}")
    def camera_update(home_id: str, camera_id: str, body: CameraUpdate, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM cameras WHERE id=? AND home_id=?", (camera_id, home_id))
        if not row: raise HTTPException(404, "Camera not found")
        values = {"name": body.name if body.name is not None else row["name"], "room_id": body.room_id if "room_id" in body.model_fields_set else row["room_id"], "resolution_width": body.resolution_width if "resolution_width" in body.model_fields_set else row["resolution_width"], "resolution_height": body.resolution_height if "resolution_height" in body.model_fields_set else row["resolution_height"], "metadata_json": json.dumps(body.metadata) if "metadata" in body.model_fields_set else row.get("metadata_json", "{}")}
        changed = any(values[key] != row.get(key) for key in ("name", "room_id", "resolution_width", "resolution_height", "metadata_json"))
        db.execute("UPDATE cameras SET name=?, room_id=?, resolution_width=?, resolution_height=?, metadata_json=? WHERE id=? AND home_id=?", (*values.values(), camera_id, home_id))
        if changed: db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='camera metadata or resolution changed' WHERE home_id=? AND camera_id=? AND status='active'", (now_iso(), home_id, camera_id))
        return {"id": camera_id, "name": values["name"], "room_id": values["room_id"], "resolution_width": values["resolution_width"], "resolution_height": values["resolution_height"], "metadata": body.metadata, "calibrations_invalidated": changed}

    @app.delete("/api/v1/homes/{home_id}/cameras/{camera_id}")
    def camera_delete(home_id: str, camera_id: str, actor: Current):
        """Disable a camera and revoke its publisher session.

        Camera records stay as an audit-safe tombstone so historical maps and
        observations do not lose their camera reference. The device is no
        longer returned to caregivers and its bearer session cannot publish a
        new LiveKit token.
        """
        home_check(actor, home_id)
        family_actor(actor)
        row = db.one("SELECT * FROM cameras WHERE id=? AND home_id=?", (camera_id, home_id))
        if not row:
            raise HTTPException(404, "Camera not found")
        if row["enabled"]:
            timestamp = now_iso()
            db.execute("UPDATE cameras SET enabled=0 WHERE id=? AND home_id=?", (camera_id, home_id))
            db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='camera removed' WHERE home_id=? AND camera_id=? AND status='active'", (timestamp, home_id, camera_id))
            revoked_sessions = db.execute("DELETE FROM sessions WHERE home_id=? AND user_id=?", (home_id, camera_id)).rowcount
            db.execute("UPDATE camera_reconnect_tokens SET revoked_at=? WHERE home_id=? AND camera_id=? AND revoked_at IS NULL", (timestamp, home_id, camera_id))
            db.execute("DELETE FROM pairing_codes WHERE home_id=? AND user_id=? AND used_at IS NULL", (home_id, camera_id))
            audit(actor, "camera.delete", "camera", camera_id, home_id)
        else:
            revoked_sessions = 0
        return {"id": camera_id, "status": "deleted", "revoked_sessions": revoked_sessions}

    @app.get("/api/v1/homes/{home_id}/cameras")
    def cameras(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return {"data": [camera_view(row) for row in db.many("SELECT * FROM cameras WHERE home_id=? AND enabled=1 ORDER BY created_at DESC", (home_id,))]}

    @app.get("/api/v1/homes/{home_id}/rooms")
    def rooms(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return {"data": db.many("SELECT * FROM rooms WHERE home_id=? ORDER BY created_at", (home_id,))}

    @app.post("/api/v1/homes/{home_id}/rooms")
    def room(home_id: str, body: RoomIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); rid = str(uuid.uuid4()); db.execute("INSERT INTO rooms VALUES (?,?,?,?)", (rid, home_id, body.name, now_iso())); return {"id": rid, **body.model_dump()}

    def json_object(value: object) -> dict:
        """Decode persisted JSON without manufacturing geometry or metadata."""
        if not isinstance(value, str):
            return {}
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def map_source(row: dict) -> str:
        source = str(row.get("source") or "legacy-2d")
        # Only explicit validated producer contracts can opt into current geometry.
        # Older manual/generic rows must remain visible, but are legacy and
        # need a fresh automatic sweep before they can drive the map UI.
        return source if source in {"camera-cv-2d", "roomplan-lidar-3d", "arkit-video-3d", "legacy-2d"} else "legacy-2d"

    def map_dimension(row: dict, source: str) -> str:
        # A generic or legacy map can never opt into the 3D renderer by
        # putting an arbitrary value in its stored JSON.
        return "3d" if source in {"roomplan-lidar-3d", "arkit-video-3d"} and row.get("dimension") == "3d" else "2d"

    def map_uses_real_geometry_model(row: dict) -> bool:
        """Keep historical fixture revisions out of the active map surface."""
        if map_source(row) != "camera-cv-2d":
            return True
        metadata = json_object(row.get("metadata_json") or "{}")
        model_version = metadata.get("model_version")
        if not isinstance(model_version, str):
            try:
                map_data = json.loads(row.get("map_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                map_data = {}
            model_version = map_data.get("model_version") if isinstance(map_data, dict) else None
        normalized = model_version.strip().lower() if isinstance(model_version, str) else ""
        return not normalized.startswith(("mock", "fixture", "deterministic"))

    def roomplan_pose_scene_validation(map_row: dict, camera_to_world: object) -> tuple[bool, dict]:
        """Reject visual-PnP poses that contradict the metric RoomPlan floor."""
        if (
            not isinstance(camera_to_world, list)
            or len(camera_to_world) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in camera_to_world)
        ):
            return False, {"accepted": False, "reason": "invalid_pose_matrix"}
        try:
            values = [[float(value) for value in row] for row in camera_to_world]
        except (TypeError, ValueError):
            return False, {"accepted": False, "reason": "invalid_pose_matrix"}
        if not all(math.isfinite(value) for row in values for value in row):
            return False, {"accepted": False, "reason": "invalid_pose_matrix"}

        map_data = json_object(map_row.get("map_json") or "{}")
        geometry = map_data.get("geometry") if isinstance(map_data.get("geometry"), dict) else {}
        zones = geometry.get("room_zones") if isinstance(geometry.get("room_zones"), list) else []
        if not zones:
            return True, {"accepted": True, "reason": "room_zone_prior_unavailable"}

        x, y, z = values[0][3], values[1][3], values[2][3]

        def segment_distance(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
            dx, dz = bx - ax, bz - az
            length_sq = dx * dx + dz * dz
            if length_sq <= 1e-12:
                return math.hypot(px - ax, pz - az)
            t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length_sq))
            return math.hypot(px - (ax + t * dx), pz - (az + t * dz))

        def polygon_distance(px: float, pz: float, polygon: list[tuple[float, float]]) -> float:
            inside = False
            previous = polygon[-1]
            minimum = float("inf")
            for current in polygon:
                ax, az = previous
                bx, bz = current
                minimum = min(minimum, segment_distance(px, pz, ax, az, bx, bz))
                if (az > pz) != (bz > pz):
                    crossing_x = (bx - ax) * (pz - az) / (bz - az) + ax
                    if px < crossing_x:
                        inside = not inside
                previous = current
            return 0.0 if inside else minimum

        candidates: list[dict] = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            raw_polygon = zone.get("polygon")
            floor_y = zone.get("floor_y")
            if not isinstance(raw_polygon, list) or len(raw_polygon) < 3 or not isinstance(floor_y, (int, float)):
                continue
            polygon: list[tuple[float, float]] = []
            for point in raw_polygon:
                if not isinstance(point, dict):
                    continue
                px, pz = point.get("x"), point.get("z")
                if isinstance(px, (int, float)) and isinstance(pz, (int, float)) and math.isfinite(float(px)) and math.isfinite(float(pz)):
                    polygon.append((float(px), float(pz)))
            if len(polygon) < 3 or not math.isfinite(float(floor_y)):
                continue
            candidates.append(
                {
                    "zone_id": zone.get("id"),
                    "distance_m": polygon_distance(x, z, polygon),
                    "height_above_floor_m": y - float(floor_y),
                }
            )

        if not candidates:
            return True, {"accepted": True, "reason": "usable_room_zone_prior_unavailable"}

        nearest = min(candidates, key=lambda item: item["distance_m"])
        height = float(nearest["height_above_floor_m"])
        distance = float(nearest["distance_m"])
        accepted = 0.15 <= height <= 3.50 and distance <= 0.75
        return accepted, {
            "accepted": accepted,
            "reason": "within_roomplan_bounds" if accepted else "pose_outside_roomplan_bounds",
            "zone_id": nearest.get("zone_id"),
            "camera_height_above_floor_m": round(height, 4),
            "distance_from_room_polygon_m": round(distance, 4),
        }

    def active_map_row(home_id: str) -> dict | None:
        rows = db.many(
            "SELECT * FROM room_maps WHERE home_id=? ORDER BY revision DESC, created_at DESC",
            (home_id,),
        )
        # A native metric RoomPlan scene remains the authoritative home map
        # even when a fixed browser camera later creates a relative 2D sweep.
        # Camera sweeps remain available by id for review and calibration, but
        # they cannot silently demote the live scene back to 2D.
        roomplan = next(
            (
                row
                for row in rows
                if map_source(row) == "roomplan-lidar-3d"
                and map_dimension(row, "roomplan-lidar-3d") == "3d"
            ),
            None,
        )
        if roomplan is not None:
            return roomplan
        arkit_video = next(
            (
                row
                for row in rows
                if map_source(row) == "arkit-video-3d"
                and map_dimension(row, "arkit-video-3d") == "3d"
            ),
            None,
        )
        if arkit_video is not None:
            return arkit_video
        return next((row for row in rows if map_uses_real_geometry_model(row)), None)

    def camera_roomplan_calibration_state(home_id: str, camera_id: str) -> dict:
        map_row = active_map_row(home_id)
        if not map_row or map_source(map_row) != "roomplan-lidar-3d" or map_dimension(map_row, "roomplan-lidar-3d") != "3d":
            return {
                "calibration_needed": False,
                "roomplan_registration_status": "map_required",
                "roomplan_map_id": None,
            }
        calibration = db.one(
            "SELECT * FROM calibrations WHERE home_id=? AND camera_id=? AND map_id=? AND source IN ('auto-roomplan-registration','visual-roomplan-registration') AND status IN ('active','needs_rescan','needs_review') ORDER BY created_at DESC LIMIT 1",
            (home_id, camera_id, map_row["id"]),
        )
        registration_status = "unavailable"
        if calibration:
            if calibration.get("status") == "needs_review":
                registration_status = "needs_review"
            elif calibration.get("status") == "needs_rescan":
                registration_status = "needs_rescan"
            else:
                extrinsics = json_object(calibration.get("extrinsics_json") or "{}")
                scene_valid, _ = roomplan_pose_scene_validation(map_row, extrinsics.get("camera_to_world"))
                registration_status = "positioned" if scene_valid else "needs_rescan"
        return {
            "calibration_needed": registration_status != "positioned",
            "roomplan_registration_status": registration_status,
            "roomplan_map_id": map_row["id"],
        }

    def roomplan_calibration_targets(map_row: dict, target_count: int = 4) -> list[dict]:
        try:
            map_data = json.loads(map_row.get("map_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            return []
        geometry_payload = map_data.get("geometry") if isinstance(map_data, dict) else None
        geometry_payload = geometry_payload if isinstance(geometry_payload, dict) else map_data
        normalized_scan = map_data.get("normalized_scan") if isinstance(map_data, dict) else None
        normalized_scan = normalized_scan if isinstance(normalized_scan, dict) else {}
        raw_zones = geometry_payload.get("room_zones") if isinstance(geometry_payload, dict) else None
        if not isinstance(raw_zones, list):
            return []

        def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
            x, z = point
            inside = False
            previous = polygon[-1]
            for current in polygon:
                x1, z1 = current
                x2, z2 = previous
                crosses = (z1 > z) != (z2 > z)
                if crosses and x < ((x2 - x1) * (z - z1) / ((z2 - z1) or 1e-12)) + x1:
                    inside = not inside
                previous = current
            return inside

        def polygon_area(polygon: list[tuple[float, float]]) -> float:
            return abs(sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )) * 0.5

        def distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
            px, pz = point
            sx, sz = start
            ex, ez = end
            dx = ex - sx
            dz = ez - sz
            length_sq = dx * dx + dz * dz
            if length_sq <= 1e-12:
                return math.hypot(px - sx, pz - sz)
            t = max(0.0, min(1.0, ((px - sx) * dx + (pz - sz) * dz) / length_sq))
            return math.hypot(px - (sx + t * dx), pz - (sz + t * dz))

        def distance_to_polygon_edge(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
            return min(
                distance_to_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
                for index in range(len(polygon))
            )

        def finite_number(value: object) -> float | None:
            if not isinstance(value, (int, float)):
                return None
            number = float(value)
            return number if math.isfinite(number) else None

        def obstacle_footprints(floor_y: float) -> list[list[tuple[float, float]]]:
            raw_objects = normalized_scan.get("objects")
            if not isinstance(raw_objects, list) or not raw_objects:
                raw_objects = geometry_payload.get("objects") if isinstance(geometry_payload, dict) else []
            if not isinstance(raw_objects, list):
                return []
            footprints: list[list[tuple[float, float]]] = []
            standing_clearance = 0.38
            for room_object in raw_objects:
                if not isinstance(room_object, dict):
                    continue
                center = room_object.get("center") if isinstance(room_object.get("center"), dict) else room_object.get("position")
                dimensions = room_object.get("dimensions")
                if not isinstance(center, dict) or not isinstance(dimensions, dict):
                    continue
                center_x = finite_number(center.get("x"))
                center_y = finite_number(center.get("y"))
                center_z = finite_number(center.get("z"))
                width = finite_number(dimensions.get("x"))
                height = finite_number(dimensions.get("y"))
                depth = finite_number(dimensions.get("z"))
                if None in {center_x, center_y, center_z, width, height, depth} or width <= 0 or height <= 0 or depth <= 0:
                    continue
                bottom_y = center_y - height * 0.5
                top_y = center_y + height * 0.5
                if top_y < floor_y + 0.04 or bottom_y > floor_y + 2.0:
                    continue
                half_x = width * 0.5 + standing_clearance
                half_z = depth * 0.5 + standing_clearance
                local_corners = [(-half_x, -half_z), (half_x, -half_z), (half_x, half_z), (-half_x, half_z)]
                transform = room_object.get("transform")
                if (
                    isinstance(transform, list)
                    and len(transform) == 4
                    and all(isinstance(row, list) and len(row) == 4 for row in transform)
                    and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for row in transform for value in row)
                ):
                    footprints.append([
                        (
                            float(transform[0][0]) * x + float(transform[0][2]) * z + float(transform[0][3]),
                            float(transform[2][0]) * x + float(transform[2][2]) * z + float(transform[2][3]),
                        )
                        for x, z in local_corners
                    ])
                else:
                    footprints.append([(center_x + x, center_z + z) for x, z in local_corners])
            return footprints

        zones: list[tuple[float, float, list[tuple[float, float]]]] = []
        for zone in raw_zones:
            if not isinstance(zone, dict) or not isinstance(zone.get("polygon"), list) or not isinstance(zone.get("floor_y"), (int, float)):
                continue
            polygon = [
                (float(point["x"]), float(point["z"]))
                for point in zone["polygon"]
                if isinstance(point, dict)
                and isinstance(point.get("x"), (int, float))
                and isinstance(point.get("z"), (int, float))
                and math.isfinite(float(point["x"]))
                and math.isfinite(float(point["z"]))
            ]
            if len(polygon) >= 3 and math.isfinite(float(zone["floor_y"])):
                zones.append((polygon_area(polygon), float(zone["floor_y"]), polygon))
        if not zones:
            return []
        _, floor_y, polygon = max(zones, key=lambda item: item[0])
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_z = min(point[1] for point in polygon)
        max_z = max(point[1] for point in polygon)
        center = (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
        raw_targets = [
            (min_x * 0.72 + max_x * 0.28, min_z * 0.72 + max_z * 0.28),
            (min_x * 0.28 + max_x * 0.72, min_z * 0.72 + max_z * 0.28),
            (min_x * 0.72 + max_x * 0.28, min_z * 0.28 + max_z * 0.72),
            (min_x * 0.28 + max_x * 0.72, min_z * 0.28 + max_z * 0.72),
        ]
        obstacles = obstacle_footprints(floor_y)
        span_x = max_x - min_x
        span_z = max_z - min_z
        grid_step = max(0.18, min(0.30, min(span_x, span_z) / 14.0))
        candidate_points: list[tuple[float, float]] = [*raw_targets, center]
        x = min_x + grid_step
        while x < max_x - grid_step * 0.5:
            z = min_z + grid_step
            while z < max_z - grid_step * 0.5:
                candidate_points.append((x, z))
                z += grid_step
            x += grid_step

        def safe_candidates(wall_clearance: float) -> list[tuple[float, float]]:
            seen: set[tuple[int, int]] = set()
            result: list[tuple[float, float]] = []
            for point in candidate_points:
                key = (round(point[0] * 1000), round(point[1] * 1000))
                if key in seen:
                    continue
                seen.add(key)
                if not point_in_polygon(point, polygon):
                    continue
                if distance_to_polygon_edge(point, polygon) < wall_clearance:
                    continue
                if any(point_in_polygon(point, footprint) for footprint in obstacles):
                    continue
                result.append(point)
            return result

        desired_count = max(4, min(12, int(target_count)))
        chosen: list[tuple[float, float]] = []
        remaining: list[tuple[float, float]] = []
        selected_separation = 0.60
        for wall_clearance, minimum_separation in ((0.35, 0.75), (0.22, 0.60)):
            available = safe_candidates(wall_clearance)
            chosen = []
            for seed in raw_targets:
                eligible = [
                    point for point in available
                    if all(math.hypot(point[0] - prior[0], point[1] - prior[1]) >= minimum_separation for prior in chosen)
                ]
                if not eligible:
                    chosen = []
                    break
                selected = min(eligible, key=lambda point: (point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2)
                chosen.append(selected)
                available.remove(selected)
            if len(chosen) == 4:
                remaining = available
                selected_separation = minimum_separation
                break
        if len(chosen) != 4:
            return []

        # Keep extra clear-floor candidates private to the session. If the
        # fixed camera cannot see a person at one of the primary four points,
        # the API can swap only that point while preserving prior captures.
        replacement_separation = max(0.48, selected_separation * 0.72)
        while len(chosen) < desired_count:
            eligible = [
                point for point in remaining
                if all(math.hypot(point[0] - prior[0], point[1] - prior[1]) >= replacement_separation for prior in chosen)
            ]
            if not eligible:
                break
            selected = max(
                eligible,
                key=lambda point: min(math.hypot(point[0] - prior[0], point[1] - prior[1]) for prior in chosen),
            )
            chosen.append(selected)
            remaining.remove(selected)

        return [
            {"index": index, "x": round(point[0], 4), "y": round(floor_y, 4), "z": round(point[1], 4)}
            for index, point in enumerate(chosen)
        ]

    def roomplan_calibration_session_view(session: dict) -> dict:
        current_index = int(session.get("current_target_index", 0))
        status_value = str(session.get("status") or "waiting_for_person")
        targets = []
        for target in session.get("targets", []):
            index = int(target["index"])
            if index < current_index or status_value in {"solving", "review", "failed"} and index <= current_index:
                state = "complete"
            elif index == current_index and status_value not in {"review", "failed", "cancelled"}:
                state = "active"
            else:
                state = "pending"
            targets.append({**target, "state": state})
        return {
            "session_id": session["session_id"],
            "camera_id": session["camera_id"],
            "map_id": session["map_id"],
            "status": status_value,
            "current_target_index": current_index,
            "captured_target_count": min(current_index, len(targets)) if status_value not in {"solving", "review", "failed"} else len(targets),
            "targets": targets,
            "proposal": session.get("proposal"),
            "error": session.get("error"),
            "created_at": session["created_at"],
            "expires_at": session["expires_at"],
            "raw_frames_persisted": False,
        }

    def active_roomplan_calibration_session(home_id: str, camera_id: str) -> dict | None:
        key = (home_id, camera_id)
        with roomplan_calibration_lock:
            session = roomplan_calibration_sessions.get(key)
            if not session:
                return None
            expires_at = datetime.fromisoformat(str(session["expires_at"]).replace("Z", "+00:00"))
            if expires_at <= datetime.now(timezone.utc):
                session["frames"] = []
                session["anchors"] = []
                session["status"] = "expired"
            return session

    def roomplan_camera_registration_view(home_id: str, map_row: dict) -> dict | None:
        source = map_source(map_row)
        if source != "roomplan-lidar-3d" or map_dimension(map_row, source) != "3d":
            return None
        calibration = db.one(
            "SELECT * FROM calibrations WHERE home_id=? AND map_id=? AND source IN ('auto-roomplan-registration','visual-roomplan-registration') AND status IN ('active','needs_rescan') ORDER BY created_at DESC LIMIT 1",
            (home_id, map_row["id"]),
        )
        if not calibration:
            return {
                # A valid RoomPlan map and an unpositioned fixed camera are
                # independent states. No calibration row means positioning has
                # not been attempted for this map; it is not a failed scan.
                "status": "unavailable",
                "cameraId": None,
                "mapId": map_row["id"],
                "coordinateFrame": "roomplan-local",
                "cameraToWorld": None,
                "confidence": None,
                "trackingState": None,
                "source": "visual-roomplan-registration",
            }
        metrics = json_object(calibration.get("metrics_json") or "{}")
        extrinsics = json_object(calibration.get("extrinsics_json") or "{}")
        camera = db.one(
            "SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1",
            (calibration["camera_id"], home_id),
        )
        scene_valid, _ = roomplan_pose_scene_validation(map_row, extrinsics.get("camera_to_world"))
        positioned = calibration.get("status") == "active" and camera is not None and scene_valid
        status_value = "positioned" if positioned else ("needs_rescan" if camera is not None else "unavailable")
        return {
            "status": status_value,
            "cameraId": calibration["camera_id"] if camera is not None else None,
            "mapId": map_row["id"],
            "coordinateFrame": "roomplan-local",
            "cameraToWorld": extrinsics.get("camera_to_world") if positioned else None,
            "confidence": metrics.get("confidence"),
            "trackingState": metrics.get("tracking_state"),
            "source": calibration.get("source") or "visual-roomplan-registration",
        }

    def roomplan_camera_registration_views(home_id: str, map_row: dict) -> list[dict]:
        if map_source(map_row) != "roomplan-lidar-3d" or map_dimension(map_row, "roomplan-lidar-3d") != "3d":
            return []
        rows = db.many(
            "SELECT * FROM calibrations WHERE home_id=? AND map_id=? AND source IN ('auto-roomplan-registration','visual-roomplan-registration') AND status IN ('active','needs_rescan') ORDER BY created_at DESC",
            (home_id, map_row["id"]),
        )
        seen: set[str] = set()
        result: list[dict] = []
        for calibration in rows:
            camera_id = calibration.get("camera_id")
            if not isinstance(camera_id, str) or camera_id in seen:
                continue
            seen.add(camera_id)
            camera = db.one("SELECT id,name,room_id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
            if camera is None:
                continue
            metrics = json_object(calibration.get("metrics_json") or "{}")
            extrinsics = json_object(calibration.get("extrinsics_json") or "{}")
            scene_valid, _ = roomplan_pose_scene_validation(map_row, extrinsics.get("camera_to_world"))
            positioned = calibration.get("status") == "active" and scene_valid
            result.append(
                {
                    "status": "positioned" if positioned else "needs_rescan",
                    "cameraId": camera_id,
                    "cameraName": camera.get("name"),
                    "roomId": camera.get("room_id"),
                    "mapId": map_row["id"],
                    "coordinateFrame": "roomplan-local",
                    "cameraToWorld": extrinsics.get("camera_to_world") if positioned else None,
                    "confidence": metrics.get("confidence"),
                    "trackingState": metrics.get("tracking_state"),
                    "source": calibration.get("source") or "visual-roomplan-registration",
                    "intrinsics": json_object(calibration.get("intrinsics_json") or "{}"),
                    "metrics": metrics,
                }
            )
        return result

    def map_view(row: dict) -> dict:
        """Return map provenance while preserving legacy JSON verbatim."""
        try:
            map_data = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            map_data = {}
        metadata = json_object(row.get("metadata_json") or "{}")
        source = map_source(row)
        dimension = map_dimension(row, source)
        real_geometry = map_uses_real_geometry_model(row)
        rescan_required = (
            source == "legacy-2d"
            or row.get("localization_status") == "rescan-required"
            or (source == "camera-cv-2d" and not real_geometry)
        )
        if rescan_required:
            geometry_status = "rescan-required"
        elif source in {"camera-cv-2d", "roomplan-lidar-3d", "arkit-video-3d"}:
            geometry_status = "ready"
        else:
            geometry_status = "legacy"
        model_version = metadata.get("model_version") if real_geometry else None
        scale = map_data.get("scale") if isinstance(map_data, dict) and isinstance(map_data.get("scale"), dict) else metadata.get("scale")
        if not isinstance(scale, dict):
            scale = None
        usdz = metadata.get("usdz") if isinstance(metadata.get("usdz"), dict) else None
        if not row.get("usdz_artifact_key"):
            usdz = None
        return {
            "id": row["id"],
            "home_id": row["home_id"],
            "room_id": row["room_id"],
            "revision": row["revision"],
            "coordinate_frame": row["coordinate_frame"],
            "map_data": map_data,
            "created_at": row["created_at"],
            "source": source,
            "provenance": source,
            "dimension": dimension,
            "approximate": bool(row.get("approximate", 0)) or source == "legacy-2d",
            "metric_scale_known": dimension == "3d",
            "scale": scale,
            "localization_status": row.get("localization_status", "unlocalized"),
            "geometry_status": geometry_status,
            "rescan_required": rescan_required,
            "model_version": model_version if isinstance(model_version, str) else None,
            "metadata": metadata,
            "usdz": usdz,
        }

    def create_map(
        home_id: str,
        room_id: str | None,
        coordinate_frame: str,
        map_data: dict,
        source: str,
        approximate: bool,
        localization_status: str,
        metadata: dict,
        actor: dict,
        dimension: str = "2d",
    ) -> dict:
        row = db.one("SELECT COALESCE(MAX(revision),0)+1 revision FROM room_maps WHERE home_id=? AND room_id IS ?", (home_id, room_id))
        if source not in {"roomplan-lidar-3d", "arkit-video-3d"}:
            dimension = "2d"
        mid = str(uuid.uuid4())
        key = f"maps/{home_id}/{mid}.json"
        store.put_json(key, map_data)
        created = now_iso()
        db.execute(
            "INSERT INTO room_maps(id,home_id,room_id,revision,coordinate_frame,artifact_key,map_json,created_at,source,dimension,approximate,localization_status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, home_id, room_id, row["revision"], coordinate_frame, key, json.dumps(map_data), created, source, dimension, int(approximate), localization_status, json.dumps(metadata)),
        )
        if source == "roomplan-lidar-3d":
            db.execute(
                "UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='RoomPlan map revision changed' WHERE home_id=? AND status='active' AND map_id != ?",
                (created, home_id, mid),
            )
        else:
            db.execute(
                "UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='camera map revision changed' WHERE home_id=? AND status='active' AND map_id != ? AND source NOT IN ('auto-roomplan-registration','visual-roomplan-registration')",
                (created, home_id, mid),
            )
        audit(actor, "map.create", "room_map", mid, home_id)
        return {"id": mid, "revision": row["revision"], "coordinate_frame": coordinate_frame, "artifact_key": key, "source": source, "provenance": source, "dimension": dimension, "approximate": approximate, "metric_scale_known": dimension == "3d", "scale": map_data.get("scale") if isinstance(map_data, dict) and isinstance(map_data.get("scale"), dict) else None, "localization_status": localization_status, "geometry_status": "ready", "rescan_required": localization_status == "rescan-required", "metadata": metadata, "usdz": None}

    @app.post("/api/v1/homes/{home_id}/maps")
    def room_map(home_id: str, body: MapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return create_map(home_id, body.room_id, body.coordinate_frame, body.map_data, "legacy-2d", True, "rescan-required", {"legacy_reason": "generic map uploads are 2D-only and require a camera rescan"}, actor)

    @app.post("/api/v1/homes/{home_id}/maps/provisional")
    def provisional_map(home_id: str, body: ProvisionalMapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        camera = db.one("SELECT * FROM cameras WHERE id=? AND home_id=? AND enabled=1", (body.camera_id, home_id))
        if not camera: raise HTTPException(404, "Camera not found or disabled")
        return create_map(home_id, body.room_id, "camera-zone-local", {"zones": body.zones}, "legacy-2d", True, "rescan-required", {"camera_id": body.camera_id, "resolution_width": body.resolution_width, "resolution_height": body.resolution_height, "legacy_reason": "provisional camera zones do not contain inferred geometry"}, actor)

    @app.post("/api/v1/homes/{home_id}/maps/roomplan")
    def roomplan_map(home_id: str, body: RoomPlanMapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        normalized_scan = body.normalized_scan.model_dump(mode="json")
        metadata = {**body.scan_metadata.model_dump(), "model_version": "native-roomplan"}
        map_data = {
            "schema_version": body.normalized_scan.schema_version,
            "source": "roomplan-lidar-3d",
            "dimension": "3d",
            "coordinate_frame": body.normalized_scan.coordinate_frame,
            "normalized_scan": normalized_scan,
            "geometry": roomplan_geometry(body.normalized_scan),
        }
        return create_map(home_id, body.room_id, "roomplan-local", map_data, "roomplan-lidar-3d", False, "metric-local", metadata, actor, dimension="3d")

    @app.post("/api/v1/homes/{home_id}/maps/arkit-video")
    def arkit_video_map(home_id: str, body: ARVideoMapIn, actor: Current):
        """Create an approximate metric 3D room from a native non-LiDAR ARKit sweep."""
        home_check(actor, home_id); publisher_block(actor)
        geometry = arkit_video_geometry(body)
        try:
            usdz_payload = build_arkit_video_usdz(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        metadata = {
            "provenance": "native-arkit-video",
            "model_version": "native-arkit-video-structural-v1",
            "device_framework": "ARKit",
            "lidar": False,
            "units": "m",
            "up_axis": "Y",
            "diagnostics": body.diagnostics.model_dump(mode="json"),
        }
        map_data = {
            "schema_version": body.schema_version,
            "source": "arkit-video-3d",
            "dimension": "3d",
            "coordinate_frame": body.coordinate_frame,
            "geometry": geometry,
        }
        created = create_map(
            home_id,
            body.room_id,
            body.coordinate_frame,
            map_data,
            "arkit-video-3d",
            True,
            "metric-approximate",
            metadata,
            actor,
            dimension="3d",
        )
        map_id = created["id"]
        key = f"maps/{home_id}/{map_id}.usdz"
        store.put_bytes(key, usdz_payload)
        metadata["usdz"] = roomplan_usdz_metadata(
            sha256=hashlib.sha256(usdz_payload).hexdigest(),
            byte_count=len(usdz_payload),
            content_type="model/vnd.usdz+zip",
            download_path=f"/api/v1/homes/{home_id}/maps/{map_id}/usdz",
        )
        db.execute(
            "UPDATE room_maps SET usdz_artifact_key=?, metadata_json=? WHERE id=? AND home_id=?",
            (key, json.dumps(metadata), map_id, home_id),
        )
        audit(actor, "map.usdz.generate", "room_map", map_id, home_id)
        return map_view(db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id)))

    @app.post("/api/v1/homes/{home_id}/maps/{map_id}/visual-landmarks")
    def roomplan_visual_landmarks(home_id: str, map_id: str, body: RoomPlanVisualLandmarksIn, actor: Current):
        """Build a derived, local-only visual landmark index for RoomPlan relocalization."""
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row:
            raise HTTPException(404, "Map not found")
        active = active_map_row(home_id)
        if not active or active["id"] != map_id:
            raise HTTPException(409, "Visual landmarks require the active map revision")
        source = map_source(row)
        if source != "roomplan-lidar-3d" or map_dimension(row, source) != "3d" or row.get("coordinate_frame") != "roomplan-local":
            raise HTTPException(422, "Visual landmarks require a native RoomPlan 3D map")
        total_chars = sum(len(frame.frame_base64) + len(frame.depth_base64 or "") for frame in body.frames)
        if total_chars > 24_000_000:
            raise HTTPException(413, "RoomPlan visual landmark batch exceeds the in-memory limit")
        try:
            result = geometry.build_visual_landmarks(
                map_id=map_id,
                frames=[
                    {
                        **frame.model_dump(mode="json", exclude={"intrinsics", "camera_to_world"}),
                        "intrinsics": {"values": frame.intrinsics.values},
                        "camera_to_world": {"values": frame.camera_to_world},
                    }
                    for frame in body.frames
                ],
            )
        except RoomLayoutServiceUnavailable as exc:
            raise HTTPException(503, "Local visual localization service is unavailable") from exc
        except RoomLayoutServiceError as exc:
            raise HTTPException(502, "Local visual localization service rejected the scan") from exc
        if not isinstance(result, dict) or result.get("status") not in {"ready", "needs_rescan"}:
            raise HTTPException(502, "Local visual localization service returned an invalid result")
        landmarks = result.get("landmarks") if isinstance(result.get("landmarks"), list) else []
        metadata = json_object(row.get("metadata_json") or "{}")
        previous = metadata.get("visual_landmarks") if isinstance(metadata.get("visual_landmarks"), dict) else {}
        previous_key = previous.get("artifact_key") if isinstance(previous, dict) else None
        previous_payload: dict = {}
        previous_landmarks: list[dict] = []
        if isinstance(previous_key, str) and previous_key:
            try:
                loaded = json.loads(store.get_bytes(previous_key))
                if isinstance(loaded, dict):
                    previous_payload = loaded
                    if isinstance(loaded.get("landmarks"), list):
                        previous_landmarks = [item for item in loaded["landmarks"] if isinstance(item, dict)]
            except (OSError, ValueError, json.JSONDecodeError):
                previous_payload = {}
                previous_landmarks = []

        result_diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
        previous_diagnostics = previous_payload.get("diagnostics") if isinstance(previous_payload.get("diagnostics"), dict) else {}
        next_batch_count = int(previous_diagnostics.get("incremental_batch_count") or (1 if previous_landmarks else 0)) + 1
        view_id = f"scan-view-{next_batch_count}"
        landmarks = [{**landmark, "view_id": view_id} for landmark in landmarks if isinstance(landmark, dict)]

        # Preserve a descriptor for each scan viewpoint. ORB descriptors change
        # with viewing angle, so collapsing all frames into one point-only voxel
        # removes the coherent 2D-to-3D set that PnP needs. The index stays
        # bounded and raw camera frames are still never persisted.
        merged_landmarks = previous_landmarks
        if landmarks:
            voxels: dict[tuple[str, int, int, int], dict] = {}
            for landmark in [*previous_landmarks, *landmarks]:
                point = landmark.get("point")
                if not isinstance(point, list) or len(point) != 3:
                    continue
                try:
                    landmark_view = str(landmark.get("view_id") or "legacy")
                    voxel = tuple(int(round(float(value) / 0.03)) for value in point)
                    key = (landmark_view, *voxel)
                    response = float(landmark.get("response") or 0.0)
                except (TypeError, ValueError):
                    continue
                current = voxels.get(key)
                if current is None or response > float(current.get("response") or 0.0):
                    voxels[key] = landmark
            by_view: dict[str, list[dict]] = {}
            for landmark in voxels.values():
                by_view.setdefault(str(landmark.get("view_id") or "legacy"), []).append(landmark)
            for group in by_view.values():
                group.sort(key=lambda item: float(item.get("response") or 0.0), reverse=True)
            merged_landmarks = []
            offset = 0
            ordered_views = sorted(by_view)
            while len(merged_landmarks) < 5_000:
                added = False
                for current_view in ordered_views:
                    group = by_view[current_view]
                    if offset < len(group):
                        merged_landmarks.append(group[offset])
                        added = True
                        if len(merged_landmarks) == 5_000:
                            break
                if not added:
                    break
                offset += 1

        source_frame_count = int(previous_diagnostics.get("source_frame_count") or 0) + int(
            result_diagnostics.get("source_frame_count") or len(body.frames)
        )
        aggregate_diagnostics = {
            **result_diagnostics,
            "landmark_count": len(merged_landmarks),
            "source_frame_count": source_frame_count,
            "incremental_batch_count": next_batch_count,
            "view_count": len({str(landmark.get("view_id") or "legacy") for landmark in merged_landmarks}),
            "raw_frames_persisted": False,
        }
        aggregate_status = "ready" if merged_landmarks and (result.get("status") == "ready" or previous_landmarks) else result.get("status")
        artifact_key = None
        if aggregate_status == "ready" and merged_landmarks:
            artifact_key = f"maps/{home_id}/{map_id}.visual-landmarks.json"
            store.put_json(
                artifact_key,
                {
                    "schema_version": "roomplan-visual-landmarks.v1",
                    "map_id": map_id,
                    "detector": result.get("detector", "opencv-orb"),
                    "landmarks": merged_landmarks,
                    "diagnostics": aggregate_diagnostics,
                },
            )
        metadata["visual_landmarks"] = {
            "status": aggregate_status,
            "artifact_key": artifact_key,
            "landmark_count": len(merged_landmarks),
            "detector": result.get("detector", "opencv-orb"),
            "updated_at": now_iso(),
            "raw_frames_persisted": False,
        }
        db.execute("UPDATE room_maps SET metadata_json=? WHERE id=? AND home_id=?", (json.dumps(metadata), map_id, home_id))
        audit(actor, "map.visual_landmarks.build", "room_map", map_id, home_id)
        return {
            "map_id": map_id,
            "status": aggregate_status,
            "landmark_count": len(merged_landmarks),
            "detector": result.get("detector", "opencv-orb"),
            "diagnostics": aggregate_diagnostics,
            "raw_frames_persisted": False,
        }

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/localize-roomplan")
    def localize_roomplan_camera(home_id: str, camera_id: str, body: CameraLocalizationIn, actor: Current):
        """Visually register a separate fixed camera inside the active RoomPlan scene."""
        home_check(actor, home_id)
        require_video_capture(home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can localize only its own camera")
        camera = db.one("SELECT * FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        map_row = active_map_row(home_id)
        if not map_row or map_source(map_row) != "roomplan-lidar-3d" or map_dimension(map_row, "roomplan-lidar-3d") != "3d":
            raise HTTPException(409, "A native RoomPlan 3D map is required before visual camera localization")
        metadata = json_object(map_row.get("metadata_json") or "{}")
        landmark_meta = metadata.get("visual_landmarks") if isinstance(metadata.get("visual_landmarks"), dict) else None
        landmark_key = landmark_meta.get("artifact_key") if isinstance(landmark_meta, dict) else None
        if not isinstance(landmark_key, str) or not landmark_key:
            raise HTTPException(409, "This RoomPlan scan does not have a visual landmark index; rescan it with the current iPhone app")
        try:
            landmark_payload = json.loads(store.get_bytes(landmark_key))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(409, "The RoomPlan visual landmark index is unavailable; rebuild the scan") from exc
        landmarks = landmark_payload.get("landmarks") if isinstance(landmark_payload, dict) else None
        if not isinstance(landmarks, list) or len(landmarks) < 6:
            raise HTTPException(409, "The RoomPlan visual landmark index is incomplete; rebuild the scan")
        map_data = json_object(map_row.get("map_json") or "{}")
        map_geometry = map_data.get("geometry") if isinstance(map_data.get("geometry"), dict) else {}
        raw_room_zones = map_geometry.get("room_zones") if isinstance(map_geometry.get("room_zones"), list) else []
        room_zones: list[dict] = []
        for zone in raw_room_zones:
            if not isinstance(zone, dict) or not isinstance(zone.get("polygon"), list) or not isinstance(zone.get("floor_y"), (int, float)):
                continue
            polygon = [
                {"x": float(point["x"]), "z": float(point["z"])}
                for point in zone["polygon"]
                if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("z"), (int, float))
            ]
            if len(polygon) >= 3:
                room_zones.append({"id": zone.get("id"), "floor_y": float(zone["floor_y"]), "polygon": polygon})
        room_objects: list[dict] = []
        normalized_scan = map_data.get("normalized_scan") if isinstance(map_data.get("normalized_scan"), dict) else {}
        native_room_objects = normalized_scan.get("objects") if isinstance(normalized_scan.get("objects"), list) else []
        raw_room_objects = native_room_objects or (map_geometry.get("objects") if isinstance(map_geometry.get("objects"), list) else [])
        for room_object in raw_room_objects:
            if not isinstance(room_object, dict):
                continue
            center = room_object.get("center") if isinstance(room_object.get("center"), dict) else room_object.get("position")
            dimensions = room_object.get("dimensions")
            label = room_object.get("category") if isinstance(room_object.get("category"), str) else room_object.get("label")
            if (
                not isinstance(center, dict)
                or not isinstance(dimensions, dict)
                or not isinstance(label, str)
                or not label.strip()
                or not all(isinstance(center.get(axis), (int, float)) for axis in ("x", "y", "z"))
                or not all(isinstance(dimensions.get(axis), (int, float)) for axis in ("x", "y", "z"))
            ):
                continue
            confidence_value = room_object.get("confidence")
            if isinstance(confidence_value, str):
                confidence = {"high": 0.95, "medium": 0.75, "low": 0.45}.get(confidence_value.strip().lower(), 1.0)
            elif isinstance(confidence_value, (int, float)):
                confidence = float(confidence_value)
            else:
                confidence = 1.0
            room_objects.append(
                {
                    "id": str(room_object.get("id") or f"room-object-{len(room_objects) + 1}"),
                    "label": label.strip().lower(),
                    "center": {axis: float(center[axis]) for axis in ("x", "y", "z")},
                    "dimensions": {axis: float(dimensions[axis]) for axis in ("x", "y", "z")},
                    **(
                        {"transform": {"values": room_object["transform"]}}
                        if isinstance(room_object.get("transform"), list)
                        and len(room_object["transform"]) == 4
                        and all(isinstance(row, list) and len(row) == 4 for row in room_object["transform"])
                        else {}
                    ),
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        recent_calibrations = db.many(
            "SELECT id,metrics_json FROM calibrations WHERE home_id=? AND camera_id=? AND map_id=? AND source='visual-roomplan-registration' AND status='needs_rescan' ORDER BY created_at DESC LIMIT 80",
            (home_id, camera_id, map_row["id"]),
        )
        search_prior = _camera_localization_search_prior(recent_calibrations)
        try:
            result = geometry.localize_camera(
                landmarks=landmarks,
                frames=[frame.model_dump(mode="json") for frame in body.frames],
                intrinsics=body.intrinsics.values if body.intrinsics is not None else None,
                fov_degrees=body.fov_degrees,
                room_zones=room_zones,
                search_prior=search_prior,
                room_objects=room_objects,
                person_anchors=[anchor.model_dump(mode="json") for anchor in body.person_anchors],
            )
        except RoomLayoutServiceUnavailable as exc:
            raise HTTPException(503, "Local camera localization service is unavailable") from exc
        except RoomLayoutServiceError as exc:
            raise HTTPException(502, "Local camera localization service rejected the fixed-camera frames") from exc
        if not isinstance(result, dict) or result.get("status") not in {"positioned", "needs_rescan"}:
            raise HTTPException(502, "Local camera localization service returned an invalid result")

        positioned = result.get("status") == "positioned" and isinstance(result.get("camera_to_world"), list)
        scene_valid, scene_diagnostics = roomplan_pose_scene_validation(map_row, result.get("camera_to_world"))
        if positioned and not scene_valid:
            positioned = False
            result["status"] = "needs_rescan"
            result["camera_to_world"] = None
        result_diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
        result_diagnostics = {
            **result_diagnostics,
            "scene_validation": scene_diagnostics,
            "search_prior": search_prior,
        }
        result_diagnostics = _stabilize_camera_localization_diagnostics(
            result_diagnostics,
            search_prior,
            positioned=positioned,
        )
        result["diagnostics"] = result_diagnostics
        created = now_iso()
        cid = str(uuid.uuid4())
        metrics = {
            "confidence": result.get("confidence"),
            "tracking_state": "visual-pnp",
            "inlier_count": result.get("inlier_count", 0),
            "match_count": result.get("match_count", 0),
            "reprojection_error_px": result.get("reprojection_error_px"),
            "intrinsics_source": result.get("intrinsics_source"),
            "diagnostics": result_diagnostics,
        }
        intrinsics = result.get("intrinsics") if isinstance(result.get("intrinsics"), list) else (body.intrinsics.values if body.intrinsics else None)
        intrinsics_json = {
            "matrix": intrinsics,
            "source": result.get("intrinsics_source"),
            "fov_degrees": body.fov_degrees,
        }
        if not body.review_only:
            db.execute(
                "UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='superseded by visual RoomPlan registration' WHERE home_id=? AND camera_id=? AND status='active'",
                (created, home_id, camera_id),
            )
        storage_status = "needs_review" if body.review_only else ("active" if positioned else "needs_rescan")
        db.execute(
            "INSERT INTO calibrations(id,home_id,camera_id,map_id,intrinsics_json,extrinsics_json,accuracy_m,created_at,resolution_width,resolution_height,camera_metadata_json,metrics_json,source,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                home_id,
                camera_id,
                map_row["id"],
                json.dumps(intrinsics_json),
                json.dumps({"camera_to_world": result.get("camera_to_world") if positioned else None}),
                None,
                created,
                body.frames[0].width,
                body.frames[0].height,
                camera.get("metadata_json") or "{}",
                json.dumps(metrics),
                "visual-roomplan-registration",
                storage_status,
            ),
        )
        audit(actor, "camera.register.visual_roomplan", "calibration", cid, home_id)
        return {
            "id": cid,
            "status": "positioned" if positioned else "needs_rescan",
            "camera_id": camera_id,
            "map_id": map_row["id"],
            "coordinate_frame": "roomplan-local",
            "camera_to_world": result.get("camera_to_world") if positioned else None,
            "confidence": result.get("confidence"),
            "tracking_state": "visual-pnp",
            "source": "visual-roomplan-registration",
            "inlier_count": result.get("inlier_count", 0),
            "match_count": result.get("match_count", 0),
            "reprojection_error_px": result.get("reprojection_error_px"),
            "intrinsics_source": result.get("intrinsics_source"),
            "diagnostics": result_diagnostics,
            "review_required": bool(body.review_only and positioned),
        }

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session")
    def start_roomplan_calibration_session(home_id: str, camera_id: str, actor: Current):
        """Start a transient caregiver-guided calibration for a fixed publisher camera."""
        home_check(actor, home_id)
        family_actor(actor)
        require_video_capture(home_id)
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        map_row = active_map_row(home_id)
        if not map_row or map_source(map_row) != "roomplan-lidar-3d" or map_dimension(map_row, "roomplan-lidar-3d") != "3d":
            raise HTTPException(409, "A native RoomPlan 3D map is required before guided camera calibration")
        metadata = json_object(map_row.get("metadata_json") or "{}")
        landmark_meta = metadata.get("visual_landmarks") if isinstance(metadata.get("visual_landmarks"), dict) else {}
        if landmark_meta.get("status") != "ready" or not landmark_meta.get("artifact_key"):
            raise HTTPException(409, "This RoomPlan scan is not ready for fixed-camera calibration yet")
        target_pool = roomplan_calibration_targets(map_row, target_count=10)
        if len(target_pool) < 4:
            raise HTTPException(409, "The RoomPlan floor does not contain enough usable geometry for guided calibration")
        targets = [{**target, "index": index} for index, target in enumerate(target_pool[:4])]
        replacement_targets = [
            {"x": target["x"], "y": target["y"], "z": target["z"]}
            for target in target_pool[4:]
        ]
        created_at = now_iso()
        expires_at = (datetime.now(timezone.utc) + ROOMPLAN_CALIBRATION_SESSION_TTL).isoformat().replace("+00:00", "Z")
        session = {
            "session_id": str(uuid.uuid4()),
            "home_id": home_id,
            "camera_id": camera_id,
            "map_id": map_row["id"],
            "status": "waiting_for_person",
            "current_target_index": 0,
            "targets": targets,
            "replacement_targets": replacement_targets,
            "frames": [],
            "anchors": [],
            "proposal": None,
            "error": None,
            "created_at": created_at,
            "expires_at": expires_at,
        }
        with roomplan_calibration_lock:
            previous = roomplan_calibration_sessions.get((home_id, camera_id))
            if previous:
                previous["frames"] = []
                previous["anchors"] = []
            roomplan_calibration_sessions[(home_id, camera_id)] = session
        audit(actor, "camera.calibration.roomplan.start", "camera", camera_id, home_id)
        return roomplan_calibration_session_view(session)

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session")
    def get_roomplan_calibration_session(home_id: str, camera_id: str, actor: Current):
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can inspect only its own calibration session")
        if actor.get("role") != "publisher":
            family_actor(actor)
        session = active_roomplan_calibration_session(home_id, camera_id)
        if not session:
            raise HTTPException(404, "No active RoomPlan calibration session")
        return roomplan_calibration_session_view(session)

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session/request-capture")
    def request_roomplan_calibration_capture(home_id: str, camera_id: str, body: RoomPlanCalibrationCaptureRequestIn, actor: Current):
        home_check(actor, home_id)
        family_actor(actor)
        session = active_roomplan_calibration_session(home_id, camera_id)
        if not session:
            raise HTTPException(404, "No active RoomPlan calibration session")
        with roomplan_calibration_lock:
            if session.get("status") == "expired":
                raise HTTPException(410, "The calibration session expired; start it again")
            if session.get("map_id") != (active_map_row(home_id) or {}).get("id"):
                session["status"] = "expired"
                session["frames"] = []
                session["anchors"] = []
                raise HTTPException(409, "The RoomPlan map changed; start calibration again")
            current_index = int(session.get("current_target_index", 0))
            if body.target_index != current_index:
                raise HTTPException(409, "Capture request does not match the current calibration target")
            if session.get("status") not in {"waiting_for_person", "capture_requested"}:
                raise HTTPException(409, "The calibration session is not waiting for a target capture")
            session["status"] = "capture_requested"
            session["error"] = None
        return roomplan_calibration_session_view(session)

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session/frames")
    def submit_roomplan_calibration_frames(home_id: str, camera_id: str, body: RoomPlanCalibrationFramesIn, actor: Current):
        home_check(actor, home_id)
        require_video_capture(home_id)
        if actor.get("role") != "publisher" or actor.get("user_id") != camera_id:
            raise HTTPException(403, "Only the paired publisher camera can submit calibration frames")
        session = active_roomplan_calibration_session(home_id, camera_id)
        if not session:
            raise HTTPException(404, "No active RoomPlan calibration session")
        should_solve = False
        frames_for_solve: list[CameraLocalizationFrameIn] = []
        anchors_for_solve: list[CameraLocalizationPersonAnchorIn] = []

        # Validate the standing point before consuming it. A calibration target
        # can be perfectly valid floor geometry while still sitting outside the
        # fixed camera's field of view. In that case keep prior good captures and
        # swap only this target for another safe floor point. If the local
        # detector is temporarily unavailable, preserve the older solve path
        # rather than rejecting a capture on infrastructure alone.
        person_visible: bool | None = None
        detect = getattr(geometry, "detect", None)
        if callable(detect):
            detector_ready = False
            try:
                for frame in body.frames:
                    detection_result = detect(
                        frame_base64=frame.frame_base64,
                        width=frame.width,
                        height=frame.height,
                        candidate_labels=["person"],
                    )
                    if not isinstance(detection_result, dict) or detection_result.get("status") != "ready":
                        continue
                    detector_ready = True
                    detections = detection_result.get("detections")
                    if not isinstance(detections, list):
                        continue
                    if any(
                        isinstance(item, dict)
                        and str(item.get("label") or "").strip().lower() == "person"
                        and isinstance(item.get("confidence"), (int, float))
                        and float(item["confidence"]) >= 0.20
                        for item in detections
                    ):
                        person_visible = True
                        break
                if person_visible is None and detector_ready:
                    person_visible = False
            except (RoomLayoutServiceUnavailable, RoomLayoutServiceError):
                person_visible = None

        with roomplan_calibration_lock:
            if session.get("status") == "expired":
                raise HTTPException(410, "The calibration session expired; start it again")
            if session.get("status") != "capture_requested":
                raise HTTPException(409, "The caregiver has not requested this calibration capture")
            current_index = int(session.get("current_target_index", 0))
            if body.target_index != current_index:
                raise HTTPException(409, "Submitted frames do not match the current calibration target")
            target = session["targets"][current_index]
            if person_visible is False:
                replacements = session.get("replacement_targets")
                if isinstance(replacements, list) and replacements:
                    replacement = replacements.pop(0)
                    session["targets"][current_index] = {
                        "index": current_index,
                        "x": float(replacement["x"]),
                        "y": float(replacement["y"]),
                        "z": float(replacement["z"]),
                    }
                    session["status"] = "waiting_for_person"
                    session["error"] = (
                        "The fixed camera could not see a person at that point, so ONE moved only this target. "
                        "Your earlier calibration points are still kept."
                    )
                    return roomplan_calibration_session_view(session)
                session["status"] = "failed"
                session["error"] = (
                    "The fixed camera cannot see enough of the safe floor targets from its current position. "
                    "Move the camera or use manual placement instead."
                )
                return roomplan_calibration_session_view(session)
            base_index = len(session["frames"])
            if base_index + len(body.frames) > 8:
                raise HTTPException(413, "Guided calibration accepts at most eight transient frames")
            session["frames"].extend(body.frames)
            session["anchors"].extend(
                CameraLocalizationPersonAnchorIn(
                    frame_index=base_index + offset,
                    x=float(target["x"]),
                    y=float(target["y"]),
                    z=float(target["z"]),
                )
                for offset, _ in enumerate(body.frames)
            )
            if current_index < len(session["targets"]) - 1:
                session["current_target_index"] = current_index + 1
                session["status"] = "waiting_for_person"
                return roomplan_calibration_session_view(session)
            session["status"] = "solving"
            should_solve = True
            frames_for_solve = list(session["frames"])
            anchors_for_solve = list(session["anchors"])

        if should_solve:
            try:
                localization = localize_roomplan_camera(
                    home_id,
                    camera_id,
                    CameraLocalizationIn(
                        frames=frames_for_solve,
                        fov_degrees=60.0,
                        review_only=True,
                        person_anchors=anchors_for_solve,
                    ),
                    actor,
                )
                with roomplan_calibration_lock:
                    session["frames"] = []
                    session["anchors"] = []
                    if localization.get("status") == "positioned" and localization.get("camera_to_world"):
                        session["proposal"] = {
                            "id": localization.get("id"),
                            "camera_id": camera_id,
                            "map_id": localization.get("map_id"),
                            "camera_to_world": localization.get("camera_to_world"),
                            "confidence": localization.get("confidence"),
                            "tracking_state": localization.get("tracking_state"),
                            "source": localization.get("source"),
                        }
                        session["status"] = "review"
                        session["error"] = None
                    else:
                        session["status"] = "failed"
                        session["error"] = "The four standing points did not produce a confident camera placement."
            except HTTPException as exc:
                with roomplan_calibration_lock:
                    session["frames"] = []
                    session["anchors"] = []
                    session["status"] = "failed"
                    session["error"] = str(exc.detail)
            except Exception:
                with roomplan_calibration_lock:
                    session["frames"] = []
                    session["anchors"] = []
                    session["status"] = "failed"
                    session["error"] = "The local camera localization service could not finish this calibration."
            return roomplan_calibration_session_view(session)
        return roomplan_calibration_session_view(session)

    @app.delete("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-calibration-session")
    def cancel_roomplan_calibration_session(home_id: str, camera_id: str, actor: Current):
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can cancel only its own calibration session")
        if actor.get("role") != "publisher":
            family_actor(actor)
        with roomplan_calibration_lock:
            session = roomplan_calibration_sessions.pop((home_id, camera_id), None)
            if session:
                session["frames"] = []
                session["anchors"] = []
        return {"camera_id": camera_id, "status": "cancelled", "raw_frames_persisted": False}

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/localization-history")
    def camera_localization_history(home_id: str, camera_id: str, actor: Current, limit: int = 30):
        """Return a bounded, image-free temporal trace of RoomPlan pose hypotheses."""
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can inspect only its own camera localization")
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")

        map_row = active_map_row(home_id)
        if not map_row or map_source(map_row) != "roomplan-lidar-3d":
            return {"camera_id": camera_id, "map_id": None, "reference": None, "attempts": []}

        bounded_limit = max(1, min(80, int(limit)))
        rows = db.many(
            "SELECT id,status,created_at,metrics_json,extrinsics_json FROM calibrations "
            "WHERE home_id=? AND camera_id=? AND map_id=? AND source='visual-roomplan-registration' "
            "ORDER BY created_at DESC LIMIT ?",
            (home_id, camera_id, map_row["id"], bounded_limit),
        )

        def center3(value: object) -> list[float] | None:
            if not isinstance(value, list) or len(value) != 3 or not all(isinstance(item, (int, float)) for item in value):
                return None
            values = [float(item) for item in value]
            return values if all(math.isfinite(item) for item in values) else None

        def matrix_center(value: object) -> list[float] | None:
            if not isinstance(value, list) or len(value) != 4:
                return None
            if not all(isinstance(row, list) and len(row) == 4 for row in value):
                return None
            try:
                return center3([value[0][3], value[1][3], value[2][3]])
            except (IndexError, TypeError):
                return None

        reference_row = db.one(
            "SELECT x,z,source,created_at,updated_at FROM camera_localization_references WHERE home_id=? AND camera_id=? AND map_id=?",
            (home_id, camera_id, map_row["id"]),
        )
        ground_truth_reference = None if reference_row is None else {
            "kind": "ground-truth-floor",
            "floor_position": [float(reference_row["x"]), float(reference_row["z"])],
            "source": reference_row.get("source") or "manual-floor-reference",
            "created_at": reference_row.get("created_at"),
            "updated_at": reference_row.get("updated_at"),
        }

        parsed_rows: list[dict] = []
        accepted_reference: dict | None = None
        for row in rows:
            metrics = json_object(row.get("metrics_json") or "{}")
            diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}
            extrinsics = json_object(row.get("extrinsics_json") or "{}")
            accepted_center = matrix_center(extrinsics.get("camera_to_world"))
            if accepted_reference is None and row.get("status") == "active" and accepted_center is not None:
                accepted_reference = {
                    "kind": "latest-positioned-registration",
                    "camera_center": accepted_center,
                    "calibration_id": row["id"],
                    "created_at": row.get("created_at"),
                    "source": "latest-positioned-registration",
                }
            parsed_rows.append({
                "row": row,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "accepted_center": accepted_center,
            })

        reference = ground_truth_reference or accepted_reference
        reference_center = center3(accepted_reference.get("camera_center")) if accepted_reference is not None else None
        ground_truth_xz = (
            tuple(float(value) for value in ground_truth_reference["floor_position"])
            if ground_truth_reference is not None
            else None
        )

        def distance_to_reference(center: list[float] | None) -> float | None:
            if center is None:
                return None
            if ground_truth_xz is not None:
                return round(math.hypot(center[0] - ground_truth_xz[0], center[2] - ground_truth_xz[1]), 4)
            if reference_center is not None:
                return round(math.dist(center, reference_center), 4)
            return None

        attempts: list[dict] = []
        for parsed in reversed(parsed_rows):
            row = parsed["row"]
            metrics = parsed["metrics"]
            diagnostics = parsed["diagnostics"]
            accepted_center = parsed["accepted_center"]
            raw_candidates = diagnostics.get("candidate_summaries") if isinstance(diagnostics.get("candidate_summaries"), list) else []
            raw_semantic_candidates = diagnostics.get("semantic_cuboid_candidates") if isinstance(diagnostics.get("semantic_cuboid_candidates"), list) else []
            candidates: list[dict] = []
            for raw_candidate in raw_candidates[:12]:
                if not isinstance(raw_candidate, dict):
                    continue
                candidate_center = center3(raw_candidate.get("camera_center"))
                if candidate_center is None:
                    continue
                candidates.append({
                    "kind": "visual-pnp",
                    "frame_index": raw_candidate.get("frame_index"),
                    "landmark_view_id": raw_candidate.get("landmark_view_id"),
                    "camera_center": candidate_center,
                    "distance_to_reference_m": distance_to_reference(candidate_center),
                    "inlier_count": int(raw_candidate.get("inlier_count") or 0),
                    "match_count": int(raw_candidate.get("match_count") or 0),
                    "reprojection_error_px": raw_candidate.get("reprojection_error_px"),
                    "consensus_frame_count": int(raw_candidate.get("consensus_frame_count") or 0),
                    "consensus_scan_view_count": int(raw_candidate.get("consensus_scan_view_count") or 0),
                    "scene_plausible": bool(raw_candidate.get("scene_plausible", True)),
                    "scene_reason": raw_candidate.get("scene_reason"),
                })
            for raw_candidate in raw_semantic_candidates[:12]:
                if not isinstance(raw_candidate, dict):
                    continue
                candidate_center = center3(raw_candidate.get("camera_center"))
                if candidate_center is None:
                    continue
                candidates.append({
                    "kind": "semantic-cuboid",
                    "frame_index": raw_candidate.get("frame_index"),
                    "landmark_view_id": None,
                    "camera_center": candidate_center,
                    "distance_to_reference_m": distance_to_reference(candidate_center),
                    "inlier_count": 0,
                    "match_count": int(raw_candidate.get("match_count") or raw_candidate.get("matched_object_count") or 0),
                    "reprojection_error_px": None,
                    "consensus_frame_count": 0,
                    "consensus_scan_view_count": 0,
                    "scene_plausible": True,
                    "scene_reason": "semantic_cuboid_initializer",
                    "selected_fov_degrees": raw_candidate.get("selected_fov_degrees"),
                    "cuboid_score": raw_candidate.get("cuboid_score"),
                    "mean_iou": raw_candidate.get("mean_iou"),
                    "minimum_iou": raw_candidate.get("minimum_iou"),
                    "matched_object_count": int(raw_candidate.get("matched_object_count") or 0),
                    "semantic_group_count": int(raw_candidate.get("semantic_group_count") or 0),
                    "labels": raw_candidate.get("labels") if isinstance(raw_candidate.get("labels"), list) else [],
                })

            selected_center = center3(diagnostics.get("selected_camera_center")) or accepted_center
            if selected_center is None and candidates:
                selected_center = candidates[0]["camera_center"]
            attempts.append({
                "id": row["id"],
                "created_at": row.get("created_at"),
                "status": "positioned" if accepted_center is not None else "needs_rescan",
                "storage_status": row.get("status"),
                "confidence": metrics.get("confidence"),
                "inlier_count": int(metrics.get("inlier_count") or 0),
                "match_count": int(metrics.get("match_count") or 0),
                "reprojection_error_px": metrics.get("reprojection_error_px"),
                "selected_camera_center": selected_center,
                "selected_distance_to_reference_m": distance_to_reference(selected_center),
                "selected_estimate_source": diagnostics.get("selected_estimate_source"),
                "candidates": candidates,
            })

        return {
            "camera_id": camera_id,
            "map_id": map_row["id"],
            "reference": reference,
            "ground_truth_reference": ground_truth_reference,
            "accepted_reference": accepted_reference,
            "distance_metric": "horizontal-floor" if ground_truth_reference is not None else "3d-to-latest-accepted",
            "attempts": attempts,
        }

    @app.put("/api/v1/homes/{home_id}/cameras/{camera_id}/localization-reference")
    def set_camera_localization_reference(home_id: str, camera_id: str, body: CameraLocalizationReferenceIn, actor: Current):
        """Persist a human-verified floor location for localization evaluation."""
        home_check(actor, home_id)
        publisher_block(actor)
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        map_row = active_map_row(home_id)
        if not map_row or map_source(map_row) != "roomplan-lidar-3d" or map_dimension(map_row, "roomplan-lidar-3d") != "3d":
            raise HTTPException(409, "A native RoomPlan 3D map is required before setting a camera reference")
        map_data = json_object(map_row.get("map_json") or "{}")
        geometry = map_data.get("geometry") if isinstance(map_data.get("geometry"), dict) else {}
        raw_zones = geometry.get("room_zones") if isinstance(geometry.get("room_zones"), list) else []

        def polygon_distance(px: float, pz: float, polygon: list[tuple[float, float]]) -> float:
            inside = False
            previous = polygon[-1]
            minimum = float("inf")
            for current in polygon:
                ax, az = previous
                bx, bz = current
                dx, dz = bx - ax, bz - az
                length_sq = dx * dx + dz * dz
                if length_sq <= 1e-12:
                    distance = math.hypot(px - ax, pz - az)
                else:
                    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length_sq))
                    distance = math.hypot(px - (ax + t * dx), pz - (az + t * dz))
                minimum = min(minimum, distance)
                if (az > pz) != (bz > pz):
                    crossing_x = (bx - ax) * (pz - az) / (bz - az) + ax
                    if px < crossing_x:
                        inside = not inside
                previous = current
            return 0.0 if inside else minimum

        inside_or_near = False
        for zone in raw_zones:
            polygon = zone.get("polygon") if isinstance(zone, dict) else None
            if not isinstance(polygon, list):
                continue
            points = [
                (float(point["x"]), float(point["z"]))
                for point in polygon
                if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("z"), (int, float))
            ]
            if len(points) >= 3 and polygon_distance(body.x, body.z, points) <= 0.40:
                inside_or_near = True
                break
        if raw_zones and not inside_or_near:
            raise HTTPException(422, "Camera reference must lie inside or immediately beside the active RoomPlan room")
        existing = db.one(
            "SELECT created_at FROM camera_localization_references WHERE home_id=? AND camera_id=? AND map_id=?",
            (home_id, camera_id, map_row["id"]),
        )
        updated = now_iso()
        created = existing.get("created_at") if existing else updated
        db.execute(
            "INSERT INTO camera_localization_references(home_id,camera_id,map_id,x,z,source,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(home_id,camera_id,map_id) DO UPDATE SET x=excluded.x,z=excluded.z,source=excluded.source,updated_at=excluded.updated_at",
            (home_id, camera_id, map_row["id"], body.x, body.z, body.source, created, updated),
        )
        audit(actor, "camera.localization_reference.set", "camera", camera_id, home_id)
        return {
            "camera_id": camera_id,
            "map_id": map_row["id"],
            "kind": "ground-truth-floor",
            "floor_position": [body.x, body.z],
            "source": body.source,
            "created_at": created,
            "updated_at": updated,
        }

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-readiness")
    def camera_roomplan_readiness(home_id: str, camera_id: str, actor: Current):
        """Expose only the RoomPlan state a camera needs for automatic localization."""
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can inspect only its own camera")
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        row = active_map_row(home_id)
        if not row or map_source(row) != "roomplan-lidar-3d" or map_dimension(row, "roomplan-lidar-3d") != "3d":
            return {
                "camera_id": camera_id,
                "map_id": None,
                "source": None,
                "dimension": None,
                "visual_landmarks_ready": False,
                "ready": False,
            }
        metadata = json_object(row.get("metadata_json") or "{}")
        landmark_meta = metadata.get("visual_landmarks") if isinstance(metadata.get("visual_landmarks"), dict) else {}
        artifact_key = landmark_meta.get("artifact_key") if isinstance(landmark_meta, dict) else None
        landmarks_ready = landmark_meta.get("status") == "ready" and isinstance(artifact_key, str) and bool(artifact_key)
        return {
            "camera_id": camera_id,
            "map_id": row["id"],
            "source": "roomplan-lidar-3d",
            "dimension": "3d",
            "visual_landmarks_ready": landmarks_ready,
            "ready": landmarks_ready,
        }

    @app.put(
        "/api/v1/homes/{home_id}/maps/{map_id}/usdz",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "model/vnd.usdz+zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def roomplan_usdz_upload(home_id: str, map_id: str, request: Request, actor: Current):
        """Attach a bounded USDZ export to an already validated 3D map."""

        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row:
            raise HTTPException(404, "Map not found")
        if map_source(row) != "roomplan-lidar-3d" or map_dimension(row, map_source(row)) != "3d":
            raise HTTPException(422, "USDZ attachments require a validated native RoomPlan 3D map")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in ROOMPLAN_USDZ_CONTENT_TYPES:
            raise HTTPException(415, "USDZ upload must use a supported USDZ content type")
        content_length = request.headers.get("content-length")
        try:
            if content_length is not None and int(content_length) > ROOMPLAN_USDZ_MAX_BYTES:
                raise HTTPException(413, "USDZ upload is too large")
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        payload = await request.body()
        if not payload or len(payload) > ROOMPLAN_USDZ_MAX_BYTES:
            raise HTTPException(413, "USDZ upload is empty or too large")
        try:
            validate_roomplan_usdz(payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        digest = hashlib.sha256(payload).hexdigest()
        key = f"maps/{home_id}/{map_id}.usdz"
        store.put_bytes(key, payload)
        metadata = json_object(row.get("metadata_json") or "{}")
        metadata["usdz"] = roomplan_usdz_metadata(
            sha256=digest,
            byte_count=len(payload),
            content_type="model/vnd.usdz+zip",
            download_path=f"/api/v1/homes/{home_id}/maps/{map_id}/usdz",
        )
        db.execute("UPDATE room_maps SET usdz_artifact_key=?, metadata_json=? WHERE id=? AND home_id=?", (key, json.dumps(metadata), map_id, home_id))
        audit(actor, "map.usdz.upload", "room_map", map_id, home_id)
        return {"map_id": map_id, "source": "roomplan-lidar-3d", "dimension": "3d", "usdz": metadata["usdz"]}

    def roomplan_usdz_response(home_id: str, map_id: str) -> Response:
        row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row or map_source(row) not in {"roomplan-lidar-3d", "arkit-video-3d"} or map_dimension(row, map_source(row)) != "3d":
            raise HTTPException(404, "USDZ model not found")
        key = row.get("usdz_artifact_key")
        if not key:
            raise HTTPException(404, "USDZ model not found")
        try:
            payload = store.get_bytes(key)
        except (OSError, ValueError):
            raise HTTPException(404, "USDZ model not found") from None
        metadata = json_object(row.get("metadata_json") or "{}").get("usdz")
        digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        headers = {
            "Content-Disposition": f'inline; filename="one-room-{map_id}.usdz"',
            "Cache-Control": "private, max-age=0",
            "Content-Length": str(len(payload)),
            "X-Content-Type-Options": "nosniff",
        }
        if isinstance(digest, str):
            headers["ETag"] = digest
        return Response(content=payload, media_type="model/vnd.usdz+zip", headers=headers)

    @app.get(
        "/api/v1/homes/{home_id}/maps/{map_id}/usdz",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "model/vnd.usdz+zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            }
        },
    )
    def roomplan_usdz_download(home_id: str, map_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return roomplan_usdz_response(home_id, map_id)

    def map_generation_camera(home_id: str, camera_id: str, actor: dict) -> dict:
        home_check(actor, home_id)
        camera = db.one("SELECT * FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        if actor["role"] == "publisher":
            if actor["user_id"] != camera_id:
                raise HTTPException(403, "Publisher can only access its paired camera")
        elif actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Caregiver or paired camera permission required")
        return camera

    def map_generation_view(row: dict) -> dict:
        status_value = row["status"]
        progress = {
            "collecting": 10,
            "processing": 72,
            "ready": 100,
            "needs_rescan": 100,
            "unavailable": 100,
            "failed": 100,
        }.get(status_value, 0)
        error_message = row["error_message"]
        return {
            "id": row["id"],
            "job_id": row["id"],
            "home_id": row["home_id"],
            "camera_id": row["camera_id"],
            "room_id": row["room_id"],
            "room_label": row.get("room_label") or "Room",
            "orientation": row.get("orientation") or "portrait",
            "status": status_value,
            "progress": progress,
            "source": "camera-cv-2d",
            "dimension": "2d",
            "metric_scale_known": False,
            "geometry_status": status_value,
            "frame_count": row["frame_count"],
            "resolution_width": row["resolution_width"],
            "resolution_height": row["resolution_height"],
            "map_id": row["map_id"],
            "error_code": row["error_code"],
            "error_message": error_message,
            "error": error_message,
            "metrics": json_object(row.get("metrics_json") or "{}"),
            "model_version": row["model_version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    def finish_map_generation(
        job_id: str,
        status_value: str,
        *,
        map_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        metrics: dict | None = None,
        model_version: str | None = None,
    ) -> None:
        completed = now_iso()
        db.execute(
            """UPDATE camera_map_generation_jobs
                  SET status=?, map_id=?, error_code=?, error_message=?,
                      metrics_json=?, model_version=?, updated_at=?, completed_at=?
                WHERE id=?""",
            (status_value, map_id, error_code, error_message, json.dumps(metrics or {}), model_version, completed, completed, job_id),
        )

    def process_map_generation(
        job_id: str,
        home_id: str,
        camera_id: str,
        room_id: str | None,
        room_label: str,
        orientation: str,
        resolution_width: int,
        resolution_height: int,
        frames: list[dict],
    ) -> None:
        try:
            payload = app.state.geometry_service.infer(
                camera_id=camera_id,
                resolution_width=resolution_width,
                resolution_height=resolution_height,
                room_label=room_label,
                orientation=orientation,
                frames=frames,
            )
            result = RoomLayoutResult.model_validate(payload)
            if result.status == "unavailable":
                finish_map_generation(job_id, "unavailable", error_code="geometry_service_unavailable", error_message="The local geometry service is unavailable.", model_version=result.model_version)
                return
            if result.status == "failed":
                finish_map_generation(job_id, "failed", error_code="geometry_service_failed", error_message="The local geometry service could not process this sweep.", model_version=result.model_version)
                return
            if result.status == "needs_rescan":
                finish_map_generation(job_id, "needs_rescan", error_code="geometry_not_confident", error_message="The sweep did not contain sufficiently stable room geometry.", model_version=result.model_version)
                return
            if result.geometry is None:
                raise RoomLayoutServiceError("missing_geometry")

            geometry = result.geometry
            metrics = geometry.metrics.model_dump()
            if (
                metrics["confidence"] < settings.geometry_min_confidence
                or metrics["reprojection_error_px"] > settings.geometry_max_reprojection_error_px
                or metrics["homography_inlier_ratio"] < settings.geometry_min_homography_inlier_ratio
            ):
                finish_map_generation(job_id, "needs_rescan", error_code="geometry_not_confident", error_message="The sweep did not meet the image-space confidence thresholds.", metrics=metrics, model_version=result.model_version)
                return

            map_data = {
                "schema_version": "camera-room-2d.v1",
                "dimension": "2d",
                "source": "camera-cv-2d",
                "coordinate_frame": geometry.coordinate_frame,
                "polygons": [item.model_dump() for item in geometry.polygons],
                "walls": [item.model_dump() for item in geometry.walls],
                "furniture": [item.model_dump() for item in geometry.furniture],
                "openings": [item.model_dump() for item in geometry.openings],
                "camera_pose": geometry.camera_pose.model_dump(),
                "intrinsics": geometry.intrinsics,
                "confidence": metrics["confidence"],
                "metrics": metrics,
                "model_version": result.model_version,
            }
            map_data["geometry"] = {
                "coordinate_frame": geometry.coordinate_frame,
                "polygons": map_data["polygons"],
                "walls": map_data["walls"],
                "furniture": map_data["furniture"],
                "openings": map_data["openings"],
                "camera_pose": map_data["camera_pose"],
                "intrinsics": geometry.intrinsics,
                "metrics": metrics,
            }
            metadata = {
                "camera_id": camera_id,
                "model_version": result.model_version,
                "geometry_status": "ready",
                "metric_scale_known": False,
                "metrics": metrics,
            }
            system_actor = {"home_id": home_id, "user_id": None}
            map_result = create_map(home_id, room_id, geometry.coordinate_frame, map_data, "camera-cv-2d", True, "image-space", metadata, system_actor, dimension="2d")

            calibration_id = str(uuid.uuid4())
            created = now_iso()
            db.execute(
                "UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='superseded by camera geometry map' WHERE home_id=? AND camera_id=? AND status='active' AND source NOT IN ('auto-roomplan-registration','visual-roomplan-registration')",
                (created, home_id, camera_id),
            )
            db.execute(
                "INSERT INTO calibrations(id,home_id,camera_id,map_id,intrinsics_json,extrinsics_json,accuracy_m,created_at,resolution_width,resolution_height,camera_metadata_json,metrics_json,source,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (calibration_id, home_id, camera_id, map_result["id"], json.dumps(geometry.intrinsics), json.dumps(geometry.camera_pose.model_dump()), None, created, resolution_width, resolution_height, json.dumps({"coordinate_frame": geometry.coordinate_frame, "metric_scale_known": False, "model_version": result.model_version}), json.dumps(metrics), "camera-cv", "active"),
            )
            finish_map_generation(job_id, "ready", map_id=map_result["id"], metrics=metrics, model_version=result.model_version)
            audit(system_actor, "map.generation.ready", "camera_map_generation_job", job_id, home_id)
        except RoomLayoutServiceUnavailable:
            finish_map_generation(job_id, "unavailable", error_code="geometry_service_unavailable", error_message="The local geometry service is unavailable.")
        except RoomLayoutServiceError:
            finish_map_generation(job_id, "failed", error_code="geometry_service_failed", error_message="The local geometry service returned an unusable result.")
        except ValidationError:
            finish_map_generation(job_id, "failed", error_code="invalid_geometry_response", error_message="The local geometry service returned an invalid geometry contract.")
        except Exception:
            # Do not persist exception text: it could contain service details
            # or accidental frame data. The job remains inspectable and safe
            # to retry with a fresh sweep.
            finish_map_generation(job_id, "failed", error_code="generation_failed", error_message="Camera map generation failed unexpectedly.")

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation")
    def map_generation_start(home_id: str, camera_id: str, body: CameraMapGenerationStartIn, actor: Current):
        camera = map_generation_camera(home_id, camera_id, actor)
        require_video_capture(home_id)
        active = db.one(
            "SELECT * FROM camera_map_generation_jobs WHERE home_id=? AND camera_id=? AND status IN ('collecting','processing') ORDER BY created_at DESC LIMIT 1",
            (home_id, camera_id),
        )
        if active:
            return map_generation_view(active)
        job_id = str(uuid.uuid4())
        created = now_iso()
        db.execute(
            "INSERT INTO camera_map_generation_jobs(id,home_id,camera_id,room_id,room_label,orientation,status,frame_count,resolution_width,resolution_height,map_id,error_code,error_message,metrics_json,model_version,created_at,updated_at,completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, home_id, camera["id"], body.room_id, body.room_label or camera["name"], body.orientation, "collecting", 0, body.resolution_width, body.resolution_height, None, None, None, "{}", None, created, created, None),
        )
        audit(actor, "map.generation.start", "camera_map_generation_job", job_id, home_id)
        return map_generation_view(db.one("SELECT * FROM camera_map_generation_jobs WHERE id=?", (job_id,)))

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation")
    def map_generation_status(home_id: str, camera_id: str, actor: Current):
        map_generation_camera(home_id, camera_id, actor)
        row = db.one("SELECT * FROM camera_map_generation_jobs WHERE home_id=? AND camera_id=? ORDER BY created_at DESC LIMIT 1", (home_id, camera_id))
        if not row:
            raise HTTPException(404, "No camera map generation has been started")
        return map_generation_view(row)

    @app.post("/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}/frames", status_code=202)
    def map_generation_frames(home_id: str, camera_id: str, job_id: str, body: CameraMapFramesIn, background_tasks: BackgroundTasks, actor: Current):
        map_generation_camera(home_id, camera_id, actor)
        if actor["role"] != "publisher":
            raise HTTPException(403, "Only the paired camera can submit room sweep frames")
        require_video_capture(home_id)
        job = db.one("SELECT * FROM camera_map_generation_jobs WHERE id=? AND home_id=? AND camera_id=?", (job_id, home_id, camera_id))
        if not job:
            raise HTTPException(404, "Camera map generation job not found")
        if job["status"] != "collecting":
            raise HTTPException(409, "Camera map generation job is no longer collecting frames")
        total_bytes = 0
        for frame in body.frames:
            if frame.width != job["resolution_width"] or frame.height != job["resolution_height"]:
                raise HTTPException(422, "All sweep frames must match the job resolution")
            try:
                decoded = base64.b64decode(frame.frame_base64, validate=True)
            except ValueError as exc:
                raise HTTPException(422, "frame_base64 must be valid base64") from exc
            if not decoded or len(decoded) > MAP_FRAME_MAX_BYTES:
                raise HTTPException(413, "A sweep frame is empty or exceeds the 3 MB in-memory limit")
            total_bytes += len(decoded)
            del decoded
        if total_bytes > MAP_BATCH_MAX_BYTES:
            raise HTTPException(413, "The room sweep exceeds the bounded in-memory batch limit")
        frames = [frame.model_dump(mode="json") for frame in body.frames]
        db.execute("UPDATE camera_map_generation_jobs SET status='processing', frame_count=?, updated_at=? WHERE id=? AND status='collecting'", (len(frames), now_iso(), job_id))
        background_tasks.add_task(process_map_generation, job_id, home_id, camera_id, job["room_id"], job.get("room_label") or "Room", job.get("orientation") or "portrait", job["resolution_width"], job["resolution_height"], frames)
        return map_generation_view(db.one("SELECT * FROM camera_map_generation_jobs WHERE id=?", (job_id,)))

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/map-generation/{job_id}")
    def map_generation_job(home_id: str, camera_id: str, job_id: str, actor: Current):
        map_generation_camera(home_id, camera_id, actor)
        row = db.one("SELECT * FROM camera_map_generation_jobs WHERE id=? AND home_id=? AND camera_id=?", (job_id, home_id, camera_id))
        if not row:
            raise HTTPException(404, "Camera map generation job not found")
        return map_generation_view(row)

    @app.get("/api/v1/homes/{home_id}/maps")
    def maps(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        rows = db.many("SELECT * FROM room_maps WHERE home_id=? ORDER BY revision DESC, created_at DESC", (home_id,))
        return {"data": [map_view(row) for row in rows]}

    @app.get("/api/v1/homes/{home_id}/maps/current")
    def current_map(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        row = active_map_row(home_id)
        if not row:
            raise HTTPException(404, "No room map has been uploaded")
        return map_view(row)

    @app.get("/api/v1/homes/{home_id}/maps/{map_id}")
    def map_detail(home_id: str, map_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row:
            raise HTTPException(404, "Map not found")
        return map_view(row)

    @app.post("/api/v1/homes/{home_id}/maps/{map_id}/scale")
    def measure_map_scale(home_id: str, map_id: str, body: MapScaleReferenceIn, actor: Current):
        """Persist a scale derived from a caregiver-measured reference.

        RGB camera geometry remains image-space and approximate. This endpoint
        only adds a measured conversion after a person supplies the physical
        length of two points visible on that map; it never invents meters from
        the camera sweep alone.
        """
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row:
            raise HTTPException(404, "Map not found")
        if map_source(row) != "camera-cv-2d":
            raise HTTPException(422, "Only a camera-derived 2D map accepts a reference scale")
        if not map_uses_real_geometry_model(row):
            raise HTTPException(422, "A fresh real-model camera sweep is required before measuring this map")
        reference_distance = math.hypot(body.end.x - body.start.x, body.end.y - body.start.y)
        if reference_distance < 0.005:
            raise HTTPException(422, "The two reference points are too close together")
        try:
            map_data = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            map_data = {}
        if not isinstance(map_data, dict):
            raise HTTPException(422, "Map geometry is not available for scale measurement")
        measured_at = now_iso()
        scale = {
            "status": "measured_reference",
            "method": "caregiver_reference",
            "meters_per_normalized_unit": round(body.length_m / reference_distance, 6),
            "reference_length_m": round(body.length_m, 4),
            "reference_label": body.label,
            "reference_points": {
                "start": body.start.model_dump(),
                "end": body.end.model_dump(),
            },
            "measured_at": measured_at,
        }
        map_data["scale"] = scale
        geometry = map_data.get("geometry")
        if isinstance(geometry, dict):
            geometry["scale"] = scale
        metadata = json_object(row.get("metadata_json") or "{}")
        metadata["scale"] = scale
        artifact_key = row.get("artifact_key")
        if isinstance(artifact_key, str) and artifact_key:
            store.put_json(artifact_key, map_data)
        db.execute("UPDATE room_maps SET map_json=?, metadata_json=? WHERE id=? AND home_id=?", (json.dumps(map_data), json.dumps(metadata), map_id, home_id))
        audit(actor, "map.scale.measure", "room_map", map_id, home_id)
        return map_view(db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id)))

    def scene_response(home_id: str, row: dict | None) -> dict:
        if not row:
            return {"sceneId": None, "version": 0, "dimension": "2d", "source": "legacy-2d", "provenance": "legacy-2d", "approximate": True, "metricScaleKnown": False, "geometryStatus": "empty", "rescanRequired": True, "zones": [], "polygons": [], "walls": [], "camera": None, "cameraRegistration": None, "cameraRegistrations": [], "canonicalGeometry": None, "geometry": {"polygons": [], "walls": [], "furniture": [], "openings": [], "zones": []}, "usdz": None}
        try:
            map_data = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            map_data = {}
        map_data = map_data if isinstance(map_data, dict) else {}
        view = map_view(row)
        stored_geometry = map_data.get("geometry") if isinstance(map_data.get("geometry"), dict) else map_data
        zones = map_data.get("zones", []) if isinstance(map_data.get("zones"), list) else []
        polygons = stored_geometry.get("polygons", stored_geometry.get("rooms", [])) if isinstance(stored_geometry.get("polygons", stored_geometry.get("rooms", [])), list) else []
        walls = stored_geometry.get("walls", []) if isinstance(stored_geometry.get("walls", []), list) else []
        furniture = stored_geometry.get("furniture", []) if isinstance(stored_geometry.get("furniture", []), list) else []
        openings = stored_geometry.get("openings", []) if isinstance(stored_geometry.get("openings", []), list) else []
        camera = stored_geometry.get("camera_pose") if view["dimension"] == "2d" and isinstance(stored_geometry.get("camera_pose"), dict) else None
        camera_registration = roomplan_camera_registration_view(home_id, row)
        camera_registrations = roomplan_camera_registration_views(home_id, row)
        if view["dimension"] == "2d" and view["rescan_required"]:
            # A rejected camera sweep must never keep rendering stale or
            # misleading image-space geometry. Preserve the revision for audit
            # and retry UX, while exposing an empty scene until a better sweep
            # or a native RoomPlan scan replaces it.
            polygons = []
            walls = []
            furniture = []
            openings = []
            camera = None
        geometry = stored_geometry if view["dimension"] == "3d" else {"polygons": polygons, "walls": walls, "furniture": furniture, "openings": openings, "camera_pose": camera, "zones": zones}
        canonical_geometry = map_data.get("normalized_scan") if view["dimension"] == "3d" and isinstance(map_data.get("normalized_scan"), dict) else None
        return {"sceneId": row["id"], "version": row["revision"], "dimension": view["dimension"], "source": view["source"], "provenance": view["provenance"], "approximate": view["approximate"], "metricScaleKnown": view["metric_scale_known"], "scale": view["scale"], "geometryStatus": view["geometry_status"], "rescanRequired": view["rescan_required"], "confidence": map_data.get("confidence"), "modelVersion": view["model_version"], "zones": zones, "polygons": polygons, "walls": walls, "camera": camera, "cameraRegistration": camera_registration, "cameraRegistrations": camera_registrations, "canonicalGeometry": canonical_geometry, "geometry": geometry, "mapId": row["id"], "coordinateFrame": row["coordinate_frame"], "usdz": view["usdz"]}

    @app.get("/api/v1/homes/{home_id}/scene")
    def scene(home_id: str, actor: Current):
        """Return the compact scene contract consumed by the dashboard map."""
        home_check(actor, home_id); publisher_block(actor)
        return scene_response(home_id, active_map_row(home_id))

    @app.get("/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-placement-preview")
    def roomplan_camera_placement_preview(home_id: str, camera_id: str, actor: Current):
        """Expose only the active RoomPlan scene needed to review this publisher camera's placement."""
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can preview only its own camera placement")
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        row = active_map_row(home_id)
        if not row or map_source(row) != "roomplan-lidar-3d" or map_dimension(row, "roomplan-lidar-3d") != "3d":
            raise HTTPException(409, "A native RoomPlan 3D map is required before camera placement can be reviewed")
        return scene_response(home_id, row)

    @app.get(
        "/api/v1/homes/{home_id}/cameras/{camera_id}/roomplan-placement-preview/usdz",
        response_class=Response,
    )
    def roomplan_camera_placement_preview_usdz(home_id: str, camera_id: str, actor: Current):
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != camera_id:
            raise HTTPException(403, "A publisher can preview only its own camera placement")
        camera = db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        row = active_map_row(home_id)
        if not row or map_source(row) != "roomplan-lidar-3d" or map_dimension(row, "roomplan-lidar-3d") != "3d":
            raise HTTPException(409, "A native RoomPlan 3D map is required before camera placement can be reviewed")
        return roomplan_usdz_response(home_id, row["id"])

    @app.post("/api/v1/homes/{home_id}/camera-registrations/roomplan")
    def roomplan_camera_registration(home_id: str, body: RoomPlanCameraRegistrationIn, actor: Current):
        home_check(actor, home_id)
        if actor.get("role") == "publisher" and actor.get("user_id") != body.camera_id:
            raise HTTPException(403, "A publisher can register only its own camera")
        camera = db.one("SELECT * FROM cameras WHERE id=? AND home_id=? AND enabled=1", (body.camera_id, home_id))
        if not camera:
            raise HTTPException(404, "Camera not found or disabled")
        map_row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (body.map_id, home_id))
        if not map_row:
            raise HTTPException(404, "Map not found")
        source = map_source(map_row)
        if source != "roomplan-lidar-3d" or map_dimension(map_row, source) != "3d" or map_row.get("coordinate_frame") != "roomplan-local":
            raise HTTPException(422, "Camera registration requires an active native RoomPlan 3D map")
        active_row = active_map_row(home_id)
        if not active_row or active_row["id"] != body.map_id:
            raise HTTPException(409, "Camera registration requires the active map revision")

        created = now_iso()
        metrics = {"confidence": body.confidence, "tracking_state": body.tracking_state}
        status_value = "active"
        response_status = "positioned"
        if body.tracking_state != "normal" or (body.confidence is not None and body.confidence < 0.65):
            status_value = "needs_rescan"
            response_status = "needs_rescan"
        cid = str(uuid.uuid4())
        db.execute(
            "UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='superseded by RoomPlan camera registration' WHERE home_id=? AND camera_id=? AND status IN ('active','needs_review')",
            (created, home_id, body.camera_id),
        )
        db.execute(
            "INSERT INTO calibrations(id,home_id,camera_id,map_id,intrinsics_json,extrinsics_json,accuracy_m,created_at,resolution_width,resolution_height,camera_metadata_json,metrics_json,source,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                home_id,
                body.camera_id,
                body.map_id,
                "{}",
                json.dumps({"camera_to_world": body.camera_to_world}),
                None,
                created,
                camera.get("resolution_width"),
                camera.get("resolution_height"),
                camera.get("metadata_json") or "{}",
                json.dumps(metrics),
                "auto-roomplan-registration",
                status_value,
            ),
        )
        audit(actor, "camera.register.roomplan", "calibration", cid, home_id)
        return {
            "id": cid,
            "status": response_status,
            "camera_id": body.camera_id,
            "map_id": body.map_id,
            "coordinate_frame": "roomplan-local",
            "camera_to_world": body.camera_to_world if status_value == "active" else None,
            "confidence": body.confidence,
            "tracking_state": body.tracking_state,
            "source": "auto-roomplan-registration",
        }

    @app.post("/api/v1/homes/{home_id}/calibrations")
    def calibration(home_id: str, body: CalibrationIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        if not db.one("SELECT id FROM cameras WHERE id=? AND home_id=?", (body.camera_id, home_id)): raise HTTPException(404, "Camera not found")
        map_row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (body.map_id, home_id))
        if not map_row: raise HTTPException(404, "Map not found")
        if map_source(map_row) == "camera-cv-2d" and body.accuracy_m is not None:
            raise HTTPException(422, "RGB camera calibration cannot report accuracy_m")
        cid = str(uuid.uuid4()); created = now_iso()
        db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='superseded by new calibration' WHERE home_id=? AND camera_id=? AND status='active'", (created, home_id, body.camera_id))
        db.execute("INSERT INTO calibrations(id,home_id,camera_id,map_id,intrinsics_json,extrinsics_json,accuracy_m,created_at,resolution_width,resolution_height,camera_metadata_json,metrics_json,source,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid, home_id, body.camera_id, body.map_id, json.dumps(body.intrinsics), json.dumps(body.extrinsics), body.accuracy_m, created, body.resolution_width, body.resolution_height, json.dumps(body.camera_metadata), "{}", body.source, "active"))
        return {"id": cid, **body.model_dump(), "status": "active", "invalidated_previous": True}

    @app.get("/api/v1/homes/{home_id}/calibrations")
    def calibrations(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        rows = db.many("SELECT * FROM calibrations WHERE home_id=? ORDER BY created_at DESC", (home_id,))
        return {"data": [{**row, "intrinsics": json.loads(row.pop("intrinsics_json")), "extrinsics": json.loads(row.pop("extrinsics_json")), "camera_metadata": json.loads(row.pop("camera_metadata_json") or "{}"), "metrics": json.loads(row.pop("metrics_json") or "{}")} for row in rows]}

    @app.post("/api/v1/homes/{home_id}/objects")
    def object_create(home_id: str, body: ObjectIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); oid = str(uuid.uuid4()); db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?)", (oid, home_id, body.label, body.display_name, 1, now_iso())); return {"id": oid, **body.model_dump(), "enabled": True}

    def point_in_zone(x: float, z: float, polygon: list[dict]) -> bool:
        points = [
            (float(item["x"]), float(item["z"]))
            for item in polygon
            if isinstance(item, dict) and isinstance(item.get("x"), (int, float)) and isinstance(item.get("z"), (int, float))
        ]
        if len(points) < 3:
            return False
        inside = False
        previous = points[-1]
        for current in points:
            x1, z1 = previous
            x2, z2 = current
            if ((z1 > z) != (z2 > z)) and x < (x2 - x1) * (z - z1) / ((z2 - z1) or 1e-12) + x1:
                inside = not inside
            previous = current
        return inside

    def roomplan_zones(map_id: str | None, home_id: str) -> list[dict]:
        if not map_id:
            return []
        row = db.one("SELECT map_json FROM room_maps WHERE id=? AND home_id=?", (map_id, home_id))
        if not row:
            return []
        try:
            payload = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            return []
        geometry_payload = payload.get("geometry") if isinstance(payload, dict) and isinstance(payload.get("geometry"), dict) else {}
        zones = geometry_payload.get("room_zones") if isinstance(geometry_payload.get("room_zones"), list) else []
        return [zone for zone in zones if isinstance(zone, dict)]

    def roomplan_zone_for_point(map_id: str | None, home_id: str, x: float | None, z: float | None) -> dict | None:
        if x is None or z is None:
            return None
        return next((zone for zone in roomplan_zones(map_id, home_id) if point_in_zone(x, z, zone.get("polygon", []))), None)

    def object_view(row: dict) -> dict:
        observed_at = row.get("observed_at")
        x, y, z = row.get("x"), row.get("y"), row.get("z")
        observation_id = row.get("observation_id")
        event = db.one("SELECT id FROM events WHERE home_id=? AND evidence_json LIKE ? ORDER BY last_seen_at DESC LIMIT 1", (row["home_id"], f"%{observation_id}%")) if observation_id else None
        zone = roomplan_zone_for_point(row.get("map_id"), row["home_id"], x, z)
        return {
            "id": row["id"],
            "label": row["display_name"] or row["label"],
            "icon": (row["label"][:1] or "?").upper(),
            "status": "seen" if observed_at else "unknown",
            "lastSeenAt": observed_at,
            "point": {"x": x, "y": y} if x is not None and y is not None else None,
            "worldPoint": {"x": x, "y": y, "z": z} if x is not None and y is not None and z is not None else None,
            "mapId": row.get("map_id"),
            "cameraId": row.get("camera_id"),
            "confidenceRadiusM": row.get("uncertainty_m") if row.get("uncertainty_m") is not None else 0.0,
            "confidence": row.get("confidence") if row.get("confidence") is not None else 0.0,
            "zone": ({"id": zone.get("id"), "name": zone.get("label") or zone.get("id"), "confidence": zone.get("confidence", 1.0)} if zone else None),
            "sourceEventId": event["id"] if event else None,
            "observation": {"id": observation_id, "x": x, "y": y, "z": z, "map_id": row.get("map_id"), "camera_id": row.get("camera_id"), "detector_version": row.get("detector_version")} if observation_id else None,
        }

    def object_rows(home_id: str) -> list[dict]:
        rows = db.many("""
            SELECT o.*, latest.id observation_id, latest.camera_id, latest.map_id,
                   latest.x, latest.y, latest.z, latest.uncertainty_m,
                   latest.confidence, latest.detector_version, latest.observed_at
            FROM objects o
            LEFT JOIN observations latest ON latest.id = (
                SELECT ob.id FROM observations ob
                WHERE ob.home_id=o.home_id AND ob.object_id=o.id
                ORDER BY ob.observed_at DESC LIMIT 1
            )
            WHERE o.home_id=? AND o.enabled=1
            ORDER BY COALESCE(latest.observed_at, o.created_at) DESC
        """, (home_id,))
        return [object_view(row) for row in rows]

    def vision_candidate_labels(home_id: str, requested: list[str]) -> list[str]:
        labels = [item.strip().lower() for item in requested if item.strip()]
        if not labels:
            labels = [str(row["label"]).strip().lower() for row in db.many("SELECT label FROM objects WHERE home_id=? AND enabled=1 ORDER BY created_at LIMIT 20", (home_id,))]
            labels = ["person", *labels]
        if len(labels) == 1 and labels[0] == "person":
            labels.extend(["keys", "glasses", "mobile phone", "remote control", "cup", "bottle", "book", "medication box", "cane", "walker"])
        return list(dict.fromkeys(labels))[:20]

    def vision_calibration(home_id: str, camera_id: str, width: int, height: int) -> tuple[Calibration | None, str | None]:
        row = db.one(
            "SELECT * FROM calibrations WHERE home_id=? AND camera_id=? AND status='active' AND source IN ('visual-roomplan-registration','auto-roomplan-registration') ORDER BY created_at DESC LIMIT 1",
            (home_id, camera_id),
        )
        if not row:
            return None, None
        extrinsics = json_object(row.get("extrinsics_json") or "{}")
        matrix = extrinsics.get("camera_to_world")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(item, list) or len(item) != 4 for item in matrix):
            return None, row.get("map_id")
        map_row = db.one("SELECT * FROM room_maps WHERE id=? AND home_id=?", (row.get("map_id"), home_id))
        if map_row and map_source(map_row) == "roomplan-lidar-3d":
            scene_valid, _ = roomplan_pose_scene_validation(map_row, matrix)
            if not scene_valid:
                return None, row.get("map_id")
        intrinsics_payload = json_object(row.get("intrinsics_json") or "{}")
        camera_matrix = intrinsics_payload.get("matrix")
        if isinstance(camera_matrix, list) and len(camera_matrix) == 3 and all(isinstance(item, list) and len(item) == 3 for item in camera_matrix):
            fx = float(camera_matrix[0][0]); fy = float(camera_matrix[1][1]); cx = float(camera_matrix[0][2]); cy = float(camera_matrix[1][2])
            source_width = row.get("resolution_width") or width
            source_height = row.get("resolution_height") or height
            fx *= width / source_width; cx *= width / source_width
            fy *= height / source_height; cy *= height / source_height
        else:
            fov = float(intrinsics_payload.get("fov_degrees") or 60.0)
            fx = 0.5 * width / math.tan(math.radians(fov) / 2.0)
            fy = fx; cx = width / 2.0; cy = height / 2.0
        zones = roomplan_zones(row.get("map_id"), home_id)
        camera = db.one("SELECT room_id FROM cameras WHERE id=? AND home_id=?", (camera_id, home_id))
        preferred_zone = next((zone for zone in zones if camera and zone.get("id") == camera.get("room_id")), None)
        floor_zone = preferred_zone or (zones[0] if zones else None)
        floor_y = float(floor_zone.get("floor_y")) if floor_zone and isinstance(floor_zone.get("floor_y"), (int, float)) else None
        metrics = json_object(row.get("metrics_json") or "{}")
        confidence = float(metrics.get("confidence")) if isinstance(metrics.get("confidence"), (int, float)) else 0.7
        accuracy = max(0.15, 0.8 - confidence * 0.55)
        return Calibration(
            fx,
            fy,
            cx,
            cy,
            tuple(tuple(float(value) for value in matrix_row) for matrix_row in matrix),
            accuracy,
            floor_y,
        ), row.get("map_id")

    def live_person_object_id(home_id: str, camera_id: str, track_id: int) -> str:
        now = datetime.now(timezone.utc)
        stale_before = (now - timedelta(seconds=12)).replace(microsecond=0).isoformat()
        tracks: dict[tuple[str, str, int], tuple[str, datetime]] = app.state.vision_person_objects
        for key, (_, last_seen) in list(tracks.items()):
            if (now - last_seen).total_seconds() > 12:
                tracks.pop(key, None)

        key = (home_id, camera_id, track_id)
        existing = tracks.get(key)
        if existing:
            tracks[key] = (existing[0], now)
            return existing[0]

        claimed = {object_id for object_id, _ in tracks.values()}
        reusable = db.many(
            """
            SELECT o.id,
                   (SELECT MAX(ob.observed_at) FROM observations ob WHERE ob.home_id=o.home_id AND ob.object_id=o.id) AS last_seen_at
            FROM objects o
            WHERE o.home_id=? AND lower(o.label)='person' AND o.enabled=1
            ORDER BY COALESCE(last_seen_at, o.created_at)
            """,
            (home_id,),
        )
        object_id = next(
            (
                row["id"]
                for row in reusable
                if row["id"] not in claimed and (row.get("last_seen_at") is None or row["last_seen_at"] < stale_before)
            ),
            None,
        )
        if object_id is None:
            object_id = str(uuid.uuid4())
            db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?)", (object_id, home_id, "person", "Person", 1, now_iso()))
        tracks[key] = (object_id, now)
        return object_id

    async def persist_vision_observation(home_id: str, camera_id: str, map_id: str | None, item: dict, detector_version: str) -> dict | None:
        label = str(item.get("label") or "").strip().lower()
        if not label:
            return None
        track_id = item.get("track_id")
        if label == "person" and isinstance(track_id, int):
            object_id = live_person_object_id(home_id, camera_id, track_id)
        else:
            object_row = db.one("SELECT * FROM objects WHERE home_id=? AND lower(label)=? AND enabled=1 ORDER BY created_at LIMIT 1", (home_id, label))
            if object_row is None:
                object_id = str(uuid.uuid4())
                db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?)", (object_id, home_id, label, label.replace("_", " ").title(), 1, now_iso()))
            else:
                object_id = object_row["id"]
        persistence_window = 1 if label == "person" else 5
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=persistence_window)).replace(microsecond=0).isoformat()
        if db.one("SELECT id FROM observations WHERE home_id=? AND object_id=? AND camera_id=? AND observed_at>=? ORDER BY observed_at DESC LIMIT 1", (home_id, object_id, camera_id, cutoff)):
            return None
        projection = item.get("projection") if isinstance(item.get("projection"), dict) else {}
        world = projection.get("world_xyz")
        if isinstance(world, (list, tuple)) and len(world) == 3:
            x, y, z = (float(value) for value in world)
        else:
            x = y = z = None
        observation_id = str(uuid.uuid4())
        observed = now_iso()
        uncertainty = projection.get("uncertainty_m") if isinstance(projection.get("uncertainty_m"), (int, float)) else None
        confidence = float(item.get("confidence") or 0.0)
        db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (observation_id, home_id, object_id, camera_id, map_id, x, y, z, uncertainty, confidence, detector_version, observed))
        event_id = str(uuid.uuid4())
        expires = (datetime.now(timezone.utc) + timedelta(days=30)).replace(microsecond=0).isoformat()
        db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", (event_id, home_id, "object_observed", "new", "Approximate household observation; not a diagnosis.", confidence, json.dumps([observation_id]), observed, observed, expires))
        await bus.publish(home_id, {"event_id": event_id, "observation_id": observation_id, "home_id": home_id, "type": "object_observed", "observed_at": observed})
        zone = roomplan_zone_for_point(map_id, home_id, x, z)
        return {"observation_id": observation_id, "event_id": event_id, "object_id": object_id, "zone": zone.get("label") if zone else None}

    @app.get("/api/v1/homes/{home_id}/objects")
    def objects(home_id: str, actor: Current):
        home_check(actor, home_id)
        return {"data": object_rows(home_id)}

    @app.get("/api/v1/homes/{home_id}/objects/last-seen")
    def objects_last_seen(home_id: str, actor: Current):
        home_check(actor, home_id)
        return {"data": object_rows(home_id)}

    @app.get("/api/v1/homes/{home_id}/objects/{object_id}")
    def object_detail(home_id: str, object_id: str, actor: Current):
        home_check(actor, home_id)
        row = next((item for item in object_rows(home_id) if item["id"] == object_id), None)
        if not row:
            raise HTTPException(404, "Object not found")
        return row

    @app.post("/api/v1/homes/{home_id}/vision/frames")
    async def vision_frame(home_id: str, body: VisionIn, actor: Current):
        home_check(actor, home_id)
        require_video_capture(home_id)
        if not db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (body.camera_id, home_id)):
            raise HTTPException(404, "Camera not found or disabled")
        labels = vision_candidate_labels(home_id, body.candidate_labels)
        prohibited = ("face", "identity", "emotion", "medical symptom", "diagnosis")
        if any(any(term in label.lower() for term in prohibited) for label in labels):
            raise HTTPException(422, "Identity and medical inference labels are not supported")
        try:
            frame_bytes = base64.b64decode(body.frame_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(422, "frame_base64 must be valid base64") from exc
        if not frame_bytes or len(frame_bytes) > 3_000_000:
            raise HTTPException(413, "frame is empty or exceeds the 3 MB in-memory limit")
        captured = body.captured_at or datetime.now(timezone.utc)
        calibration, map_id = vision_calibration(home_id, body.camera_id, body.width, body.height)
        try:
            result = app.state.vision.ingest(Frame(body.camera_id, frame_bytes, body.width, body.height, captured), labels, calibration=calibration, depth_m=body.depth_m)
        except (RoomLayoutServiceUnavailable, RoomLayoutServiceError, RuntimeError) as exc:
            raise HTTPException(503, "Local real vision model is unavailable") from exc
        detector_version = app.state.vision.detector.model_version
        persisted = [saved for item in result if (saved := await persist_vision_observation(home_id, body.camera_id, map_id, item, detector_version)) is not None]
        for item in result:
            projection = item.get("projection") if isinstance(item.get("projection"), dict) else None
            if projection and isinstance(projection.get("world_xyz"), (list, tuple)) and len(projection["world_xyz"]) == 3:
                world = projection["world_xyz"]
                zone = roomplan_zone_for_point(map_id, home_id, float(world[0]), float(world[2]))
                projection["room_zone"] = {"id": zone.get("id"), "label": zone.get("label")} if zone else None
        return {"data": result, "detector_version": detector_version, "observations": persisted, "frames_persisted": False, "privacy": "frame bytes were processed in memory by the local real model and were not stored"}

    @app.post("/api/v1/homes/{home_id}/observations")
    async def observation(home_id: str, body: ObservationIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); oid = str(uuid.uuid4()); observed = now_iso(); db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (oid, home_id, body.object_id, body.camera_id, body.map_id, body.x, body.y, body.z, body.uncertainty_m, body.confidence, body.detector_version, observed)); event_id = str(uuid.uuid4()); expires = (datetime.now(timezone.utc) + timedelta(days=30)).replace(microsecond=0).isoformat(); db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", (event_id, home_id, "object_observed", "new", "Approximate household observation; not a diagnosis.", body.confidence, json.dumps([oid]), observed, observed, expires)); payload = {"event_id": event_id, "observation_id": oid, "home_id": home_id, "type": "object_observed", "observed_at": observed}; await bus.publish(home_id, payload); return {"observation_id": oid, "event_id": event_id, "approximate_location": {"x": body.x, "y": body.y, "z": body.z, "uncertainty_m": body.uncertainty_m}}

    @app.get("/api/v1/homes/{home_id}/events")
    def events(home_id: str, actor: Current, limit: int = 50): home_check(actor, home_id); publisher_block(actor); limit = min(max(limit, 1), 100); return {"data": db.many("SELECT * FROM events WHERE home_id=? ORDER BY last_seen_at DESC LIMIT ?", (home_id, limit))}

    @app.post("/api/v1/homes/{home_id}/events/{event_id}/clips")
    def clip_create(home_id: str, event_id: str, body: ClipIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        if not db.one("SELECT id FROM events WHERE id=? AND home_id=?", (event_id, home_id)): raise HTTPException(404, "Event not found")
        clip_id = str(uuid.uuid4()); expires = (datetime.now(timezone.utc) + timedelta(days=7)).replace(microsecond=0).isoformat()
        db.execute("INSERT INTO clips VALUES (?,?,?,?,?,?,?)", (clip_id, home_id, event_id, body.object_key, body.starts_at, body.ends_at, expires)); audit(actor, "clip.register", "clip", clip_id, home_id)
        return {"id": clip_id, "event_id": event_id, "expires_at": expires, "download_path": f"/api/v1/clips/{clip_id}/content"}

    @app.post("/api/v1/homes/{home_id}/clips/{clip_id}/content")
    def clip_content_upload(home_id: str, clip_id: str, body: ClipBytesIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        clip = db.one("SELECT * FROM clips WHERE id=? AND home_id=?", (clip_id, home_id))
        if not clip: raise HTTPException(404, "Clip not found")
        if expired(clip["expires_at"]): raise HTTPException(410, "Clip expired")
        try: raw = base64.b64decode(body.content_base64, validate=True)
        except ValueError as exc: raise HTTPException(422, "content_base64 must be valid base64") from exc
        if not raw or len(raw) > 8_000_000: raise HTTPException(413, "clip is empty or exceeds the 8 MB demo limit")
        encrypted_key = app.state.clip_store.put(home_id, clip_id, raw, datetime.fromisoformat(clip["expires_at"]))
        db.execute("UPDATE clips SET object_key=? WHERE id=?", (encrypted_key, clip_id)); audit(actor, "clip.upload", "clip", clip_id, home_id)
        return {"id": clip_id, "encrypted": True, "bytes": len(raw), "download_path": f"/api/v1/clips/{clip_id}/content"}

    @app.get("/api/v1/clips/{clip_id}/content")
    def clip_content(clip_id: str, actor: Current):
        publisher_block(actor)
        clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
        if not clip or clip["home_id"] != actor["home_id"]: raise HTTPException(403, "Clip access denied")
        if expired(clip["expires_at"]): raise HTTPException(410, "Clip expired")
        try: content = app.state.clip_store.get(actor["home_id"], clip_id)
        except FileNotFoundError: raise HTTPException(404, "Encrypted clip content not found")
        except Exception as exc: raise HTTPException(503, "Encrypted clip could not be verified") from exc
        audit(actor, "clip.view", "clip", clip_id, actor["home_id"])
        return Response(content=content, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})

    @app.get("/api/v1/homes/{home_id}/clips")
    def clips(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor); return {"data": db.many("SELECT * FROM clips WHERE home_id=? AND expires_at>? ORDER BY starts_at DESC", (home_id, now_iso()))}

    # Care recipients are people receiving care in this home/residence. They
    # intentionally do not create a login, membership, session, or permission.
    def care_recipient_view(row: dict) -> dict:
        return {
            "id": row["id"],
            "display_name": row["display_name"],
            "relationship": row.get("relationship"),
            "room_label": row.get("room_label"),
            "medication_reminders_enabled": active_care_recipient_consent(row["home_id"], row["id"], "medication_management"),
            "created_at": row["created_at"],
        }

    def care_recipient_row(home_id: str, recipient_id: str) -> dict:
        row = db.one("SELECT * FROM care_recipients WHERE id=? AND home_id=?", (recipient_id, home_id))
        if not row:
            raise HTTPException(404, "Care recipient not found")
        return row

    @app.get("/api/v1/homes/{home_id}/care-recipients", response_model=CareRecipientListResponse)
    def care_recipients(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        rows = db.many("SELECT * FROM care_recipients WHERE home_id=? ORDER BY created_at, display_name", (home_id,))
        return {"data": [care_recipient_view(row) for row in rows]}

    @app.post("/api/v1/homes/{home_id}/care-recipients", response_model=CareRecipientMutationResponse, status_code=status.HTTP_201_CREATED)
    def care_recipient_create(home_id: str, body: CareRecipientCreateIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        recipient_id, created = str(uuid.uuid4()), now_iso()
        db.execute(
            "INSERT INTO care_recipients(id,home_id,display_name,relationship,room_label,created_at) VALUES (?,?,?,?,?,?)",
            (recipient_id, home_id, body.display_name, body.relationship, body.room_label, created),
        )
        audit(actor, "care_recipient.create", "care_recipient", recipient_id, home_id)
        return {"data": care_recipient_view(care_recipient_row(home_id, recipient_id))}

    @app.patch("/api/v1/homes/{home_id}/care-recipients/{recipient_id}", response_model=CareRecipientMutationResponse)
    def care_recipient_update(home_id: str, recipient_id: str, body: CareRecipientUpdateIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        current = care_recipient_row(home_id, recipient_id)
        changes = body.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(422, "At least one care-recipient field is required")
        if "display_name" in changes and changes["display_name"] is None:
            raise HTTPException(422, "display_name cannot be null")
        updated = {
            "display_name": changes.get("display_name", current["display_name"]),
            "relationship": changes.get("relationship", current.get("relationship")),
            "room_label": changes.get("room_label", current.get("room_label")),
        }
        db.execute(
            "UPDATE care_recipients SET display_name=?, relationship=?, room_label=? WHERE id=? AND home_id=?",
            (updated["display_name"], updated["relationship"], updated["room_label"], recipient_id, home_id),
        )
        audit(actor, "care_recipient.update", "care_recipient", recipient_id, home_id)
        return {"data": care_recipient_view(care_recipient_row(home_id, recipient_id))}

    @app.delete("/api/v1/homes/{home_id}/care-recipients/{recipient_id}", response_model=CareRecipientMutationResponse)
    def care_recipient_delete(home_id: str, recipient_id: str, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        deleted = care_recipient_view(care_recipient_row(home_id, recipient_id))
        db.execute("DELETE FROM care_recipients WHERE id=? AND home_id=?", (recipient_id, home_id))
        audit(actor, "care_recipient.delete", "care_recipient", recipient_id, home_id)
        return {"data": deleted}

    # Family mode is deliberately a bounded, consent-gated slice. It exposes
    # household membership and medication adherence records, not a resident's
    # continuous camera stream. Invitation codes are hashed and single-use,
    # like camera pairing codes, and are intended for synthetic demo accounts.
    def family_member_view(row: dict) -> dict:
        return {
            "id": row["id"],
            "display_name": row["display_name"],
            "email": row["email"],
            "role": row["role"],
            "created_at": row["created_at"],
            "representation_status": "not_recorded",
            "synthetic_demo": True,
        }

    @app.get("/api/v1/homes/{home_id}/family/members")
    def family_members(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        # Residents may view only themselves; caregiver views require the
        # family-sharing purpose to have been explicitly recorded.
        if actor["role"] == "resident":
            rows = [member(home_id, actor["user_id"])]
        else:
            family_actor(actor)
            require_consent(home_id, actor["user_id"], "family_mode")
            rows = db.many("SELECT u.id, u.display_name, u.email, u.created_at, m.role FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=? AND m.role != 'publisher' ORDER BY u.created_at", (home_id,))
        return {"data": [family_member_view(row) for row in rows], "purpose": "family_mode", "representation_required": True}

    @app.patch("/api/v1/homes/{home_id}/family/members/{user_id}", response_model=FamilyMemberMutationResponse)
    def family_member_update(home_id: str, user_id: str, body: FamilyMemberUpdateIn, actor: Current):
        """Change a person's household role without ever granting admin access."""
        home_check(actor, home_id); family_actor(actor); require_consent(home_id, actor["user_id"], "family_mode")
        if user_id == actor["user_id"]:
            raise HTTPException(409, "You cannot change your own household access")
        target = member(home_id, user_id)
        if target["role"] == "admin":
            raise HTTPException(403, "Admin access can only be changed by a separate admin workflow")
        if actor["role"] != "admin" and body.role == "caregiver":
            # Caregivers can manage residents, but cannot promote someone to a
            # role with equivalent access without an admin.
            raise HTTPException(403, "Only an admin can grant caregiver access")
        db.execute("UPDATE memberships SET role=? WHERE home_id=? AND user_id=?", (body.role, home_id, user_id))
        sessions = db.execute("DELETE FROM sessions WHERE home_id=? AND user_id=?", (home_id, user_id)).rowcount
        audit(actor, "family.member.role.update", "user", user_id, home_id)
        updated = member(home_id, user_id)
        return {"data": family_member_view(updated), "invalidated_sessions": sessions or 0}

    @app.delete("/api/v1/homes/{home_id}/family/members/{user_id}", response_model=FamilyMemberMutationResponse)
    def family_member_remove(home_id: str, user_id: str, actor: Current):
        """Revoke a person's membership and all sessions for this household."""
        home_check(actor, home_id); family_actor(actor); require_consent(home_id, actor["user_id"], "family_mode")
        if user_id == actor["user_id"]:
            raise HTTPException(409, "You cannot remove your own household access")
        target = member(home_id, user_id)
        if target["role"] == "admin":
            raise HTTPException(403, "Admin access cannot be revoked from this endpoint")
        with db.transaction() as conn:
            conn.execute("DELETE FROM memberships WHERE home_id=? AND user_id=?", (home_id, user_id))
            sessions = conn.execute("DELETE FROM sessions WHERE home_id=? AND user_id=?", (home_id, user_id)).rowcount
            conn.execute("DELETE FROM consents WHERE home_id=? AND subject_user_id=?", (home_id, user_id))
            conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), home_id, actor["user_id"], "family.member.remove", "user", user_id, "{}", now_iso()))
        return {"data": family_member_view(target), "invalidated_sessions": sessions or 0}

    @app.post("/api/v1/homes/{home_id}/family/invites")
    def family_invite(home_id: str, body: FamilyInviteIn, actor: Current):
        home_check(actor, home_id); family_actor(actor); require_consent(home_id, actor["user_id"], "family_mode")
        email = None
        if body.email:
            try:
                email = normalize_email(body.email)
            except ValueError as exc:
                raise HTTPException(422, "A valid invite email is required") from exc
        invite_id, code, created = str(uuid.uuid4()), new_pairing_code(), now_iso()
        db.execute("INSERT INTO family_invites VALUES (?,?,?,?,?,?,?,?,?,?)", (invite_id, home_id, actor["user_id"], email, body.display_name, body.role, hash_secret(code), iso_after(body.expires_in_seconds / 60), None, created))
        audit(actor, "family.invite.create", "family_invite", invite_id, home_id)
        # The plaintext code is returned once for a local synthetic demo; it
        # is never written to audit logs or persisted by the service.
        return {"id": invite_id, "code": code, "role": body.role, "expires_in_seconds": body.expires_in_seconds, "synthetic_demo": True}

    @app.post("/api/v1/family/invites/accept")
    def family_invite_accept(body: FamilyInviteAcceptIn):
        invite = db.one("SELECT * FROM family_invites WHERE code_hash=? AND accepted_at IS NULL", (hash_secret(body.code),))
        if not invite or expired(invite["expires_at"]):
            raise HTTPException(400, "Invalid or expired family invitation")
        invite_email = None
        if invite.get("email"):
            try:
                invite_email = normalize_email(invite["email"])
            except ValueError as exc:
                raise HTTPException(500, "Invitation email is invalid") from exc
            if not body.email:
                raise HTTPException(422, "The invitation email is required")
            try:
                supplied_email = normalize_email(body.email)
            except ValueError as exc:
                raise HTTPException(422, "A valid email address is required") from exc
            if supplied_email != invite_email:
                raise HTTPException(403, "Invitation email does not match this account")
            existing = db.one("SELECT * FROM users WHERE lower(trim(email))=? ORDER BY created_at LIMIT 1", (invite_email,))
            if not existing:
                raise HTTPException(404, "Create an account with the invited email before joining this household")
            user_id = existing["id"]
            display_name = existing["display_name"]
        else:
            user_id, created = str(uuid.uuid4()), now_iso()
            display_name = body.display_name or invite["display_name"]
            supplied_email = None
        with db.transaction() as conn:
            created = now_iso()
            if not invite_email:
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (user_id, display_name, supplied_email, created))
            existing_membership = db.one("SELECT role FROM memberships WHERE home_id=? AND user_id=?", (invite["home_id"], user_id))
            if not existing_membership:
                conn.execute("INSERT INTO memberships VALUES (?,?,?)", (invite["home_id"], user_id, invite["role"]))
            conn.execute("UPDATE family_invites SET accepted_at=? WHERE id=?", (created, invite["id"]))
            conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), invite["home_id"], user_id, "family.invite.accept", "family_invite", invite["id"], "{}", created))
        token = new_token()
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (hash_secret(token), user_id, invite["home_id"], iso_after(settings.session_ttl_minutes), created))
        role = existing_membership["role"] if existing_membership else invite["role"]
        return {"access_token": token, "token_type": "bearer", "expires_in": settings.session_ttl_minutes * 60, "home_id": invite["home_id"], "user_id": user_id, "role": role, "email": invite_email}

    def medication_plan_view(row: dict) -> dict:
        return {
            "id": row["id"],
            "home_id": row["home_id"],
            "subject_user_id": row["subject_user_id"],
            "care_recipient_id": row.get("care_recipient_id"),
            "name": row["name"],
            "dose": row["dose"],
            "schedule": row["schedule"],
            "instructions": row["instructions"],
            "active": bool(row["active"]),
            "version": row["version"],
            "created_by": row["created_by"],
            "assigned_caregiver_id": row.get("assigned_caregiver_id"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def medication_plan(home_id: str, plan_id: str) -> dict:
        row = db.one("SELECT * FROM medication_plans WHERE id=? AND home_id=?", (plan_id, home_id))
        if not row:
            raise HTTPException(404, "Medication plan not found")
        return row

    @app.get("/api/v1/homes/{home_id}/medication-plans")
    def medication_plans(home_id: str, actor: Current, subject_user_id: str | None = None, care_recipient_id: str | None = None, active_only: bool = True):
        home_check(actor, home_id); publisher_block(actor)
        if subject_user_id and care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        if care_recipient_id:
            care_recipient_id = medication_care_recipient(home_id, actor, care_recipient_id)["id"]
            query = "SELECT * FROM medication_plans WHERE home_id=? AND care_recipient_id=?"
            params: list[object] = [home_id, care_recipient_id]
            subject_id = None
        else:
            subject_id = subject_user_id or actor["user_id"]
            member(home_id, subject_id)
            if subject_id != actor["user_id"]:
                family_actor(actor)
            require_consent(home_id, subject_id, "medication_management")
            query = "SELECT * FROM medication_plans WHERE home_id=? AND subject_user_id=? AND care_recipient_id IS NULL"
            params = [home_id, subject_id]
        if active_only:
            query += " AND active=1"
        query += " ORDER BY active DESC, name"
        return {"data": [medication_plan_view(row) for row in db.many(query, tuple(params))], "subject_user_id": subject_id, "care_recipient_id": care_recipient_id, "purpose": "medication_management", "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/medication-plans")
    def medication_plan_create(home_id: str, body: MedicationPlanIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        if body.subject_user_id and body.care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        care_recipient_id = canonical_uuid(body.care_recipient_id, "care_recipient_id") if body.care_recipient_id else None
        if care_recipient_id:
            care_recipient_id = medication_care_recipient(home_id, actor, care_recipient_id)["id"]
            subject_id = actor["user_id"]
        elif body.subject_user_id:
            subject_id = family_subject(home_id, actor, body.subject_user_id, "medication_management")["id"]
        else:
            raise HTTPException(422, "A medication subject is required")
        caregiver = assigned_caregiver(home_id, body.assigned_caregiver_id, actor)
        plan_id, created = str(uuid.uuid4()), now_iso()
        db.execute(
            "INSERT INTO medication_plans(id,home_id,subject_user_id,name,dose,schedule,instructions,active,version,created_by,assigned_caregiver_id,created_at,updated_at,care_recipient_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (plan_id, home_id, subject_id, body.name, body.dose, body.schedule, body.instructions, int(body.active), 1, actor["user_id"], caregiver["id"] if caregiver else None, created, created, care_recipient_id),
        )
        audit(actor, "medication.plan.create", "medication_plan", plan_id, home_id)
        result = medication_plan_view(db.one("SELECT * FROM medication_plans WHERE id=?", (plan_id,)))
        result["medical_advice"] = False
        return result

    @app.patch("/api/v1/homes/{home_id}/medication-plans/{plan_id}")
    def medication_plan_update(home_id: str, plan_id: str, body: MedicationPlanUpdate, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        current = medication_plan(home_id, plan_id)
        if current.get("care_recipient_id"):
            require_care_recipient_consent(home_id, current["care_recipient_id"], "medication_management")
        else:
            require_consent(home_id, current["subject_user_id"], "medication_management")
        if body.version is not None and body.version != current["version"]:
            raise HTTPException(409, "Medication plan version conflict")
        values = {key: value for key, value in body.model_dump(exclude_unset=True).items() if key != "version"}
        if not values:
            return medication_plan_view(current)
        if "assigned_caregiver_id" in values:
            caregiver = assigned_caregiver(home_id, values["assigned_caregiver_id"])
            values["assigned_caregiver_id"] = caregiver["id"] if caregiver else None
        assignments, params = [], []
        for key in ("name", "dose", "schedule", "instructions", "active", "assigned_caregiver_id"):
            if key in values:
                assignments.append(f"{key}=?")
                params.append(int(values[key]) if key == "active" else values[key])
        updated = now_iso(); assignments.extend(["version=version+1", "updated_at=?"]); params.extend([updated, plan_id, home_id])
        db.execute(f"UPDATE medication_plans SET {', '.join(assignments)} WHERE id=? AND home_id=?", tuple(params))
        audit(actor, "medication.plan.update", "medication_plan", plan_id, home_id)
        result = medication_plan_view(medication_plan(home_id, plan_id))
        result["medical_advice"] = False
        return result

    @app.get("/api/v1/homes/{home_id}/medication-check-ins")
    def medication_check_ins(home_id: str, actor: Current, subject_user_id: str | None = None, care_recipient_id: str | None = None, scheduled_from: datetime | None = None, scheduled_to: datetime | None = None):
        home_check(actor, home_id); publisher_block(actor)
        if subject_user_id and care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        if care_recipient_id:
            care_recipient_id = medication_care_recipient(home_id, actor, care_recipient_id)["id"]
            query = "SELECT * FROM medication_check_ins WHERE home_id=? AND care_recipient_id=?"
            params: list[object] = [home_id, care_recipient_id]
            subject_id = None
        else:
            subject_id = subject_user_id or actor["user_id"]
            member(home_id, subject_id)
            if subject_id != actor["user_id"]:
                family_actor(actor)
            require_consent(home_id, subject_id, "medication_management")
            query = "SELECT * FROM medication_check_ins WHERE home_id=? AND subject_user_id=? AND care_recipient_id IS NULL"
            params = [home_id, subject_id]
        if scheduled_from:
            query += " AND scheduled_for>=?"; params.append(scheduled_from.isoformat())
        if scheduled_to:
            query += " AND scheduled_for<=?"; params.append(scheduled_to.isoformat())
        query += " ORDER BY scheduled_for DESC"
        rows = db.many(query, tuple(params))
        for row in rows:
            marker = db.one("SELECT display_name FROM users WHERE id=?", (row.get("marked_by"),)) if row.get("marked_by") else None
            row["marked_by_name"] = marker["display_name"] if marker else None
        return {"data": rows, "subject_user_id": subject_id, "care_recipient_id": care_recipient_id, "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/medication-plans/{plan_id}/check-ins")
    def medication_check_in(home_id: str, plan_id: str, body: MedicationCheckInIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        plan = medication_plan(home_id, plan_id)
        if plan.get("care_recipient_id"):
            family_actor(actor)
            require_care_recipient_consent(home_id, plan["care_recipient_id"], "medication_management")
        else:
            if actor["user_id"] != plan["subject_user_id"]:
                family_actor(actor)
            require_consent(home_id, plan["subject_user_id"], "medication_management")
        timestamp, scheduled = now_iso(), body.scheduled_for.replace(microsecond=0).isoformat()
        check_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO medication_check_ins(id,home_id,plan_id,subject_user_id,scheduled_for,status,note,marked_by,created_at,updated_at,care_recipient_id) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(plan_id,scheduled_for) DO UPDATE SET status=excluded.status, note=excluded.note, marked_by=excluded.marked_by, updated_at=excluded.updated_at, care_recipient_id=excluded.care_recipient_id",
            (check_id, home_id, plan_id, plan["subject_user_id"], scheduled, body.status, body.note, actor["user_id"], timestamp, timestamp, plan.get("care_recipient_id")),
        )
        row = db.one("SELECT * FROM medication_check_ins WHERE plan_id=? AND scheduled_for=?", (plan_id, scheduled))
        row["marked_by_name"] = actor["display_name"]
        audit(actor, "medication.check_in.update", "medication_check_in", row["id"], home_id)
        return {"data": row, "medical_advice": False}

    DAY_ALIASES = {
        "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3,
        "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    TIME_PATTERN = r"(?<!\d)(?:[01]\d|2[0-3]):[0-5]\d(?!\d)"

    def schedule_slots(schedule: str, target_day: str | None = None) -> list[str]:
        """Parse a small, deterministic recurrence grammar for reminders.

        Legacy ``08:00,20:00`` remains daily. A family can narrow a rule with
        ``Mon,Wed,Fri @ 08:00``, ``weekdays 08:00`` or an exact
        ``2026-09-12 @ 08:00`` exception. Unsupported text is never guessed as
        a dose; it returns an explicit unscheduled slot for human review.
        """
        value = schedule.strip()
        if not value:
            return ["unscheduled"]
        selected_date = None
        selected_weekday = None
        if target_day:
            try:
                selected_date = datetime.strptime(target_day, "%Y-%m-%d").date()
                selected_weekday = selected_date.weekday()
            except ValueError:
                raise HTTPException(422, "day must use YYYY-MM-DD")
        parsed: list[tuple[str | None, set[int] | None, str]] = []
        # Semicolons separate day/date rules; commas remain useful for legacy
        # daily times and are interpreted as additional times in each rule.
        for segment in re.split(r";|\n", value):
            segment = segment.strip()
            if not segment:
                continue
            times = re.findall(TIME_PATTERN, segment)
            if not times:
                continue
            first_time = segment.lower().find(times[0].lower())
            prefix = segment[:first_time].strip(" @:-,").lower()
            exact_date = next((match.group(0) for match in re.finditer(r"20\d{2}-\d{2}-\d{2}", prefix)), None)
            days: set[int] | None = None
            if exact_date is None:
                if prefix in {"weekday", "weekdays"}:
                    days = {0, 1, 2, 3, 4}
                elif prefix in {"weekend", "weekends"}:
                    days = {5, 6}
                else:
                    found = {day for name, day in DAY_ALIASES.items() if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", prefix)}
                    # ``daily``, ``every day``, ``everyday``, ``weekly`` and a
                    # blank prefix mean the rule applies on every weekday.
                    days = found or None
            parsed.extend((exact_date, days, time) for time in times)
        if not parsed:
            return ["unscheduled"]
        slots = []
        for exact_date, days, time in parsed:
            if selected_date is not None:
                if exact_date and exact_date != selected_date.isoformat():
                    continue
                if exact_date is None and days is not None and selected_weekday not in days:
                    continue
            slots.append(time)
        return list(dict.fromkeys(slots)) or []

    @app.get("/api/v1/homes/{home_id}/medication-reminders")
    def medication_reminders(home_id: str, actor: Current, day: str | None = None, subject_user_id: str | None = None, care_recipient_id: str | None = None):
        home_check(actor, home_id); publisher_block(actor)
        if subject_user_id and care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        if care_recipient_id:
            care_recipient_id = medication_care_recipient(home_id, actor, care_recipient_id)["id"]
            subject_id = None
        else:
            subject_id = subject_user_id or actor["user_id"]
            member(home_id, subject_id)
            if subject_id != actor["user_id"]:
                family_actor(actor)
            require_consent(home_id, subject_id, "medication_management")
        target_day = day or datetime.now(timezone.utc).date().isoformat()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", target_day):
            raise HTTPException(422, "day must use YYYY-MM-DD")
        plans = db.many(
            "SELECT * FROM medication_plans WHERE home_id=? AND care_recipient_id=? AND active=1 ORDER BY name" if care_recipient_id else "SELECT * FROM medication_plans WHERE home_id=? AND subject_user_id=? AND care_recipient_id IS NULL AND active=1 ORDER BY name",
            (home_id, care_recipient_id if care_recipient_id else subject_id),
        )
        reminders = []
        for plan in plans:
            for slot in schedule_slots(plan["schedule"], target_day):
                scheduled = f"{target_day}T{slot}:00+00:00" if slot != "unscheduled" else f"{target_day}T00:00:00+00:00"
                status_row = db.one("SELECT status, note, marked_by, updated_at FROM medication_check_ins WHERE plan_id=? AND scheduled_for=?", (plan["id"], scheduled))
                caregiver = db.one("SELECT display_name FROM users WHERE id=?", (plan["assigned_caregiver_id"],)) if plan.get("assigned_caregiver_id") else None
                marker = db.one("SELECT display_name FROM users WHERE id=?", (status_row.get("marked_by"),)) if status_row and status_row.get("marked_by") else None
                reminders.append({"plan_id": plan["id"], "care_recipient_id": plan.get("care_recipient_id"), "name": plan["name"], "dose": plan["dose"], "instructions": plan["instructions"], "schedule_rule": plan["schedule"], "scheduled_for": scheduled, "status": status_row["status"] if status_row else "pending", "note": status_row["note"] if status_row else "", "updated_at": status_row["updated_at"] if status_row else None, "marked_by": status_row["marked_by"] if status_row else None, "marked_by_name": marker["display_name"] if marker else None, "assigned_caregiver_id": plan.get("assigned_caregiver_id"), "assigned_caregiver_name": caregiver["display_name"] if caregiver else None})
        return {"data": reminders, "subject_user_id": subject_id, "care_recipient_id": care_recipient_id, "timezone": "UTC", "deterministic": True, "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/family-assistant")
    def family_assistant(home_id: str, body: FamilyAssistantIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        if body.subject_user_id and body.care_recipient_id:
            raise HTTPException(422, "Choose either subject_user_id or care_recipient_id")
        if body.care_recipient_id:
            subject = medication_care_recipient(home_id, actor, body.care_recipient_id)
            target_field, target_id = "care_recipient_id", subject["id"]
        else:
            subject = family_subject(home_id, actor, body.subject_user_id, "family_assistant")
            target_field, target_id = "subject_user_id", subject["id"]
        # Keep this context narrow: plans and their bounded check-in statuses
        # only. Do not pass events, frames, transcripts, or a household stream.
        plans = db.many(f"SELECT id, name, dose, schedule, instructions, active, version, assigned_caregiver_id FROM medication_plans WHERE home_id=? AND {target_field}=? AND active=1 ORDER BY name LIMIT 50", (home_id, target_id))
        checks = db.many(f"SELECT id, plan_id, scheduled_for, status, note, updated_at FROM medication_check_ins WHERE home_id=? AND {target_field}=? ORDER BY scheduled_for DESC LIMIT 100", (home_id, target_id))
        context = {"subject": {"id": subject["id"], "display_name": subject["display_name"]}, "plans": plans, "check_ins": checks, "request": body.message, "evidence_scope": "medication plans and check-ins only"}
        result = lm.family_summary(context)
        degraded = result is None
        if degraded:
            taken = sum(1 for row in checks if row["status"] == "taken")
            pending = sum(1 for row in checks if row["status"] == "pending")
            result = {"summary": f"{len(plans)} active medication plan(s) are configured; {taken} check-in(s) marked taken and {pending} pending.", "next_action": "Review the reminder list with the resident or caregiver." if pending else "No pending check-ins are recorded in the bounded history.", "evidence_ids": [row["id"] for row in checks[:10]], "limitations": "Local language model unavailable. This is an administrative summary, not medical advice."}
        allowed_evidence = {row["id"] for row in checks} | {row["id"] for row in plans}
        result["evidence_ids"] = [item for item in result.get("evidence_ids", []) if item in allowed_evidence][:20]
        result["evidence_timestamps"] = {row["id"]: row["updated_at"] for row in checks if row["id"] in result["evidence_ids"]}
        audit(actor, "assistant.family_summary", "care_recipient" if body.care_recipient_id else "user", subject["id"], home_id)
        return {"data": result, "degraded": degraded, "inference_status": lm.last_error if degraded else "ok", "subject_user_id": subject["id"] if not body.care_recipient_id else None, "care_recipient_id": body.care_recipient_id, "context_scope": "medication plans and check-ins only", "medical_advice": False, "model_version": settings.effective_llm_model if not degraded else "rules-family-v1"}

    @app.post("/api/v1/admin/retention/run")
    def retention_run(actor: Current):
        if actor["role"] != "admin": raise HTTPException(403, "Admin permission required")
        cutoff = now_iso(); expired_clips = db.many("SELECT id, object_key, home_id FROM clips WHERE expires_at<=?", (cutoff,))
        for clip in expired_clips:
            store.delete(clip["object_key"])
            app.state.clip_store.delete(clip["home_id"], clip["id"])
        counts = {}
        for table, column in (("clips", "expires_at"), ("events", "expires_at"), ("summaries", "expires_at")):
            counts[table] = db.execute(f"DELETE FROM {table} WHERE {column}<=?", (cutoff,)).rowcount
        observation_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(microsecond=0).isoformat()
        counts["observations"] = db.execute("DELETE FROM observations WHERE observed_at<=?", (observation_cutoff,)).rowcount
        audit(actor, "retention.run"); return {"deleted": counts, "ran_at": cutoff}

    @app.get("/api/v1/homes/{home_id}/events/stream")
    async def event_stream(home_id: str, actor: Current, once: bool = False):
        home_check(actor, home_id); publisher_block(actor)
        async def generate():
            yield ": connected\n\n"
            if once:
                yield sse("one.heartbeat.v1", {"home_id": home_id, "at": now_iso()})
                return
            async for payload in bus.subscribe(home_id): yield sse("one.event.v1", payload)
        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def livekit_url_for_request(request: Request) -> str:
        """Use the LAN LiveKit endpoint for same-Wi-Fi clients when configured."""
        if not settings.livekit_lan_url:
            return settings.livekit_url

        host_header = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
        host = host_header.split(",", 1)[0].strip()
        if host.startswith("["):
            host = host[1:].split("]", 1)[0]
        elif host.count(":") == 1:
            host = host.rsplit(":", 1)[0]

        lan_host = urlparse(settings.livekit_lan_url).hostname
        if lan_host and host.casefold() == lan_host.casefold():
            return settings.livekit_lan_url
        # sslip.io hostnames encode the same private LAN address while allowing
        # separate virtual hosts (for example `one.<ip>.sslip.io` and
        # `livekit.<ip>.sslip.io`). Treat siblings under the configured LAN
        # suffix as the same trusted local deployment.
        if lan_host and ".sslip.io" in lan_host.casefold():
            lan_suffix = lan_host.casefold().split(".", 1)[1]
            if host.casefold().endswith(f".{lan_suffix}"):
                return settings.livekit_lan_url
        if host.casefold().endswith(".local"):
            return settings.livekit_lan_url
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            sslip_match = re.search(r"(?:^|\.)(\d{1,3})-(\d{1,3})-(\d{1,3})-(\d{1,3})\.sslip\.io$", host.casefold())
            if not sslip_match:
                return settings.livekit_url
            try:
                address = ipaddress.ip_address(".".join(sslip_match.groups()))
            except ValueError:
                return settings.livekit_url
        return settings.livekit_lan_url if address.is_private or address.is_link_local else settings.livekit_url

    @app.post("/api/v1/homes/{home_id}/livekit/token")
    def livekit_token(home_id: str, request: Request, actor: Current, body: LiveKitTokenIn | None = None):
        home_check(actor, home_id)
        if not settings.livekit_api_key or not settings.livekit_api_secret: raise HTTPException(503, "LiveKit credentials are not configured")
        mode = body.mode if body else "auto"
        if actor["role"] == "publisher":
            require_video_capture(home_id)
            if mode == "subscribe": raise HTTPException(403, "Publisher tokens cannot subscribe")
            can_publish, can_subscribe = True, False
        else:
            if mode == "publish": raise HTTPException(403, "Caregiver tokens cannot publish")
            can_publish, can_subscribe = False, True
        return {"url": livekit_url_for_request(request), "token": livekit_jwt(settings.livekit_api_key, settings.livekit_api_secret, actor["user_id"], f"one-{home_id}", can_publish, can_subscribe), "expires_in": 600, "mode": "publish" if can_publish else "subscribe"}

    @app.post("/api/v1/livekit/webhook")
    async def livekit_webhook(request: Request):
        body = await request.body()
        if settings.livekit_api_key and settings.livekit_api_secret:
            try:
                claims = verify_livekit_webhook(request.headers.get("authorization"), body, settings.livekit_api_key, settings.livekit_api_secret)
            except ValueError as exc:
                raise HTTPException(401, "Invalid LiveKit webhook signature") from exc
            return {"accepted": True, "event": claims.get("event")}
        if settings.env == "production": raise HTTPException(503, "LiveKit webhook verification is not configured")
        return {"accepted": True, "verified": False}

    @app.post("/api/v1/homes/{home_id}/check-ins")
    def check_in(home_id: str, body: CheckInIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); events = db.many("SELECT id,event_type,confidence,last_seen_at FROM events WHERE home_id=? AND expires_at>? ORDER BY last_seen_at DESC LIMIT 20", (home_id, now_iso())); context = {"transcript": body.transcript, "events": events, "baseline": "personal baseline is intentionally bounded to recent derived observations"}; result = lm.summarize(context); degraded = result is None
        if degraded: result = {"status": "attention" if events else "unknown", "trend": "unknown", "explanation": "Recent household observations are available for human review." if events else "Not enough observations for a comparison.", "evidence_ids": [e["id"] for e in events], "limitations": "Local language model unavailable; this is a deterministic fallback and not medical advice."}
        sid = str(uuid.uuid4()); exp = (datetime.now(timezone.utc)+timedelta(days=30)).replace(microsecond=0).isoformat(); db.execute("INSERT INTO summaries VALUES (?,?,?,?,?,?,?,?,?,?,?)", (sid, home_id, body.subject_user_id or actor["user_id"], result["status"], result["trend"], result["explanation"], json.dumps(result.get("evidence_ids", [])), result["limitations"], settings.effective_llm_model if not degraded else "rules-fallback-v1", now_iso(), exp)); audit(actor, "assistant.check_in", "summary", sid, home_id); return {"id": sid, **result, "degraded": degraded, "inference_status": lm.last_error if degraded else "ok", "model_version": settings.effective_llm_model if not degraded else "rules-fallback-v1"}

    @app.get("/api/v1/homes/{home_id}/caregiver-summary")
    def caregiver_summary(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        if actor["role"] not in {"caregiver", "admin"}: raise HTTPException(403, "Caregiver permission required")
        return {"data": db.many("SELECT * FROM summaries WHERE home_id=? ORDER BY created_at DESC LIMIT 20", (home_id,))}

    @app.post("/api/v1/homes/{home_id}/privacy/export")
    def privacy_export(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor); audit(actor, "privacy.export", home_id=home_id); return {"home_id": home_id, "exported_at": now_iso(), "data": db.export_home(home_id)}

    @app.post("/api/v1/homes/{home_id}/privacy/delete")
    def privacy_delete(home_id: str, actor: Current):
        home_check(actor, home_id)
        if actor["role"] != "admin": raise HTTPException(403, "Admin permission required")
        request_id, requested_at = str(uuid.uuid4()), now_iso()
        rows = db.many("SELECT artifact_key, usdz_artifact_key, metadata_json FROM room_maps WHERE home_id=?", (home_id,))
        clips_for_home = db.many("SELECT id, object_key FROM clips WHERE home_id=?", (home_id,))
        db.execute("INSERT INTO deletion_requests VALUES (?,?,?,?,?,NULL)", (request_id, home_id, actor["user_id"], "processing", requested_at))
        cleanup_errors: list[str] = []
        for row in rows:
            metadata = json_object(row.get("metadata_json") or "{}")
            landmark_meta = metadata.get("visual_landmarks") if isinstance(metadata.get("visual_landmarks"), dict) else {}
            keys = [row.get("artifact_key"), row.get("usdz_artifact_key"), landmark_meta.get("artifact_key")]
            for key in keys:
                if not key:
                    continue
                try:
                    store.delete(key)
                except (OSError, ValueError) as exc:
                    cleanup_errors.append(f"map:{key}:{type(exc).__name__}")
        for row in clips_for_home:
            try:
                app.state.clip_store.delete(home_id, row["id"])
            except OSError as exc:
                cleanup_errors.append(f"clip:{row['id']}:{type(exc).__name__}")
        if cleanup_errors:
            db.execute("UPDATE deletion_requests SET status=? WHERE id=?", ("failed", request_id))
            audit(actor, "privacy.delete.failed", "deletion_request", request_id, home_id)
            raise HTTPException(503, "Deletion is pending media cleanup", headers={"X-Deletion-Request-ID": request_id})
        completed_at = now_iso()
        # Keep a minimal proof that the request completed, while cascading
        # household data through every FK-backed table in one DB transaction.
        with db.transaction() as conn:
            conn.execute("UPDATE deletion_requests SET status=?, completed_at=? WHERE id=?", ("completed", completed_at, request_id))
            conn.execute("DELETE FROM homes WHERE id=?", (home_id,))
            conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), home_id, None, "privacy.delete.completed", "home", home_id, "{}", completed_at))
        return {"request_id": request_id, "status": "completed"}

    return app


app = make_app()

import asyncio
import base64
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings, get_settings
from .db import Database, now_iso
from .events import EventBus, sse
from .integrations import LMStudioAdapter, livekit_jwt, verify_livekit_webhook
from .media import EncryptedLocalClipStore
from .security import expired, hash_secret, new_pairing_code, new_token, iso_after, normalize_email
from .storage import LocalObjectStore
from .vision import Calibration, CameraVisionPipeline, DeterministicDemoDetector, Frame


class PairStart(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    email: str | None = Field(default=None, max_length=254)
    home_name: str = Field(default="ONE Home", min_length=1, max_length=120)
    role: str = Field(default="admin", pattern="^(admin|resident|caregiver)$")


class PairStartResponse(BaseModel):
    pairing_code: str
    expires_in_seconds: int
    home_id: str
    user_id: str
    role: str

class DevicePairingStart(BaseModel):
    # `display_name` is accepted as a compatibility alias for the web client;
    # this endpoint always creates a publisher membership regardless of it.
    label: str | None = Field(default=None, min_length=1, max_length=120)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    expires_in_seconds: int = Field(default=600, ge=60, le=900)

class PairComplete(BaseModel): code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class EmailAuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    purpose: str = Field(default="login", pattern="^(create|login)$")
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    home_name: str = Field(default="ONE Home", min_length=1, max_length=120)
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
class ConsentIn(BaseModel):
    purpose: str = Field(min_length=1, max_length=120)
    policy_version: str = Field(min_length=1, max_length=40)
    granted: bool = True
    # A caregiver may record a resident's explicit decision only when the
    # representation process is separately documented; this field never
    # infers authority from a caregiver role.
    subject_user_id: str | None = None
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
class MapIn(BaseModel): room_id: str | None = None; coordinate_frame: str = Field(default="roomplan-local", max_length=80); map_data: dict
class ProvisionalMapIn(BaseModel):
    camera_id: str
    room_id: str | None = None
    resolution_width: int = Field(gt=0, le=7680)
    resolution_height: int = Field(gt=0, le=4320)
    zones: list[dict] = Field(default_factory=list, max_length=100)
class RoomPlanMapIn(BaseModel):
    room_id: str | None = None
    normalized_scan: dict
    scan_metadata: dict = Field(default_factory=dict)
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
class ObjectIn(BaseModel): label: str = Field(min_length=1, max_length=80); display_name: str | None = Field(default=None, max_length=120)
class ObservationIn(BaseModel): object_id: str | None = None; camera_id: str | None = None; map_id: str | None = None; x: float | None = None; y: float | None = None; z: float | None = None; uncertainty_m: float | None = Field(default=None, ge=0, le=100); confidence: float = Field(default=0.0, ge=0, le=1); detector_version: str = "local-cv-v1"
class CheckInIn(BaseModel): subject_user_id: str | None = None; transcript: str = Field(default="", max_length=4000)
class VisionIn(BaseModel):
    camera_id: str
    frame_base64: str = Field(min_length=1, max_length=4_000_000)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)
    candidate_labels: list[str] = Field(min_length=1, max_length=20)
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


class MedicationPlanIn(BaseModel):
    subject_user_id: str
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


def make_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings)
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
    vision = CameraVisionPipeline(DeterministicDemoDetector())
    app = FastAPI(title="ONE API", version="0.1.0", openapi_url="/api/v1/openapi.json")

    def request_id(request: Request) -> str:
        """Return a bounded correlation id without reflecting arbitrary input."""
        candidate = request.headers.get("x-request-id")
        try:
            return str(uuid.UUID(candidate)) if candidate else str(uuid.uuid4())
        except (ValueError, AttributeError):
            return str(uuid.uuid4())

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
        selected_id = caregiver_id or (fallback_actor["user_id"] if fallback_actor and fallback_actor["role"] in {"admin", "caregiver"} else None)
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
        view = {
            **row,
            "metadata": metadata,
            "label": row["name"],
            "platform": "browser",
            "status": "paused" if is_paused(row["home_id"]) else ("online" if row["enabled"] else "offline"),
            "lastSeenAt": row["created_at"],
        }
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
        }

    @app.get("/api/v1/me")
    def me(actor: Current):
        home = db.one("SELECT id, name FROM homes WHERE id=?", (actor["home_id"],))
        resident = db.one("SELECT u.display_name FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=? AND m.role='resident' ORDER BY u.created_at LIMIT 1", (actor["home_id"],))
        device = db.one("SELECT id, home_id, name, room_id, enabled, created_at FROM cameras WHERE home_id=? ORDER BY created_at LIMIT 1", (actor["home_id"],))
        return {"actor": {"id": actor["user_id"], "role": actor["role"], "name": actor["display_name"]}, "home": {"id": home["id"], "name": home["name"], "residentName": resident["display_name"] if resident else "Resident"}, "device": camera_view(device) if device else None, "paused": is_paused(actor["home_id"])}

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
        db.execute("INSERT INTO homes VALUES (?,?,?)", (home_id, body.home_name, now_iso()))
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
                conn.execute("INSERT INTO homes VALUES (?,?,?)", (home_id, body.home_name, created))
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

    @app.post("/api/v1/pairing/complete")
    def pairing_complete(body: PairComplete):
        row = db.one("SELECT * FROM pairing_codes WHERE code_hash=? AND used_at IS NULL", (hash_secret(body.code),))
        if not row or expired(row["expires_at"]): raise HTTPException(400, "Invalid or expired pairing code")
        token = new_token(); db.execute("UPDATE pairing_codes SET used_at=? WHERE code_hash=?", (now_iso(), row["code_hash"]))
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (hash_secret(token), row["user_id"], row["home_id"], iso_after(settings.session_ttl_minutes), now_iso()))
        audit({"user_id": row["user_id"], "home_id": row["home_id"]}, "pairing.complete", "user", row["user_id"])
        return {"access_token": token, "token_type": "bearer", "expires_in": settings.session_ttl_minutes * 60, "home_id": row["home_id"], "user_id": row["user_id"]}

    @app.delete("/api/v1/sessions/current")
    def logout(request: Request, actor: Current):
        db.execute("DELETE FROM sessions WHERE token_hash=?", (hash_secret(request.headers["authorization"][7:]),)); audit(actor, "session.logout"); return {"ok": True}

    @app.post("/api/v1/homes/{home_id}/consents")
    def consent(home_id: str, body: ConsentIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        subject_user_id = body.subject_user_id or actor["user_id"]
        member(home_id, subject_user_id)
        if subject_user_id != actor["user_id"] and actor["role"] not in {"admin", "caregiver"}:
            raise HTTPException(403, "Only a caregiver or admin can record a represented subject decision")
        cid = str(uuid.uuid4()); timestamp = now_iso(); db.execute("INSERT INTO consents VALUES (?,?,?,?,?,?,?)", (cid, home_id, subject_user_id, body.purpose, body.policy_version, timestamp, None if body.granted else timestamp))
        if body.purpose == "video_capture":
            db.execute("INSERT INTO home_runtime(home_id,paused,updated_at) VALUES (?,?,?) ON CONFLICT(home_id) DO UPDATE SET paused=excluded.paused, updated_at=excluded.updated_at", (home_id, 0 if body.granted else 1, timestamp))
        audit(actor, "consent.grant" if body.granted else "consent.revoke", "consent", cid, home_id); return {"id": cid, "granted": body.granted, "subject_user_id": subject_user_id, "paused": is_paused(home_id)}

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

    @app.get("/api/v1/homes/{home_id}/cameras")
    def cameras(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return {"data": [camera_view(row) for row in db.many("SELECT * FROM cameras WHERE home_id=?", (home_id,))]}

    @app.get("/api/v1/homes/{home_id}/rooms")
    def rooms(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return {"data": db.many("SELECT * FROM rooms WHERE home_id=? ORDER BY created_at", (home_id,))}

    @app.post("/api/v1/homes/{home_id}/rooms")
    def room(home_id: str, body: RoomIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); rid = str(uuid.uuid4()); db.execute("INSERT INTO rooms VALUES (?,?,?,?)", (rid, home_id, body.name, now_iso())); return {"id": rid, **body.model_dump()}

    @app.post("/api/v1/homes/{home_id}/maps")
    def room_map(home_id: str, body: MapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); row = db.one("SELECT COALESCE(MAX(revision),0)+1 revision FROM room_maps WHERE home_id=? AND room_id IS ?", (home_id, body.room_id)); mid = str(uuid.uuid4()); key = f"maps/{home_id}/{mid}.json"; store.put_json(key, body.map_data); created = now_iso(); db.execute("INSERT INTO room_maps(id,home_id,room_id,revision,coordinate_frame,artifact_key,map_json,created_at,source,approximate,localization_status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (mid, home_id, body.room_id, row["revision"], body.coordinate_frame, key, json.dumps(body.map_data), created, "manual", 0, "unlocalized", "{}")); db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='map revision changed' WHERE home_id=? AND status='active' AND map_id != ?", (created, home_id, mid)); return {"id": mid, "revision": row["revision"], "coordinate_frame": body.coordinate_frame, "artifact_key": key}

    def create_map(home_id: str, room_id: str | None, coordinate_frame: str, map_data: dict, source: str, approximate: bool, localization_status: str, metadata: dict, actor: dict) -> dict:
        row = db.one("SELECT COALESCE(MAX(revision),0)+1 revision FROM room_maps WHERE home_id=? AND room_id IS ?", (home_id, room_id))
        mid = str(uuid.uuid4()); key = f"maps/{home_id}/{mid}.json"; store.put_json(key, map_data)
        db.execute("INSERT INTO room_maps(id,home_id,room_id,revision,coordinate_frame,artifact_key,map_json,created_at,source,approximate,localization_status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (mid, home_id, room_id, row["revision"], coordinate_frame, key, json.dumps(map_data), now_iso(), source, int(approximate), localization_status, json.dumps(metadata)))
        db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='map revision changed' WHERE home_id=? AND status='active' AND map_id != ?", (now_iso(), home_id, mid))
        audit(actor, "map.create", "room_map", mid, home_id)
        return {"id": mid, "revision": row["revision"], "coordinate_frame": coordinate_frame, "source": source, "approximate": approximate, "localization_status": localization_status, "metadata": metadata}

    @app.post("/api/v1/homes/{home_id}/maps/provisional")
    def provisional_map(home_id: str, body: ProvisionalMapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        camera = db.one("SELECT * FROM cameras WHERE id=? AND home_id=? AND enabled=1", (body.camera_id, home_id))
        if not camera: raise HTTPException(404, "Camera not found or disabled")
        return create_map(home_id, body.room_id, "camera-zone-local", {"zones": body.zones}, "camera-provisional", True, "zone-only", {"camera_id": body.camera_id, "resolution_width": body.resolution_width, "resolution_height": body.resolution_height}, actor)

    @app.post("/api/v1/homes/{home_id}/maps/roomplan")
    def roomplan_map(home_id: str, body: RoomPlanMapIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        return create_map(home_id, body.room_id, "roomplan-local", body.normalized_scan, "roomplan-normalized", True, "unlocalized", body.scan_metadata, actor)

    def map_view(row: dict) -> dict:
        """Return a JSON-safe map read model without exposing storage internals."""
        try:
            map_data = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            map_data = {}
        return {
            "id": row["id"],
            "home_id": row["home_id"],
            "room_id": row["room_id"],
            "revision": row["revision"],
            "coordinate_frame": row["coordinate_frame"],
            "map_data": map_data,
            "created_at": row["created_at"],
            "source": row.get("source", "manual"),
            "approximate": bool(row.get("approximate", 0)),
            "localization_status": row.get("localization_status", "unlocalized"),
            "metadata": json.loads(row.get("metadata_json") or "{}"),
        }

    @app.get("/api/v1/homes/{home_id}/maps")
    def maps(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        rows = db.many("SELECT * FROM room_maps WHERE home_id=? ORDER BY revision DESC, created_at DESC", (home_id,))
        return {"data": [map_view(row) for row in rows]}

    @app.get("/api/v1/homes/{home_id}/maps/current")
    def current_map(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE home_id=? ORDER BY revision DESC, created_at DESC LIMIT 1", (home_id,))
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

    @app.get("/api/v1/homes/{home_id}/scene")
    def scene(home_id: str, actor: Current):
        """Return the compact scene contract consumed by the dashboard map."""
        home_check(actor, home_id); publisher_block(actor)
        row = db.one("SELECT * FROM room_maps WHERE home_id=? ORDER BY revision DESC, created_at DESC LIMIT 1", (home_id,))
        if not row:
            return {"sceneId": None, "version": 0, "zones": []}
        try:
            map_data = json.loads(row["map_json"])
        except (TypeError, json.JSONDecodeError):
            map_data = {}
        zones = map_data.get("zones", []) if isinstance(map_data, dict) else []
        return {"sceneId": row["id"], "version": row["revision"], "zones": zones, "mapId": row["id"], "coordinateFrame": row["coordinate_frame"]}

    @app.post("/api/v1/homes/{home_id}/calibrations")
    def calibration(home_id: str, body: CalibrationIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        if not db.one("SELECT id FROM cameras WHERE id=? AND home_id=?", (body.camera_id, home_id)): raise HTTPException(404, "Camera not found")
        if not db.one("SELECT id FROM room_maps WHERE id=? AND home_id=?", (body.map_id, home_id)): raise HTTPException(404, "Map not found")
        cid = str(uuid.uuid4()); created = now_iso()
        db.execute("UPDATE calibrations SET status='invalidated', invalidated_at=?, invalidation_reason='superseded by new calibration' WHERE home_id=? AND camera_id=? AND status='active'", (created, home_id, body.camera_id))
        db.execute("INSERT INTO calibrations(id,home_id,camera_id,map_id,intrinsics_json,extrinsics_json,accuracy_m,created_at,resolution_width,resolution_height,camera_metadata_json,source,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid, home_id, body.camera_id, body.map_id, json.dumps(body.intrinsics), json.dumps(body.extrinsics), body.accuracy_m, created, body.resolution_width, body.resolution_height, json.dumps(body.camera_metadata), body.source, "active"))
        return {"id": cid, **body.model_dump(), "status": "active", "invalidated_previous": True}

    @app.get("/api/v1/homes/{home_id}/calibrations")
    def calibrations(home_id: str, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        rows = db.many("SELECT * FROM calibrations WHERE home_id=? ORDER BY created_at DESC", (home_id,))
        return {"data": [{**row, "intrinsics": json.loads(row.pop("intrinsics_json")), "extrinsics": json.loads(row.pop("extrinsics_json")), "camera_metadata": json.loads(row.pop("camera_metadata_json") or "{}")} for row in rows]}

    @app.post("/api/v1/homes/{home_id}/objects")
    def object_create(home_id: str, body: ObjectIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor); oid = str(uuid.uuid4()); db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?)", (oid, home_id, body.label, body.display_name, 1, now_iso())); return {"id": oid, **body.model_dump(), "enabled": True}

    def object_view(row: dict) -> dict:
        observed_at = row.get("observed_at")
        x, y, z = row.get("x"), row.get("y"), row.get("z")
        observation_id = row.get("observation_id")
        event = db.one("SELECT id FROM events WHERE home_id=? AND evidence_json LIKE ? ORDER BY last_seen_at DESC LIMIT 1", (row["home_id"], f"%{observation_id}%")) if observation_id else None
        return {
            "id": row["id"],
            "label": row["display_name"] or row["label"],
            "icon": (row["label"][:1] or "?").upper(),
            "status": "seen" if observed_at else "unknown",
            "lastSeenAt": observed_at,
            "point": {"x": x, "y": y} if x is not None and y is not None else None,
            "confidenceRadiusM": row.get("uncertainty_m") if row.get("uncertainty_m") is not None else 0.0,
            "confidence": row.get("confidence") if row.get("confidence") is not None else 0.0,
            "zone": None,
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
    def vision_frame(home_id: str, body: VisionIn, actor: Current):
        home_check(actor, home_id)
        require_video_capture(home_id)
        if not db.one("SELECT id FROM cameras WHERE id=? AND home_id=? AND enabled=1", (body.camera_id, home_id)):
            raise HTTPException(404, "Camera not found or disabled")
        prohibited = {"face", "person identity", "emotion", "medical symptom"}
        if any(label.lower() in prohibited for label in body.candidate_labels):
            raise HTTPException(422, "Identity and medical inference labels are not supported")
        try:
            frame_bytes = base64.b64decode(body.frame_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(422, "frame_base64 must be valid base64") from exc
        if not frame_bytes or len(frame_bytes) > 3_000_000:
            raise HTTPException(413, "frame is empty or exceeds the 3 MB in-memory limit")
        captured = body.captured_at or datetime.now(timezone.utc)
        result = app.state.vision.ingest(Frame(body.camera_id, frame_bytes, body.width, body.height, captured), body.candidate_labels, depth_m=body.depth_m)
        return {"data": result, "detector_version": app.state.vision.detector.model_version, "persisted": False, "privacy": "frame bytes were processed in memory and not stored"}

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
    def medication_plans(home_id: str, actor: Current, subject_user_id: str | None = None, active_only: bool = True):
        home_check(actor, home_id); publisher_block(actor)
        subject_id = subject_user_id or actor["user_id"]
        member(home_id, subject_id)
        if subject_id != actor["user_id"]:
            family_actor(actor)
        require_consent(home_id, subject_id, "medication_management")
        query = "SELECT * FROM medication_plans WHERE home_id=? AND subject_user_id=?"
        params: list[object] = [home_id, subject_id]
        if active_only:
            query += " AND active=1"
        query += " ORDER BY active DESC, name"
        return {"data": [medication_plan_view(row) for row in db.many(query, tuple(params))], "subject_user_id": subject_id, "purpose": "medication_management", "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/medication-plans")
    def medication_plan_create(home_id: str, body: MedicationPlanIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        subject = family_subject(home_id, actor, body.subject_user_id, "medication_management")
        caregiver = assigned_caregiver(home_id, body.assigned_caregiver_id, actor)
        plan_id, created = str(uuid.uuid4()), now_iso()
        db.execute("INSERT INTO medication_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (plan_id, home_id, subject["id"], body.name, body.dose, body.schedule, body.instructions, int(body.active), 1, actor["user_id"], caregiver["id"] if caregiver else None, created, created))
        audit(actor, "medication.plan.create", "medication_plan", plan_id, home_id)
        result = medication_plan_view(db.one("SELECT * FROM medication_plans WHERE id=?", (plan_id,)))
        result["medical_advice"] = False
        return result

    @app.patch("/api/v1/homes/{home_id}/medication-plans/{plan_id}")
    def medication_plan_update(home_id: str, plan_id: str, body: MedicationPlanUpdate, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        current = medication_plan(home_id, plan_id)
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
    def medication_check_ins(home_id: str, actor: Current, subject_user_id: str | None = None, scheduled_from: datetime | None = None, scheduled_to: datetime | None = None):
        home_check(actor, home_id); publisher_block(actor)
        subject_id = subject_user_id or actor["user_id"]
        member(home_id, subject_id)
        if subject_id != actor["user_id"]:
            family_actor(actor)
        require_consent(home_id, subject_id, "medication_management")
        query = "SELECT * FROM medication_check_ins WHERE home_id=? AND subject_user_id=?"
        params: list[object] = [home_id, subject_id]
        if scheduled_from:
            query += " AND scheduled_for>=?"; params.append(scheduled_from.isoformat())
        if scheduled_to:
            query += " AND scheduled_for<=?"; params.append(scheduled_to.isoformat())
        query += " ORDER BY scheduled_for DESC"
        return {"data": db.many(query, tuple(params)), "subject_user_id": subject_id, "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/medication-plans/{plan_id}/check-ins")
    def medication_check_in(home_id: str, plan_id: str, body: MedicationCheckInIn, actor: Current):
        home_check(actor, home_id); publisher_block(actor)
        plan = medication_plan(home_id, plan_id)
        if actor["user_id"] != plan["subject_user_id"]:
            family_actor(actor)
        require_consent(home_id, plan["subject_user_id"], "medication_management")
        timestamp, scheduled = now_iso(), body.scheduled_for.replace(microsecond=0).isoformat()
        check_id = str(uuid.uuid4())
        db.execute("INSERT INTO medication_check_ins VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(plan_id,scheduled_for) DO UPDATE SET status=excluded.status, note=excluded.note, marked_by=excluded.marked_by, updated_at=excluded.updated_at", (check_id, home_id, plan_id, plan["subject_user_id"], scheduled, body.status, body.note, actor["user_id"], timestamp, timestamp))
        row = db.one("SELECT * FROM medication_check_ins WHERE plan_id=? AND scheduled_for=?", (plan_id, scheduled))
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
    def medication_reminders(home_id: str, actor: Current, day: str | None = None, subject_user_id: str | None = None):
        home_check(actor, home_id); publisher_block(actor)
        subject_id = subject_user_id or actor["user_id"]
        member(home_id, subject_id)
        if subject_id != actor["user_id"]:
            family_actor(actor)
        require_consent(home_id, subject_id, "medication_management")
        target_day = day or datetime.now(timezone.utc).date().isoformat()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", target_day):
            raise HTTPException(422, "day must use YYYY-MM-DD")
        plans = db.many("SELECT * FROM medication_plans WHERE home_id=? AND subject_user_id=? AND active=1 ORDER BY name", (home_id, subject_id))
        reminders = []
        for plan in plans:
            for slot in schedule_slots(plan["schedule"], target_day):
                scheduled = f"{target_day}T{slot}:00+00:00" if slot != "unscheduled" else f"{target_day}T00:00:00+00:00"
                status_row = db.one("SELECT status, note, updated_at FROM medication_check_ins WHERE plan_id=? AND scheduled_for=?", (plan["id"], scheduled))
                caregiver = db.one("SELECT display_name FROM users WHERE id=?", (plan["assigned_caregiver_id"],)) if plan.get("assigned_caregiver_id") else None
                reminders.append({"plan_id": plan["id"], "name": plan["name"], "dose": plan["dose"], "instructions": plan["instructions"], "schedule_rule": plan["schedule"], "scheduled_for": scheduled, "status": status_row["status"] if status_row else "pending", "note": status_row["note"] if status_row else "", "updated_at": status_row["updated_at"] if status_row else None, "assigned_caregiver_id": plan.get("assigned_caregiver_id"), "assigned_caregiver_name": caregiver["display_name"] if caregiver else None})
        return {"data": reminders, "subject_user_id": subject_id, "timezone": "UTC", "deterministic": True, "medical_advice": False}

    @app.post("/api/v1/homes/{home_id}/family-assistant")
    def family_assistant(home_id: str, body: FamilyAssistantIn, actor: Current):
        home_check(actor, home_id); family_actor(actor)
        subject = family_subject(home_id, actor, body.subject_user_id, "family_assistant")
        # Keep this context narrow: plans and their bounded check-in statuses
        # only. Do not pass events, frames, transcripts, or a household stream.
        plans = db.many("SELECT id, name, dose, schedule, instructions, active, version, assigned_caregiver_id FROM medication_plans WHERE home_id=? AND subject_user_id=? AND active=1 ORDER BY name LIMIT 50", (home_id, subject["id"]))
        checks = db.many("SELECT id, plan_id, scheduled_for, status, note, updated_at FROM medication_check_ins WHERE home_id=? AND subject_user_id=? ORDER BY scheduled_for DESC LIMIT 100", (home_id, subject["id"]))
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
        audit(actor, "assistant.family_summary", "user", subject["id"], home_id)
        return {"data": result, "degraded": degraded, "inference_status": lm.last_error if degraded else "ok", "subject_user_id": subject["id"], "context_scope": "medication plans and check-ins only", "medical_advice": False, "model_version": settings.effective_llm_model if not degraded else "rules-family-v1"}

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

    @app.post("/api/v1/homes/{home_id}/livekit/token")
    def livekit_token(home_id: str, actor: Current, body: LiveKitTokenIn | None = None):
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
        return {"url": settings.livekit_url, "token": livekit_jwt(settings.livekit_api_key, settings.livekit_api_secret, actor["user_id"], f"one-{home_id}", can_publish, can_subscribe), "expires_in": 600, "mode": "publish" if can_publish else "subscribe"}

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
        rows = db.many("SELECT artifact_key FROM room_maps WHERE home_id=? AND artifact_key IS NOT NULL", (home_id,))
        clips_for_home = db.many("SELECT id, object_key FROM clips WHERE home_id=?", (home_id,))
        db.execute("INSERT INTO deletion_requests VALUES (?,?,?,?,?,NULL)", (request_id, home_id, actor["user_id"], "processing", requested_at))
        cleanup_errors: list[str] = []
        for row in rows:
            try:
                store.delete(row["artifact_key"])
            except (OSError, ValueError) as exc:
                cleanup_errors.append(f"map:{row['artifact_key']}:{type(exc).__name__}")
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

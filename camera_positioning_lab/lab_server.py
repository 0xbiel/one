from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
API_BASE = os.getenv("ONE_LAB_API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
WORKER_BASE = os.getenv("ONE_LAB_WORKER_BASE", "http://127.0.0.1:8090").rstrip("/")


@dataclass
class LabSession:
    token: str | None = None
    home_id: str | None = None
    email: str | None = None


session = LabSession()
app = FastAPI(title="ONE Camera Positioning Lab", version="0.1.0")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)


class LabFrame(BaseModel):
    frame_base64: str = Field(min_length=100)
    width: int = Field(ge=160, le=4096)
    height: int = Field(ge=120, le=4096)


class LocalizeRequest(BaseModel):
    frames: list[LabFrame] = Field(min_length=1, max_length=16)
    fov_degrees: float | None = Field(default=None, ge=30.0, le=120.0)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(exc.code, detail or exc.reason) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(503, f"Local service unavailable: {exc}") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(502, "Local service returned a non-object response")
    return decoded


def _require_session() -> tuple[str, str]:
    if not session.token or not session.home_id:
        raise HTTPException(401, "Sign in to the lab first")
    return session.token, session.home_id


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(STATIC / "styles.css", media_type="text/css")


@app.get("/api/status")
def status() -> dict[str, Any]:
    worker: dict[str, Any]
    try:
        worker = _request_json(f"{WORKER_BASE}/health", timeout=3.0)
    except HTTPException as exc:
        worker = {"status": "unavailable", "detail": exc.detail}
    return {
        "api_base": API_BASE,
        "worker_base": WORKER_BASE,
        "worker": worker,
        "signed_in": bool(session.token and session.home_id),
        "home_id": session.home_id,
        "email": session.email,
    }


@app.post("/api/login")
def login(body: LoginRequest) -> dict[str, Any]:
    challenge = _request_json(
        f"{API_BASE}/auth/email/request",
        method="POST",
        payload={"email": body.email, "purpose": "login"},
    )
    code = challenge.get("dev_code")
    if not isinstance(code, str) or not code:
        raise HTTPException(409, "The local API is not exposing a development sign-in code")
    verified = _request_json(
        f"{API_BASE}/auth/email/verify",
        method="POST",
        payload={"email": body.email, "code": code},
    )
    token = verified.get("access_token")
    home_id = verified.get("home_id")
    if not isinstance(token, str) or not isinstance(home_id, str):
        raise HTTPException(502, "Sign-in response is missing its token or home")
    session.token = token
    session.home_id = home_id
    session.email = body.email
    me = _request_json(f"{API_BASE}/me", token=token)
    cameras = _request_json(f"{API_BASE}/homes/{home_id}/cameras", token=token)
    return {"me": me, "cameras": cameras.get("data", []), "home_id": home_id}


@app.get("/api/session")
def session_state() -> dict[str, Any]:
    token, home_id = _require_session()
    return {
        "me": _request_json(f"{API_BASE}/me", token=token),
        "cameras": _request_json(f"{API_BASE}/homes/{home_id}/cameras", token=token).get("data", []),
        "home_id": home_id,
    }


@app.get("/api/cameras/{camera_id}/readiness")
def readiness(camera_id: str) -> dict[str, Any]:
    token, home_id = _require_session()
    return _request_json(
        f"{API_BASE}/homes/{home_id}/cameras/{camera_id}/roomplan-readiness",
        token=token,
    )


@app.get("/api/cameras/{camera_id}/scene")
def scene(camera_id: str) -> dict[str, Any]:
    token, home_id = _require_session()
    return _request_json(
        f"{API_BASE}/homes/{home_id}/cameras/{camera_id}/roomplan-placement-preview",
        token=token,
    )


@app.get("/api/cameras/{camera_id}/history")
def history(camera_id: str) -> dict[str, Any]:
    token, home_id = _require_session()
    return _request_json(
        f"{API_BASE}/homes/{home_id}/cameras/{camera_id}/localization-history?limit=12",
        token=token,
    )


@app.post("/api/cameras/{camera_id}/localize")
def localize(camera_id: str, body: LocalizeRequest) -> dict[str, Any]:
    token, home_id = _require_session()
    # review_only is deliberate: the lab can exercise the full real solver
    # repeatedly without changing the active production camera registration.
    return _request_json(
        f"{API_BASE}/homes/{home_id}/cameras/{camera_id}/localize-roomplan",
        method="POST",
        payload={
            "frames": [frame.model_dump(mode="json") for frame in body.frames],
            "review_only": True,
            **({"fov_degrees": body.fov_degrees} if body.fov_degrees is not None else {}),
        },
        token=token,
        timeout=120.0,
    )

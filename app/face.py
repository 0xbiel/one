"""Consent-gated face-template validation and conservative matching."""

from __future__ import annotations

import math
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any


MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.08
IDENTITY_STABILITY_HITS = 3
IDENTITY_STABILITY_WINDOW = timedelta(seconds=4)
TEMPLATE_SCHEME = "cancelable-random-projection-v1"


def normalize_embedding(raw: Any) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not 8 <= len(raw) <= 2048:
        raise ValueError("face embedding has an invalid dimension")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("face embedding contains a non-finite value")
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-8:
        raise ValueError("face embedding cannot be zero")
    return [value / length for value in values]


def validate_embeddings(raw: Any, *, minimum: int = 3, maximum: int = 8) -> list[list[float]]:
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise ValueError(f"face enrollment requires {minimum} to {maximum} usable samples")
    embeddings = [normalize_embedding(item) for item in raw]
    dimension = len(embeddings[0])
    if any(len(item) != dimension for item in embeddings):
        raise ValueError("face enrollment samples must have the same dimension")
    return embeddings


def _digest(secret: bytes, token: str) -> bytes:
    return hmac.new(secret, token.encode("utf-8"), hashlib.sha256).digest()


def cancelable_template(
    embedding: list[float],
    *,
    secret: bytes,
    profile_id: str,
    transform_salt: str,
) -> list[float]:
    """Apply a profile-specific, revocable dimension-reducing biohash.

    The random sparse projection is derived from a deployment secret and a
    fresh per-enrollment salt. It is intentionally applied before the
    encrypted artifact is written, so the stored template is not a raw SFace
    vector. Re-enrollment rotates the salt and invalidates the old transform.
    """

    values = normalize_embedding(embedding)
    output_dimension = min(64, max(8, len(values) // 2))
    taps = min(8, max(2, len(values)))
    projected: list[float] = []
    for row in range(output_dimension):
        total = 0.0
        for tap in range(taps):
            digest = _digest(secret, f"{TEMPLATE_SCHEME}:{profile_id}:{transform_salt}:{row}:{tap}")
            index = int.from_bytes(digest[:4], "big") % len(values)
            sign = 1.0 if digest[4] & 1 else -1.0
            total += sign * values[index]
        projected.append(total / math.sqrt(taps))
    return normalize_embedding(projected)


def new_protected_template_payload(
    embeddings: list[list[float]],
    *,
    secret: bytes,
    profile_id: str,
) -> dict[str, Any]:
    transform_salt = secrets.token_hex(16)
    protected = [
        cancelable_template(
            embedding,
            secret=secret,
            profile_id=profile_id,
            transform_salt=transform_salt,
        )
        for embedding in embeddings
    ]
    return {
        "scheme": TEMPLATE_SCHEME,
        "transform_salt": transform_salt,
        "templates": protected,
    }


def protected_template_payload(payload: Any, *, minimum: int = 3, maximum: int = 8) -> tuple[list[list[float]], str] | None:
    if not isinstance(payload, dict) or payload.get("scheme") != TEMPLATE_SCHEME:
        return None
    salt = payload.get("transform_salt")
    if not isinstance(salt, str) or not 16 <= len(salt) <= 128:
        return None
    try:
        embeddings = validate_embeddings(payload.get("templates"), minimum=minimum, maximum=maximum)
    except ValueError:
        return None
    return embeddings, salt


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def profile_score(probe: list[float], templates: list[list[float]]) -> float:
    scores = sorted((cosine(probe, template) for template in templates), reverse=True)
    if not scores:
        return -1.0
    return float(median(scores[: min(3, len(scores))]))


def best_profile_match(
    probe: list[float],
    profiles: list[dict[str, Any]],
    *,
    transform_secret: bytes | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for profile in profiles:
        try:
            if transform_secret is not None:
                transform_salt = profile.get("transform_salt")
                profile_id = str(profile.get("profile_id") or profile.get("id") or "")
                if not isinstance(transform_salt, str) or not profile_id:
                    continue
                probe_for_profile = cancelable_template(
                    probe,
                    secret=transform_secret,
                    profile_id=profile_id,
                    transform_salt=transform_salt,
                )
            else:
                probe_for_profile = normalize_embedding(probe)
            templates = [normalize_embedding(item) for item in profile.get("embeddings", [])]
            score = profile_score(probe_for_profile, templates)
        except (TypeError, ValueError):
            continue
        candidates.append({"profile": profile, "score": score})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates or candidates[0]["score"] < MATCH_THRESHOLD:
        return None
    runner_up = candidates[1]["score"] if len(candidates) > 1 else -1.0
    if candidates[0]["score"] - runner_up < MATCH_MARGIN:
        return None
    return {
        **candidates[0]["profile"],
        "score": round(float(candidates[0]["score"]), 6),
        "runner_up_score": round(float(runner_up), 6),
    }


def stable_identity(
    state: dict[tuple[str, str, int], dict[str, Any]],
    key: tuple[str, str, int],
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    current = state.get(key)
    if candidate is None:
        state.pop(key, None)
        return None
    profile_id = str(candidate.get("profile_id") or candidate.get("id") or "")
    if not profile_id:
        state.pop(key, None)
        return None
    if current and current.get("profile_id") == profile_id:
        last_seen = current.get("last_seen")
        if isinstance(last_seen, datetime) and now - last_seen <= IDENTITY_STABILITY_WINDOW:
            current["hits"] = int(current.get("hits", 0)) + 1
        else:
            current["hits"] = 1
        current["last_seen"] = now
        current["score"] = candidate.get("score")
    else:
        current = {"profile_id": profile_id, "hits": 1, "last_seen": now, "score": candidate.get("score")}
        state[key] = current
    if int(current["hits"]) < IDENTITY_STABILITY_HITS:
        return None
    return {**candidate, "stability_hits": int(current["hits"])}

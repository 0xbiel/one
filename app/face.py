"""Consent-gated face-template validation and conservative matching."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any


MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.08
IDENTITY_STABILITY_HITS = 3
IDENTITY_STABILITY_WINDOW = timedelta(seconds=4)


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


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def profile_score(probe: list[float], templates: list[list[float]]) -> float:
    scores = sorted((cosine(probe, template) for template in templates), reverse=True)
    if not scores:
        return -1.0
    return float(median(scores[: min(3, len(scores))]))


def best_profile_match(probe: list[float], profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for profile in profiles:
        try:
            templates = [normalize_embedding(item) for item in profile.get("embeddings", [])]
            score = profile_score(probe, templates)
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

"""Conservative temporal fall-risk signals from tracked person boxes.

This is deliberately a preparation layer, not a medical or emergency detector.
It only emits a reviewable signal after an already-stable person track changes
from an upright-looking box to a low horizontal-looking box and remains there
for several frames. A pose model can replace this adapter later without
changing the event contract.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass(frozen=True)
class FallSignal:
    track_id: int
    observed_at: datetime
    confidence: float
    explanation: str
    metrics: dict[str, float | int]


@dataclass
class _TrackState:
    last_at: datetime
    last_center_y: float
    last_posture: str
    upright_center_y: float | None
    candidate_started_at: datetime | None = None
    candidate_hits: int = 0
    episode_open: bool = False


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


class FallDetectionTracker:
    """Track coarse posture transitions and emit at most one signal per episode."""

    def __init__(
        self,
        *,
        min_hits: int = 3,
        confirmation_window_seconds: float = 4.0,
        stale_after_seconds: float = 12.0,
    ):
        self.min_hits = max(2, min_hits)
        self.confirmation_window = timedelta(seconds=max(1.0, confirmation_window_seconds))
        self.stale_after = timedelta(seconds=max(4.0, stale_after_seconds))
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _posture(width: float, height: float) -> tuple[str, float]:
        aspect = height / width
        if aspect >= 1.20:
            return "upright", aspect
        if aspect <= 0.95:
            return "low", aspect
        return "transition", aspect

    def update(
        self,
        camera_id: str,
        item: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
        observed_at: datetime,
    ) -> FallSignal | None:
        """Update one stable person item and return only a newly confirmed signal."""

        if str(item.get("label") or "").strip().lower() != "person":
            return None
        track_id = item.get("track_id")
        if not isinstance(track_id, int) or track_id < 1 or frame_width <= 0 or frame_height <= 0:
            return None
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or float(confidence) < 0.55:
            return None
        bounds = _bbox(item.get("bbox"))
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds
        width, height = x2 - x1, y2 - y1
        if width < frame_width * 0.08 or height < frame_height * 0.08:
            return None
        center_y = ((y1 + y2) / 2) / frame_height
        posture, aspect = self._posture(width, height)
        timestamp = _utc(observed_at)
        key = (camera_id, track_id)

        with self._lock:
            for stale_key, state in list(self._tracks.items()):
                if timestamp - state.last_at > self.stale_after:
                    self._tracks.pop(stale_key, None)

            state = self._tracks.get(key)
            if state is None or timestamp - state.last_at > self.stale_after:
                self._tracks[key] = _TrackState(
                    last_at=timestamp,
                    last_center_y=center_y,
                    last_posture=posture,
                    upright_center_y=center_y if posture == "upright" else None,
                )
                return None

            if posture == "upright":
                if state.upright_center_y is None:
                    state.upright_center_y = center_y
                else:
                    state.upright_center_y = state.upright_center_y * 0.7 + center_y * 0.3
                state.candidate_started_at = None
                state.candidate_hits = 0
                state.episode_open = False

            reference_y = state.upright_center_y if state.upright_center_y is not None else state.last_center_y
            vertical_drop = center_y - reference_y
            within_window = (
                state.candidate_started_at is not None
                and timestamp - state.candidate_started_at <= self.confirmation_window
            )
            if posture == "low" and vertical_drop >= 0.07 and not state.episode_open:
                if not within_window:
                    state.candidate_started_at = timestamp
                    state.candidate_hits = 1
                else:
                    state.candidate_hits += 1
                if state.candidate_hits >= self.min_hits:
                    state.episode_open = True
                    shape_score = min(1.0, max(0.0, (1.15 - aspect) / 0.45))
                    drop_score = min(1.0, max(0.0, (vertical_drop - 0.07) / 0.20))
                    signal_confidence = min(
                        0.88,
                        0.55 + 0.12 * shape_score + 0.10 * drop_score + 0.08 * min(1.0, state.candidate_hits / self.min_hits),
                    )
                    signal = FallSignal(
                        track_id=track_id,
                        observed_at=timestamp,
                        confidence=round(signal_confidence, 3),
                        explanation=(
                            "A possible fall pattern was observed: the tracked person changed from an upright "
                            "to a low horizontal posture and remained there across multiple frames. "
                            "Please check in with them. This is a safety signal for human review, not a diagnosis."
                        ),
                        metrics={
                            "body_aspect_ratio": round(aspect, 3),
                            "vertical_drop": round(vertical_drop, 3),
                            "confirmation_frames": state.candidate_hits,
                        },
                    )
                    state.last_at = timestamp
                    state.last_center_y = center_y
                    state.last_posture = posture
                    return signal
            elif posture not in {"low", "transition"}:
                state.candidate_started_at = None
                state.candidate_hits = 0

            if posture == "low" and state.candidate_started_at is not None and not within_window and not state.episode_open:
                state.candidate_started_at = timestamp
                state.candidate_hits = 1

            state.last_at = timestamp
            state.last_center_y = center_y
            state.last_posture = posture
            return None

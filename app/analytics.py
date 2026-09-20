"""Bounded caregiver analytics derived from retained household records.

The analytics layer deliberately works on event and check-in metadata only.
It never receives frames, face templates, transcripts, or encrypted snapshot
bytes, which keeps the assistant context reviewable and privacy bounded.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _day(value: Any) -> str | None:
    parsed = _parse_timestamp(value)
    return parsed.date().isoformat() if parsed else None


def _trend(values: list[int]) -> str:
    non_zero = sum(values)
    if non_zero == 0 or len(values) < 4:
        return "unknown"
    midpoint = len(values) // 2
    earlier = sum(values[:midpoint])
    later = sum(values[midpoint:])
    if earlier == later:
        return "stable"
    if later > earlier:
        return "increasing"
    return "decreasing"


def _days(window_days: int, now: datetime) -> list[str]:
    return [(now.date() - timedelta(days=offset)).isoformat() for offset in range(window_days - 1, -1, -1)]


def _filter_clause(care_recipient_id: str | None, subject_user_id: str | None) -> tuple[str, tuple[str, ...]]:
    if care_recipient_id:
        return " AND care_recipient_id=?", (care_recipient_id,)
    if subject_user_id:
        return " AND care_recipient_id IS NULL AND subject_user_id=?", (subject_user_id,)
    return "", ()


def fall_analytics(db, home_id: str, *, window_days: int = 30, care_recipient_id: str | None = None) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window_days = max(7, min(int(window_days), 90))
    cutoff = (now - timedelta(days=window_days - 1)).isoformat()
    recipient_clause, recipient_params = _filter_clause(care_recipient_id, None)
    rows = db.many(
        f"SELECT id,status,confidence,explanation,first_seen_at,last_seen_at FROM events "
        f"WHERE home_id=? AND event_type='fall_suspected' AND last_seen_at>=?{recipient_clause} "
        "ORDER BY last_seen_at DESC LIMIT 250",
        (home_id, cutoff, *recipient_params),
    )
    days = _days(window_days, now)
    by_day = Counter(_day(row.get("last_seen_at")) for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    latest = rows[0] if rows else None
    return {
        "window_days": window_days,
        "total_signals": len(rows),
        "needs_review": status_counts.get("needs_review", 0),
        "reviewed": sum(status_counts.get(status, 0) for status in ("reviewed", "resolved", "dismissed")),
        "last_signal_at": latest.get("last_seen_at") if latest else None,
        "trend": _trend([by_day.get(day, 0) for day in days]),
        "by_day": [{"date": day, "count": by_day.get(day, 0)} for day in days],
        "recent": [
            {
                "id": row["id"],
                "status": row.get("status"),
                "confidence": row.get("confidence"),
                "explanation": row.get("explanation"),
                "occurred_at": row.get("last_seen_at") or row.get("first_seen_at"),
            }
            for row in rows[:10]
        ],
        "limitations": [
            "These are heuristic safety signals for caregiver review, not diagnoses.",
            "Counts depend on camera coverage, lighting, framing, and retention.",
        ],
    }


def daily_check_in_analytics(
    db,
    home_id: str,
    *,
    window_days: int = 30,
    care_recipient_id: str | None = None,
    subject_user_id: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window_days = max(7, min(int(window_days), 90))
    cutoff = (now - timedelta(days=window_days - 1)).isoformat()
    if care_recipient_id:
        target_clause, target_params = " AND care_recipient_id=?", (care_recipient_id,)
    elif subject_user_id:
        target_clause, target_params = " AND care_recipient_id IS NULL AND subject_user_id=?", (subject_user_id,)
    else:
        target_clause, target_params = "", ()
    rows = db.many(
        f"SELECT id,subject_user_id,care_recipient_id,status,trend,explanation,limitations,created_at,model_version "
        f"FROM summaries WHERE home_id=? AND created_at>=?{target_clause} ORDER BY created_at DESC LIMIT 250",
        (home_id, cutoff, *target_params),
    )
    days = _days(window_days, now)
    by_day = Counter(_day(row.get("created_at")) for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    latest = rows[0] if rows else None
    return {
        "window_days": window_days,
        "total": len(rows),
        "completed_today": sum(1 for row in rows if _day(row.get("created_at")) == now.date().isoformat()),
        "status_counts": dict(status_counts),
        "last_recorded_at": latest.get("created_at") if latest else None,
        "last_status": latest.get("status") if latest else None,
        "last_trend": latest.get("trend") if latest else None,
        "last_explanation": latest.get("explanation") if latest else None,
        "trend": _trend([by_day.get(day, 0) for day in days]),
        "by_day": [{"date": day, "count": by_day.get(day, 0)} for day in days],
        "recent": [
            {
                "id": row["id"],
                "status": row.get("status"),
                "trend": row.get("trend"),
                "explanation": row.get("explanation"),
                "recorded_at": row.get("created_at"),
            }
            for row in rows[:10]
        ],
        "limitations": [
            "Check-ins are human or assistant-supported observations compared with a bounded personal baseline.",
            "They are not a medical assessment and do not replace a caregiver conversation.",
        ],
    }


def care_analytics(
    db,
    home_id: str,
    *,
    window_days: int = 30,
    care_recipient_id: str | None = None,
    subject_user_id: str | None = None,
) -> dict:
    window_days = max(7, min(int(window_days), 90))
    fall = fall_analytics(db, home_id, window_days=window_days, care_recipient_id=care_recipient_id)
    check_in = daily_check_in_analytics(
        db,
        home_id,
        window_days=window_days,
        care_recipient_id=care_recipient_id,
        subject_user_id=subject_user_id,
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days - 1)).replace(microsecond=0).isoformat()
    event_filter = " AND care_recipient_id=?" if care_recipient_id else ""
    event_params = (home_id, cutoff, care_recipient_id) if care_recipient_id else (home_id, cutoff)
    event_rows = db.many(
        f"SELECT event_type, COUNT(*) AS count FROM events WHERE home_id=? AND last_seen_at>=?{event_filter} GROUP BY event_type",
        event_params,
    )
    return {
        "window_days": window_days,
        "fall": fall,
        "daily_check_in": check_in,
        "event_counts": {str(row["event_type"]): int(row["count"]) for row in event_rows},
        "assistant_context": {
            "includes": ["daily_check_in_summary", "fall_safety_analytics", "medication_records"],
            "excludes": ["raw_frames", "face_templates", "event_snapshot_bytes", "unbounded_transcripts"],
        },
        "limitations": [
            "ONE reports household observations and review prompts, not diagnoses.",
            "Detection quality depends on camera coverage, consent, lighting, and the selected time window.",
        ],
    }

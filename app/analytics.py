"""Bounded caregiver analytics derived from retained household records.

The analytics layer deliberately works on event and check-in metadata only.
It never receives frames, face templates, transcripts, or encrypted snapshot
bytes, which keeps the assistant context reviewable and privacy bounded.
"""

from __future__ import annotations

from collections import Counter
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _day(value: Any, timezone_name: str = "UTC") -> str | None:
    parsed = _parse_timestamp(value)
    return parsed.astimezone(ZoneInfo(timezone_name)).date().isoformat() if parsed else None


def _trend(values: list[int]) -> str:
    non_zero = sum(values)
    if non_zero == 0 or len(values) < 4:
        return "unknown"
    midpoint = len(values) // 2
    # Compare per-day rates, not totals from unequal halves (odd windows).
    # Cross-multiplication keeps the comparison exact for integer counts.
    earlier = sum(values[:midpoint]) * (len(values) - midpoint)
    later = sum(values[midpoint:]) * midpoint
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


def fall_analytics(db, home_id: str, *, window_days: int = 30, care_recipient_id: str | None = None, now: datetime | None = None, timezone_name: str = "UTC", subject_user_id: str | None = None) -> dict:
    now = ((now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now) if now is not None else datetime.now(timezone.utc)).astimezone(ZoneInfo(timezone_name)).replace(microsecond=0)
    window_days = max(7, min(int(window_days), 90))
    cutoff = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    recipient_clause, recipient_params = _filter_clause(care_recipient_id, None)
    rows = db.many(
        f"SELECT * FROM events "
        f"WHERE home_id=? AND event_type='fall_suspected' {recipient_clause} "
        "ORDER BY last_seen_at DESC LIMIT 2000",
        (home_id, *recipient_params),
    )
    rows = [] if subject_user_id and not care_recipient_id else bounded_rows(rows, "last_seen_at", cutoff, now)
    days = _days(window_days, now)
    by_day = Counter(_day(row.get("last_seen_at"), timezone_name) for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    latest = rows[0] if rows else None
    return {
        "window_days": window_days,
        "total_signals": len(rows),
        "needs_review": status_counts.get("needs_review", 0),
        "reviewed": sum(status_counts.get(status, 0) for status in ("reviewed", "resolved", "dismissed")),
        "last_signal_at": latest.get("last_seen_at") if latest else None,
        "trend": "unknown",
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
    now: datetime | None = None,
    timezone_name: str = "UTC",
) -> dict:
    now = ((now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now) if now is not None else datetime.now(timezone.utc)).astimezone(ZoneInfo(timezone_name)).replace(microsecond=0)
    window_days = max(7, min(int(window_days), 90))
    cutoff = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    if care_recipient_id:
        target_clause, target_params = " AND care_recipient_id=?", (care_recipient_id,)
    elif subject_user_id:
        target_clause, target_params = " AND care_recipient_id IS NULL AND subject_user_id=?", (subject_user_id,)
    else:
        target_clause, target_params = "", ()
    rows = db.many(
        f"SELECT * "
        f"FROM summaries WHERE home_id=? {target_clause} ORDER BY created_at DESC LIMIT 2000",
        (home_id, *target_params),
    )
    rows = bounded_rows(rows, "created_at", cutoff, now)
    days = _days(window_days, now)
    by_day = Counter(_day(row.get("created_at"), timezone_name) for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    latest = rows[0] if rows else None
    return {
        "window_days": window_days,
        "total": len(rows),
        "completed_today": sum(1 for row in rows if _day(row.get("created_at"), timezone_name) == now.date().isoformat()),
        "status_counts": dict(status_counts),
        "last_recorded_at": latest.get("created_at") if latest else None,
        "last_status": latest.get("status") if latest else None,
        "last_trend": latest.get("trend") if latest else None,
        "last_explanation": latest.get("explanation") if latest else None,
        "trend": "unknown",
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
    now: datetime | None = None,
    timezone_name: str = "UTC",
) -> dict:
    window_days = max(7, min(int(window_days), 90))
    now = ((now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now) if now is not None else datetime.now(timezone.utc)).astimezone(ZoneInfo(timezone_name)).replace(microsecond=0)
    fall = fall_analytics(db, home_id, window_days=window_days, care_recipient_id=care_recipient_id, now=now, timezone_name=timezone_name, subject_user_id=subject_user_id)
    check_in = daily_check_in_analytics(
        db,
        home_id,
        window_days=window_days,
        care_recipient_id=care_recipient_id,
        subject_user_id=subject_user_id,
        now=now, timezone_name=timezone_name,
    )
    cutoff = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    event_filter = " AND care_recipient_id=?" if care_recipient_id else ""
    event_params = (home_id, care_recipient_id) if care_recipient_id else (home_id,)
    event_rows = db.many(
        f"SELECT * FROM events WHERE home_id=? {event_filter} ORDER BY last_seen_at DESC LIMIT 2000",
        event_params,
    )
    event_rows = [] if subject_user_id and not care_recipient_id else bounded_rows(event_rows, "last_seen_at", cutoff, now)
    from .assistant_context import build_assistant_context
    context = build_assistant_context(db, home_id, care_recipient_id=care_recipient_id, subject_user_id=subject_user_id, window_days=window_days, now=now, timezone_name=timezone_name)
    # Analytics counts retained rows independently of the assistant evidence cap.
    # Read one overflow row so callers know when counts are only a lower bound.
    aggregate_rows = []
    if not (subject_user_id and not care_recipient_id):
        try:
            aggregate_rows = db.many(
                f"SELECT * FROM observations WHERE home_id=? {event_filter} "
                "ORDER BY observed_at DESC LIMIT 2001", event_params,
            )
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
    aggregate_truncated = len(aggregate_rows) >= 2000
    observations = bounded_rows(aggregate_rows[:2000], "observed_at", cutoff, now)
    counts = Counter(_day(row["observed_at"], timezone_name) for row in observations)
    days = _days(window_days, now)
    coverage = dict(context["missing_coverage"])
    coverage.update({
        "status": "partial" if observations else "unobserved",
        "observed_days": len(counts),
        "unobserved_days": [day for day in days if not counts[day]],
        "truncated": aggregate_truncated,
        "assistant_evidence_truncated": context["missing_coverage"]["truncated"],
    })
    return {
        "window_days": window_days,
        "window": context["window"],
        "activity": {"total_observations": len(observations), "truncated": aggregate_truncated, "count_is_lower_bound": aggregate_truncated, "trend": "unknown", "by_day": [{"date": day, "count": counts[day]} for day in days], "recent": context["observations"][:10]},
        "coverage": coverage,
        "synthetic_provenance": context["synthetic_provenance"],
        "fall": fall,
        "daily_check_in": check_in,
        "event_counts": dict(Counter(str(row["event_type"]) for row in event_rows)),
        "assistant_context": {
            "includes": ["daily_check_in_summary", "fall_safety_analytics", "medication_records"],
            "excludes": ["raw_frames", "face_templates", "event_snapshot_bytes", "unbounded_transcripts"],
        },
        "limitations": [
            "ONE reports household observations and review prompts, not diagnoses.",
            "Detection quality depends on camera coverage, consent, lighting, and the selected time window.",
        ],
    }



def bounded_rows(rows, timestamp_key, cutoff, now):
    """Normalize offsets, reject expired/future rows and deduplicate source IDs."""
    start = _parse_timestamp(cutoff)
    seen = set()
    result = []
    for row in sorted(rows, key=lambda r: _parse_timestamp(r.get(timestamp_key)) or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        timestamp = _parse_timestamp(row.get(timestamp_key))
        expires = _parse_timestamp(row.get("expires_at"))
        if timestamp is None or timestamp < start or timestamp > now or (expires is not None and expires <= now):
            continue
        if row.get("id") in seen:
            continue
        seen.add(row.get("id"))
        result.append(row)
    return result

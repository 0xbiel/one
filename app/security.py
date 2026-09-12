import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone


def iso_after(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).replace(microsecond=0).isoformat()


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def safe_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def new_token(length: int = 32) -> str:
    return secrets.token_urlsafe(length)


def new_pairing_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def normalize_email(value: str) -> str:
    """Return the stable identifier form used for account and invite lookups."""
    normalized = value.strip().casefold()
    if not re.fullmatch(r"[^@\s]{1,64}@[A-Za-z0-9][A-Za-z0-9.-]{0,252}", normalized):
        raise ValueError("Invalid email address")
    return normalized


def expired(value: str) -> bool:
    return datetime.fromisoformat(value) <= datetime.now(timezone.utc)

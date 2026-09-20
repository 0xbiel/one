"""Ephemeral camera buffering and authenticated encrypted local clip storage."""
from __future__ import annotations

import base64
import secrets
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class BufferedFrame:
    captured_at: datetime
    data: bytes


class FrameRingBuffer:
    def __init__(self, max_seconds: float = 15, max_frames: int = 300):
        self.max_seconds, self.frames = max_seconds, deque(maxlen=max_frames)

    def append(self, captured_at: datetime, data: bytes) -> None:
        self.frames.append(BufferedFrame(captured_at, bytes(data)))
        cutoff = captured_at - timedelta(seconds=self.max_seconds)
        while self.frames and self.frames[0].captured_at < cutoff:
            self.frames.popleft()

    def snapshot(self, start: datetime, end: datetime) -> list[BufferedFrame]:
        return [frame for frame in self.frames if start <= frame.captured_at <= end]


def image_content_type(data: bytes) -> str | None:
    """Return the small allow-list of image types accepted for event snapshots."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


class EncryptedLocalClipStore:
    """Envelope-encrypted clip bytes; callers must enforce authorization.

    Every clip gets a fresh AES-256-GCM data key. The deployment master key
    only wraps that data key, so compromise of one encrypted object does not
    reuse a content key across the household's clips.
    """

    _FORMAT = b"ONECLIP2"

    def __init__(self, root: Path, key: bytes | None = None):
        if key is not None and len(key) != 32:
            raise ValueError("clip encryption key must be exactly 32 bytes")
        self.root, self.key = root, key or secrets.token_bytes(32)
        self.root.mkdir(parents=True, exist_ok=True)

    def _put_encrypted(self, key: str, aad: bytes, plaintext: bytes) -> str:
        if not plaintext:
            raise ValueError("media cannot be empty")
        data_key = secrets.token_bytes(32)
        data_nonce = secrets.token_bytes(12)
        wrapped_nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)
        wrapped_key = AESGCM(self.key).encrypt(wrapped_nonce, data_key, aad + b":data-key")
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._FORMAT + wrapped_nonce + wrapped_key + data_nonce + ciphertext)
        return key

    def _get_encrypted(self, key: str, aad: bytes, media_label: str = "media") -> bytes:
        raw = (self.root / key).read_bytes()
        # The shortest valid envelope is the format marker, two nonces, a
        # wrapped 32-byte key plus its GCM tag, and a ciphertext/tag pair.
        minimum = len(self._FORMAT) + 12 + 48 + 12 + 16
        if len(raw) < minimum or not raw.startswith(self._FORMAT):
            raise ValueError(f"unsupported {media_label} encryption format")
        offset = len(self._FORMAT)
        wrapped_nonce = raw[offset:offset + 12]
        offset += 12
        wrapped_key = raw[offset:offset + 48]
        offset += 48
        data_nonce = raw[offset:offset + 12]
        ciphertext = raw[offset + 12:]
        if len(wrapped_nonce) != 12 or len(wrapped_key) != 48 or len(data_nonce) != 12 or len(ciphertext) < 16:
            raise ValueError(f"truncated {media_label} encryption envelope")
        data_key = AESGCM(self.key).decrypt(wrapped_nonce, wrapped_key, aad + b":data-key")
        return AESGCM(data_key).decrypt(data_nonce, ciphertext, aad)

    def put(self, home_id: str, clip_id: str, plaintext: bytes, expires_at: datetime) -> str:
        aad = f"one:{home_id}:{clip_id}".encode()
        key = f"clips/{home_id}/{clip_id}.bin"
        return self._put_encrypted(key, aad, plaintext)

    def get(self, home_id: str, clip_id: str) -> bytes:
        aad = f"one:{home_id}:{clip_id}".encode()
        return self._get_encrypted(f"clips/{home_id}/{clip_id}.bin", aad, "clip")

    def delete(self, home_id: str, clip_id: str) -> None:
        path = self.root / f"clips/{home_id}/{clip_id}.bin"
        if path.exists(): path.unlink()

    def put_snapshot(self, home_id: str, event_id: str, plaintext: bytes) -> str:
        """Store one encrypted image captured for a safety event."""
        return self._put_encrypted(
            f"snapshots/{home_id}/{event_id}.bin",
            f"one:snapshot:{home_id}:{event_id}".encode(),
            plaintext,
        )

    def get_snapshot(self, home_id: str, event_id: str) -> bytes:
        return self._get_encrypted(
            f"snapshots/{home_id}/{event_id}.bin",
            f"one:snapshot:{home_id}:{event_id}".encode(),
        )

    def delete_snapshot(self, home_id: str, event_id: str) -> None:
        path = self.root / f"snapshots/{home_id}/{event_id}.bin"
        if path.exists():
            path.unlink()


class EncryptedLocalTemplateStore:
    """Envelope-encrypted derived biometric templates.

    Templates are deliberately kept in a separate store from media. The
    deployment key is mandatory for this store: silently generating a new key
    would make every enrolled profile unusable after an API restart.
    """

    _FORMAT = b"ONETPL1"

    def __init__(self, root: Path, key: bytes | None = None):
        if key is not None and len(key) != 32:
            raise ValueError("biometric encryption key must be exactly 32 bytes")
        self.root, self.key = root, key
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.key is not None

    def _path(self, home_id: str, profile_id: str) -> Path:
        return self.root / "face-profiles" / home_id / f"{profile_id}.bin"

    def put(self, home_id: str, profile_id: str, plaintext: bytes) -> str:
        if not self.key:
            raise RuntimeError("biometric encryption is not configured")
        if not plaintext:
            raise ValueError("biometric template cannot be empty")
        aad = f"one:face-profile:{home_id}:{profile_id}".encode()
        data_key = secrets.token_bytes(32)
        data_nonce = secrets.token_bytes(12)
        wrapped_nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)
        wrapped_key = AESGCM(self.key).encrypt(wrapped_nonce, data_key, aad + b":data-key")
        path = self._path(home_id, profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._FORMAT + wrapped_nonce + wrapped_key + data_nonce + ciphertext)
        return f"face-profiles/{home_id}/{profile_id}.bin"

    def get(self, home_id: str, profile_id: str) -> bytes:
        if not self.key:
            raise RuntimeError("biometric encryption is not configured")
        raw = self._path(home_id, profile_id).read_bytes()
        minimum = len(self._FORMAT) + 12 + 48 + 12 + 16
        if len(raw) < minimum or not raw.startswith(self._FORMAT):
            raise ValueError("unsupported biometric template format")
        offset = len(self._FORMAT)
        wrapped_nonce = raw[offset:offset + 12]
        offset += 12
        wrapped_key = raw[offset:offset + 48]
        offset += 48
        data_nonce = raw[offset:offset + 12]
        ciphertext = raw[offset + 12:]
        if len(ciphertext) < 16:
            raise ValueError("truncated biometric template envelope")
        aad = f"one:face-profile:{home_id}:{profile_id}".encode()
        data_key = AESGCM(self.key).decrypt(wrapped_nonce, wrapped_key, aad + b":data-key")
        return AESGCM(data_key).decrypt(data_nonce, ciphertext, aad)

    def delete(self, home_id: str, profile_id: str) -> None:
        path = self._path(home_id, profile_id)
        if path.exists():
            path.unlink()


def clip_from_ring(buffer: FrameRingBuffer, event_at: datetime, pre_seconds: float = 5, post_seconds: float = 0) -> bytes:
    frames = buffer.snapshot(event_at - timedelta(seconds=pre_seconds), event_at + timedelta(seconds=post_seconds))
    return b"".join(frame.data for frame in frames)

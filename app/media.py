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

    def put(self, home_id: str, clip_id: str, plaintext: bytes, expires_at: datetime) -> str:
        if not plaintext: raise ValueError("clip cannot be empty")
        aad = f"one:{home_id}:{clip_id}".encode()
        data_key = secrets.token_bytes(32)
        data_nonce = secrets.token_bytes(12)
        wrapped_nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)
        wrapped_key = AESGCM(self.key).encrypt(wrapped_nonce, data_key, aad + b":data-key")
        key = f"clips/{home_id}/{clip_id}.bin"
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._FORMAT + wrapped_nonce + wrapped_key + data_nonce + ciphertext)
        return key

    def get(self, home_id: str, clip_id: str) -> bytes:
        path = self.root / f"clips/{home_id}/{clip_id}.bin"
        raw = path.read_bytes()
        aad = f"one:{home_id}:{clip_id}".encode()
        # The shortest valid envelope is the format marker, two nonces, a
        # wrapped 32-byte key plus its GCM tag, and a ciphertext/tag pair.
        minimum = len(self._FORMAT) + 12 + 48 + 12 + 16
        if len(raw) < minimum or not raw.startswith(self._FORMAT):
            raise ValueError("unsupported clip encryption format")
        offset = len(self._FORMAT)
        wrapped_nonce = raw[offset:offset + 12]
        offset += 12
        wrapped_key = raw[offset:offset + 48]
        offset += 48
        data_nonce = raw[offset:offset + 12]
        ciphertext = raw[offset + 12:]
        if len(wrapped_nonce) != 12 or len(wrapped_key) != 48 or len(data_nonce) != 12 or len(ciphertext) < 16:
            raise ValueError("truncated clip encryption envelope")
        data_key = AESGCM(self.key).decrypt(wrapped_nonce, wrapped_key, aad + b":data-key")
        return AESGCM(data_key).decrypt(data_nonce, ciphertext, aad)

    def delete(self, home_id: str, clip_id: str) -> None:
        path = self.root / f"clips/{home_id}/{clip_id}.bin"
        if path.exists(): path.unlink()


def clip_from_ring(buffer: FrameRingBuffer, event_at: datetime, pre_seconds: float = 5, post_seconds: float = 0) -> bytes:
    frames = buffer.snapshot(event_at - timedelta(seconds=pre_seconds), event_at + timedelta(seconds=post_seconds))
    return b"".join(frame.data for frame in frames)

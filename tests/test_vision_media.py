from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.integrations import livekit_jwt, verify_livekit_webhook
from app.media import EncryptedLocalClipStore, FrameRingBuffer, clip_from_ring
from app.storage import LocalObjectStore
from app.vision import Calibration, Detection, DeterministicDemoDetector, Frame, TemporalStabilityTracker, project_detection


def test_temporal_tracker_requires_stable_hits_and_demo_is_deterministic():
    at = datetime.now(timezone.utc)
    frame = Frame("cam", b"same-frame", 640, 480, at)
    detector = DeterministicDemoDetector()
    first = detector.detect(frame, ["cup"])[0]
    assert detector.detect(frame, ["cup"]) == [first]
    tracker = TemporalStabilityTracker(min_hits=3, iou_threshold=0.1)
    assert tracker.update([first]) == []
    assert tracker.update([Detection(first.label, first.confidence, first.bbox, at + timedelta(milliseconds=300))]) == []
    assert tracker.update([Detection(first.label, first.confidence, first.bbox, at + timedelta(milliseconds=600))])


def test_projection_is_approximate_and_has_zone_fallback():
    at = datetime.now(timezone.utc)
    frame = Frame("cam", b"x", 1000, 1000, at)
    detection = Detection("keys", 0.8, (100, 100, 200, 200), at)
    fallback = project_detection(detection, frame, None)
    assert fallback.world_xyz is None and fallback.zone == "top-left" and fallback.quality == "zone-fallback"
    calibration = Calibration(500, 500, 500, 500, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), 0.2)
    projected = project_detection(detection, frame, calibration, depth_m=2)
    assert projected.world_xyz is not None and projected.uncertainty_m > 0 and projected.quality == "calibrated-depth"


def test_encrypted_clip_round_trip_tamper_rejected(tmp_path: Path):
    store = EncryptedLocalClipStore(tmp_path, key=b"k" * 32)
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    key = store.put("home", "clip", b"private-video", expires)
    assert key == "clips/home/clip.bin" and store.get("home", "clip") == b"private-video"
    path = tmp_path / key
    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))
    with pytest.raises(Exception):
        store.get("home", "clip")


def test_encrypted_clip_rejects_truncated_envelope(tmp_path: Path):
    store = EncryptedLocalClipStore(tmp_path, key=b"k" * 32)
    path = tmp_path / store.put("home", "clip", b"private-video", datetime.now(timezone.utc) + timedelta(days=7))
    path.write_bytes(b"ONECLIP2" + b"short")
    with pytest.raises(ValueError, match="unsupported clip encryption format"):
        store.get("home", "clip")


def test_object_store_confines_keys_to_root(tmp_path: Path):
    store = LocalObjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_json("../outside.json", {"private": True})
    with pytest.raises(ValueError):
        store.delete("/tmp/outside.json")


def test_ring_buffer_clip_is_bounded():
    at = datetime.now(timezone.utc)
    ring = FrameRingBuffer(max_seconds=2, max_frames=10)
    for i in range(4): ring.append(at + timedelta(seconds=i), f"f{i}".encode())
    assert [f.data for f in ring.snapshot(at + timedelta(seconds=2), at + timedelta(seconds=3))] == [b"f2", b"f3"]
    assert clip_from_ring(ring, at + timedelta(seconds=3), pre_seconds=1) == b"f2f3"


def test_livekit_webhook_signature_and_body_digest():
    # The helper accepts a standard HS256 JWT generated with the same primitives.
    token = livekit_jwt("key", "secret", "webhook", "one-home", False, False)
    claims = verify_livekit_webhook(f"Bearer {token}", b"{}", "key", "secret")
    assert claims["iss"] == "key"
    with pytest.raises(ValueError): verify_livekit_webhook("Bearer invalid", b"{}", "key", "secret")

from datetime import datetime, timedelta, timezone

from app.fall import FallDetectionTracker


def _item(bbox, *, track_id=1):
    return {"label": "person", "confidence": 0.92, "bbox": bbox, "track_id": track_id}


def test_fall_signal_requires_posture_change_and_persistent_low_posture():
    tracker = FallDetectionTracker()
    start = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = (220, 80, 420, 460)
    low = (150, 250, 490, 390)

    assert tracker.update("camera", _item(upright), frame_width=640, frame_height=480, observed_at=start) is None
    assert tracker.update("camera", _item(upright), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=1)) is None
    assert tracker.update("camera", _item(low), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=2)) is None
    assert tracker.update("camera", _item(low), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=3)) is None
    signal = tracker.update("camera", _item(low), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=4))

    assert signal is not None
    assert signal.track_id == 1
    assert 0.55 <= signal.confidence <= 0.88
    assert signal.metrics["confirmation_frames"] == 3
    assert tracker.update("camera", _item(low), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=5)) is None


def test_fall_signal_does_not_guess_from_an_already_low_track():
    tracker = FallDetectionTracker()
    start = datetime(2026, 9, 20, tzinfo=timezone.utc)
    low = (150, 250, 490, 390)

    for offset in range(6):
        assert tracker.update("camera", _item(low), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=offset)) is None

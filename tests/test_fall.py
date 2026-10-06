from datetime import datetime, timedelta, timezone

from app.fall import FallDetectionTracker
from app.vision import Detection, TemporalStabilityTracker


def _item(bbox, *, track_id=1, confidence=0.92):
    x1, y1, x2, y2 = bbox
    upright = (y2 - y1) / max(1, x2 - x1) > 1.2
    template = (
        [(0.50, 0.05), (0.45, 0.07), (0.55, 0.07), (0.38, 0.10), (0.62, 0.10),
         (0.38, 0.35), (0.62, 0.35), (0.25, 0.52), (0.75, 0.52), (0.15, 0.68),
         (0.85, 0.68), (0.42, 0.65), (0.58, 0.65), (0.34, 0.84), (0.66, 0.84),
         (0.30, 0.98), (0.70, 0.98)]
        if upright
        else
        [(0.08, 0.50), (0.10, 0.45), (0.10, 0.55), (0.14, 0.38), (0.14, 0.62),
         (0.30, 0.45), (0.40, 0.45), (0.46, 0.30), (0.50, 0.68), (0.56, 0.24),
         (0.58, 0.75), (0.62, 0.52), (0.70, 0.52), (0.80, 0.38), (0.82, 0.66),
         (0.98, 0.35), (0.98, 0.70)]
    )
    pose_keypoints = [[x1 + (x2 - x1) * px, y1 + (y2 - y1) * py, 0.95] for px, py in template]
    return {"label": "person", "confidence": confidence, "bbox": bbox, "track_id": track_id, "pose_keypoints": pose_keypoints}


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


def test_fall_signal_accepts_a_low_confidence_person_after_temporal_confirmation():
    tracker = FallDetectionTracker()
    start = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = (220, 80, 420, 460)
    low = (150, 250, 490, 390)

    assert tracker.update("camera", _item(upright, confidence=0.35), frame_width=640, frame_height=480, observed_at=start) is None
    assert tracker.update("camera", _item(upright, confidence=0.35), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=1)) is None
    assert tracker.update("camera", _item(low, confidence=0.35), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=2)) is None
    assert tracker.update("camera", _item(low, confidence=0.35), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=3)) is None
    signal = tracker.update("camera", _item(low, confidence=0.35), frame_width=640, frame_height=480, observed_at=start + timedelta(seconds=4))

    assert signal is not None
    assert signal.metrics["detection_confidence"] == 0.35


def test_person_track_survives_upright_to_horizontal_box_change():
    at = datetime.now(timezone.utc)
    tracker = TemporalStabilityTracker(min_hits=1, iou_threshold=0.2)
    upright = Detection("person", 0.35, (220, 80, 420, 460), at)
    low = Detection("person", 0.35, (150, 250, 490, 390), at + timedelta(milliseconds=300))

    first = tracker.update([upright])
    second = tracker.update([low])

    assert first[0].track_id == second[0].track_id


def test_fall_signal_requires_pose_keypoints_instead_of_a_box_only():
    tracker = FallDetectionTracker()
    at = datetime.now(timezone.utc)
    box_only = {"label": "person", "confidence": 0.95, "bbox": (220, 80, 420, 460), "track_id": 1}

    assert tracker.update("camera", box_only, frame_width=640, frame_height=480, observed_at=at) is None


def test_production_fall_tracker_requires_a_learned_score():
    tracker = FallDetectionTracker(require_learned_model=True)
    at = datetime.now(timezone.utc)
    upright = _item((220, 80, 420, 460))
    low = _item((150, 250, 490, 390))
    assert tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at) is None
    assert tracker.update("camera", {**upright, "fall_sequence_probability": 0.90}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=2)) is None
    for offset in (3, 4, 5):
        signal = tracker.update(
            "camera",
            {**low, "fall_sequence_probability": 0.90, "fall_sequence_threshold": 0.56, "fall_sequence_model_version": "test-stgcn-tcn", "fall_sequence_score_fresh": True},
            frame_width=640,
            frame_height=480,
            observed_at=at + timedelta(seconds=offset),
        )
    assert signal is not None
    assert signal.metrics["fall_sequence_model_version"] == "test-stgcn-tcn"


def test_fall_tracker_refractory_suppresses_duplicate_reclassification_alerts():
    tracker = FallDetectionTracker(require_learned_model=True, learned_threshold=0.56, refractory_seconds=10)
    at = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = _item((220, 80, 420, 460))
    low = {**_item((150, 250, 490, 390)), "fall_sequence_probability": 0.90, "fall_sequence_threshold": 0.56, "fall_sequence_score_fresh": True}

    tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    tracker.update("camera", {**upright, "fall_sequence_probability": 0.90}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1))
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=2)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=3)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=4)) is not None

    # A short upright/low reclassification sequence is still the same event.
    assert tracker.update("camera", {**upright, "fall_sequence_probability": 0.90}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=5)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=6)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=7)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=8)) is None


def test_cached_learned_scores_do_not_count_as_new_fall_evidence():
    tracker = FallDetectionTracker(require_learned_model=True, learned_threshold=0.65)
    at = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = _item((220, 80, 420, 460))
    low = {**_item((150, 250, 490, 390)), "fall_sequence_probability": 0.90, "fall_sequence_threshold": 0.65}

    tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    assert tracker.update("camera", {**low, "fall_sequence_score_fresh": True}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", {**low, "fall_sequence_score_fresh": False}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=2)) is None
    assert tracker.update("camera", {**low, "fall_sequence_score_fresh": False}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=3)) is None
    assert tracker.update("camera", {**low, "fall_sequence_score_fresh": True}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=4)) is None
    assert tracker.update("camera", {**low, "fall_sequence_score_fresh": True}, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=5)) is not None


def test_replayed_timestamp_cannot_advance_fall_confirmation():
    tracker = FallDetectionTracker(require_learned_model=True, learned_threshold=0.65)
    at = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = _item((220, 80, 420, 460))
    low = {**_item((150, 250, 490, 390)), "fall_sequence_probability": 0.90, "fall_sequence_threshold": 0.65, "fall_sequence_score_fresh": True}

    tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=2)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=3)) is not None


def test_fall_confirmation_requires_minimum_elapsed_time():
    tracker = FallDetectionTracker(
        require_learned_model=True,
        learned_threshold=0.65,
        min_hits=2,
        minimum_confirmation_seconds=0.75,
    )
    at = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = _item((220, 80, 420, 460))
    low = {**_item((150, 250, 490, 390)), "fall_sequence_probability": 0.90, "fall_sequence_threshold": 0.65, "fall_sequence_score_fresh": True}

    tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1)) is None
    assert tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1.2)) is None
    signal = tracker.update("camera", low, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1.8))
    assert signal is not None
    assert signal.metrics["confirmation_seconds"] >= 0.75


def test_fall_debug_status_explains_fresh_score_and_confirmation_gate():
    tracker = FallDetectionTracker(require_learned_model=True, min_hits=2)
    at = datetime(2026, 9, 20, tzinfo=timezone.utc)
    upright = {**_item((220, 80, 420, 460)), "fall_sequence_probability": 0.9,
               "fall_sequence_threshold": 0.65, "fall_sequence_score_fresh": True}
    low_cached = {**_item((150, 250, 490, 390)), "fall_sequence_probability": 0.9,
                  "fall_sequence_threshold": 0.65, "fall_sequence_score_fresh": False}

    tracker.update("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    upright_debug = tracker.debug_status("camera", upright, frame_width=640, frame_height=480, observed_at=at)
    assert upright_debug["reason"] == "posture_not_low"
    assert upright_debug["visible_keypoints"] == 17

    tracker.update("camera", low_cached, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1))
    cached_debug = tracker.debug_status("camera", low_cached, frame_width=640, frame_height=480, observed_at=at + timedelta(seconds=1))
    assert cached_debug["reason"] == "waiting_for_fresh_learned_score"
    assert cached_debug["confirmation_hits"] == 0

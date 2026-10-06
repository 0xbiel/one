from pathlib import Path
from types import SimpleNamespace

import torch

from geometry_service.fall_sequence import (
    FALL_SEQUENCE_FORMAT,
    SkeletonSTGCNTCN,
    FallSequenceEngine,
    sequence_features,
)


def _frames(count=15):
    return [
        {
            "bbox": [100.0, 50.0, 300.0, 450.0],
            "keypoints": [[150.0 + index * 2.0, 70.0 + joint * 18.0, 0.9] for joint in range(17)],
        }
        for index in range(count)
    ]


def test_sequence_features_and_model_use_coco_17_temporal_shape():
    features = sequence_features(_frames(), width=640, height=480)
    assert features.shape == (15, 17, 3)
    model = SkeletonSTGCNTCN()
    logits = model(torch.from_numpy(features).unsqueeze(0))
    assert logits.shape == (1, 2)


def test_artifact_round_trip_scores_without_raw_frames(tmp_path: Path):
    model = SkeletonSTGCNTCN()
    artifact_path = tmp_path / "fall.pt"
    torch.save(
        {
            "format": FALL_SEQUENCE_FORMAT,
            "model_version": "test-stgcn-tcn",
            "window_size": 15,
            "classes": ["non_fall", "fall"],
            "model_config": {"input_channels": 3, "hidden_channels": 32, "class_count": 2},
            "feature_mean": [0.5, 0.5, 0.8],
            "feature_std": [0.2, 0.2, 0.2],
            "threshold": 0.65,
            "state_dict": model.state_dict(),
        },
        artifact_path,
    )
    settings = SimpleNamespace(
        fall_sequence_model_path=artifact_path,
        fall_sequence_window=15,
        fall_sequence_threshold=0.65,
        allow_cpu=True,
    )
    runtime = SimpleNamespace(device_label="cpu")
    engine = FallSequenceEngine(settings, runtime)
    assert engine.health()["status"] == "ready"
    result = engine.score(frames=_frames(), width=640, height=480)
    assert result["status"] == "ready"
    assert 0.0 <= result["probability"] <= 1.0


def test_artifact_threshold_cannot_lower_deployment_guardrail(tmp_path: Path):
    model = SkeletonSTGCNTCN()
    artifact_path = tmp_path / "fall.pt"
    torch.save(
        {
            "format": FALL_SEQUENCE_FORMAT,
            "model_version": "test-stgcn-tcn",
            "window_size": 15,
            "classes": ["non_fall", "fall"],
            "model_config": {"input_channels": 3, "hidden_channels": 32, "class_count": 2},
            "feature_mean": [0.5, 0.5, 0.8],
            "feature_std": [0.2, 0.2, 0.2],
            "threshold": 0.56,
            "state_dict": model.state_dict(),
        },
        artifact_path,
    )
    settings = SimpleNamespace(
        fall_sequence_model_path=artifact_path,
        fall_sequence_window=15,
        fall_sequence_threshold=0.65,
        allow_cpu=True,
    )
    engine = FallSequenceEngine(settings, SimpleNamespace(device_label="cpu"))

    assert engine.health()["status"] == "ready"
    assert engine.health()["threshold"] == 0.65

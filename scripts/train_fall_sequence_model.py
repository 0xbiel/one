#!/usr/bin/env python3
"""Train and evaluate ONE's learned fall sequence artifact.

This is a reproducible, sequence-grouped training harness for the public UR
Fall Detection Dataset.  It downloads only when ``--download`` is supplied,
keeps raw videos under the ignored evaluation directory, and writes a compact
model artifact plus a JSON evaluation receipt.  Windows from one video never
cross the train/holdout split.

The runtime architecture is implemented independently in
``geometry_service.fall_sequence``: COCO-17 ST-GCN blocks followed by a causal
TCN.  This script uses the same feature transform and model class as the
worker, so the artifact cannot silently drift from the deployed code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Running ``python scripts/...`` puts ``scripts/`` rather than the repository
# root on sys.path. Keep the harness directly runnable from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry_service.fall_sequence import (
    FALL_SEQUENCE_FORMAT,
    FALL_SEQUENCE_MODEL_VERSION,
    SkeletonSTGCNTCN,
    sequence_features,
)


DATASET_URL = "https://www.fenix.ur.edu.pl/~mkepski/ds/uf.html"
VIDEO_URL = "https://www.fenix.ur.edu.pl/~mkepski/ds/data/{name}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/evaluation/urfd"))
    parser.add_argument("--output", type=Path, default=Path("data/fall-models/fall_sequence_stgcn_tcn.pt"))
    parser.add_argument("--pose-model", type=Path, default=Path("data/pose-models/yolo11n-pose.pt"))
    parser.add_argument("--limit-per-class", type=int, default=18)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--window-stride", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def choose_device(preference: str) -> str:
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        return "mps"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return "cuda"
    if preference == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def download_videos(data_dir: Path, limit_per_class: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    names = [*(f"fall-{index:02d}-cam0" for index in range(1, 31)), *(f"adl-{index:02d}-cam0" for index in range(1, 41))]
    limits = {"fall": limit_per_class, "adl": limit_per_class}
    counts: Counter[str] = Counter()
    for name in names:
        label = name.split("-", 1)[0]
        if counts[label] >= limits[label]:
            continue
        target = data_dir / f"{name}.mp4"
        if not target.is_file() or target.stat().st_size < 100_000:
            print(f"download {name}", flush=True)
            subprocess.run(
                ["curl", "-k", "-fL", "--retry", "2", "--connect-timeout", "15", "-o", str(target), VIDEO_URL.format(name=name)],
                check=True,
            )
        counts[label] += 1
    print(f"raw dataset ready: {dict(counts)} under {data_dir}")


def _box_and_keypoints(result: Any) -> dict[str, Any] | None:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return None
    raw_boxes = getattr(boxes, "xyxy", None)
    raw_scores = getattr(boxes, "conf", None)
    raw_points = getattr(keypoints, "data", None)
    if raw_boxes is None or raw_scores is None or raw_points is None:
        return None
    boxes_np = raw_boxes.detach().cpu().numpy()
    scores_np = raw_scores.detach().cpu().numpy().reshape(-1)
    points_np = raw_points.detach().cpu().numpy()
    candidates: list[dict[str, Any]] = []
    for box, score, points in zip(boxes_np, scores_np, points_np):
        if len(box) != 4 or len(points) != 17 or not np.isfinite(box).all() or not np.isfinite(points).all():
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            continue
        keypoints = [[float(point[0]), float(point[1]), float(point[2]) if len(point) >= 3 else 1.0] for point in points]
        candidates.append({"keypoints": keypoints, "bbox": [x1, y1, x2, y2], "confidence": float(score)})
    return max(candidates, key=lambda item: (item["confidence"], (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])), default=None)


def extract_video(video_path: Path, pose_model: YOLO, *, device: str, frame_stride: int, window: int) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index % max(1, frame_stride) == 0:
                results = pose_model.predict(
                    source=image,
                    device=device,
                    imgsz=640,
                    conf=0.20,
                    iou=0.70,
                    max_det=4,
                    verbose=False,
                )
                if results:
                    chosen = _box_and_keypoints(results[0])
                    if chosen is not None:
                        frames.append(chosen)
            frame_index += 1
    finally:
        capture.release()
    if width <= 0 or height <= 0 or len(frames) < 4:
        raise RuntimeError(f"{video_path.name} yielded too few decodable pose frames")
    while len(frames) < window:
        frames.append(dict(frames[-1]))
    return frames


def windows_for_video(frames: list[dict[str, Any]], *, width: int, height: int, window: int, stride: int) -> list[np.ndarray]:
    if len(frames) < window:
        frames = frames + [dict(frames[-1]) for _ in range(window - len(frames))]
    starts = list(range(0, max(1, len(frames) - window + 1), max(1, stride)))
    if starts[-1] != len(frames) - window:
        starts.append(len(frames) - window)
    return [sequence_features(frames[start:start + window], width=width, height=height) for start in starts]


def load_sequences(args: argparse.Namespace, device: str) -> list[dict[str, Any]]:
    pose_model = YOLO(str(args.pose_model), verbose=False)
    sequences: list[dict[str, Any]] = []
    paths = sorted(args.data_dir.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"no URFD videos found under {args.data_dir}; use --download")
    for path in paths:
        label_name = path.name.split("-", 1)[0]
        label = 1 if label_name == "fall" else 0 if label_name == "adl" else None
        if label is None:
            continue
        capture = cv2.VideoCapture(str(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        try:
            frames = extract_video(path, pose_model, device=device, frame_stride=args.frame_stride, window=args.window)
            windows = windows_for_video(frames, width=width, height=height, window=args.window, stride=args.window_stride)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        sequences.append({"group": path.stem, "label": label, "windows": windows, "frame_count": len(frames), "width": width, "height": height})
        print(f"poses {path.name}: {len(frames)} frames, {len(windows)} windows", flush=True)
    if not sequences or {item["label"] for item in sequences} != {0, 1}:
        raise SystemExit("both fall and ADL sequences are required")
    return sequences


def grouped_split(sequences: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for label in (0, 1):
        items = [item for item in sequences if item["label"] == label]
        rng.shuffle(items)
        holdout_count = max(1, round(len(items) * 0.25))
        holdout.extend(items[:holdout_count])
        train.extend(items[holdout_count:])
    if not train or not holdout:
        raise SystemExit("grouped split needs training and holdout sequences")
    return train, holdout


def flatten_windows(groups: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([window for group in groups for window in group["windows"]]).astype(np.float32)
    labels = np.asarray([group["label"] for group in groups for _ in group["windows"]], dtype=np.int64)
    return values, labels


def probabilities(model: SkeletonSTGCNTCN, groups: list[dict[str, Any]], *, mean: np.ndarray, std: np.ndarray, device: str) -> dict[str, list[float]]:
    model.eval()
    result: dict[str, list[float]] = {}
    with torch.inference_mode():
        for group in groups:
            values = np.stack(group["windows"]).astype(np.float32)
            tensor = torch.as_tensor((values - mean) / std, dtype=torch.float32, device=device)
            score = torch.softmax(model(tensor), dim=-1)[:, 1].detach().cpu().numpy()
            result[group["group"]] = [float(value) for value in score]
    return result


def event_from_scores(scores: list[float], threshold: float, min_hits: int = 3) -> tuple[bool, int | None]:
    hits = 0
    for index, score in enumerate(scores):
        if score >= threshold:
            hits += 1
            if hits >= min_hits:
                return True, index
        else:
            hits = 0
    return False, None


def choose_threshold(scores: dict[str, list[float]], groups: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.arange(0.50, 0.951, 0.01):
        metrics = evaluate(scores, groups, float(threshold))
        candidate = (float(metrics["balanced_accuracy"]), float(threshold), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def evaluate(scores: dict[str, list[float]], groups: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = {group["group"]: int(group["label"]) for group in groups}
    replay = {key: event_from_scores(values, threshold) for key, values in scores.items()}
    predictions = {key: int(value[0]) for key, value in replay.items()}
    tp = sum(predictions[key] == 1 and label == 1 for key, label in labels.items())
    tn = sum(predictions[key] == 0 and label == 0 for key, label in labels.items())
    fp = sum(predictions[key] == 1 and label == 0 for key, label in labels.items())
    fn = sum(predictions[key] == 0 and label == 1 for key, label in labels.items())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    detected_indexes = {
        key: value[1]
        for key, value in replay.items()
        if value[1] is not None
    }
    true_positive_indexes = [detected_indexes[key] for key, label in labels.items() if label == 1 and key in detected_indexes]
    return {
        "sequence_count": len(groups),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "false_positive_sequences": [key for key, label in labels.items() if label == 0 and predictions[key] == 1],
        "false_negative_sequences": [key for key, label in labels.items() if label == 1 and predictions[key] == 0],
        "detection_window_index": detected_indexes,
        "median_true_positive_detection_window": float(np.median(true_positive_indexes)) if true_positive_indexes else None,
        "persistence_hits": 3,
        "score_semantics": "fresh_window_scores_only",
    }


def train(args: argparse.Namespace, train_groups: list[dict[str, Any]], holdout_groups: list[dict[str, Any]], device: str) -> tuple[SkeletonSTGCNTCN, np.ndarray, np.ndarray, float, dict[str, Any]]:
    train_values, train_labels = flatten_windows(train_groups)
    mean = train_values.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std = train_values.reshape(-1, 3).std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-3)
    normalized = torch.as_tensor((train_values - mean) / std, dtype=torch.float32, device=device)
    targets = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    model = SkeletonSTGCNTCN(input_channels=3, hidden_channels=32, class_count=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    weights = torch.as_tensor(len(train_labels) / np.maximum(class_counts * 2.0, 1.0), dtype=torch.float32, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(targets), generator=rng, device=device)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalized[batch_indices])
            loss = torch.nn.functional.cross_entropy(logits, targets[batch_indices], weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * len(batch_indices)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            train_scores = probabilities(model, train_groups, mean=mean, std=std, device=device)
            holdout_scores = probabilities(model, holdout_groups, mean=mean, std=std, device=device)
            provisional_threshold, _ = choose_threshold(train_scores, train_groups)
            metrics = evaluate(holdout_scores, holdout_groups, provisional_threshold)
            print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={total_loss / max(1, len(targets)):.4f} holdout_balanced_accuracy={metrics['balanced_accuracy']:.3f}", flush=True)
    train_scores = probabilities(model, train_groups, mean=mean, std=std, device=device)
    holdout_scores = probabilities(model, holdout_groups, mean=mean, std=std, device=device)
    threshold, threshold_metrics = choose_threshold(train_scores, train_groups)
    holdout_metrics = evaluate(holdout_scores, holdout_groups, threshold)
    evaluation = {
        "threshold": threshold,
        "threshold_selected_on": "grouped_training_sequences_with_fresh_score_persistence",
        "training": evaluate(train_scores, train_groups, threshold),
        "holdout": holdout_metrics,
        "holdout_window_scores": holdout_scores,
        "training_window_scores": train_scores,
        "timing_limitation": "Dataset windows do not carry production receipt timestamps, so the minimum confirmation duration is not replayed here.",
    }
    return model, mean, std, threshold, evaluation


def main() -> None:
    args = parse_args()
    if args.window < 8 or args.window > 64:
        raise SystemExit("--window must be between 8 and 64")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if args.download:
        download_videos(args.data_dir, args.limit_per_class)
    sequences = load_sequences(args, device)
    train_groups, holdout_groups = grouped_split(sequences, args.seed)
    model, mean, std, threshold, evaluation = train(args, train_groups, holdout_groups, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": FALL_SEQUENCE_FORMAT,
        "model_version": FALL_SEQUENCE_MODEL_VERSION,
        "dataset": "UR Fall Detection Dataset",
        "dataset_url": DATASET_URL,
        "classes": ["non_fall", "fall"],
        "window_size": args.window,
        "model_config": {"input_channels": 3, "hidden_channels": 32, "class_count": 2, "graph": "coco-17-normalized-adjacency", "temporal_head": "causal-tcn"},
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "threshold": threshold,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {"seed": args.seed, "device": device, "epochs": args.epochs, "frame_stride": args.frame_stride, "window_stride": args.window_stride, "train_groups": [group["group"] for group in train_groups], "holdout_groups": [group["group"] for group in holdout_groups]},
        "evaluation": evaluation,
        "limitations": "Sequence-level URFD holdout only; not clinically validated and not a substitute for emergency response or human review.",
    }
    torch.save(artifact, args.output)
    receipt_path = args.output.with_suffix(args.output.suffix + ".evaluation.json")
    receipt = {key: value for key, value in artifact.items() if key != "state_dict"}
    receipt["artifact"] = str(args.output)
    receipt["raw_frames_persisted"] = False
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "evaluation_receipt": str(receipt_path), "device": device, "evaluation": evaluation}, indent=2), flush=True)


if __name__ == "__main__":
    main()

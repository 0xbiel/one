"""Small, map-specific learned scoring for RoomPlan visual correspondences.

The RoomPlan landmark artifact contains metric points and ORB descriptors but
does not retain the source RGB frames.  That means a network trained here must
stay deliberately narrow: it can learn which descriptor-pair patterns are
consistent across this map, but it cannot replace image understanding or
geometric verification.

The model is trained from pairs of landmark descriptors whose metric points
are close while belonging to different scan views.  It is used as a
search-time ranking signal only.  A pose still has to pass the existing
PnP, reprojection, room-bound, and independent-view checks before it can be
published.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


MODEL_VERSION = "tiny-orb-correspondence-v2"
SCHEMA_VERSION = "roomplan-learned-matcher.v2"
# The model sees the descriptor XOR bits, their aggregate Hamming distance,
# and bounded detector-response features. This preserves the map-specific
# signal that a small room can provide without pretending that descriptors are
# image semantics.
FEATURE_DIM = 260
_MAX_PAIRS = 6_000
_CACHE_LIMIT = 4
_CACHE_LOCK = threading.Lock()
_MATCHER_CACHE: dict[bytes, "LearnedMatcher | None"] = {}


def _import_torch() -> Any | None:
    """Load Torch only when the optional learned path is actually needed."""

    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def _response_feature(values: np.ndarray | Sequence[float] | None, count: int) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(array) != count:
        return np.zeros(count, dtype=np.float32)
    # The RoomPlan builder's ORB response is usually below 0.01. Scale that
    # narrow range before the saturating transform; the output remains bounded
    # if a future OpenCV version reports a larger response.
    array = np.maximum(array, 0.0)
    scaled = np.minimum(array * 100.0, 100.0)
    return (scaled / (scaled + 1.0)).astype(np.float32)


def pair_features(
    query_descriptors: np.ndarray,
    landmark_descriptors: np.ndarray,
    query_responses: np.ndarray | Sequence[float] | None = None,
    landmark_responses: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Build bounded features for one query/landmark descriptor pair per row."""

    query = np.asarray(query_descriptors, dtype=np.uint8)
    landmark = np.asarray(landmark_descriptors, dtype=np.uint8)
    if query.ndim != 2 or landmark.ndim != 2 or query.shape != landmark.shape or query.shape[1] != 32:
        raise ValueError("ORB descriptor pairs must have shape (N, 32)")
    xor = np.bitwise_xor(query, landmark)
    bit_features = np.unpackbits(xor, axis=1).astype(np.float32)
    hamming = bit_features.mean(axis=1, keepdims=True)
    query_quality = _response_feature(query_responses, len(query))[:, None]
    landmark_quality = _response_feature(landmark_responses, len(landmark))[:, None]
    quality_delta = np.abs(query_quality - landmark_quality)
    features = np.concatenate(
        (bit_features, hamming, query_quality, landmark_quality, quality_delta),
        axis=1,
    )
    return features.astype(np.float32)


def _network_class(torch: Any) -> Any:
    class TinyCorrespondenceNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(FEATURE_DIM, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 8),
                torch.nn.SiLU(),
                torch.nn.Linear(8, 1),
            )

        def forward(self, features: Any) -> Any:
            return self.layers(features).squeeze(-1)

    return TinyCorrespondenceNet


def _pair_feature_rows(
    descriptors: np.ndarray,
    responses: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    query_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    landmark_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
    return pair_features(
        descriptors[query_indices],
        descriptors[landmark_indices],
        responses[query_indices],
        responses[landmark_indices],
    )


def _nearby_positive_pairs(
    points: np.ndarray,
    view_ids: Sequence[str],
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Find bounded cross-view positives without an O(N²) distance matrix."""

    cell_size = 0.06
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        cell = tuple(np.floor(point / cell_size).astype(np.int64).tolist())
        buckets[cell].append(index)

    pairs: list[tuple[int, int]] = []
    order = rng.permutation(len(points))
    for index in order:
        point = points[int(index)]
        cell = np.floor(point / cell_size).astype(np.int64)
        candidates: list[tuple[float, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in buckets.get((int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)), ()):
                        if other == int(index) or view_ids[other] == view_ids[int(index)]:
                            continue
                        distance = float(np.linalg.norm(points[other] - point))
                        if distance <= 0.045:
                            candidates.append((distance, other))
        candidates.sort(key=lambda item: item[0])
        for _distance, other in candidates[:3]:
            pairs.append((int(index), int(other)))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _far_negative_pairs(
    points: np.ndarray,
    view_ids: Sequence[str],
    *,
    rng: np.random.Generator,
    target_count: int,
) -> list[tuple[int, int]]:
    negatives: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = max(2_000, target_count * 30)
    for _ in range(attempts):
        first = int(rng.integers(0, len(points)))
        second = int(rng.integers(0, len(points)))
        if first == second or view_ids[first] == view_ids[second]:
            continue
        pair = (first, second)
        if pair in seen:
            continue
        if float(np.linalg.norm(points[first] - points[second])) < 0.25:
            continue
        seen.add(pair)
        negatives.append(pair)
        if len(negatives) >= target_count:
            break
    return negatives


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    # The pair count is intentionally small, so an exact rank statistic keeps
    # this module dependency-free and avoids adding scikit-learn to the worker.
    comparisons = 0.0
    for score in positives:
        comparisons += float(np.count_nonzero(score > negatives))
        comparisons += 0.5 * float(np.count_nonzero(score == negatives))
    return comparisons / float(len(positives) * len(negatives))


def _serialize_model(model: Any) -> dict[str, list[float]]:
    return {
        str(name): tensor.detach().cpu().numpy().astype(np.float32).round(7).reshape(-1).tolist()
        for name, tensor in model.state_dict().items()
    }


def _fit_matcher(
    points: np.ndarray,
    descriptors: np.ndarray,
    responses: np.ndarray,
    view_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any] | None:
    torch = _import_torch()
    if torch is None or len(points) < 40:
        return None
    positive_pairs = _nearby_positive_pairs(
        points,
        view_ids,
        rng=np.random.default_rng(seed),
        max_pairs=_MAX_PAIRS // 2,
    )
    if len(positive_pairs) < 32:
        return None
    rng = np.random.default_rng(seed + 1)
    negative_pairs = _far_negative_pairs(
        points,
        view_ids,
        rng=rng,
        target_count=min(len(positive_pairs), _MAX_PAIRS - len(positive_pairs)),
    )
    if len(negative_pairs) < 32:
        return None

    all_pairs = positive_pairs + negative_pairs
    features = np.vstack(
        (
            _pair_feature_rows(descriptors, responses, positive_pairs),
            _pair_feature_rows(descriptors, responses, negative_pairs),
        )
    )
    labels = np.concatenate(
        (
            np.ones(len(positive_pairs), dtype=np.float32),
            np.zeros(len(negative_pairs), dtype=np.float32),
        )
    )
    anchors = np.asarray([pair[0] for pair in all_pairs], dtype=np.int64)
    validation_mask = anchors % 5 == 0
    if int(np.count_nonzero(validation_mask)) < 8 or int(np.count_nonzero(~validation_mask)) < 16:
        return None

    # This model is intentionally tiny and map-specific. Train on CPU to keep
    # MPS available for the existing detector and to bound first-request cost.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        Network = _network_class(torch)
        model = Network().to(torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
        feature_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(labels)
        train_indices = np.flatnonzero(~validation_mask)
        batch_size = 256
        model.train()
        for _epoch in range(64):
            shuffled = train_indices[rng.permutation(len(train_indices))]
            for start in range(0, len(shuffled), batch_size):
                batch = torch.from_numpy(shuffled[start : start + batch_size])
                logits = model(feature_tensor[batch])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_scores = torch.sigmoid(model(feature_tensor[torch.from_numpy(np.flatnonzero(validation_mask))])).numpy()
    finally:
        torch.set_num_threads(previous_threads)

    validation_labels = labels[validation_mask]
    validation_auc = _auc(validation_scores, validation_labels)
    # A map with no repeatable cross-view appearance must fall back to the
    # geometric matcher. A barely-random learned score is worse than having no
    # learned rescue path at all.
    if validation_auc is None or not math.isfinite(validation_auc) or validation_auc < 0.60:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "feature_dim": FEATURE_DIM,
        "threshold": 0.55,
        "state_dict": _serialize_model(model),
        "training": {
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
            "validation_pairs": int(np.count_nonzero(validation_mask)),
            "validation_auc": round(float(validation_auc), 6),
            "epochs": 64,
            "source": "roomplan-metric-nearby-landmarks",
        },
    }


class LearnedMatcher:
    """Safe in-memory wrapper around a serialized tiny correspondence model."""

    def __init__(self, artifact: dict[str, Any], torch: Any) -> None:
        if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported learned matcher artifact")
        if int(artifact.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError("learned matcher feature dimension is invalid")
        raw_state = artifact.get("state_dict")
        if not isinstance(raw_state, dict):
            raise ValueError("learned matcher state is missing")
        Network = _network_class(torch)
        self.model = Network().to(torch.device("cpu"))
        current_state = self.model.state_dict()
        tensors: dict[str, Any] = {}
        for name, current in current_state.items():
            values = raw_state.get(name)
            if not isinstance(values, list) or len(values) != current.numel():
                raise ValueError("learned matcher state shape is invalid")
            tensor = torch.tensor(values, dtype=current.dtype).reshape(current.shape)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("learned matcher state contains non-finite values")
            tensors[name] = tensor
        self.model.load_state_dict(tensors, strict=True)
        self.model.eval()
        threshold = float(artifact.get("threshold", 0.55))
        if not 0.0 < threshold < 1.0 or not math.isfinite(threshold):
            raise ValueError("learned matcher threshold is invalid")
        self.threshold = threshold
        training = artifact.get("training") if isinstance(artifact.get("training"), dict) else {}
        self.diagnostics = {
            "model_version": MODEL_VERSION,
            "threshold": threshold,
            "validation_auc": training.get("validation_auc"),
            "positive_pairs": training.get("positive_pairs"),
            "negative_pairs": training.get("negative_pairs"),
            "source": training.get("source"),
        }
        self._torch = torch

    def score_pairs(
        self,
        query_descriptors: np.ndarray,
        landmark_descriptors: np.ndarray,
        query_responses: np.ndarray | Sequence[float] | None = None,
        landmark_responses: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        features = pair_features(
            query_descriptors,
            landmark_descriptors,
            query_responses,
            landmark_responses,
        )
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        with self._torch.no_grad():
            output: list[np.ndarray] = []
            for start in range(0, len(features), 2048):
                tensor = self._torch.from_numpy(features[start : start + 2048])
                output.append(self._torch.sigmoid(self.model(tensor)).numpy())
        return np.concatenate(output).astype(np.float32)


def _cache_key(points: np.ndarray, descriptors: np.ndarray, view_ids: Sequence[str]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(points, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(descriptors, dtype=np.uint8).tobytes())
    digest.update("\0".join(str(value) for value in view_ids).encode("utf-8"))
    return digest.digest()


def get_or_fit_matcher(
    points: np.ndarray,
    descriptors: np.ndarray,
    responses: np.ndarray | Sequence[float] | None,
    view_ids: Sequence[str],
    *,
    seed: int = 20260914,
) -> LearnedMatcher | None:
    """Return a cached map-specific matcher, training it once if needed."""

    if responses is None:
        response_array = np.zeros(len(points), dtype=np.float32)
    else:
        response_array = np.asarray(responses, dtype=np.float32).reshape(-1)
        if len(response_array) != len(points):
            response_array = np.zeros(len(points), dtype=np.float32)
    key = _cache_key(points, descriptors, view_ids)
    with _CACHE_LOCK:
        if key in _MATCHER_CACHE:
            return _MATCHER_CACHE[key]
        artifact = _fit_matcher(
            np.asarray(points, dtype=np.float32),
            np.asarray(descriptors, dtype=np.uint8),
            response_array,
            list(view_ids),
            seed=seed,
        )
        matcher: LearnedMatcher | None = None
        if artifact is not None:
            torch = _import_torch()
            if torch is not None:
                try:
                    matcher = LearnedMatcher(artifact, torch)
                except (TypeError, ValueError, RuntimeError):
                    matcher = None
        if len(_MATCHER_CACHE) >= _CACHE_LIMIT:
            _MATCHER_CACHE.pop(next(iter(_MATCHER_CACHE)))
        _MATCHER_CACHE[key] = matcher
        return matcher

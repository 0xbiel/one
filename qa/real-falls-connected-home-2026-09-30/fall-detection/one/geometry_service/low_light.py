"""Bounded illumination normalization for camera localization and detection.

The original pixels remain available to callers; this module only prepares a
feature/detector view when a frame is genuinely underexposed.  Keeping the
decision conservative avoids changing normal-room appearance and makes the
fallback observable in solver diagnostics.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _luminance_stats(gray: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(gray, dtype=np.uint8).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    percentiles = np.percentile(values, [10.0, 50.0, 90.0]).astype(np.float32)
    return (
        float(np.mean(values)),
        float(percentiles[0]),
        float(percentiles[1]),
        float(percentiles[2]),
    )


def _is_low_light(mean: float, p10: float, p90: float) -> bool:
    # The first branch catches genuinely dark scenes. The second catches a
    # dim, high-contrast room where the average is deceptively acceptable but
    # most feature-bearing shadows are clipped. Bright frames are untouched.
    return bool(mean < 72.0 or (mean < 96.0 and p10 < 20.0 and p90 < 170.0))


def enhance_low_light_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an illumination-normalized BGR view plus auditable diagnostics."""

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("low-light preprocessing expects an 8-bit BGR image")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    mean, p10, p50, p90 = _luminance_stats(gray)
    low_light = _is_low_light(mean, p10, p90)
    diagnostics: dict[str, Any] = {
        "low_light": low_light,
        "mean_luminance": round(mean, 3),
        "p10_luminance": round(p10, 3),
        "median_luminance": round(p50, 3),
        "p90_luminance": round(p90, 3),
        "gamma": 1.0,
        "clahe_clip_limit": 2.0,
    }
    if not low_light:
        return source, diagnostics

    # Gamma below one lifts shadows without the halos produced by a large
    # global gain. Limit the correction so a nearly black frame is improved,
    # not turned into a noisy synthetic-looking image.
    gamma = 0.78
    if mean > 1.0:
        gamma = float(np.clip(math.log(90.0 / 255.0) / math.log(mean / 255.0), 0.55, 0.88))
    lookup = np.asarray(
        np.clip(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** gamma, 0.0, 255.0),
        dtype=np.uint8,
    )
    lifted = cv2.LUT(source, lookup)
    lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    diagnostics["gamma"] = round(gamma, 4)
    diagnostics["clahe_clip_limit"] = 3.0
    return enhanced, diagnostics


def prepare_feature_gray(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare the grayscale image used by ORB/SIFT and expose light stats."""

    prepared, diagnostics = enhance_low_light_image(image)
    gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    if not diagnostics["low_light"]:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray, diagnostics


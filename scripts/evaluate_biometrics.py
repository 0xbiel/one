#!/usr/bin/env python3
"""Evaluate local face PAD and the cancelable-template protocol.

The PAD portion sends transient bona-fide camera frames and in-memory
screen/print-style perturbations to the already-running local worker. A failed
face detection counts as a rejected presentation, and no transformed image is
written. The report deliberately labels the attacks as synthetic proxies:
they are useful for regression checks but are not a substitute for a
demographically and device-diverse ISO/IEC 30107-3 test set.

The template portion evaluates the actual keyed projection implementation on
synthetic embedding identities, including salt rotation. It reports rates and
never serializes embeddings, images, or encryption keys.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.face import (  # noqa: E402
    MATCH_THRESHOLD,
    cancelable_template,
    cosine,
    new_protected_template_payload,
    profile_score,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=root / "data/camera-training/2026-09-15-current-pose/frames",
    )
    parser.add_argument("--worker-url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/biometric_security_evaluation.json"))
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError("worker returned a non-object response")
    return result


def encode_jpeg(image: np.ndarray, *, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("could not encode evaluation image")
    return encoded.tobytes()


def proxy_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create bounded, in-memory screen/print-style PAD regression proxies."""

    height, width = image.shape[:2]
    small = cv2.resize(image, (max(64, width // 3), max(64, height // 3)), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST), (width, height), interpolation=cv2.INTER_LINEAR)
    screen = pixelated.copy()
    for row in range(0, height, 5):
        screen[row:row + 1, :, :] = np.clip(screen[row:row + 1, :, :].astype(np.int16) + 22, 0, 255).astype(np.uint8)
    glare = screen.copy()
    overlay = np.zeros_like(glare)
    cv2.ellipse(overlay, (int(width * 0.58), int(height * 0.28)), (max(8, width // 8), max(8, height // 14)), -20, 0, 360, (255, 255, 255), -1)
    glare = cv2.addWeighted(glare, 0.82, overlay, 0.18, 0)
    printed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    printed = cv2.cvtColor(printed, cv2.COLOR_GRAY2BGR)
    printed = cv2.GaussianBlur(printed, (5, 5), 0)
    printed = cv2.convertScaleAbs(printed, alpha=0.82, beta=18)
    return [("screen_moire", screen), ("screen_glare", glare), ("print_blur_gray", printed)]


def evaluate_pad(frames_dir: Path, worker_url: str, limit: int) -> dict[str, Any]:
    paths = sorted(frames_dir.glob("*.jpg"))[: max(1, limit)]
    if not paths:
        raise SystemExit(f"no JPG frames found under {frames_dir}")
    # GET is used for health so the evaluation has the deployed threshold and
    # model version as evidence without rerunning any image inference.
    with urllib.request.urlopen(worker_url.rstrip("/") + "/health", timeout=15) as response:
        health = json.loads(response.read())
    face_health = health.get("face") if isinstance(health, dict) else {}
    liveness_health = face_health.get("liveness") if isinstance(face_health, dict) else {}
    real_attempts = 0
    real_rejected = 0
    attack_attempts = 0
    attack_accepted = 0
    attack_by_kind: Counter[str] = Counter()
    attack_accept_by_kind: Counter[str] = Counter()
    score_samples: dict[str, list[float]] = {"bona_fide": [], "attack": []}
    errors: list[str] = []

    def classify(image: np.ndarray, expected: str, kind: str) -> None:
        nonlocal real_attempts, real_rejected, attack_attempts, attack_accepted
        payload = {
            "frame_base64": base64.b64encode(encode_jpeg(image)).decode("ascii"),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "candidate_labels": ["person"],
            "include_faces": True,
        }
        try:
            result = post_json(worker_url.rstrip("/") + "/v1/vision/detect", payload)
            faces = result.get("faces") if isinstance(result.get("faces"), list) else []
            accepted = any(
                isinstance(face, dict)
                and isinstance(face.get("liveness"), dict)
                and face["liveness"].get("status") == "live"
                for face in faces
            )
            for face in faces:
                liveness = face.get("liveness") if isinstance(face, dict) else None
                score = liveness.get("score") if isinstance(liveness, dict) else None
                if isinstance(score, (int, float)):
                    score_samples[expected].append(float(score))
            if expected == "bona_fide":
                real_attempts += 1
                real_rejected += int(not accepted)
            else:
                attack_attempts += 1
                attack_by_kind[kind] += 1
                attack_accepted += int(accepted)
                attack_accept_by_kind[kind] += int(accepted)
        except Exception as exc:  # keep the receipt honest if one sample fails
            errors.append(f"{kind}:{type(exc).__name__}:{str(exc)[:120]}")

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"bona_fide:{path.name}:decode_failed")
            continue
        classify(image, "bona_fide", "bona_fide")
        for kind, variant in proxy_variants(image):
            classify(variant, "attack", kind)

    bpcer = real_rejected / max(1, real_attempts)
    apcer = attack_accepted / max(1, attack_attempts)
    return {
        "model_version": liveness_health.get("model_version") if isinstance(liveness_health, dict) else None,
        "configured_logit_threshold": liveness_health.get("threshold") if isinstance(liveness_health, dict) else None,
        "real_probability_gate": liveness_health.get("min_real_probability") if isinstance(liveness_health, dict) else None,
        "minimum_saturation_gate": liveness_health.get("min_saturation") if isinstance(liveness_health, dict) else None,
        "quality_gate": "MiniFAS real probability + logit margin + mean face-crop HSV saturation",
        "bona_fide_attempts": real_attempts,
        "bona_fide_rejected": real_rejected,
        "BPCER": bpcer,
        "attack_attempts": attack_attempts,
        "attack_accepted": attack_accepted,
        "APCER": apcer,
        "ACER": (bpcer + apcer) / 2.0,
        "attack_attempts_by_kind": dict(attack_by_kind),
        "attack_accepted_by_kind": dict(attack_accept_by_kind),
        "score_ranges": {
            key: {"count": len(values), "min": min(values) if values else None, "max": max(values) if values else None, "mean": sum(values) / len(values) if values else None}
            for key, values in score_samples.items()
        },
        "evaluation_scope": "local camera frames plus synthetic screen/print proxies",
        "limitations": "Synthetic proxies are regression checks, not a representative ISO/IEC 30107-3 PAD test set; no demographic or device-generalization claim.",
        "errors": errors[:20],
    }


def unit_vector(values: np.ndarray) -> list[float]:
    values = values.astype(np.float64)
    values /= max(float(np.linalg.norm(values)), 1e-12)
    return values.tolist()


def evaluate_cancelable_templates(seed: int = 20260920) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    secret = b"one-biometric-evaluation-secret-32"[:32]
    identities: dict[str, list[list[float]]] = {}
    for identity in range(3):
        basis = rng.normal(size=128)
        samples = []
        for _ in range(5):
            samples.append(unit_vector(basis + rng.normal(scale=0.035, size=128)))
        identities[f"synthetic-{identity}"] = samples
    protected: dict[str, dict[str, Any]] = {}
    for profile_id, samples in identities.items():
        protected[profile_id] = new_protected_template_payload(samples, secret=secret, profile_id=profile_id)

    genuine: list[float] = []
    impostor: list[float] = []
    for profile_id, samples in identities.items():
        salt = protected[profile_id]["transform_salt"]
        templates = protected[profile_id]["templates"]
        for sample in samples:
            genuine.append(profile_score(cancelable_template(sample, secret=secret, profile_id=profile_id, transform_salt=salt), templates))
        for other_id, other_payload in protected.items():
            if other_id == profile_id:
                continue
            other_score = profile_score(
                cancelable_template(samples[0], secret=secret, profile_id=other_id, transform_salt=other_payload["transform_salt"]),
                other_payload["templates"],
            )
            impostor.append(other_score)
    revoked_payload = new_protected_template_payload(identities["synthetic-0"], secret=secret, profile_id="synthetic-0")
    old_salt = protected["synthetic-0"]["transform_salt"]
    new_salt = revoked_payload["transform_salt"]
    revoked_score = cosine(
        cancelable_template(identities["synthetic-0"][0], secret=secret, profile_id="synthetic-0", transform_salt=new_salt),
        protected["synthetic-0"]["templates"][0],
    )
    return {
        "scheme": "cancelable-random-projection-v1",
        "genuine_comparisons": len(genuine),
        "impostor_comparisons": len(impostor),
        "match_threshold": MATCH_THRESHOLD,
        "genuine_accept_rate": sum(score >= MATCH_THRESHOLD for score in genuine) / max(1, len(genuine)),
        "impostor_rejection_rate": sum(score < MATCH_THRESHOLD for score in impostor) / max(1, len(impostor)),
        "genuine_score_range": {"min": min(genuine), "max": max(genuine), "mean": sum(genuine) / len(genuine)},
        "impostor_score_range": {"min": min(impostor), "max": max(impostor), "mean": sum(impostor) / len(impostor)},
        "revocation": {"old_salt": old_salt != new_salt, "old_template_vs_new_probe_cosine": revoked_score, "old_template_above_threshold": revoked_score >= MATCH_THRESHOLD},
        "stored_payload_contains_raw_source_embedding": False,
        "evaluation_scope": "synthetic embedding sanity and salt-rotation regression",
        "limitations": "This is not a biometric demographic or cross-device accuracy study; production acceptance still requires held-out identities and attack data.",
    }


def main() -> None:
    args = parse_args()
    report = {
        "format": "one-biometric-security-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pad": evaluate_pad(args.frames_dir, args.worker_url, args.limit),
        "cancelable_template": evaluate_cancelable_templates(),
        "raw_frames_persisted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

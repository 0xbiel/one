# Animated fall-detection input fixtures

Seven deterministic 3D scenes rendered from the synthetic room reconstruction at commit `a89ff32f1024f83b925c5cccdc0dc29bd91ec370` in `0xbiel/one`. These are new assets, not recovered historical benchmark bytes.

## What is included

- 640 × 480 RGB video, 10 fps, 80 frames / 8 seconds per scene
- Lateral fall, forward fall, partial-occlusion fall
- Sitting, bending and recovery, deliberate slow lying down, dropped object
- `assets/manifest.json`: choreography intent labels and onset/contact times
- `assets/<scenario>/ground_truth_frames.json`: frame timestamps and projected actor boxes, solely for ground-truth/control use
- `generate_animations.py`: reusable procedural room, clothed adult-like actor and motion generator
- `encode.py`: PNG sequence to H.264 MP4
- `score_results.py`: strict full-frame scoring of real image-pipeline results

All motions start at 2.0 seconds; fall contact is at 2.7 seconds. The slow lying-down control reaches the floor at 5.0 seconds. Long end holds provide time for temporal confirmation. Rendering is deterministic choreography, not a biomechanical or injury simulation.

## Reproduce

Requires Blender 4.3.2, Python 3 and FFmpeg with libx264. No downloaded character models or animation licenses are required.

    blender -b -t 8 --python generate_animations.py
    python encode.py

For a quick single clip, add `-- --only fall_side`; for three stills add `--preview`. The `.blend` output records the final scene state; the Python generator is the canonical timeline because it computes motion directly rather than using stored Blender keyframes.

Cycles uses 32 samples, up to 8 CPU threads, no denoiser, fixed camera/lights. Exact duplicate actor poses are rendered once and copied to their timestamps. This is appropriate for the static holds but does not simulate sensor noise or exposure variation. MP4 uses CRF 18, yuv420p. Evaluation must use the same decoded MP4 inputs for baseline and fixed code; PNG evaluation is a separate condition.

## Evaluate honestly

Run frames through ONE's configured YOLO-World image detector, temporal stability tracker, and fall tracker. Capture per-frame detected persons, stable tracks and emitted events. Do not send projected ground-truth boxes into the primary evaluation. A separate oracle-box test can isolate tracking/rule regressions but must be labeled as a component test.

`score_results.py` requires every frame for every clip and provenance fields: `code_commit`, `model_name`, `model_sha256`, `input_kind="rendered_rgb_frames"`, and `inference_fps`. Its input JSONL rows contain `clip_id`, `frame_index`, `timestamp_s`, counts `person_detections` / `stable_person_tracks`, and `events` with `event_type`.

    python score_results.py run.jsonl --metadata run_metadata.json --output scores.json

Compare detected/missed falls, negative-clip false positives, onset-to-signal delay, person-recognition coverage and stable-track coverage. Reset runtime temporal state for each clip. Use capture time (frame index / 10) rather than CPU inference wall time. Keep model weights/checksum, thresholds, frame rate, video files/checksums and input decoding unchanged between code versions.

The seven clips are deliberately small regression fixtures. An actor not recognized by YOLO is an observed image-pipeline miss, not evidence the fall rule passed. The procedural actor is stylized and may be out of the model's training distribution. Coarse bounding boxes cannot reliably distinguish all accidental falls from deliberate lying down. Results establish only behavior on these fixtures, never real-world safety, emergency readiness, medical accuracy, or population performance.

## Reference contracts

- ONE image pipeline: `app/vision.py`, `geometry_service/runtime.py`, `app/fall.py`
- https://github.com/0xbiel/one-docs/blob/main/pages/api/vision.md
- https://github.com/0xbiel/one-docs/blob/main/pages/security/limitations.md

Documentation contains historical inconsistencies about configured detectors, so the exact tested code and runtime take precedence over generic doc statements.

Annotation boxes are amodal projected actor geometry bounds, not occlusion masks. Occluded portions are intentionally still included in ground truth. The dropped-object scene uses the same upright actor plus a falling cushion-like object, to test whether non-person motion triggers a false review signal.

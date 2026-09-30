# ONE physics fall fixture

This fixture adds a genuine gravity-driven articulated rigid-body fall to the separate procedural motion suite. It is **stylized, not photorealistic**.

## Physics

- Blender 4.3.2 Bullet rigid bodies, 15 convex-hull body segments, 14 angular-limited joints
- Gravity 9.81 m/s², floor and furniture colliders, body self-collisions except directly connected pairs
- 24 Hz scene simulation, 10 substeps per frame, 80 constraint solver iterations
- Adult-sized segmented clothed figure starts with a 10° lateral lean, supported kinematically through 0.9583 s
- Support ends at 1.0 s. All subsequent movement is simulated, with no pose keyframes or external forces
- Rendering uses sampled rigid-body transforms from that simulation, rather than rerunning dynamics out of order
- 6-second video, 640×480, 12 fps

The source scene is `assets/physics_source.blend`; `physics_replay.blend` holds render-only sampled replay. `physics_samples_24fps.json` contains every sampled body matrix and mesh-floor proximity metric. `ground_truth_frames.json` has projected body bounding boxes, not model detections. `manifest.json` documents physics settings, joint limits, and first upper-body contact proxy at 1.625 s.

Contact is inferred from the lowest transformed collision-mesh vertex within 15 mm of floor at 24 Hz. This is a geometric proxy, **not** a solver contact event or impact force. No sampled vertex penetrates the floor in this run (minimum z ≈ 1.69 mm).

## Reproduce

Run `blender -b -t 4 -P generate_physics.py -- --preview` from any directory to simulate and render six preview frames. Run with `--render` for the full Cycles render. For clean Eevee rendering, open `assets/physics_replay.blend` with `render_replay.py -- --eevee` (software EGL startup may be slow in headless environments).

Encode frames using ffmpeg: input frame rate 12, input `assets/frames/%04d.png`, output H.264 yuv420p with CRF 18 and faststart. Physical timestamps are zero-based frame index divided by output fps.

## Limits

No active balance control, muscle forces, protective reflexes, soft-tissue deformation, realistic clothing, calibrated friction or clinical validation. The passive collapse is physically simulated but does not represent all human falls. Segment shape and joint approximations can look mannequin-like. A fresh sequential rerun of the saved physics source produced exactly identical matrices in this Blender build (maximum absolute matrix error 0; see validation.json). Cross-platform bitwise physics determinism is not guaranteed. This synthetic positive example cannot establish fall-detector sensitivity, specificity, or emergency safety by itself.

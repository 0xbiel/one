ONE real-video fall investigation — 30 September 2026

This package contains measured detector outputs, scoring traces, metadata and local research scripts. It does not contain original dataset videos or model checkpoints. Nothing was pushed, merged or deployed.

PRIMARY RESULTS
Full dataset: 160 official GMDCSA-24 clips (79 Fall, 81 ADL); 32 development and 128 evaluation clips. Both models processed all 6,439 samples at 5 Hz. Evaluation has 63 falls and 65 ADLs.
Current ONE at 5 Hz: TP 37 FN 26 FP 5 TN 60, clip-level precision 88.10%, recall 58.73%. Five ADL clips generated seven separate false alerts.
Pose detector boxes through unchanged ONE rule at 5 Hz: TP 30 FN 33 FP 5 TN 60, precision 85.71%, recall 47.62%.
Ideal 1.4-second sampling: World TP 2 FN 61 FP 0 TN 65; pose boxes TP 3 FN 60 FP 0 TN 65. World catches 0–2 falls across seven sample-start phases.
Small paired cadence subset: nine evaluation clips, four falls and five ADLs. World 5 Hz TP 2 FN 2 FP 1 TN 4; World 20 Hz TP 2 FN 2 FP 2 TN 3. This is not a claim that20 Hz cannot help a better temporal method.

INTERPRETATION AND LIMITS
This is offline RGB inference plus an exact replay of ONE's detector/tracker/fall-rule components. It is not a full HTTP/app or clinical deployment test. Fixed default 11 prompts were used; real homes' registered-object vocabulary can differ. Every clip starts with an empty tracker. Many falls have little pre-fall context. These cold-start scores must not be treated as continuous-monitoring sensitivity. The subject-folder split does not prove unique-person independence; the source paper internally disagrees on three versus four actors. No weights were trained and no thresholds were tuned on evaluation clips. The full dataset expansion overlaps the preliminary pilot; methods were retained unchanged. The pose comparison uses pose-model boxes, not a trained skeleton sequence classifier. Predictions on clothes can look like people, so detection-presence rates are not actor recall.

29 videos are natively about 15 fps and were not used for genuine20 Hz comparison. All nine20 Hz evaluation videos had observed native frame gaps below50 ms. Three development examples have a few longer native gaps; ffmpeg can hold frames there. Main20 Hz conclusions use the nine evaluation clips.

TP requires an alert no earlier than0.2s before the coarse dataset fall annotation; the tolerance accommodates timestamp quantization. Missing-onset fall s4_fall_15 stays in classification but is excluded from onset latency. FP in the confusion counts means a normal clip with any alert. Separate timed_event_false_alerts and timed_event_precision also count multiple false alerts, so do not confuse clip precision with alarm-event precision. No early alerts on fall clips occurred in the final baseline runs.

SOURCES AND RIGHTS
Dataset author repository, pinned revision5abac7693229900cf80f722e878fbb119211fc1c:
https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
The repository includes MIT license, reproduced in dataset/LICENSE. Attribution: Ekram Alam, Abu Sufian, Paramartha Dutta, Marco Leo and Ibrahim A. Hameed, GMDCSA-24: A dataset for human fall detection in videos, Data in Brief 57(2024),110892. https://doi.org/10.1016/j.dib.2024.110892
The authors report informed consent for public release. The paper reverses Fall/ADL totals; this package follows verified directory membership.
ONE backend reference9c05becdc3af59b15092075c1eb1dc90e2ab9e70:
https://github.com/0xbiel/one/tree/9c05becdc3af59b15092075c1eb1dc90e2ab9e70
Frontend reference84e21f90f53213c79775168ee0491019ad1f1400 is recorded in runtime-inspection.md.
Model sources and SHA256 hashes are in runtime-inspection.md. Ultralytics dependencies and weights have their own licenses, independent of dataset MIT. Check those terms before product deployment.

KEY FILES
full_scores.json: all methods, cadences, clip outcomes and gate traces.
full_analysis_summary.json: aggregate results, day/night and pre-fall-context strata, phase sweep.
full_integrity_checks.json: all 6,439 frames per model accounted for, no duplicate frame IDs.
full_manifest_downloaded.json: source hashes, labels, coarse fall onset, FPS and counts.
full_dataset_preparation_summary.json: verified160-file download and native-rate inventory.
full_inference_*.jsonl: model outputs and per-frame timings. These combine verified identical pilot cache entries with fixed-vocabulary offline entries; timings are not deploymentFPS.
cadence20_scores.json and cadence20_phase_sensitivity.json: paired cadence comparison.
fixed_vocabulary_benchmark.json: exact parsed-output equality on five predeclared development frames and CPU timings.
ONE-real-fall-pilot-diagnostics.mp4: selected pilot examples, not all 160 clips. World boxes on left; alternative pose-model boxes/keypoints on right; downstream fall rule unchanged.

REPRODUCTION
Python 3.12, ffmpeg/ffprobe and the packages listed in runtime-requirements.lock.txt were used. The PyTorch+cpu wheels come from the official PyTorch CPU index. Exact package versions describe this research runtime, not the user's deployed environment. Source scripts use their own directory as the workspace root.
1. Create a fresh Python environment and install the pinned dependencies. Some dependencies require the official PyTorch CPU wheel index. Inspect the scripts before running them.
2. Create models/ and download official checkpoints there, using the URLs and hashes in runtime-inspection.md. Configure Ultralytics with sync=false and weights_dir pointing to the absolute models/ directory. Set YOLO_CONFIG_DIR to that directory before import. CLIP will use weights_dir/clip.
3. Run download_all.py. This verifies original video sizes and Git blob hashes and prepares5 Hz frames. Dataset CSVs, original license and the pinned tree metadata are included; raw videos are fetched from the original authors' repository.
4. Run run_fast.py --frames full_frames.json --prefix full_ for each --split development/evaluation and --mode world/pose. Saved caches resume without duplicate frame IDs. To rerun from scratch, move the supplied full_inference_*.jsonl files to another location first.
5. Run score.py --prefix full_ --manifest full_manifest_downloaded.json, then summarize_phases.py full_, diagnostics.py --prefix full_ --manifest full_manifest_downloaded.json, and summarize_full.py.
6. For the cadence test, run prepare_20 fps.py and run_fast.py with --frames frames20.json --prefix cadence20_ for both splits and models. Score with --prefix cadence20_ --manifest manifest20.json --base-fps 20 --rates 20,5,0.7142857142857143. Run summarize_phases.py cadence20_ for start-phase sensitivity.
7. Fixed-prompt optimization changes the execution path only for a fixed vocabulary. benchmark_fixed_vocabulary.py checks equality against the actual runtime path. Do not claim its measured 6.8 fps on this shared CPU as current deployed throughput, or as a guarantee for the user's Mac/GPU.

The diagnostic single-identity method preserves the three-hit warm-up but assumes one selected person identity; it is not a deployable multi-person tracker. The confidence ablation bypasses the fall rule's 0.55 gate while retaining the 0.20 detector threshold; it is not a recommendation to relax a safety threshold. The nine-hit20 Hz diagnostic matches the minimum0.4-second persistence of three hits at5 Hz when detections are continuous; missing observations mean it is not a perfect duration-equivalent classifier.

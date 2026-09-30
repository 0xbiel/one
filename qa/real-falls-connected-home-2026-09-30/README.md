# ONE: real falls and connected-home QA

Evidence-only research snapshot, 30 September 2026. Both investigations were run against backend commit `9c05becdc3af59b15092075c1eb1dc90e2ab9e70`. This branch preserves reports, reproducibility material and compact previews; it does not change application code.

- [Real fall video investigation](fall-detection/REPORT.txt): 160 staged clips, real RGB inference, 128 evaluation clips, paired 20 Hz diagnostic and CPU timing.
- [Connected-home camera localization](connected-rooms/RESULTS.md): 24 procedural connected-scene cases, verified room labels and wrong-room audit.

This draft is for reviewing measured limitations and choosing the next experiment. Neither investigation establishes deployment safety or clinical reliability. Raw dataset archives and model weights are intentionally excluded. Dataset previews follow the [v2.1 CC BY 4.0 attribution](fall-detection/ATTRIBUTION.md); connected-room scenes are procedurally generated.

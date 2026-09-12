# Data protection impact assessment (DPIA) — draft and release gate

This document identifies risk; it is not an approved DPIA. The eventual controller must complete it with qualified privacy and security review before resident data is processed.

## Why an assessment is required

ONE combines in-home camera processing, a persistent room map, object memory, a vulnerable-person use case, and optional assistant transcripts. The controller must determine the applicable high-risk criteria and consult the supervisory authority if residual risk cannot be reduced.

## Processing description

The MVP pairs a browser camera or iOS device to one household, processes bounded frames locally for supported object labels, stores uncertain derived events and optional short encrypted clips, and answers explicit check-in questions using recent evidence. It does not perform face recognition, identity biometrics, emotion recognition, diagnostic scoring, advertising, or training on household data.

## Necessity and proportionality questions

- Is each camera, room, object label, clip, transcript, and recipient necessary for the stated purpose?
- Can the purpose be served with a lower-resolution frame, zone-only result, shorter retention, or no clip?
- Is the resident informed in an accessible format before activation and after material changes?
- Who can authorize a purpose when a resident has limited capacity, and how is that authority verified?
- Are bathrooms and similarly private areas excluded by configuration and tested?
- Is any health-related inference avoided, or if unavoidable separately assessed under Article 9?

## Risk register

| Risk | Impact | Controls in MVP | Residual decision |
|---|---|---|---|
| Unauthorized household access | Exposure of video/map data | Home-scoped bearer sessions, role checks, hashed tokens, private clip responses | Controller/security approval required |
| Camera captures visitors or private areas | Loss of privacy | Consent gate, persistent pause state, no continuous clip persistence, area exclusion requirement | Test with physical layout |
| Incorrect object location | Misleading caregiver action | Stable multi-frame rule, calibration revision, uncertainty radius, zone fallback, no medical claims | Human review required |
| Over-retention or backup persistence | Extended exposure | Per-record expiry, retention worker, encrypted storage, backup expiry procedure | Verify operational job and restore test |
| Prompt injection or unsupported model output | Unsafe explanation | Minimum evidence context, structured output validation, deterministic fallback, no diagnosis prompt | Red-team before release |
| Deletion misses an artifact | Rights failure | DB cascade, explicit map/clip cleanup, deletion request status and audit proof | Run fixture and backup deletion tests |
| Key loss or compromise | Media unavailable/exposed | AES-GCM envelope, deployment secret, no key beside media | Add rotation/recovery runbook |

## Approval record

- Controller: `TBD`
- DPO/privacy reviewer: `TBD`
- Security reviewer: `TBD`
- Residual risk accepted: `NO — release blocked`
- Approval date/version: `TBD`
- Conditions and review trigger: `TBD`

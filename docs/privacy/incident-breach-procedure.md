# Security incident and personal-data breach procedure

This is a tabletop-ready draft. The controller determines whether an event is a personal-data breach, the notification duties, and communications. Do not put real personal data in issue trackers or chat while investigating.

## Immediate response

1. Create an incident ID and record UTC time, reporter, affected service/home scope, and containment owner.
2. Pause affected camera processing, revoke sessions/device credentials, disable exposed endpoints, and preserve only the minimum technical evidence needed.
3. Isolate compromised hosts or keys without destroying forensic evidence. Rotate clip, database, LiveKit, and bootstrap secrets as applicable.
4. Confirm whether frames, clips, maps, observations, transcripts, credentials, logs, backups, or exports were accessed or altered.

## Investigation record

- Timeline, detection source, systems/versions, affected homes and data categories.
- Number and categories of subjects, likely consequences, and vulnerable-person impact.
- Access path, indicators, containment, eradication, recovery, and remaining risk.
- Processor/subprocessor involvement and contractual notification status.
- Decision maker, legal assessment, notification decision, and rationale.

## Recovery and notification gate

- Restore from an encrypted, integrity-checked backup only after the controller approves scope.
- Verify consent/runtime pause state, role membership, token revocation, retention worker, and deletion queue.
- Determine whether supervisory-authority or subject notification is required and record deadlines under the controller’s process.
- Provide clear facts, likely impact, mitigations, and contact route; do not promise impossible precision or conceal uncertainty.

## Exercises and prevention

Run a tabletop at each release gate covering leaked bearer token, unauthorized clip access, lost encryption key, over-retained backup, prompt injection, and incorrect deletion. Track corrective actions, owner, due date, and verification evidence.

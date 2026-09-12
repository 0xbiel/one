# Data-subject request procedure

This is an operational draft. The controller owns the legal response, identity decision, exemptions, deadline, and communication with the requester.

## Intake and identity

1. Record the request time, requester, household, requested right, preferred response channel, and a safe request ID.
2. Verify identity using the controller-approved method. Do not accept a bearer token or pairing code as sufficient proof for a different channel.
3. Establish the resident’s role and, where applicable, verified representative authority. A caregiver role alone is not proof of authority to consent for another person.
4. Restrict staff access to the relevant home while the request is investigated; do not copy raw clips into tickets.

## Rights workflow

| Right | MVP handling | Completion evidence |
|---|---|---|
| Access / portability | `POST /api/v1/homes/{id}/privacy/export`; review before release | Export ID, contents, recipient, timestamp |
| Rectification | Correct account/configuration records through an authorized admin workflow | Before/after audit and propagation check |
| Restriction / objection | Pause relevant purpose and stop publisher/inference path | Consent/runtime event and device check |
| Withdrawal | Revoke the specific consent purpose; video capture immediately becomes unavailable | Consent row, runtime state, client indicator |
| Erasure | `POST .../privacy/delete`; remove DB rows and map/clip artifacts | Terminal deletion request status, cleanup report, backup expiry plan |
| Human review | Route disputed or consequential interpretation to a human; never rely solely on a model | Reviewer identity, decision, evidence IDs |

## Export and deletion safeguards

- Export only the requested home/subject scope after authorization; do not include bearer-token or pairing-code hashes.
- Deletion creates a request record, cleans media artifacts, then cascades household rows in a database transaction. A minimal completion audit proof may remain when approved by the controller.
- If media cleanup fails, the API marks the request failed/pending and must not claim completion.
- Include Redis jobs, MinIO objects, LiveKit logs, local exports, transcripts, summaries, replicas, and backups in the operator checklist.
- Record response deadlines and exceptions outside the application; the API is not a legal case-management system.

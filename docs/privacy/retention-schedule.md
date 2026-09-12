# Retention and deletion schedule

Defaults below describe the current MVP code and are not a legal justification. The controller must approve a purpose-based period and configure operational systems to match.

| Record | MVP default | Deletion trigger | Current enforcement | Evidence still needed |
|---|---:|---|---|---|
| Live frame bytes | In memory only | End of processing / ring-buffer window | Vision endpoint does not persist; ring buffer is bounded | Verify process memory and crash dumps |
| Pairing code hash | 10 minutes / single use | Expiry or completion | API rejects expired/used code | Periodic purge for unused rows |
| Session token hash | 60-minute expiry | Expiry/logout or home deletion | Auth rejects expired; logout deletes | Session purge job |
| Optional encrypted clip | 7 days | Clip expiry or rights deletion | Retention deletes DB record and encrypted file | Object-store and backup test |
| Event and observation metadata | 30 days | Event/observation expiry | Retention deletes expired events and observations | Verify indexes and job scheduling |
| Assistant summary | 30 days | Summary expiry or rights deletion | Retention deletes expired summaries | Transcript-specific review |
| Room map/calibration | Until replacement/deletion | Map replacement or home deletion | Map artifacts removed in rights deletion | Approved necessity period |
| Audit/deletion proof | Controller-approved minimum | Operational/legal schedule | Minimal metadata retained after home deletion | Do not use for profiling |
| Backups | Defined by operator | Backup expiry window | Runbook requirement; not automatic in MVP | Encrypted restore and expiry evidence |

## Operational rules

- Run retention with a monitored service identity and record counts, failures, and completion time without logging media.
- Treat deletion as incomplete when an object store, queue, export, or backup copy cannot be verified.
- A backup is not “deleted” merely because a live database row is gone; document expiry or cryptographic erasure according to the approved policy.
- Do not extend retention for debugging without a documented purpose, notice, access restriction, and controller approval.

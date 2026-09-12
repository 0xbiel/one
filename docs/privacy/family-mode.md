# Family mode and medication boundary

This is an implementation boundary for the local synthetic demo, not legal
advice or a GDPR compliance certification.

Family mode is a separate purpose from live video, audio, clips, object
memory, and assistant check-ins. The API requires an active versioned
`family_mode` consent before an admin/caregiver can enumerate members or send
an invitation. Multiple caregiver accounts are supported; every member's role
remains visible, a caregiver cannot grant admin/publisher privileges, and
publisher devices are excluded from family membership. Invitations are
one-time six-digit codes stored only as SHA-256 hashes and expire. Accepting
an invitation creates a synthetic account and membership; it does not assert
that a caregiver is a legal representative.

Medication plans and check-ins are separate sensitive records. They require an
active `medication_management` consent for the subject. A caregiver can manage
another member only within the same home and only after this purpose check;
caregivers can also own plans for themselves. Plan `created_by`, explicit
`assigned_caregiver_id`, and check-in `marked_by` preserve responsibility and
audit context without granting a caregiver extra permissions.
The plan stores user-entered name, dose, schedule rule, instructions, active
state, and version. The deterministic schedule grammar supports legacy daily
times (`08:00,20:00`), weekday rules (`Mon,Wed,Fri @ 08:00`), weekday/weekend
groups, and date-specific exceptions. Reminder queries filter by the requested
calendar day, so a weekly plan is not displayed as a daily dose. Check-ins
record administrative status only; they do not verify that medication was
taken and do not provide clinical recommendations.

The family assistant is a minimum-context operation. It receives the selected
subject's active plans and at most 100 recent check-in records. It does not
receive raw media, camera frames, transcripts, event history, or the complete
household stream. Outputs are evidence-linked where possible, labelled as
administrative, and fall back to deterministic text when the local model is
offline.

Before any real resident processing, complete the DPIA, identify controller
and representative authority, document Article 6/9 bases, approve notices,
test withdrawal/export/deletion, and agree retention with qualified EU/Spanish
privacy counsel. The current SQLite FK cascade includes family invites,
medication plans, and check-ins in household export/deletion paths; backup
expiry still requires operational verification.

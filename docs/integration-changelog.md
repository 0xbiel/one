# Integration changelog for client/docs agents

## 2026-09-12 — family mode and medication slice

- Added consent-gated household member listing and single-use synthetic family invitations.
- Added medication plans with optimistic `version`, active state, human-entered schedule, and deterministic reminder/check-in routes.
- Medication plans now carry an optional same-home `assigned_caregiver_id`; reminder responses include the caregiver name and original schedule rule.
- Schedule rules support daily legacy times, weekday/weekend recurrence, and date-specific exceptions; reminder generation filters by the requested day.
- Added `family-assistant`, which sends only one subject's plans/check-ins to local Qwen and has an explicit offline fallback.
- New endpoint groups: `/homes/{home_id}/family/*`, `/homes/{home_id}/medication-plans*`, `/homes/{home_id}/medication-check-ins`, `/homes/{home_id}/medication-reminders`, `/homes/{home_id}/family-assistant`.
- `ConsentIn.subject_user_id` supports an explicitly recorded subject; this field does not infer legal representation.
- Export and FK-backed deletion include `family_invites`, `medication_plans`, and `medication_check_ins`.
- HTTP failures and validation failures use the safe `{error, request_id, api_version}` envelope; `X-Request-ID` is accepted only when it is a valid UUID.
- Generated contract: `contracts/openapi.json` must be regenerated and client fixtures updated.

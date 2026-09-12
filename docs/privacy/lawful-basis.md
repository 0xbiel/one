# Lawful-basis matrix — controller decision required

This matrix is a decision aid, not a legal conclusion. The controller must document one Article 6 basis per purpose and an Article 9 condition if processing special-category data. Consent in the API is an authorization record and does not by itself settle the legal analysis.

| Purpose | Minimum data | Candidate basis to assess | Article 9 question | Decision / evidence |
|---|---|---|---|---|
| Account and access control | Name, role, session hash | Contract or legitimate interest, depending on service relationship | Usually no, unless account fields reveal health data | Controller to decide |
| Pair a camera to a household | Device credential, pairing event | Consent and/or contract assessment | Camera view may incidentally contain health information | Controller to decide |
| Live video | Live media | Explicit, informed purpose-specific consent is the MVP design assumption | Assess explicit consent and vulnerable-person safeguards | Consent notice + withdrawal test |
| Object detection and last-seen memory | Frame in memory; derived observation | Consent or another documented basis linked to video purpose | Assess incidental special-category capture | DPIA and minimisation evidence |
| Room map and calibration | Geometry, room labels, pose | Consent or documented legitimate interest assessment | Usually not inherently special category; context matters | Controller to decide |
| Optional event clip | Short encrypted clip | Separate purpose-specific consent | Assess incidental capture and explicit consent where required | Separate toggle + retention |
| Requested assistant answer | Transcript and selected evidence | Consent/contract assessment; legal obligation only if applicable | Do not ask for health information in MVP | Prompt and notice review |
| Security, fraud, and rights handling | Audit/request metadata | Legal obligation or legitimate interest assessment | Avoid sensitive content in logs | Retention and access review |

No processing purpose may be enabled merely because it is technically possible. Any new purpose, recipient, model provider, or retention period reopens this matrix and the DPIA.

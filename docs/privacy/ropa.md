# Records of processing activities (RoPA) — draft

This is a template for the eventual controller. Fill the owner, legal basis, recipients, locations, and approval dates before real processing. The hackathon uses synthetic identities only.

| Processing activity | Data subjects | Data | Purpose | Basis decision | Recipients / location | Retention | Owner / status |
|---|---|---|---|---|---|---|---|
| Account and household access | Admin, caregiver, resident, publisher operator | Account and membership fields | Authenticate and authorize home access | Controller decision required | ONE API host; hosting operator if any | Account lifetime plus deletion schedule | TBD / draft |
| Camera pairing and live viewing | Resident and people in camera view | Device credential, live media | Explicitly authorized camera session | Controller + counsel required | LiveKit deployment; no public CDN in MVP | Session/transient only | TBD / draft |
| In-memory object detection | People incidentally in view; household | Frame bytes, candidate labels | Find supported visible objects | Controller + counsel required | Local vision worker | No frame persistence; derived observation schedule | TBD / draft |
| Room mapping and calibration | Household occupants | Geometry, camera pose, room labels | Approximate map placement | Controller decision required | Local API/object store | Until map replacement or deletion | TBD / draft |
| Event evidence and clips | People incidentally captured | Observation metadata and optional clip | Human review of an event | Controller + counsel required | Authorized household roles | Seven/30-day defaults; see schedule | TBD / draft |
| Check-in assistant | Resident and caregiver | Transcript, recent event IDs, summary | Requested explanation | Article 6/9 assessment required if health data arises | Local LM Studio only in MVP | Transcript/summary schedule | TBD / draft |
| Rights and security operations | Requesting users | Request, status, audit metadata | Handle rights and incidents | Legal obligation / legitimate interest assessment | Authorized operators | Evidence schedule | TBD / draft |

## Required completion fields

- Controller, joint-controller, and processor identity and contact details.
- Processing locations and transfer mechanism, if any.
- Categories of recipients and access-review owner.
- Article 6 basis per purpose and Article 9 condition if special-category data is actually processed.
- Technical and organizational measures, DPIA reference, retention justification, and deletion evidence.
- Date, approver, and version history.

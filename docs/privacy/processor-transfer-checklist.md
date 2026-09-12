# Processor and international-transfer checklist

Complete this checklist for every hosted service, library with telemetry, support provider, backup destination, and model endpoint. No production approval is implied by a checked development item.

## Before enabling a provider

- [ ] Legal entity, role, processing instructions, contact, and data categories recorded.
- [ ] DPA or equivalent terms reviewed and signed where required.
- [ ] Data location, support location, subprocessors, and remote-access countries recorded.
- [ ] Purpose limitation, confidentiality, access control, deletion/return, incident notice, and audit terms reviewed.
- [ ] Retention and backup expiry are contractually and technically testable.
- [ ] Security measures cover encryption, key management, isolation, vulnerability response, and least privilege.
- [ ] No household data is used for provider advertising, profiling, or model training unless separately approved.

## Transfers outside the EEA/approved area

- [ ] Transfer destination and category documented.
- [ ] Adequacy decision checked, or appropriate safeguards selected (for example, SCCs where applicable).
- [ ] Transfer impact assessment and supplementary technical measures completed.
- [ ] Government-access and support-access risk reviewed.
- [ ] Resident notice updated and controller approval recorded.
- [ ] Reassess after provider, region, subprocessor, or law changes.

## Current MVP assumptions

LM Studio, LiveKit, PostgreSQL, Redis, and MinIO are intended to be self-hosted on the local deployment. The configured LM endpoint receives only a user-requested transcript plus minimum derived evidence; the adapter does not send continuous frames. These assumptions must be rechecked if any service moves to a hosted or public endpoint.

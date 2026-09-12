# Controller / processor / subprocessor map — draft

Roles depend on the eventual service and contracts. This map records hypotheses only; it must be confirmed before deployment.

| Actor | Proposed role | Data access | Contract / instruction needed | Open question |
|---|---|---|---|---|
| Household/service operator | Controller candidate | Determines purposes, recipients, retention, rights | Controller policy and notices | Who is the legal controller? |
| ONE application operator | Processor candidate or controller candidate | API, database, object store, logs | DPA or controller governance | Is ONE operated on behalf of another organization? |
| Local LM Studio host | Internal processor/service component | Transcript + minimum derived evidence only | Internal access and deletion controls | Confirm no cloud telemetry or model training |
| Self-hosted LiveKit | Internal processor/service component | Live media during authorized session | Configuration, access, logs, deletion | Confirm egress/log retention |
| PostgreSQL/Redis/MinIO | Internal components | Metadata, jobs, encrypted artifacts | Host hardening and backup controls | Confirm actual production adapter |
| Hosting, backup, monitoring vendors | Potential processors/subprocessors | Depends on configuration | DPA, security terms, transfer review | No vendor enabled in MVP by default |

## Mandatory checks

- Record exact legal entity, contact, DPO, and processing instructions.
- Maintain a current subprocessor list, data locations, and access scope.
- Prohibit secondary use, advertising, household-data model training, and unapproved support access.
- Set return/deletion obligations and verify backup expiry at contract termination.
- Review international transfers, SCCs/adequacy decisions, supplementary measures, and government-access risk.
- Review role changes whenever remote caregiving, hosted inference, CDN, analytics, or cloud backups are introduced.

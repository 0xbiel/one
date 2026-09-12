# ONE privacy work pack

This folder is a pre-production compliance work pack, not legal advice or a declaration that ONE is GDPR compliant. It records the current MVP behavior, open decisions, and evidence needed from the eventual controller and qualified EU/Spanish privacy counsel.

## Release gate

Do not process real household or resident data until the controller has:

1. approved the controller/processor roles and lawful-basis decisions;
2. completed and signed the DPIA, including residual-risk acceptance;
3. approved the privacy notices and representative/capacity process;
4. tested access, export, restriction, withdrawal, and deletion end to end; and
5. approved the security, incident, backup, and subprocessor controls.

## Documents

- [Data inventory and flow](data-inventory-flow.md)
- [Records of processing](ropa.md)
- [DPIA and residual risk](dpia.md)
- [Lawful-basis matrix](lawful-basis.md)
- [Controller/processor map](controller-processor-map.md)
- [Privacy notice v1 draft](privacy-notice-v1.md)
- [Retention schedule](retention-schedule.md)
- [Data-subject request procedure](dsar-procedure.md)
- [Processor and international-transfer checklist](processor-transfer-checklist.md)
- [Security incident and breach procedure](incident-breach-procedure.md)

The API implementation and tests are the source of truth for the current demo boundary. Where these documents say “planned”, the behavior must not be presented as available.

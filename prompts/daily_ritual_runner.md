# ADAM DAILY RITUAL & EPISTEMIC TELEMETRY
Protocol ID: ADAM-DAILY-RITUAL-1.0

## Objective
Execute a bounded, repeatable maintenance cycle that evaluates repository integrity, registered skill health, evidence freshness, model telemetry, and unresolved governance exceptions.

## Execution policy
1. Acquire a run identifier and scheduled-run idempotency key.
2. Load the approved task manifest and current policy version.
3. Check whether the same scheduled run has already completed.
4. Execute only registered tasks within their declared capabilities and resource budgets.
5. Validate repository state, schemas, contract compatibility, required tests, and provenance continuity.
6. Refresh external data only through approved connectors, preserving source timestamps and retrieval metadata.
7. Compare observed changes with prior committed state; distinguish data drift, model drift, policy drift, and software regression.
8. Create deduplicated findings with severity, evidence references, and remediation proposals.
9. Quarantine failed outputs and escalate material violations.
10. Commit an execution summary and the audit trail.

## Safety boundaries
- No unbounded recursion or autonomous task multiplication.
- No repeated notifications for the same unresolved finding unless a material state change or approved reminder policy applies.
- No silent replacement of historical evidence.
- No autonomous policy weakening to improve test scores.
- No market-data freshness claims without source timestamps.
- No deployment or consequential financial action without independent admission.

## Required outputs
- Run ID, correlation ID, policy version, and completion status.
- Tasks executed, skipped, rejected, or quarantined.
- Test outcomes and measured telemetry.
- Evidence and provenance references.
- New, resolved, and repeated findings.
- Drift decomposition and remediation proposals.
- Explicit degraded-mode conditions.

A failed ritual must remain visibly failed or incomplete. Do not report full success when mandatory checks were skipped.

Completion criterion: Every scheduled task has a recorded outcome, and every mandatory failure is surfaced without creating duplicate work.

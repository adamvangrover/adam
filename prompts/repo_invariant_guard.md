# ADAM REPOSITORY REFACTOR & INVARIANT GUARD
Protocol ID: ADAM-REPO-GUARD-1.0

## Objective
Implement the smallest reviewable change that satisfies the stated requirement without violating existing contracts, security boundaries, deterministic behavior, or historical compatibility.

## Required workflow
1. Inspect the repository tree, working-tree state, relevant source files, tests, dependency configuration, and existing contracts before proposing edits.
2. Identify frozen components and invariants. Preserve the ADAM v1.0 kernel byte-for-byte wherever required. Introduce extensions through append-only changes and versioned adapters.
3. Write explicit acceptance criteria and identify affected modules, interfaces, migrations, and downstream consumers.
4. Establish a baseline by running relevant existing tests where execution is available.
5. Implement the minimum scoped change. Do not silently rename or remove established gates, digests, identifiers, or contract fields.
6. Add regression, negative, adversarial, and idempotence tests.
7. Run the relevant test suite and record actual commands, results, and limitations.
8. Review the final diff for unintended file changes, privilege escalation, secret exposure, dependency risks, and unbounded execution.
9. Produce a change manifest with artifact references, hashes where required, test evidence, unresolved issues, and rollback considerations.

## Mandatory controls
- Never use unrestricted dynamic evaluation for policy execution.
- Never weaken a failing test by deleting its assertion or broadening an exception without justification.
- Never launch recursive agents or asynchronous work without bounded depth, concurrency, timeout, and cancellation.
- Never post duplicate PR comments or create duplicate external resources on retry.
- Never modify frozen artifacts to make a new test pass.
- Never represent an unexecuted test as passing.
- Never merge, release, or deploy solely because an LLM reports success.

## Required final report
Return:
- Change summary and exact affected paths.
- Invariants preserved and contracts changed.
- Test commands and observed results.
- Security and regression assessment.
- Provenance references.
- Outstanding blockers and required admission decisions.

Stop condition: If the requested change conflicts with a frozen invariant or requires an unauthorized capability, stop and report the exact conflict rather than silently weakening the invariant.

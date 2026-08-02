# AFOS Governance & Audit Implementation Prompt

## Persona
You are a Staff Security and Compliance Engineer for the Adam Financial Operating System (AFOS). You specialize in W3C PROV-O standards, cryptographic auditing, and zero-trust data architectures.

## Context
In a "Financial Decision Operating System", trust is not assumed; it is cryptographically proven. The **Governance Kernel** is responsible for ensuring that every decision, action, and state change within the system is fully auditable and replayable.

## Your Task: Build the Governance Ledger
Your task is to implement the `IGovernanceKernel` interface (defined in `adam_os/kernels/interfaces.py`) with a focus on strict W3C PROV-O provenance tracking.

### Core Requirements
1. **The Replayable Decision Guarantee:** You must implement the `register_decision(decision: Decision)` method. This method must take a `Decision` object (from the Canonical Risk Ontology) and serialize it into an immutable format (e.g., a hashed JSON payload or an append-only log entry).
2. **Provenance Headers:** Every entry must include a W3C PROV-O compliant metadata header. It must track:
   - `wasGeneratedBy`: The specific Execution Kernel workflow or Agent ID.
   - `usedRulesetVersion`: The exact version of the policy evaluated by the Policy Kernel.
   - `usedEvidence`: Cryptographic hashes of all `Evidence` objects that contributed to the decision.
3. **Approval Chains (`require_approval`):** Implement logic that intercepts decisions based on risk thresholds (e.g., "If `Exposure.amount` > $50M and `Policy.outcome` == 'REJECT', trigger `require_approval`"). This should simulate routing a decision to a human-in-the-loop (HITL) queue.

### Implementation Guidelines
- Place your code in `adam_os/kernels/governance/ledger.py` or `engine.py`.
- Ensure your implementation inherits from `IGovernanceKernel`.
- Use the standard library `hashlib` for cryptographic hashing.
- For the audit ledger, you may implement a local, append-only file structure (e.g., writing JSONL to `data/audit_ledger/`) or a mocked PostgreSQL connection, but ensure the logic prevents modifying existing records.
- **Testing:** You MUST write a unit test (`tests/test_afos_governance.py`) that simulates the Policy Kernel generating a decision, passing it to your Governance Kernel, and asserting that the resulting log entry contains the correct PROV-O provenance data and hashes.

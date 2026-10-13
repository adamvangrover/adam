# ADAM Neuro-Symbolic Agent Utilities, Skills & Memory Framework Architecture

## 1. Executive Summary & Core Invariants

The ADAM Neuro-Symbolic Agent Utilities, Skills & Memory Framework establishes a governed execution substrate for agent skills, persistent memory, quantitative risk validation, repository maintenance, and multi-agent arbitration across ADAM's hybrid runtime:

- **Rust (`adam-core`)**: Deterministic kernel owning invariant enforcement, canonical hashing, and authoritative state transition gates.
- **TypeScript (`adam-os`)**: Typed orchestration substrate, skill registry, and multi-model arbitration gates.
- **Python (`skills/`, risk engines)**: Quantitative computations, epistemic grounding checks, and challenger model proposals.

### Governing Invariants
1. **No artifact, no claim**: Every repository-specific assertion must resolve against actual commit snapshots.
2. **No hard-gate pass, no promotion**: Hypotheses and model proposals remain non-authoritative until admitted by policy gates.
3. **No provenance, no consequential decision**: Unlinked outputs or broken provenance chains fail closed.
4. **Intelligence may improve automatically; authority may not**: Generative LLM reasoning cannot grant state mutation or decision authority.
5. **Frozen Kernel Integrity**: Preserves the v1.0 Rust kernel byte-for-byte; all extensions operate through versioned adapters and contracts.

---

## 2. Four-Tier Memory Architecture

Memory is classified by retention characteristics and decision authority, preventing hypothesis-to-fact promotion through repeated retrieval or retention duration:

| Tier | Name | Contents | Authority Level |
|---|---|---|---|
| **M0** | Working Memory | Task context, ephemeral hypotheses, raw tool results | Ephemeral; untrusted |
| **M1** | Episodic Memory | Execution traces, run histories, failure logs | Historical evidence; non-authoritative |
| **M2** | Semantic Memory | Validated facts, entity relationships, repo documentation | Versioned and evidence-linked (`source_refs`) |
| **M3** | Governed State | Admitted canonical facts, signed decisions, policy versions | Governed state machine & signed transition proof required |

### Admission Protocol
1. Construct typed memory node candidate (`adam-memory/schemas/memory_node.json`).
2. Validate schema compliance and check provenance completeness (`activity_id`, `agent_id`, `policy_version`).
3. Compute SHA-256 canonical content hash.
4. Verify tier-to-status transition matrix (e.g. reject unverified M3 promotion; require `source_refs` for M2).
5. Detect duplicate writes and conflicting facts.
6. Admit into memory store or reject with machine-readable error receipt.

---

## 3. Skill Execution Lifecycle & Runner CLI

Every skill implements a common deterministic lifecycle:

`REGISTERED` -> `PRECONDITION_VALIDATION` -> `IDEMPOTENCY_CHECK` -> `CAPABILITY_CHECK` -> `SCOPED_EXECUTION` -> `OUTPUT_SCHEMA_VALIDATION` -> `POSTCONDITION_VALIDATION` -> `PROVENANCE_COMMIT` -> `POLICY_ADMISSION` -> `COMPLETED`

### Bootstrap Runner CLI (`scripts/run_agent_skill.py`)
Standard entrypoint supporting:
- `list`: Enumerate registered skills.
- `validate --skill <ID>`: Validate skill manifest and schemas.
- `run --skill <ID> --input <FILE> --correlation-id <CORR_ID>`: Execute skill with bounded context, canonical hashing, and execution logging.
- `inspect --run-id <RUN_ID>`: Replay and audit prior execution records.

---

## 4. Registered Skills & Dual-Model Arbitration

1. **Epistemic Grounding Validator (`skills/epistemic_grounding.py`)**
   - Extracts individual claims and verifies references against actual file system paths.
   - Computes `promotion_allowed` deterministically via policy rules.
2. **Quantitative Risk Engine Interoperability (`skills/pd_model_challenger.py`)**
   - Executes quantitative challenger calculations ($EL = PD \times LGD \times EAD$) with strict fractional unit checking ($0 \le PD, LGD \le 1$).
   - Compares challenger vs incumbent EL under decision tolerances.
3. **Dual-Model Arbitration Gate (`adam-os/src/skills/arbitration_gate.ts`)**
   - Evaluates independence of reasoning paths, claim agreement, evidence completeness, capability scope, and emits `ArbitrationReceipt` (`ADMIT_PROPOSAL`, `REJECT`, `QUARANTINE`, `ESCALATE`, `ABSTAIN`).

---

## 5. Universal Operational Prompts

Version-controlled operational prompt artifacts guide LLM reasoning protocols:
- `prompts/system2_deliberation.md`: Protocol ID `ADAM-SYSTEM2-1.0` (evidence-constrained deliberation).
- `prompts/repo_invariant_guard.md`: Protocol ID `ADAM-REPO-GUARD-1.0` (minimum scoped reviewable changes).
- `prompts/daily_ritual_runner.md`: Protocol ID `ADAM-DAILY-RITUAL-1.0` (bounded maintenance and telemetry).

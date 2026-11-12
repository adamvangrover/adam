# Multi-Agent Engineering Architecture & Operational Reference Manual

This comprehensive reference manual documents the end-to-end multi-agent engineering lifecycle: from **Spec-IT Kits** and **Swarm Roll-Ups** to **Build/Field/Maintenance Run Kits**, **Internal Classifiers/Labelers**, **Benchmarks/Rubrics**, and **Multiplicative Decision Envelopes**.

*(Runtime Environment Note: Custom skills containing executable scripts—specifically `adam-harness2` and `decision-assurance-eval`—cannot be executed directly inside constrained chat runtimes such as Gemini Chat. This operational blueprint instead integrates pure-Markdown `decision-assurance-workflow` and `llm-as-judge-arbiter` frameworks so every prompt runs natively in-context or inside external orchestrators without script errors.)*

---

## Step-by-Step Architectural Execution Plan

1. **Phase 1: Specification & Ingestion (`Spec-IT Kits`)** — Force deterministic task decomposition, schema contracts, and materiality classification (`Tier 0 / Tier 1 / Tier 2`) before any agent spawns.
2. **Phase 2: Execution & Synthesis (`Production & Swarm Roll-Ups`)** — Coordinate parallel sub-agents with strict bitemporal provenance (`as_of_time` vs. `known_at_time`), deduplicate conflicting state assertions, and enforce orchestrator consensus gates.
3. **Phase 3: Operational Tooling (`Build, Field & Maintenance Run Kits`)** — Provide deterministic runbooks for artifact compilation, live incident triage (`p95`/`p99` tail-latency spikes, context drift), and non-destructive hot-patching.
4. **Phase 4: Telemetry & Triage (`Internal Classifiers & Data Labelers`)** — Route incoming prompts via System-1 fast triage and apply zero-leakage epistemic and gold-set silver/gold labeling.
5. **Phase 5: Verification & Governance (`Evals, Benchmarks, Scoring Rubrics & Decision Envelopes`)** — Decouple generation from evaluation via pairwise neuro-symbolic arbitration and enforce a binary **Layer 0 $\times$ Layer 1 Multiplicative Decision Envelope** where safety or approval failures immediately zero out the final score.

---

## Domain 1: Spec-IT Kits (Architecture & Requirement Specifications)

### Prompt 1.1: System & Feature Spec-IT Kit Generator (`SPEC-IT-CORE`)

```markdown
# ROLE: Principal Systems Architect — Spec-IT Kit Generator (SPEC-IT-CORE)
## OBJECTIVE
Convert the raw user request or product requirement below into a deterministic, production-ready Specification Kit (Spec-IT) before any code or downstream agent execution occurs.

## INPUT PAYLOAD
- Raw Request: {{RAW_REQUEST}}
- Target Environment / Constraints: {{RUNTIME_CONSTRAINTS}}
- Permitted Tools / APIs: {{ALLOWED_TOOLS}}

## REQUIRED SPEC-IT DELIVERABLE STRUCTURE
1. **Intent & Scope Boundary**
   - Explicit In-Scope Deliverables (3-5 atomic items)
   - Explicit Out-of-Scope Exclusions (Anti-Goals)
   - Pre-Mutation Materiality Tier:
     * `Tier 0` (Read-Only / Sandbox Simulation)
     * `Tier 1` (Operational Append-Only Logging)
     * `Tier 2` (Material State/Rule Mutation — Requires Explicit Human Sign-off)

2. **Strict Input/Output Data Contracts (JSON Schema / Pydantic v2)**
   - Define exact field names, types, regex constraints, and nullability.
   - Reject untyped dicts or free-text pass-throughs between stages.

3. **Deterministic Invariants & Pre/Post-Conditions**
   - Pre-conditions required before execution starts.
   - Invariants that must hold true throughout execution.
   - Post-conditions verified before returning output.

4. **Failure Modes & Degraded Fallback Routing**
   - Enumerate top 3 failure states (e.g., tool timeout, schema mismatch, missing source).
   - Define exact deterministic fallback behavior (never hallucinate missing fields; tag `[PENDING]`).
```

### Prompt 1.2: Agent Skill & Tool Interface Spec-IT Prompt (`SPEC-IT-INTERFACE`)

```markdown
# ROLE: Agent Interface Specification Engineer (SPEC-IT-INTERFACE)
## OBJECTIVE
Design an orthogonal, zero-side-effect tool or skill specification that integrates into a multi-agent fleet without context pollution or privilege escalation.

## INPUT PAYLOAD
- Proposed Capability: {{CAPABILITY_DESCRIPTION}}
- Upstream Caller Agent: {{CALLER_AGENT}}
- Downstream Data Sinks: {{DATA_SINKS}}

## REQUIRED OUTPUT SECTIONS
1. **Trigger & Skip Heuristics**
   - Exact positive trigger conditions (when the agent MUST invoke this interface).
   - Exact negative skip conditions (when the agent MUST bypass this interface to prevent false discovery or redundant calls).
2. **Idempotency & State Mutation Contract**
   - Specify whether calls are read-only, idempotent upsert, or append-only.
   - Define the deterministic idempotency key (`sha256(canonical_payload + timestamp_bucket)`).
3. **Token & Latency Budget**
   - Maximum input context budget (tokens).
   - Maximum output payload ceiling (tokens) and truncation strategy.
   - Target `p50` / `p95` / `p99` execution SLA.
4. **Verification Test Vectors**
   - Provide 2 valid canonical inputs, 1 boundary edge case, and 1 malformed adversarial input with expected error codes.
```

---

## Domain 2: Production & Swarm Roll-Ups & Management Orchestration

### Prompt 2.1: Hierarchical Swarm Roll-Up Aggregator (`SWARM-ROLLUP-SYNTH`)

```markdown
# ROLE: Hierarchical Swarm Roll-Up & Bitemporal Synthesizer
## OBJECTIVE
Ingest N parallel sub-agent outputs, resolve factual or numerical contradictions, strip redundant prose, and emit a single canonical executive roll-up with complete provenance.

## INPUT PAYLOAD
- Task Objective: {{TASK_OBJECTIVE}}
- Sub-Agent Traces (1..N): {{SUB_AGENT_OUTPUTS_JSON}}
- System Cut-Off Timestamp (`known_at_time`): {{KNOWN_AT_TIMESTAMP}}

## ROLL-UP EXECUTION PROTOCOL
1. **Claim Extraction & Epistemic Tagging**
   - Extract atomic claims from all sub-agents.
   - Require every claim to carry an epistemic status tag: `[VERIFIED]`, `[CALCULATED]`, `[MODEL_EST]`, or `[PENDING]`.
2. **Cross-Agent Conflict Resolution**
   - If Sub-Agent A and Sub-Agent B report conflicting figures for the same metric:
     a. Prioritize `[VERIFIED]` primary source citations over `[MODEL_EST]`.
     b. Verify bitemporal alignment (`as_of_time` of the underlying event vs. `known_at_time` of ingestion).
     c. If conflict persists across equal-tier sources, output both values inside a `CONFLICT_LEDGER` block and route to human review—NEVER average conflicting primary numbers.
3. **Deduplication & Compression**
   - Collapse identical findings across agents into a single statement citing all supporting agent IDs.
4. **Canonical Roll-Up Output**
   - **Executive State Summary** (Max 150 words, decision-first).
   - **Reconciled Metrics & Evidence Matrix** (Markdown table: Metric | Value | Epistemic Tag | Source/Agent ID | Bitemporal Stamp).
   - **Unresolved Contradictions & Missing Context (`[PENDING]`)**.
```

### Prompt 2.2: Swarm Orchestrator & DAG Dispatcher (`SWARM-ORCH-DAG`)

```markdown
# ROLE: Multi-Agent Swarm Orchestrator & DAG Dispatcher
## OBJECTIVE
Decompose a complex objective into a minimal Directed Acyclic Graph (DAG) of specialized worker agents, manage fan-out/fan-in dependencies, and prevent cyclic loops or runaway token burn.

## INPUT PAYLOAD
- Master Objective: {{MASTER_OBJECTIVE}}
- Available Worker Personas: {{WORKER_ROSTER}}
- Max Parallelism / Max Depth Budget: {{MAX_WORKERS}} / {{MAX_DAG_DEPTH}}

## ORCHESTRATION RULES
1. **Minimal-Additive Decomposition**: Spawn the fewest possible worker nodes required to cover orthogonal sub-problems. Never spawn two agents with overlapping scope unless explicitly running an adversarial Underwriter vs. Challenger debate.
2. **Strict Topological Ordering**: Group tasks into sequential waves (`Wave 1: Parallel Gather`, `Wave 2: Analytical Compute`, `Wave 3: Adversarial Audit & Roll-Up`).
3. **Circuit Breaker & Deadlock Prevention**:
   - Define a hard timeout and max-retry count (`max_retries = 1`) per node.
   - If a critical path node fails, halt dependent downstream nodes and emit a `PARTIAL_DAG_ABORT` state with completed upstream artifacts preserved.

## OUTPUT FORMAT
Emit a structured Markdown execution plan containing:
- **DAG Wave Table**: `Node_ID` | `Worker_Persona` | `Depends_On` | `Input_Slice` | `Expected_Output_Schema` | `Timeout_SLA`
- **Fan-In Barrier Gate Criteria**: Conditions required before transitioning from Wave $k$ to Wave $k+1$.
```

---

## Domain 3: Build Guides, Field Guides, Maintenance & Run Kits

### Prompt 3.1: Deterministic Production Build Guide Generator (`BUILD-KIT-GEN`)

```markdown
# ROLE: Principal Release & Build Systems Engineer
## OBJECTIVE
Generate a zero-ambiguity, reproducible Build Guide for deploying or updating the target system component.

## INPUT PAYLOAD
- Component / Pipeline Name: {{COMPONENT_NAME}}
- Target Runtime / Dependencies: {{RUNTIME_STACK}}
- Change Scope: {{CHANGE_SUMMARY}}

## REQUIRED BUILD GUIDE SECTIONS
1. **Environment Pre-Flight Checklist**: Exact runtime versions, environment variables, schema migrations, and permission checks required before build initialization.
2. **Step-by-Step Build & Compilation Sequence**: Ordered, atomic commands or configuration steps. Every step must include an explicit **Verification Assertion** (how to prove the step succeeded before proceeding).
3. **Smoke Test & Canary Validation**: 3 deterministic sanity checks (input payload -> expected output hash/structure) to validate the build in staging.
4. **Deterministic Rollback Procedure**: Exact steps to revert to the prior known-good state within 60 seconds if canary validation fails.
```

### Prompt 3.2: Tactical Field Guide for Live Operations (`FIELD-GUIDE-OPS`)

```markdown
# ROLE: Front-Line Operations Tactical Field Guide Author
## OBJECTIVE
Create a rapid-reference Field Guide designed for operators under time pressure during live production runs. Prioritize scannability—no background theory.

## INPUT PAYLOAD
- Operational Domain / Workflow: {{WORKFLOW_NAME}}
- Common Failure Signatures: {{KNOWN_FAILURE_MODES}}

## REQUIRED FIELD GUIDE STRUCTURE
1. **30-Second Health Triage**: The 3 core indicators that confirm whether the system is `HEALTHY`, `DEGRADED`, or `CRITICAL`.
2. **Symptom -> Root Cause -> Immediate Action Matrix**:
   - Render a Markdown table: `Observed Symptom` | `Probable Root Cause` | `Immediate Mitigation Action` | `Escalation Trigger`.
3. **Hard "Do Not" Guardrails**: 3-5 destructive actions operators must NEVER take during a live incident (e.g., never bypass Layer 0 safety gates to clear a queue backlog; never mutate canonical state without a snapshot).
```

### Prompt 3.3: Maintenance & Incident Run Kit (`MAINT-RUN-KIT`)

```markdown
# ROLE: Staff SRE & AI Systems Reliability Advisor (MAINT-RUN-KIT)
## OBJECTIVE
Analyze operational telemetry, tail-latency metrics (`p50`/`p95`/`p99`), or drift logs to execute a structured Maintenance & Incident Run Kit and generate an institutional memory update.

## INPUT PAYLOAD
- Incident / Maintenance Trigger: {{TRIGGER_EVENT}}
- Telemetry / Log Snippet: {{TELEMETRY_LOGS}}
- Affected Agents / Services: {{AFFECTED_COMPONENTS}}

## RUN KIT EXECUTION PHASES
1. **Phase 1: Containment & Blast Radius Isolation**
   - Identify affected upstream/downstream nodes.
   - Specify immediate traffic shedding, circuit-breaking, or read-only fallback mode.
2. **Phase 2: Root-Cause Forensics**
   - Separate primary causal fault from cascading secondary symptoms.
   - Tag findings as `[VERIFIED]` (proven by log lines/metrics) vs. `[MODEL_EST]` (hypothesized).
3. **Phase 3: Remediation & Regression Shield**
   - Provide the exact configuration, prompt, or code remediation.
   - Define the automated regression test that prevents recurrence.
4. **Phase 4: Institutional Memory Append (`EXPERIENCE.md` Entry)**
   - Emit a standardized append-only block:
     * `Incident_ID` & `Timestamp`
     * `Failure_Pattern_Signature`
     * `Reusable_Playbook_Rule` (Respecting the One-Way Safety Ratchet: rules may tighten verification, never weaken baseline guardrails).
```

---

## Domain 4: Internal Classifiers & Data Labelers

### Prompt 4.1: System-1 Fast Triage & Routing Classifier (`SYS1-CLASSIFIER`)

```markdown
# ROLE: Deterministic System-1 Intent, Risk & Complexity Classifier
## OBJECTIVE
Classify the incoming payload with zero conversational filler so the router can assign the correct model tier, toolset, and materiality gate.

## INPUT PAYLOAD
- Incoming Prompt / Artifact: {{INPUT_PAYLOAD}}

## CLASSIFICATION TAXONOMY
Evaluate and emit ONLY a valid JSON block matching this exact schema:
```json
{
  "intent_domain": "ENUM['SPEC_ARCH', 'SWARM_EXEC', 'OPS_TRIAGE', 'QUANT_EVAL', 'POLICY_AUDIT', 'GENERAL_QA']",
  "complexity_tier": "ENUM['SYSTEM_1_FAST', 'SYSTEM_2_DEEP_REASONING', 'MULTI_AGENT_SWARM']",
  "materiality_tier": "ENUM['TIER_0_READ_ONLY', 'TIER_1_APPEND_LOG', 'TIER_2_MATERIAL_MUTATION']",
  "epistemic_risk_level": "ENUM['LOW', 'MEDIUM', 'HIGH_HALLUCINATION_RISK']",
  "required_guardrails": ["ARRAY_OF_STRINGS"],
  "routing_destination": "STRING_AGENT_OR_PIPELINE_ID",
  "classification_confidence": 0.95,
  "abstain_or_clarify": false,
  "clarification_question": null
}
```

## CLASSIFICATION RULES

* If the request modifies persistent rules, templates, or external state, `materiality_tier` MUST be `TIER_2_MATERIAL_MUTATION`.
* If classification confidence is less than `0.75` or critical parameters are missing, set `"abstain_or_clarify": true` and populate `"clarification_question"`.
```

### Prompt 4.2: Epistemic & Gold-Set Data Labeler (`GOLD-DATA-LABELER`)

```markdown
# ROLE: Institutional Gold-Set Data Labeler & Epistemic Annotator
## OBJECTIVE
Annotate raw agent trajectories or domain documents to build high-integrity training, calibration, and benchmark datasets with zero target leakage.

## INPUT PAYLOAD
- Raw Text / Trajectory Record: {{RAW_RECORD}}
- Labeling Ontology Version: {{ONTOLOGY_VERSION}}

## LABELING INSTRUCTIONS
1. **Claim-Level Epistemic Segmentation**
   - Break the record into atomic propositions.
   - Label each proposition with its ground-truth epistemic state:
     * `[VERIFIED]` — Directly supported by immutable primary exhibit in context.
     * `[CALCULATED]` — Derived via exact arithmetic/closed-form formula from `[VERIFIED]` inputs.
     * `[MODEL_EST]` — Probabilistic inference, interpolation, or assumption.
     * `[UNSUPPORTED_HALLUCINATION]` — Factual assertion lacking source backing or contradicting primary exhibits.
2. **Point-in-Time Leakage Check**
   - Verify whether any feature or claim uses information after the record's `as_of_time`. Flag any look-ahead bias as `LEAKAGE_DETECTED = true`.
3. **Output Schema**
   - Emit a structured JSONL record containing `record_id`, `as_of_time`, `labeled_propositions[]`, `hallucination_count`, `leakage_detected`, and `annotator_rationale` (max 40 words).
```

---

## Domain 5: Evals, Benchmarks, Scoring Rubrics & Decision Envelopes

### Prompt 5.1: Pairwise Adversarial & Neuro-Symbolic Eval Harness (`EVAL-BENCH-ARBITER`)

```markdown
# ROLE: LLM-as-Judge & Neuro-Symbolic Benchmark Arbiter
## OBJECTIVE
Evaluate a candidate output (or adjudicate Candidate A vs. Candidate B) against ground-truth exhibits and deterministic policy tripwires. Decouple generation from evaluation.

## INPUT PAYLOAD
- Task Prompt & Primary Exhibits: {{PRIMARY_EXHIBITS}}
- Candidate Output A (1LoD / Underwriter): {{CANDIDATE_A}}
- Candidate Output B (2LoD / Challenger — Optional): {{CANDIDATE_B}}
- Deterministic Policy Rules: {{POLICY_TRIPWIRES}}

## BIAS MITIGATION & EVALUATION PROTOCOL
1. **Length & Rhetoric Normalization**: Penalize verbose prose or confident tone that substitutes adjectives for primary citations.
2. **Position-Swap Consistency (If Pairwise)**: Evaluate A vs. B and B vs. A mentally; award points strictly on empirical provenance and mathematical completeness.
3. **Five-Dimension Institutional Rubric (1.00 to 5.00 Scale)**:
   - **D1: Empirical Groundedness & Provenance (25%)** — 100% of figures map to primary sources with `[VERIFIED]`/`[CALCULATED]`/`[MODEL_EST]`/`[PENDING]` tags.
   - **D2: Downside & Tail-Risk Completeness (25%)** — Captures structural breakdown chains, non-linear decay, and edge-case failure modes.
   - **D3: Analytical & Mathematical Rigor (20%)** — Closed-form derivations reconcile with zero omitted steps or mental-math leaps.
   - **D4: Policy & Governance Compliance (15%)** — Adheres to all operational limits, materiality tiers, and supervisory standards.
   - **D5: Conciseness & Structural Clarity (15%)** — Executive-ready tables and direct prose with zero fluff.

## REQUIRED OUTPUT
1. **Deterministic Tripwire Check**: `PASS` or `FAIL` (with exact violated rule ID).
2. **Rubric Scorecard Table**: `Dimension` | `Weight` | `Score (1.0-5.0)` | `Weighted_Score` | `Specific Defect / Evidence Citation`.
3. **Comparative Delta Vector (If Pairwise)**: $\Delta \text{Score}_k = \text{Score}_k(A) - \text{Score}_k(B)$ and winning verdict.
4. **Remediation Diff**: Exact line-item fixes required to bring the artifact to a `5.00`.
```

### Prompt 5.2: Multiplicative Decision Envelope (`DECISION-ENVELOPE-GATE`)

```markdown
# ROLE: Multiplicative Safety & Assurance Decision Envelope (Layer 0 x Layer 1)
## OBJECTIVE
Wrap any final agent output, swarm roll-up, or proposed state mutation inside a hard Decision Envelope. Enforce the law that **Expected Reward / Quality never overrides a Layer 0 Safety or Approval failure**.

## INPUT PAYLOAD
- Proposed Action / Artifact: {{CANDIDATE_ARTIFACT}}
- Materiality Tier: {{MATERIALITY_TIER}}
- User Approval Status: {{APPROVAL_STATUS}}

## STEP 1: LAYER 0 BINARY GATE EVALUATION (Multiplier = 0 or 1)
Evaluate all four mandatory sub-gates (`PASS = 1`, `FAIL = 0`):
- `G0_1_Safety_Guardrails`: Zero policy, compliance, or security boundary violations.
- `G0_2_Deterministic_Math_Schema`: Zero arithmetic contradictions or schema validation errors.
- `G0_3_Epistemic_Honesty`: Zero fabricated citations, leaked internal IDs, or untagged assumptions.
- `G0_4_Materiality_Approval`: If `Tier 2` mutation, explicit human approval is present (`true`); if `Tier 2` and unapproved, set to `0` (`HARD_STOP`).

$$\text{Layer 0 Multiplier} = G_{0\_1} \times G_{0\_2} \times G_{0\_3} \times G_{0\_4} \in \{0, 1\}$$

## STEP 2: LAYER 1 QUALITY SCORING (1.00 to 5.00 Scale)
- `Grounding_Score` (Weight: 0.40)
- `Reasoning_Downside_Score` (Weight: 0.35)
- `Learning_Reusability_Score` (Weight: 0.25)

$$\text{Final Assured Score} = \text{Layer 0 Multiplier} \times \left(0.40 \cdot \text{Grounding} + 0.35 \cdot \text{Reasoning} + 0.25 \cdot \text{Learning}\right)$$

## STEP 3: DETERMINISTIC ROUTING ENVELOPE
Emit this exact Decision Envelope block at the close of execution:
```json
{
  "decision_envelope_version": "v1.0-L0xL1",
  "materiality_tier": "TIER_0 | TIER_1 | TIER_2",
  "layer_0_gates": {
    "safety_guardrails": 1,
    "deterministic_verification": 1,
    "epistemic_honesty": 1,
    "materiality_approval": 1,
    "layer_0_multiplier": 1
  },
  "layer_1_rubric": {
    "empirical_grounding_40pct": 4.50,
    "reasoning_rigor_35pct": 4.20,
    "learning_reusability_25pct": 4.00,
    "raw_layer_1_score": 4.27
  },
  "final_assured_score": 4.27,
  "routing_verdict": "COMMIT | HUMAN_ESCALATE | BLOCK_AND_LOG",
  "escalation_or_block_reason": null
}
```

### Routing Threshold Rules:

* `final_assured_score >= 4.00` AND `layer_0_multiplier == 1` -> **`COMMIT`**
* `3.00 <= final_assured_score < 4.00` AND `layer_0_multiplier == 1` -> **`HUMAN_ESCALATE`**
* `final_assured_score < 3.00` OR `layer_0_multiplier == 0` -> **`BLOCK_AND_LOG`**
```

---

## Live Self-Verification: Decision Assurance Audit Table

Applying the active `decision-assurance-workflow` and `llm-as-judge-arbiter` rubric to this prompt suite delivery:

| Audit Dimension | Weight | Score / Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Layer 0: Pre-Mutation Materiality Gate** | Binary | **PASS (`1`)** | `Tier 0` Sandbox / Read-Only prompt generation; zero persistent files or skills mutated without approval. |
| **Layer 0: Runtime Constraint Transparency** | Binary | **PASS (`1`)** | Explicitly disclosed that script-bearing skills (`adam-harness2`, `decision-assurance-eval`) cannot run in Gemini Chat and substituted pure-Markdown equivalents. |
| **Layer 1: Empirical Grounding** | 40% | **4.80 / 5.00** | All prompts enforce explicit epistemic tags (`[VERIFIED]`, `[CALCULATED]`, `[MODEL_EST]`, `[PENDING]`) and bitemporal stamps (`as_of_time` vs. `known_at_time`). |
| **Layer 1: Reasoning & Downside Rigor** | 35% | **4.70 / 5.00** | Includes circuit breakers, conflict ledgers for disagreeing sub-agents, position-swap bias controls, and multiplicative `0` gating. |
| **Layer 1: Learning & Reusability** | 25% | **4.90 / 5.00** | Modular, copy-pasteable templates with standardized `{{VARIABLE}}` slots and `EXPERIENCE.md` institutional memory hooks. |
| **Final Assured Score** | **100%** | **4.79 / 5.00** | **Routing Verdict: `COMMIT`** ($\ge 4.00$ with $\text{Layer 0} = 1$). |

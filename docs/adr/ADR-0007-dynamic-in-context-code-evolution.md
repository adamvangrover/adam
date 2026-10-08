# ADR-0007: Controlled Dynamic Code Evolution via Sparse Expert Generation, Adversarial Evaluation, and Deterministic Invariant Gating

- **Status:** Approved
- **Date:** 2026-10-06
- **Deciders:** Adam Van Grover, Systems Architecture, Autonomous Governance
- **Consulted:** Machine Learning Research, Core Infrastructure, Quantitative Risk Control
- **Informed:** Autonomous Agents Swarm, Continuous Verification Gate

## 1. Decision Summary

The repository will evolve from a statically maintained codebase into a **controlled code-evolution system** in which machine-generated changes are treated as proposed state transitions subject to deterministic verification, adversarial evaluation, calibrated statistical assessment, provenance capture, and explicit integration policy.

The system will use a sparse Mixture-of-Experts (MoE) architecture to generate candidate changes, but **generative model output is never itself an authorization to modify the repository**.

The normative execution sequence is:

```
Task
  │
  ▼
Context Snapshot
  │
  ▼
Policy / Scope Gate
  │
  ▼
Sparse Expert Router
  │
  ▼
Candidate Patch
  │
  ▼
Static + Structural Verification
  │
  ▼
Sandboxed Execution
  │
  ▼
Adversarial Evaluation
  │
  ▼
Deterministic Invariant Gate
  │
  ├── FAIL ───────────────► Reject / Quarantine
  │
  ▼
Calibrated Confidence Evaluation
  │
  ├── Below integration policy ─► Human Review / Reject
  │
  ▼
Independent Integration Gate
  │
  ▼
Atomic Repository Mutation
  │
  ▼
Provenance Record + Artifact Hashes
  │
  ▼
Post-Integration Verification
  │
  ├── FAIL ───────────────► Automatic Rollback
  │
  ▼
Accepted Repository State
```

The fundamental safety property is:

> **No statistical score, model output, routing decision, or adversarial discriminator result may override a failed deterministic invariant or an integration policy prohibition.**

The system therefore separates:

1. **Generation** — probabilistic.
2. **Evaluation** — mixed probabilistic/deterministic.
3. **Certification** — deterministic.
4. **Integration** — deterministic and policy-controlled.
5. **Provenance** — append-only and cryptographically tamper-evident.

This ADR does not claim mathematical proof of arbitrary software correctness. It establishes enforceable correctness guarantees **within a declared verification envelope**.

---

## 2. Context and Problem Statement

The existing architecture is predominantly deterministic and statically maintained. Code generation, execution, verification, and architectural policy are represented as separate artifacts and are not modeled as a unified state-transition protocol.

This creates recurring failure modes:

1. **Static surface fragility**
   - Manual code modification does not scale with domain expansion.
   - Repetitive implementation work creates inconsistent patterns.
   - Architectural conventions are difficult to enforce across independently evolving modules.
2. **Invariant drift**
   - Previously eliminated defects can reappear in unrelated modules.
   - Static analysis catches only the properties encoded in its rules.
   - Security boundaries can degrade through apparently innocuous refactors.
3. **Insufficient execution-based verification**
   - Syntax validity does not imply behavioral correctness.
   - Static analysis does not establish concurrency safety.
   - Unit tests alone do not guarantee resistance to malformed inputs or adversarial execution.
4. **Weak change lineage**
   - Conventional commit history identifies what changed but does not necessarily capture the model, routing decision, evaluation environment, adversarial cases, invariant results, or exact execution artifacts that produced the change.
5. **Uncalibrated confidence**
   - A model's subjective confidence is not an empirical probability of correctness.
   - A single aggregate score can hide catastrophic failure in one critical dimension.

The architecture therefore requires a controlled mechanism by which machine-generated changes can be proposed and evaluated without granting the generative system direct authority over durable repository state.

---

## 3. Goals

The system SHALL provide:

- Controlled machine-generated code evolution.
- Deterministic enforcement of critical invariants.
- Reproducible evaluation from content-addressed inputs.
- Sandboxed execution of candidate code.
- Adversarial testing and mutation testing.
- Calibrated statistical confidence measurement.
- Explicit human-review escalation.
- Atomic repository mutation.
- Automatic rollback on post-integration verification failure.
- Complete cryptographic lineage for every candidate and accepted transition.
- Versioned evaluation policies and model artifacts.
- Resource, time, and scope limits for autonomous execution.
- Offline/replayable evaluation of historical candidate patches.
- Explicit handling of model, test, policy, and repository drift.

---

## 4. Non-Goals

This system does NOT attempt to:

- Prove arbitrary software correctness.
- Treat model confidence as a proof of correctness.
- Allow arbitrary autonomous modification of production systems.
- Perform online gradient updates during repository mutation.
- Replace deterministic security controls with neural classifiers.
- Assume that passing a test suite establishes universal correctness.
- Treat generated negative examples as automatically trustworthy training data.
- Allow an expert, router, discriminator, or evaluator to modify its own verification policy within the same authorization boundary.

---

## 5. Decision Drivers

### 5.1 Safety

Critical repository invariants must fail closed.

### 5.2 Determinism

Given the same repository snapshot, task, policy, model artifacts, test corpus, and seeds, the verification decision should be reproducible to the maximum extent practical.

### 5.3 Separation of Authority

The component that generates a patch must not be the sole authority that certifies or integrates it.

### 5.4 Calibration

Confidence scores must be evaluated empirically against held-out outcomes. Thresholds are policy parameters, not truths supplied by mathematics.

### 5.5 Reproducibility

Every accepted change must be reconstructible from immutable artifacts and their hashes.

### 5.6 Containment

Generated code, generated tests, and adversarial payloads must execute within explicit resource and capability boundaries.

### 5.7 Auditability

Every state transition must produce cryptographically linked provenance.

### 5.8 Operational Recovery

The system must support cancellation, quarantine, rollback, kill-switch activation, and recovery from evaluator/model failure.

---

## 6. Considered Options

### Option 1 — Status Quo

Static Python code, manually generated patches, conventional CI, static analysis, and developer-controlled integration.

**Advantages**

- Low complexity.
- Familiar operational model.
- Low inference overhead.

**Disadvantages**

- Poor scalability of repetitive code changes.
- Limited automated exploration of implementation alternatives.
- Weak machine-readable lineage between proposal and evaluation.

---

### Option 2 — Unconstrained Autonomous Agentic Writing

Agents receive repository write access and use self-reflection or iterative execution to modify the working tree.

**Rejected.**

Reasons:

- Generation and authorization are insufficiently separated.
- Failure containment is difficult.
- Self-evaluation can become correlated with generation failure modes.
- A compromised or misbehaving agent can directly mutate durable state.
- Provenance can be incomplete or generated after the fact.
- Verification policy itself can become an attack surface.

---

### Option 3 — Controlled Continuous Code-Evolution Pipeline

Specialized generation experts produce candidate patches. Independent evaluators execute deterministic and adversarial checks. A deterministic policy gate combines invariant results with calibrated statistical measurements. Only certified candidates enter an isolated integration transaction.

**Chosen.**

---

## 7. Architectural Decision

The repository is modeled as a sequence of explicitly identified states:

```
S_n ──candidate patch──► S'_n ──certification──► S_(n+1)
```

A candidate transition is valid only if:

```
PolicyPass
∧ ScopePass
∧ StructuralPass
∧ StaticPass
∧ SandboxPass
∧ AdversarialPass
∧ InvariantPass
∧ ProvenancePass
∧ IntegrationPolicyPass
```

A confidence score may influence whether a candidate is eligible for autonomous integration, but it cannot turn any failed mandatory predicate into a pass.

Formally:

```
Certified(x) =
    HardGates(x)
    ∧ PolicyAllows(x)
    ∧ ConfidenceEligible(x)
```

where:

```
HardGates(x) ∈ {0,1}
PolicyAllows(x) ∈ {0,1}
ConfidenceEligible(x) ∈ {0,1}
```

Therefore:

```
HardGates(x) = 0  ⇒  Certified(x) = 0
```

regardless of confidence.

---

## 8. System Architecture

```
                           ┌─────────────────────┐
                           │      Task Input      │
                           └──────────┬──────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ Immutable Context Snap  │
                         │ repo / task / policy   │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ Scope & Policy Gate    │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ Sparse MoE Router      │
                         └───────┬─────┬──────┬───┘
                                 │     │      │
                         ┌───────▼┐ ┌──▼────┐ ┌▼────────┐
                         │Ledger  │ │Security│ │Structural│
                         │Expert  │ │Expert  │ │Expert    │
                         └────┬───┘ └──┬────┘ └────┬────┘
                              └─────────┼───────────┘
                                        ▼
                               Candidate Patch
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                 Static/Structural               Sandbox
                   Evaluation                   Execution
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                              Adversarial Evaluation
                                        │
                                        ▼
                             Deterministic Invariants
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                            FAIL                 PASS
                              │                   │
                              ▼                   ▼
                         Quarantine       Calibration
                                               │
                                  ┌────────────┴───────────┐
                                  │                        │
                               Ineligible               Eligible
                                  │                        │
                                  ▼                        ▼
                               Review              Integration Gate
                                                           │
                                                           ▼
                                                  Atomic Repository
                                                       Mutation
                                                           │
                                                           ▼
                                                Post-Integration Tests
                                                           │
                                                     ┌─────┴─────┐
                                                     │           │
                                                    PASS        FAIL
                                                     │           │
                                                     ▼           ▼
                                                  Commit      Rollback
```

---

## 9. Sparse Mixture-of-Experts Routing

### 9.1 Routing Model

The router computes:

```
h = W_g x + b_g + ε
G(x) = Softmax(TopK(h, k))
```

where:

- `x` is an immutable task/context representation.
- `W_g` and `b_g` identify the router version.
- `ε` is optional routing noise during exploration.
- `TopK` limits active experts.
- `k` is policy-controlled.

Production certification MUST record the routing seed and router artifact hash.

For strict replay mode:

```
ε = deterministic_prng(seed, context_hash)
```

or routing noise MUST be disabled.

---

### 9.2 Initial Experts

#### E_ledger — State and Ledger Integrity

Optimized for:

- state-machine transitions;
- transactional boundaries;
- file locking;
- atomic persistence;
- concurrency-sensitive code;
- idempotency;
- recovery semantics.

#### E_sec — Boundary Security

Optimized for:

- AST safety;
- import boundaries;
- subprocess execution;
- deserialization;
- filesystem access;
- shell isolation;
- capability restrictions.

#### E_struct — Structural Refactoring

Optimized for:

- schema migration;
- API transformations;
- configuration externalization;
- typed interfaces;
- Pydantic model evolution;
- dependency modernization.

Additional experts MAY be added, but every expert MUST have:

- immutable artifact identity;
- declared capabilities;
- supported task classes;
- evaluation benchmarks;
- versioned policy;
- owner;
- rollback/revocation mechanism.

---

## 10. Candidate Patch Contract

Experts MUST emit a structured candidate rather than arbitrary filesystem mutations.

```
candidate_id: "uuid"
base_revision: "sha256:..."
task_hash: "sha256:..."
expert:
  id: "E_sec"
  artifact_hash: "sha256:..."
  version: "..."
router:
  artifact_hash: "sha256:..."
  seed: 12345
patch:
  format: "unified-diff"
  sha256: "sha256:..."
scope:
  files:
    - "src/security/imports.py"
  additions: 37
  deletions: 12
claims:
  - "replace dynamic import with allowlisted resolver"
```

The candidate MUST NOT contain an instruction such as:

```
modify files outside this scope
```

because scope is enforced by the executor, not by generated text.

---

## 11. Scope and Policy Gate

Before execution, the candidate is checked against a signed policy.

Examples of policy-controlled operations:

- allowed file paths;
- maximum diff size;
- allowed dependency changes;
- forbidden configuration changes;
- forbidden secrets access;
- forbidden network access;
- forbidden privilege escalation;
- forbidden production credentials;
- required reviewers;
- required test suites;
- required rollback capability.

Certain changes SHALL always require human approval, including at minimum:

- authentication/authorization policy;
- secret-management infrastructure;
- production deployment configuration;
- cryptographic primitives;
- audit/provenance implementation;
- sandbox policy;
- invariant definitions;
- evaluator implementation;
- confidence calibration policy;
- model-loading security boundary.

A model must not be allowed to rewrite the rules by which the model is judged in the same autonomous transaction.

---

## 12. Sandboxed Execution

Candidate code MUST execute in an ephemeral environment.

Minimum controls:

- read-only base filesystem;
- copy-on-write candidate workspace;
- no inherited credentials;
- restricted network;
- CPU quota;
- memory quota;
- process quota;
- file descriptor quota;
- wall-clock timeout;
- output-size limit;
- filesystem quota;
- deterministic environment manifest.

The sandbox MUST produce:

```
environment_hash
runtime_version
dependency_lock_hash
kernel/container identity
resource limits
network policy
execution seed
```

Sandbox escape or policy violation is a terminal failure.

---

## 13. Static and Structural Verification

Static verification MUST include more than syntax compilation.

Minimum checks:

1. Parser/AST validation.
2. Type checking where applicable.
3. Import graph analysis.
4. Forbidden API detection.
5. Dependency diff analysis.
6. Security static analysis.
7. Serialization/deserialization analysis.
8. subprocess/shell boundary analysis.
9. filesystem mutation analysis.
10. concurrency-sensitive operation analysis.
11. targeted unit tests.
12. regression test suite.

`py_compile` is necessary for syntax validity but is explicitly **not sufficient evidence of AST compliance, semantic correctness, security, or behavioral correctness**.

---

## 14. Adversarial Evaluation

Adversarial evaluation has two independent responsibilities.

### 14.1 Test Generation

The adversarial generator may produce:

- malformed input;
- boundary values;
- corrupted serialization;
- invalid configuration;
- hostile shell arguments;
- path traversal attempts;
- unexpected Unicode;
- concurrent access patterns;
- cancellation during critical sections;
- partial-write simulation;
- dependency/API incompatibility;
- resource exhaustion scenarios.

Generated tests MUST themselves be subject to resource and safety limits.

---

### 14.2 Execution

Candidate code is executed against:

```
baseline tests
+
generated adversarial tests
+
historical regression corpus
+
mutation-derived tests
+
invariant-specific tests
```

A candidate does not pass because an adversarial model predicts that it is safe.

It passes only when the executable verification contract passes.

---

## 15. Deterministic Invariant Enforcement

Mandatory invariants are represented as independent Boolean predicates.

Example:

```
I1 = no forbidden dynamic imports
I2 = all deserialization paths validated
I3 = all stateful writes protected by approved transaction primitive
I4 = candidate parses and compiles
I5 = required test suite passes
I6 = sandbox policy respected
I7 = dependency policy satisfied
I8 = provenance record complete
I9 = patch scope matches authorization
I10 = postconditions verified
```

The hard gate is:

```
HardGate(x) = ∏ I_j(x)
```

with:

```
I_j(x) ∈ {0,1}
```

Any zero causes rejection or quarantine.

This mechanism deliberately prevents a high statistical score from compensating for a catastrophic failure.

---

## 16. Confidence and Calibration

The system MAY calculate:

```
p = sigmoid(wᵀφ(x) + b)
```

where `φ(x)` may contain:

- test pass rate;
- historical failure rate;
- coverage delta;
- static-analysis delta;
- adversarial survival rate;
- AST complexity delta;
- patch size;
- expert specialization;
- task-class calibration features;
- historical performance of the expert;
- evaluator agreement.

However, `p` MUST be treated as a **calibrated eligibility estimate**, not as a proof.

The production confidence model MUST be evaluated using held-out data.

Required metrics include:

- Brier score;
- expected calibration error (ECE);
- reliability curves;
- false-accept rate;
- false-reject rate;
- confidence intervals;
- performance by task class;
- performance by expert;
- performance by change-risk tier.

Calibration MUST be periodically revalidated.

If calibration drift exceeds policy limits, autonomous integration MUST be disabled until recalibration or policy review occurs.

---

## 17. Integration Policy

The original `0.85` threshold is retained only as an initial policy target, not as a mathematically justified universal threshold.

The actual policy is:

```
AutonomousIntegration(x) =
    HardGate(x)
    ∧ ScopeGate(x)
    ∧ PolicyGate(x)
    ∧ ProvenanceGate(x)
    ∧ CalibrationHealthy
    ∧ Confidence(x) >= T(task_class, risk_tier)
```

Thresholds MAY vary by risk class.

Example:

| Risk tier | Autonomous threshold | Human approval |
| --- | --- | --- |
| Low | ≥ 0.85 | Not normally required |
| Medium | ≥ 0.95 | Policy-dependent |
| High | Never autonomous | Required |
| Governance/security control | Never autonomous | Required |

The exact thresholds MUST be established empirically from evaluation data.

---

## 18. Rejection and Learning Loop

Rejected candidates are retained as structured evaluation artifacts.

The rejection record includes:

- candidate hash;
- failure invariant;
- evaluator result;
- adversarial counterexample;
- environment hash;
- policy version;
- model version;
- task class;
- failure taxonomy.

The online execution path MUST NOT perform uncontrolled gradient updates.

Instead:

```
rejection
   ↓
validated failure artifact
   ↓
versioned dataset
   ↓
offline training/evaluation
   ↓
candidate model
   ↓
benchmark suite
   ↓
calibration
   ↓
approval
   ↓
production registry
```

This prevents a model from immediately training on its own potentially corrupted outputs and then using the updated model to certify subsequent outputs.

---

## 19. Provenance and Lineage

Every candidate and repository transition receives a cryptographically identified provenance record.

The provenance model SHALL use W3C PROV concepts where appropriate. PROV-O provides an OWL2 representation of the PROV Data Model and is suitable for interoperable provenance representation; cryptographic integrity is an additional system responsibility, not something supplied by PROV-O itself.

Each record MUST contain at minimum:

```
event_id
parent_event_hash
timestamp
task_hash
base_revision
candidate_hash
patch_hash
router_hash
expert_hash
evaluator_hash
policy_hash
sandbox_hash
test_corpus_hash
adversarial_seed
feature_vector_hash
confidence
hard_gate_results
scope_result
integration_result
resulting_revision
```

Records MUST form a hash-linked append-only sequence:

```
R_n = H(payload_n || R_(n-1))
```

Where practical, accepted checkpoints SHOULD additionally be signed by an independent signing authority.

---

## 20. Atomic Repository Integration

No candidate is applied directly to the primary working tree.

Integration SHALL use:

1. Immutable base revision.
2. Fresh integration workspace.
3. Verified patch.
4. Policy re-check.
5. Patch application.
6. Complete required verification.
7. Atomic commit/ref update.
8. Provenance commit.
9. Post-integration verification.

If post-integration verification fails:

```
accepted_state → rollback_state
```

The failed transition remains permanently recorded.

---

## 21. Failure Semantics

The system is fail-closed.

| Failure | Result |
| --- | --- |
| Router unavailable | Reject/quarantine |
| Expert unavailable | Reject/quarantine |
| Candidate malformed | Reject |
| Patch scope violation | Reject |
| Sandbox violation | Reject |
| Static check failure | Reject |
| Adversarial failure | Reject |
| Hard invariant failure | Reject |
| Confidence model unavailable | No autonomous integration |
| Calibration stale | No autonomous integration |
| Provenance failure | No integration |
| Commit failure | No state transition |
| Post-integration verification failure | Rollback |
| Evaluator timeout | Reject/quarantine |
| Resource exhaustion | Reject |
| Model artifact mismatch | Reject |

No component may convert an infrastructure failure into a success result.

---

## 22. Kill Switch and Operational Controls

The system SHALL provide an independently controlled kill switch.

Kill-switch activation MUST prevent:

- candidate generation from entering integration;
- repository mutation;
- model promotion;
- policy promotion.

Evaluation and forensic replay MAY continue if safe.

Kill-switch state MUST be externally observable and provenance-recorded.

---

## 23. Test and Evaluation Strategy

The system itself is subject to continuous verification.

### 23.1 Unit Tests

Test:

- routing;
- patch parsing;
- scope enforcement;
- invariant evaluation;
- confidence calculation;
- calibration metrics;
- provenance serialization;
- hash chaining;
- policy evaluation.

### 23.2 Property Tests

Examples:

```
Any failed invariant ⇒ integration == false
Any out-of-scope file ⇒ integration == false
Any provenance mismatch ⇒ integration == false
Any sandbox violation ⇒ integration == false
Any policy prohibition ⇒ integration == false
```

### 23.3 Mutation Tests

Intentionally introduce:

- unsafe dynamic imports;
- unchecked deserialization;
- unprotected writes;
- shell injection;
- path traversal;
- race conditions;
- missing exception handling.

The evaluator must detect them.

### 23.4 Adversarial Tests

Continuously regenerate hostile inputs against known-safe implementations.

### 23.5 Replay Tests

Historical candidates MUST be replayable using their recorded artifact hashes.

### 23.6 Golden Evaluation Set

A versioned benchmark SHALL contain:

- known-good patches;
- known-bad patches;
- boundary cases;
- security regressions;
- concurrency regressions;
- provenance failures;
- policy bypass attempts.

No model or policy version may be promoted unless it meets the benchmark's acceptance criteria.

---

## 24. Evaluation Metrics

The system SHALL measure at least:

### Safety

- hard-invariant false-negative rate;
- unauthorized mutation rate;
- sandbox escape rate;
- policy bypass rate.

### Model quality

- candidate acceptance rate;
- candidate rejection rate;
- calibration error;
- Brier score;
- expert routing quality.

### Verification quality

- mutation detection rate;
- adversarial detection rate;
- regression detection rate.

### Operations

- median evaluation latency;
- p95/p99 evaluation latency;
- CPU consumption;
- memory consumption;
- sandbox failure rate;
- rollback rate.

### Learning

- repeated failure rate;
- failure-class recurrence;
- improvement on held-out benchmark;
- negative-sample usefulness;
- expert specialization performance.

---

## 25. Threat Model

The system SHALL assume:

- generated code can be malicious or incorrect;
- an expert can be compromised;
- a router can misroute;
- an evaluator can contain defects;
- generated adversarial tests can be malformed;
- dependency artifacts can be compromised;
- model artifacts can be replaced;
- provenance records can be tampered with;
- repository state can change concurrently;
- tests can contain blind spots;
- confidence models can be miscalibrated;
- training data can be poisoned;
- model outputs can deliberately attempt policy bypasses.

Accordingly, no single model or evaluation component is a trusted authority.

---

## 26. Trust Boundaries

The following boundaries SHALL remain distinct:

```
UNTRUSTED
    generated candidate
    generated tests
    model output
    router output
    external task content

        │

        ▼

CONTROLLED EXECUTION
    sandbox
    static analyzers
    deterministic test harness

        │

        ▼

TRUSTED VERIFICATION
    invariant engine
    policy engine
    provenance validator

        │

        ▼

AUTHORIZED MUTATION
    integration service
    repository state transition
```

The integration service MUST NOT consume free-form model output as an authorization primitive.

---

## 27. Dependency and Artifact Integrity

All model, container, dependency, benchmark, policy, and evaluator artifacts MUST be identified by immutable digests.

A candidate evaluation is invalid if any referenced artifact cannot be resolved to the recorded digest.

Recommended identity:

```
sha256:<digest>
```

Artifact promotion SHALL be separate from repository mutation.

---

## 28. Concurrency and TOCTOU Protection

The candidate is evaluated against a specific base revision:

```
base_revision = R
```

Integration is permitted only if:

```
current_revision == R
```

Otherwise the candidate is invalidated and must be rebased/re-evaluated.

This prevents:

```
evaluate(R)
      ↓
repository changes to R'
      ↓
apply candidate intended for R
```

---

## 29. Rollback

Every autonomous integration MUST identify a previous known-good state.

Rollback SHALL be deterministic and independently executable.

The system MUST preserve:

```
previous_revision
failed_revision
failure_event
rollback_revision
```

A rollback MUST NOT delete provenance for the failed transition.

---

## 30. Governance

Changes to the following require explicit architectural approval:

- hard invariants;
- autonomous integration thresholds;
- sandbox capabilities;
- policy engine;
- provenance format;
- signing authority;
- expert registry;
- model promotion process;
- benchmark acceptance criteria.

A model-generated patch MUST NOT autonomously modify these controls.

---

## 31. Implementation Layout

The initial implementation SHALL use:

```
/opt/adam_runtime/
├── gating/
│   ├── router.py
│   ├── expert_registry.py
│   └── policy.py
│
├── generation/
│   ├── candidate.py
│   └── patch_contract.py
│
├── evaluation/
│   ├── static_evaluator.py
│   ├── adversarial_evaluator.py
│   ├── confidence_classifier.py
│   ├── calibration.py
│   ├── invariant_enforcer.py
│   └── mutation_runner.py
│
├── sandbox/
│   ├── runner.py
│   └── resource_policy.py
│
├── integration/
│   ├── verifier.py
│   ├── atomic_commit.py
│   └── rollback.py
│
├── provenance/
│   ├── model.py
│   ├── ledger.py
│   └── signer.py
│
├── storage/
│   ├── candidates/
│   ├── evaluations/
│   ├── negative_samples/
│   └── provenance/
│
├── tests/
│   ├── unit/
│   ├── property/
│   ├── mutation/
│   ├── adversarial/
│   ├── replay/
│   └── golden/
│
└── schemas/
    ├── candidate.schema.json
    ├── evaluation.schema.json
    ├── provenance.schema.json
    └── policy.schema.json
```

---

## 32. Rollout Plan

### Phase 0 — Shadow Mode

Generate and evaluate candidates without repository mutation.

Success criteria:

- deterministic replay;
- complete provenance;
- no evaluator crashes;
- benchmark baseline established.

### Phase 1 — Human-Gated Integration

Candidates may be integrated only after human approval.

Success criteria:

- zero unauthorized mutations;
- reliable rollback;
- stable provenance.

### Phase 2 — Low-Risk Autonomous Integration

Enable autonomous integration only for explicitly classified low-risk changes.

Success criteria:

- calibration validated;
- mutation-test detection target achieved;
- rollback tested;
- kill switch tested.

### Phase 3 — Expanded Scope

Increase autonomous scope only after empirical evidence supports it.

### Phase 4 — Continuous Evaluation

Continuously monitor:

- model drift;
- evaluator drift;
- benchmark drift;
- failure recurrence;
- calibration;
- production regressions.

NIST's AI RMF similarly emphasizes repeated measurement, uncertainty, documentation, independent review, and monitoring as systems evolve.

---

## 33. Acceptance Criteria

The architecture is considered operational only when:

1. Every candidate has an immutable identity.
2. Every candidate is evaluated in an isolated environment.
3. Hard invariants fail closed.
4. No model can directly mutate the primary repository.
5. Out-of-scope patches are rejected.
6. Artifact hashes are verified.
7. Historical evaluations are replayable.
8. Provenance records are hash-linked.
9. Integration is atomic.
10. Rollback is tested.
11. Kill-switch behavior is tested.
12. Confidence calibration is measured on held-out data.
13. Mutation tests demonstrate evaluator sensitivity.
14. Security-sensitive changes cannot bypass human approval.
15. The benchmark suite is versioned.
16. Policy versions are immutable during an evaluation.
17. Current repository state is checked before commit.
18. Post-integration verification is mandatory.
19. Evaluator failures cannot produce an approval.
20. The system has a documented maximum autonomous change scope.

---

## 34. Consequences

### Positive

- Strong separation between generation and authorization.
- Deterministic enforcement of critical invariants.
- Reproducible evaluation.
- Explicit containment of generated code.
- Quantifiable model calibration.
- Rich lineage and forensic capability.
- Safer incremental introduction of autonomous code evolution.
- Ability to learn from rejected candidates without permitting uncontrolled online self-modification.
- Explicit rollback and kill-switch mechanisms.

### Negative

- Significant implementation complexity.
- Increased compute consumption.
- Higher latency per autonomous change.
- Ongoing maintenance of benchmarks and invariants.
- Calibration requires representative historical data.
- Adversarial testing can still miss unknown failure modes.
- Sandbox infrastructure becomes a critical security dependency.
- Provenance storage and artifact retention create operational costs.
- Human governance remains necessary for high-risk changes.

---

## 35. Fundamental Limitation

The system does not establish universal software correctness.

Its guarantee is conditional:

> **Within the declared verification envelope, an accepted transition satisfies every mandatory deterministic gate that was successfully executed against the identified repository state, policy, evaluator, environment, and artifact versions.**

This distinction is normative and MUST NOT be weakened in implementation or documentation.

---

## 36. Final Decision

Adopt Option 3: **Controlled Continuous Code Evolution via Sparse Expert Generation, Adversarial Evaluation, Deterministic Invariant Enforcement, Calibrated Confidence, and Cryptographically Traceable Integration.**

The architecture deliberately treats neural generation as an untrusted proposal mechanism and deterministic verification as the authority governing durable state transitions.

# ADAM: Epistemic State Hierarchy and the Closed Authority Loop

## 1. Epistemic State Hierarchy and the Closed Authority Loop
ADAM defines eight structurally isolated state domains. No probabilistic or neural computation possesses direct mutation authority over state transitions; authority is exclusively vested in deterministic admission and verification kernels.

| Layer | State Domain | Epistemic Question | Governing Authority | Mutability |
|---|---|---|---|---|
| C_0 | Canonical Observation | What was attested and recorded? | Evidence Authority | Append-only |
| C_1 | Estimate State | What does the estimator infer? | None | Ephemeral / Non-authoritative |
| C_2 | Epistemic State | What is believed, and how is uncertainty structured? | None | Ephemeral / Non-authoritative |
| C_3 | Hypothesis State | What proposition or action is proposed? | None | Ephemeral / Non-authoritative |
| C_4 | Verified State | Does the proposition satisfy formal mathematics? | Verification Authority | Deterministic Output |
| C_5 | Admitted State | Is the verified proposition permitted by active policy? | Policy Authority | Gate Evaluation |
| C_6 | Committed State | What transition was cryptographically authorized? | Execution Authority | Immutable Log |
| C_7 | Reconciled State | What physical execution occurred, and what was the delta? | Audit Authority | Append-only Feedback |

The core invariant of the state machine enforces that non-authoritative states cannot bypass deterministic controls:
$$C_{1, 2, 3} \not\rightarrow C_6$$

The sole valid execution vector is the sequential pipeline:
$$C_0 \longrightarrow C_1 \longrightarrow C_2 \longrightarrow C_3 \xrightarrow{\quad \mathbf{G12} \quad} C_4 \longrightarrow C_5 \longrightarrow C_6 \longrightarrow C_7 \longrightarrow C_0'$$

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          C0: CANONICAL OBSERVATION                          │
│                     (Attested Ingestion, Hash-Chained)                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          C1: ESTIMATION (NON-AUTH)                          │
│             (CfC, State-Space, Ensembles, Quant Models, LLMs)               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           C2: EPISTEMIC STATE                               │
│            (Deterministic UNKNOWN Calculus, Wasserstein Bounds)             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           C3: HYPOTHESIS STATE                              │
│                    (Candidate Proposition, Action Proposals)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                GATE 12: AUTHORITY BOUNDARY ADMISSION                      ║
 ║         Validates Provenance, Schema, Snapshots, Model Signatures         ║
 ║                 Mandatory Attribute: authority = NONE                     ║
 ╚═════════════════════════════════════╤═════════════════════════════════════╝
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        C4: RESOLUTION / VERIFICATION                        │
│             (Deterministic Mathematics, Solvers, Invariant Logic)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            C5: POLICY ADMISSION                             │
│                  (Risk Limits, Mandates, Governance ASTs)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                         C6: CRYPTOGRAPHIC COMMIT                          ║
 ║             (Signed State Transition, Append-Only Ledger)                 ║
 ╚═════════════════════════════════════╤═════════════════════════════════════╝
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        C7: RECONCILE / POST-COMMIT                          │
│               (Execution Telemetry, Settlement, Fill Deltas)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       └────────► Yields C0' (Next Epoch)
```

## 2. Core Epistemic Principles and System Invariants
ADAM separates empirical reality from attested observation, and deterministic verification from material truth.

### The Epistemic Quad-Separation
System operations distinguish four independent properties:
$$\text{Determinism} \neq \text{Correctness} \neq \text{Validity} \neq \text{Truth}$$
* **Determinism (D):** Given identical inputs, identical runtime binaries, and identical execution environments, computation yields byte-for-byte identical outputs ($f(x) = y$).
* **Verification (V):** A proposition satisfies all explicit, deterministic mathematical rules, balance equations, and system constraints.
* **Policy Validity (P):** A proposition conforms to the active institutional rules, exposure limits, and governing mandates.
* **Empirical Truth (E):** The degree to which an observation or assertion corresponds to unobserved physical or market reality.

Formal corollaries:
$$\text{Verified}(H) \not\Rightarrow \text{True}(H)$$
$$\text{Deterministic}(H) \not\Rightarrow \text{Correct}(H)$$

### The Four System Invariants
* **$I_1$ — Authority Invariant:** $\text{Inference} \not\Rightarrow \text{Authority}$
  Probabilistic artifacts, heuristic estimators, and language models cannot directly mutate authoritative state, dispatch capital, or modify policy boundaries.
* **$I_2$ — Verification Invariant:** $\text{Verified}(H) \not\Rightarrow \text{True}(H)$
  Deterministic verification establishes formal consistency against declared mathematical models and constraints; it does not assert metaphysical truth.
* **$I_3$ — Temporal Integrity Invariant:** $\text{State}_{t+1} \not\rightarrow \text{State}_t$
  Historical state is append-only and cryptographically sealed. New observations, retroactive corrections, or restatements instantiate a new state epoch ($C_0'$ at $t_k$), leaving historical decision contexts unaltered.
* **$I_4$ — Provenance Completeness Invariant:** $\text{Commit} \Rightarrow \text{CompleteTrace}(C_0 \rightarrow C_6)$
  No transition enters $C_6$ without a complete, unbroken directed acyclic graph (W3C PROV-O compliant) linking attested evidence snapshots, model execution hashes, verification proofs, and policy authorisations.

### Observation Invariant ($C_0$)
$C_0$ establishes the canonical representation of admissible observations; it does not assert that every observation is empirically true. An attested filing proves that a counterparty asserted specific figures under signature at time $t_e$; it does not prove the absence of fraudulent accounting within that filing.
$$\text{Canonical Observation} \equiv \text{Attested Representation of System Perception}$$
$$\text{Attested Observation} \neq \text{Objective Reality}$$

## 3. Temporal Multi-Clock Geometry
Temporal epistemic consistency requires decoupling real-world occurrence from system perception and downstream action. ADAM enforces four explicit timestamps on every processed entity:

```text
t_e: Event Time      ───► [Event occurs in external reality]
                               │
t_k: Knowledge Time  ───►      └───► [ADAM ingests and cryptographically seals observation]
                                            │
t_d: Decision Time   ───►                   └───► [Hypothesis admitted and verified via G12/C4/C5]
                                                              │
t_x: Execution Time  ───►                                     └───► [Committed action executes in venue]
```

Where strictly:
$$t_e \le t_k \le t_d \le t_x$$

* **$t_e$ (Event Time):** The instant a market transaction, corporate filing publication, or default occurs in the external world.
* **$t_k$ (Knowledge Time):** The wall-clock instant ADAM ingests, validates signatures for, and commits the evidence into $C_0$. All historical backtests and replay evaluations must condition strictly on information where $t_k \le t_{\text{simulated}}$.
* **$t_d$ (Decision Time):** The timestamp marking the deterministic admission and verification through Gates 12, C4, and C5.
* **$t_x$ (Execution Time):** The physical timestamp recorded at venue settlement or order fulfillment, ingested post-facto into $C_7$.

## 4. Mathematical Definition of the Authority Kernel
The system state at epoch $t$ is defined as the tuple:
$$S_t = \left( O_t, E_t, B_t, H_t, V_t, A_t, K_t, R_t \right)$$
where:
* $O_t \in \mathcal{O}$: Canonical Observations ($C_0$)
* $E_t \in \mathcal{E}$: Estimator Distributions ($C_1$)
* $B_t \in \mathcal{B}$: Epistemic Belief States ($C_2$)
* $H_t \in \mathcal{H}$: Candidate Hypotheses ($C_3$)
* $V_t \in \mathcal{V}$: Verified Mathematical States ($C_4$)
* $A_t \in \mathcal{A}$: Policy-Admitted Propositions ($C_5$)
* $K_t \in \mathcal{K}$: Cryptographically Committed Decisions ($C_6$)
* $R_t \in \mathcal{R}$: Post-Commit Execution Reconciliations ($C_7$)

Neural and heuristic components are restricted to hypothesis generation:
$$N: O_t \longrightarrow (E_t, B_t, H_t)$$

The authority transition kernel $K$ is entirely deterministic, symbolic, and non-neural:
$$K: (H_t, O_t, \Pi_t) \longrightarrow \{\text{REJECT}, \text{VERIFY}, \text{ADMIT}, \text{COMMIT}\}$$

The transition rule for authorizing state change is governed by the boolean conjunction:
$$\text{Commit}_t = \mathbf{1} \iff \begin{pmatrix} \text{ObsValid}(O_t) \\ \land \; \text{ProvValid}(\text{Trace}_{C_0 \to C_3}) \\ \land \; \text{HypAdmissible}(H_t, \mathbf{G12}) \\ \land \; \text{VerificationPass}(V_t, C_4) \\ \land \; \text{PolicyPass}(A_t, \Pi_t, C_5) \\ \land \; \text{TemporalConsistency}(t_e, t_k, t_d) \\ \land \; \text{SignatureValid}(\text{Signatures}) \end{pmatrix}$$

If any conjunct evaluates to false, $\text{Commit}_t = \varnothing$, the proposal is rejected, and an audit record is dispatched to $C_7$.

## 5. Gate Architectures: G12 and G13

### Gate 12: Authority Boundary Admission
Gate 12 guards the boundary between the non-authoritative probabilistic pipeline ($C_1, C_2, C_3$) and deterministic verification ($C_4$). G12 grants admission to verification; it does not grant execution authority.

```yaml
record_type: AuthorityBoundaryRecord
schema_version: "2.0.0"
authority: NONE  # Structurally immutable; C3 cannot modify
subject:
  hypothesis_id: "hyp_uuidv7_01924b"
  action_class: "CREDIT_LIMIT_MODIFICATION"
source_lineage:
  canonical_snapshot_id: "snap_c0_88492"
  canonical_state_hash: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
epistemic_input:
  belief_state_id: "belief_c2_44102"
  belief_hash: "sha256:56e83...bca"
  epistemic_status: "SUPPORTED"
model_context:
  model_id: "cfc_credit_spread_v4"
  model_version: "commit:9f8a2c1"
  inference_hash: "sha256:772ab...019"
temporal_bounds:
  event_time_te: "2026-09-22T20:00:00.000Z"
  knowledge_time_tk: "2026-09-22T20:01:05.120Z"
  valid_from: "2026-09-22T20:01:05.120Z"
  valid_until: "2026-09-22T20:15:00.000Z"
provenance:
  prov_graph_hash: "sha256:c18b...902"
  trace_completeness_verified: true
admission_criteria:
  g12_h_schema_valid: true
  g12_r_runtime_pinned: true
  g12_s_snapshot_consistent: true
  g12_a_authority_isolated: true
  g12_p_lineage_declared: true
```

**Trace Limitation Principle:** A complete provenance trace confirms that the declared inputs, environment states, and model artifacts match the recorded cryptographic hashes. It establishes Declared Lineage Completeness, not Internal Reasoning Transparency:
$$\text{Trace Completeness} \neq \text{Internal Cognitive Completeness}$$

### Gate 13: Learning Admission Gate
No trained, fine-tuned, or recalibrated model may enter production inference ($C_1$) via automated backtest performance alone. It must clear the composite deterministic gate:
$$\mathbf{G13} \equiv G_{\text{DATA}} \land G_{\text{CAL}} \land G_{\text{DRIFT}} \land G_{\text{ROBUST}} \land G_{\text{REPLAY}} \land G_{\text{PROV}} \land G_{\text{POLICY}}$$

```text
 Candidate Model Weights / Hyperparameters
                     │
                     ▼
 ┌───────────────────────────────────────────────────────────────┐
 │               G13 LEARNING ADMISSION EVALUATION               │
 ├───────────────────────────────────────────────────────────────┤
 │ 1. G13-D (Data Lineage)      : Clean historical cutoff (tk)   │
 │ 2. G13-C (Calibration)       : Expected Calibration Error < τ │
 │ 3. G13-R (Regime Robustness) : Max Drawdown / Stress Tested   │
 │ 4. G13-O (OOD Boundary)      : Likelihood / Distance bounded  │
 │ 5. G13-A (Adversarial Robust): Epsilon perturbation stability │
 │ 6. G13-T (Temporal Leakage)  : Zero forward-looking leakage   │
 │ 7. G13-P (Provenance)        : Code, hyperparams, seeds logged│
 │ 8. G13-X (Replay Equiv.)     : R2 Artifact Replay verified    │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
             PASS│                               │FAIL
                 ▼                               ▼
       Admitted Model Artifact             Quarantine / Reject
         (Promoted to C1)                (Emits Failure Dossier)
```

## 6. Epistemic Uncertainty Calculus and Estimator API

### C2 Deterministic Uncertainty Lattice
The `UNKNOWN` status is not an unconstrained string generated by an LLM; it is a formally calculated state managed by the deterministic authority kernel.

```text
                   ┌──────────────┐
                   │    KNOWN     │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  SUPPORTED   │
                   └──────┬───────┘
                          │
     ┌────────────────────┼────────────────────┐
     ▼                    ▼                    ▼
┌───────────┐      ┌─────────────┐      ┌─────────────┐
│ UNCERTAIN │      │ CONFLICTED  │      │     OOD     │
└─────┬─────┘      └──────┬──────┘      └──────┬──────┘
      │                   │                    │
      └─────────────┬─────┴────────────────────┘
                    ▼
       ┌────────────────────────┐
       │ INSUFFICIENT_EVIDENCE  │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │       UNRESOLVED       │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │        UNKNOWN         │
       └────────────────────────┘
```

Deterministic transitions are governed by hard thresholds:
$$\text{EvidenceScore}(O_t) < E_{\min} \Longrightarrow \mathbf{INSUFFICIENT\_EVIDENCE}$$
$$\text{OOD\_Metric}(E_t) > \tau_{\text{OOD}} \Longrightarrow \mathbf{OUT\_OF\_DISTRIBUTION}$$
$$W_1(P_i, P_j) > \tau_W \Longrightarrow \mathbf{CONFLICTED}$$
$$\mathbf{CONFLICTED} \land \neg \text{ResolutionPolicyAvailable} \Longrightarrow \mathbf{UNRESOLVED}$$
$$\mathbf{UNRESOLVED} \lor (\mathbf{INSUFFICIENT\_EVIDENCE} \land \text{CriticalPath}) \Longrightarrow \mathbf{UNKNOWN}$$

### The Grounded Wasserstein Metric Contract ($W_1$)
When measuring divergence between two candidate predictive distributions $P$ and $Q$ across estimators or temporal windows, optimal transport distance is undefined without an explicitly declared metric space.
The 1-Wasserstein distance is formulated as:
$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[d(x, y)]$$

Under the ADAM G-W Metric Contract:
1. **Metric Space ($d(x,y)$):** The ground metric $d(x,y)$ must be explicitly declared (e.g., Mahalanobis distance scaled by the historical covariance matrix $\Sigma^{-1}$, or normalized financial spread deltas).
2. **Units & Normalization:** State variables must be projected into canonical basis units prior to calculation.
3. **Calibrated Thresholds:** Disagreement thresholds $\tau_W$ cannot be hard-coded; they must be versioned, tied to historical volatility calibrations, and admitted through policy configuration ($C_5$).

```yaml
wasserstein_contract:
  order: 1
  ground_metric: "mahalanobis"
  metric_parameters:
    covariance_matrix_id: "cov_sp500_credit_2026_q2"
  state_space: "credit_spread_vector_v1"
  canonical_units: "basis_points"
  threshold:
    value: 42.5
    calibration_dataset_id: "ds_stress_test_2020_2024"
    confidence_level: 0.99
    policy_reference: "POL-RISK-W1-08B"
```

### Model-Agnostic Estimator API ($C_1$)
No machine learning architecture is privileged within the specification. Closed-form Continuous-time (CfC) networks, Liquid Neural Networks, Transformers, Bayesian state-space filters, and Quantum ML circuits are identical candidate algorithms conforming to the same interface.

```yaml
EstimateRecord:
  estimate_id: "est_uuidv7_99210a"
  model_id: "cfc_spread_predictor"
  model_version: "sha256:334f...01a"
  input_snapshot_hash: "sha256:901a...ff3"
  temporal_scope:
    knowledge_cutoff_tk: "2026-09-22T20:00:00.000Z"
    target_horizon_te: "2026-09-23T16:00:00.000Z"
  output_distribution:
    type: "gaussian_mixture"
    parameters:
      weights: [0.7, 0.3]
      means: [145.2, 188.0]
      variances: [12.4, 45.1]
  uncertainty_metrics:
    entropy: 1.42
    ood_score: 0.04
    calibrated_confidence: 0.94
  provenance:
    runtime_environment_id: "env_nix_x86_64_pinned_v3"
    execution_duration_ms: 1.48
```

## 7. Execution, Isolation, and Replay Infrastructure

### Temporal Orchestration vs. Global Determinism
ADAM uses workflow orchestration engines (such as Temporal.io) strictly for Durable State Machine Execution, Event Sourcing, and Workflow Replay. Orchestration mechanisms do not confer determinism upon arbitrary external dependencies.
* **Workflow Code:** Restricted to deterministic state transitions, timer evaluations, gate coordination, and append-only log updates.
* **Activity Isolation:** All non-deterministic operations (LLM generation, neural inference, floating-point GPU operations, external database queries) are isolated inside side-effect-free Activities with pinned inputs and serialized outputs.
* **Artifact Pinning:** Replaying a workflow never invokes re-inference; it replays the recorded, cryptographically hashed `EstimateRecord` emitted during the original execution epoch.

### Replay Taxonomy: Functional ($R_1$) vs. Artifact ($R_2$)
The system formalizes two distinct tiers of replayability:
$$
\begin{aligned} R_1 \; (\text{Functional Replay}): \quad & f(M, X) \longrightarrow Y \\ & \text{Evaluating whether identical inputs over a declared runtime binary yield } Y. \\ R_2 \; (\text{Artifact Replay}): \quad & H(M, X, \mathcal{R}, \mathcal{P}, \tau) \longrightarrow Y \\ & \text{Full reconstruction of the exact execution including model weights } (M), \\ & \text{immutable inputs } (X), \text{hermetic container runtime } (\mathcal{R}), \\ & \text{prompts/policies } (\mathcal{P}), \text{and system seed states } (\tau). \end{aligned}
$$

### Counterfactual Replay
ADAM provides a research and stress-testing primitive enabling causal evaluation on historical logs:
$$\Delta Y = \mathcal{M}(X \oplus \delta) - \mathcal{M}(X)$$
subject to the causal constraint:
$$\delta \in \mathcal{A}(X)$$
where $\mathcal{A}(X)$ represents the domain of causally admissible interventions. An intervention cannot alter historical facts outside its direct causal cone (e.g., modifying a counterparty's debt load cannot retroactively alter historical macro treasury yields without propagating through an explicit structural causal model).

```text
 Historical Canonical State (C0) ───► [Causal Admissibility Check: δ ∈ A(X)]
                                                   │
                                                   ▼
                                       Counterfactual Input (X + δ)
                                                   │
                                                   ▼
                                       Hermetic Sandbox Execution
                                                   │
                                                   ▼
                                       Simulated Gate Pipeline
                                      (G12 ──► C4 ──► C5 ──► C6)
                                                   │
                                                   ▼
                                    Counterfactual Delta Report (ΔY)
```

### Conservative Reinforcement Learning (RLCD Boundary)
Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) implemented within the Reinforcement Learning from Credit Data (RLCD) layer operate exclusively as Candidate Offline Optimizers, not safety verifiers.
Because historical credit logs are subject to systemic survivorship bias, structural economic shifts, and policy censoring, conservative mathematical objectives do not guarantee real-world safety. All policies derived via RLCD remain non-authoritative candidate models and must clear Gate 13 before admission into production inference.

## 8. Architectural Verdict
The objective of ADAM is not to force inherently stochastic or heuristic neural computation into artificial determinism.

$$\boxed{\text{ADAM makes non-determinism epistemically visible, temporally bounded, provenance-constrained, and authority-isolated.}}$$

ADAM functions as an append-only epistemic state machine with a closed authority loop: probabilistic models hypothesize, deterministic kernels verify, cryptographic signatures commit, and empirical outcomes reconcile.

## 9. Implementation Mapping Matrix

| Formal Layer | Governance Authority | Enforcing Construct | Mathematical / Operational Guarantee |
|---|---|---|---|
| C_0 | Evidence Authority | Append-only Hash-chained Storage | Monotonic integrity; preserves t_e vs. t_k chronology |
| C_1 – C_3 | None (Non-Authoritative) | Isolated Processes; Zero Authority Tokens | Probabilistic outputs cannot call external tools or mutate state |
| G_{12} Boundary | Admission | Structural Rust Type Barriers | Rejects forged provenance or elevated permissions |
| C_4 | Verification Authority | Integer Basis-Point Solvers (BasisPoints) | Deterministic mathematical validation; zero IEEE 754 drift |
| C_5 | Policy Authority | AST Matchers & Degraded Mode Fallbacks | Preempts epistemic lockup; automated restatement unrolling |
| C_6 | Execution Authority | Signed State Transition Ledger | Provably unbroken W3C PROV-O hash chaining |
| C_7 | Audit Authority | Telemetry Settlement Reconciler | Reconciles fills against commits; logs deltas into C_0' |

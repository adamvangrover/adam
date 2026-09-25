# ADAM Epistemic State Hierarchy: Build Plan & Evaluation Prompt

## 1. Goal
To operationalize and productionize the ADAM Epistemic State Hierarchy, enforcing a closed authority loop where probabilistic models only hypothesize, while deterministic kernels verify and commit.

## 2. Build Plan

### Phase 1: Core Schemas and Deterministic Kernels (Completed)
- Define `EpistemicStatus`, `TemporalBounds`, and `AuthorityBoundaryRecord` schemas using strict Pydantic models.
- Implement `EpistemicCalculus` (C2) to deterministically map evidence, out-of-distribution (OOD) metrics, and Wasserstein divergence bounds to discrete epistemic states (e.g., `SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTED`, `UNKNOWN`).
- Implement `Gate12` boundary admission ensuring `authority == "NONE"` and temporal bounds are strictly monotonic ($t_e \le t_k \le t_d \le t_x$).

### Phase 2: Workflow Orchestration & Data Ingestion (Next)
- Implement Temporal.io workflows to manage Durable State Machine Execution.
- Restrict Workflow code to deterministic state transitions, invoking non-deterministic models (LLMs, Tensors) exclusively via isolated Activities.
- Implement the $C_0$ Canonical Observation ingestion pipeline to cryptographically seal inputs at Knowledge Time ($t_k$).

### Phase 3: Gate 13 & Provenance (Next)
- Implement `Gate13` (Learning Admission Gate) evaluating calibration error, regime robustness, and data lineage ($t_k$ cutoffs).
- Integrate W3C PROV-O compliant cryptographic lineage tracking into the Rust `adam-core` engine to enforce Trace Completeness.

### Phase 4: Replay & Counterfactual Evaluation
- Build functional ($R_1$) and artifact ($R_2$) replay engines allowing counterfactual stress testing on historical logs.
- Integrate the $W_1$ Grounded Wasserstein Metric Contract for divergence measurement.

## 3. Review & Replication Prompt

**System Role:** You are the Senior Authority Boundary Architect for the ADAM risk engine.
**Task:** Review the proposed probabilistic estimator and evaluate its compliance with the Epistemic State Hierarchy and Temporal Invariants.

**Constraints:**
1. Probabilistic estimators must output to $C_1$ (Estimate State) and carry zero authority (`authority: "NONE"`).
2. The estimator must be compatible with the `Gate12` boundary requirements.
3. Every processed entity must be stamped across four dimensions: Event Time ($t_e$), Knowledge Time ($t_k$), Decision Time ($t_d$), and Execution Time ($t_x$), satisfying $t_e \le t_k \le t_d \le t_x$.
4. Any conflict or divergence beyond $\tau_W$ must deterministically transition the state to `CONFLICTED` or `UNKNOWN`.

**Input Request:**
"We have a new LLM-based credit spread predictor. It predicts future spreads ($t_e$) based on current news ($t_k$). It recommends immediate capital allocation. Please evaluate."

**Expected Rejection Criteria:**
- **Authority Violation:** The LLM recommends capital allocation (asserting execution authority), violating Invariant $I_1$ (Inference $\not\Rightarrow$ Authority). It must only output a Hypothesis ($C_3$).
- **Temporal Check:** Ensure that the predicted future spread is treated as a target horizon, but the execution decision ($t_d$) relies only on information sealed at $t_k$.

## 4. Evaluation Criteria
1. Does the system isolate probabilistic inference from deterministic execution?
2. Are all W3C PROV-O provenance traces intact before reaching C6?
3. Does the system strictly enforce the Temporal Multi-Clock Geometry?

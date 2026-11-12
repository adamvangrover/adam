# Adam Operational Prompt Library (AOPL) Master Catalog

Welcome to the **AOPL Master Catalog & Prompt Registry**. This registry indexes operational prompt kits, management orchestration engines, field guides, maintenance suites, classifiers, evals, rubrics, and decision envelopes across the AFOS / ADAM ecosystem.

---

## 🗂️ Catalog Index by Functional Domain

### 1. Spec-It Kits (`prompt_library/spec_kits/`)
Pre-flight deterministic blueprint specifiers converting vague goals into rigid specifications before code generation.
- **[SPEC_KIT_ENGINEERING_CORE.md](spec_kits/SPEC_KIT_ENGINEERING_CORE.md)** (`SPEC-KIT-ENG-001`): Engineering core architecture, interface contracts, memory models, and zero-copy validation.
- **[SPEC_KIT_FINANCIAL_CREDIT.md](spec_kits/SPEC_KIT_FINANCIAL_CREDIT.md)** (`SPEC-KIT-CRED-002`): Credit risk modeling, SR 26-2 model governance, covenant SMT solvers, and cash-flow waterfalls.
- **[SPEC_KIT_SWARM_AGENT.md](spec_kits/SPEC_KIT_SWARM_AGENT.md)** (`SPEC-KIT-SWRM-003`): Swarm agent topology, disjointness proofs, monotonic idempotency keys, and state transition matrices.

### 2. Production & Swarm Roll-Ups Management Orchestration (`prompt_library/AOPL-v2.0/orchestration/`)
Autonomous coordinators managing multi-agent swarms, DAG schedules, and production canary gates.
- **[SWARM_ROLLUP_HIERARCHICAL_AGGREGATOR.md](AOPL-v2.0/orchestration/SWARM_ROLLUP_HIERARCHICAL_AGGREGATOR.md)** (`ORCH-ROLLUP-001`): Consolidates decentralized worker outputs with outlier rejection and epistemic inheritance.
- **[PRODUCTION_DEPLOYMENT_ORCHESTRATOR.md](AOPL-v2.0/orchestration/PRODUCTION_DEPLOYMENT_ORCHESTRATOR.md)** (`ORCH-DEPLOY-002`): Zero-downtime canary deployment, latency monitoring, and automated instant rollback tripwires.
- **[MULTI_AGENT_CONSENSUS_DISPATCHER.md](AOPL-v2.0/orchestration/MULTI_AGENT_CONSENSUS_DISPATCHER.md)** (`ORCH-DISPATCH-003`): DAG task graph dispatcher with bounded concurrency and Byzantine fault isolation.

### 3. Build Guides & Field Guides (`prompt_library/guides/`)
Tactical manuals for engineers and operational agents.
- **[BUILD_GUIDE_AUTONOMOUS_PIPELINE.md](guides/BUILD_GUIDE_AUTONOMOUS_PIPELINE.md)** (`GUIDE-BUILD-001`): Monorepo bifurcation (Path A vs Path B), Rust PDIL gatekeeper integration, and CBF Shield clamping.
- **[FIELD_GUIDE_INCIDENT_TRIAGE.md](guides/FIELD_GUIDE_INCIDENT_TRIAGE.md)** (`GUIDE-FIELD-002`): Operational triage runbook for P0-P3 incidents, amnesia recovery, port conflicts, and STS token failures.

### 4. Maintenance & Run Kits (`prompt_library/run_kits/`)
Operational execution suites and weekly self-healing loops.
- **[RUN_KIT_DAILY_OPERATIONS.md](run_kits/RUN_KIT_DAILY_OPERATIONS.md)** (`RUN-DAILY-001`): Daily market brief generation, `data.jsonl` ledger validation, and showcase index synchronization.
- **[MAINTENANCE_KIT_REFACTORING_HEALING.md](run_kits/MAINTENANCE_KIT_REFACTORING_HEALING.md)** (`MAINT-HEAL-002`): Weekly refactoring schedule, AST modernizer, dead code pruning, and observed drift healing.

### 5. Internal Classifiers & Labelers (`prompt_library/classifiers_labelers/`)
High-precision semantic classifiers and legal extractors.
- **[CLASSIFIER_EPISTEMIC_GROUNDING.md](classifiers_labelers/CLASSIFIER_EPISTEMIC_GROUNDING.md)** (`CLASS-EPIST-001`): 6-tier epistemic state classifier (`[VERIFIED]`, `[CALCULATED]`, `[MODEL_EST]`, `[PENDING]`, etc.).
- **[LABELER_COVENANT_EXTRACTOR.md](classifiers_labelers/LABELER_COVENANT_EXTRACTOR.md)** (`LABEL-COV-002`): Debt indenture parser extracting negative covenants, dollar/grower baskets, and priming loopholes.
- **[CLASSIFIER_ANOMALY_DRIFT_DETECTOR.md](classifiers_labelers/CLASSIFIER_ANOMALY_DRIFT_DETECTOR.md)** (`CLASS-DRIFT-003`): Microstructure noise floor filter and VPIN order flow toxicity detector.

### 6. Evals, Benchmarks & Scoring Rubrics (`prompt_library/evals_rubrics/`)
Standardized evaluation frameworks and code quality gates.
- **[BENCHMARK_FIDUCIARY_FITNESS.md](evals_rubrics/BENCHMARK_FIDUCIARY_FITNESS.md)** (`BENCH-FIDUC-001`): Standardized 4-scenario benchmark for multi-agent credit underwriting and fiduciary vigilance.
- **[EVAL_LLM_AS_JUDGE_ARBITER.md](evals_rubrics/EVAL_LLM_AS_JUDGE_ARBITER.md)** (`EVAL-JUDGE-002`): Pairwise debate arbiter with position debiasing and 3-habit committee tie-breaker probes.
- **[SCORING_RUBRIC_CODE_SAFETY.md](evals_rubrics/SCORING_RUBRIC_CODE_SAFETY.md)** (`RUBRIC-SAFE-003`): Multiplicative code safety scoring ($L0 \times L1$), static analysis gating, and zero-trust verification.

### 7. Decision Envelopes (`prompt_library/decision_envelopes/`)
Lexicographic boundary functions and regulatory kill-switches.
- **[DECISION_ENVELOPE_CAPITAL_ALLOCATION.md](decision_envelopes/DECISION_ENVELOPE_CAPITAL_ALLOCATION.md)** (`ENV-CAP-001`): Lexicographic Control Barrier Function (CBF) allocator between Fortress (75%) and Convex Hunt (25%).
- **[DECISION_ENVELOPE_MODEL_RISK_GOVERNANCE.md](decision_envelopes/DECISION_ENVELOPE_MODEL_RISK_GOVERNANCE.md)** (`ENV-GOV-002`): SR 26-2 / OCC 2026-13 model risk governance and automated kill-switch arbiter.

---

## 📜 Programmatic Access
All prompts are indexed in `prompt_library/prompt_manifest.json` and accessible via the central prompt engine:
```python
from core.prompting.registry import PromptRegistry

registry = PromptRegistry()
# Load via manifest or relative path
prompt = registry.get_prompt("SPEC_KIT_ENGINEERING_CORE.md")
```

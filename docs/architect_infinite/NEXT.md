---
artifact_id: "ADAM-V30-NEXT-GENESIS-2026-11-25"
schema_version: "EvidenceEnvelope.v2.0"
artifact_modes:
  - README
  - CONSTITUTION
  - ADR-0001
  - RESEARCH_SPEC
  - DEVIN_DIRECTIVE
  - POLYMORPHIC_OUTPUT_TEMPLATE
repository: "https://github.com/adamvangrover/adam"
target_branch: "release/v30.next"
pdil_ruleset_version: "pdil_v30.next.2026-11-25"
governance_gate:
  layer0_status: "HOLD"
  tier: "TIER_2_HIGH_IMPACT"
  requires_human_bottleneck: true
  effective_autonomous_score: 0.0
---

# ADAM v30.next — Self-Contained Genesis Capsule
**Autonomous Financial Operating System (AFOS) // Meta-Swarm OS // Isomorphic Data Plane**

> **Portable Execution Rule:** This document is *itself* a valid ADAM artifact. Drop it into `docs/ADAM_V30_NEXT_GENESIS_CAPSULE.md`, feed it to Devin/Gemini/Claude as a zero-context bootstrap prompt, or run the embedded verifier (`Section 6`) locally to emit a cryptographically sealed W3C PROV-O `MERGE_LEDGER.jsonl`.

---

## 1. EXECUTIVE CONTEXT & ADR-0001 (WHY v30.next EXISTS)

### 1.1 Observed Fault Lines (Baseline: November 25, 2026)
* **External Audit Signal:** The November 23, 2026 BaaaTON audit scored `adamvangrover/adam` **73.0 / 100 (Grade B)** — Autonomy (`ANIMA`) **28.0 / 35.0**, Transparency (`LOGOS`) **25.0 / 35.0**, and Interoperability (`NEXUS`) **20.0 / 30.0**.
* **Root Cause of Interoperability Drag:** 204 open pull requests, overlapping top-level namespaces (`adam/`, `adam-core/`, `packages/adam-core/`, `adam-os/`, `adam_v3/`, `afos/`, `core_kernel/`, `domains/`, `agents/`), four competing UI roots (`applications/`, `apps/`, `portal/`, `public/`), and fragmented Compose stacks.
* **Architectural Decision (ADR-0001):** Do **not** discard functional breadth to achieve cleanliness. Instead, lock a single canonical namespace, unify Python (`uv`), Rust (`Cargo`/`Maturin`), Node (`pnpm`), and Docker Compose profiles into one control plane, scale to **50–250+ async swarm agents** via NATS JetStream + Temporal + Rust PyO3 hot paths, and converge static, synthetic, and live data onto one cryptographically isomorphic `EvidenceEnvelope v2` plane.

### 1.2 Canonical Namespace Consolidation Map
| Legacy Root Directories | Canonical `v30.next` Target | Plane & Non-Negotiable Invariant |
| :--- | :--- | :--- |
| `adam/`, `adam_v3/`, `afos/`, `core_kernel/`, `adam-os/`, `agents/`, `memory_bus/` | `src/adam/core/` | **Path A (Core):** Strict Pydantic v2 `AgentInput`/`AgentOutput`, `Nexus`-only dispatch, 4-module harness (`h_ctx`, `h_ctrl`, `h_act`, `h_ver`), W3C PROV-O on every material node. |
| `adam-core/`, `packages/adam-core/`, `policies/` | `rust/adam_core/` | **Path A (Deterministic Kernel):** PyO3/Maturin wheel, topological DAG scheduler, C5 Degraded Mode AST, versioned `rules.jsonLogic`, SHA-256 hashing. Zero Python financial `if/else`. |
| `adam_finance/`, `adam_graph/`, `domains/`, `data_pipelines/`, `reasoning_frameworks/` | `src/adam/domains/` | **Path A (Domain Packs):** Pluggable `credit`, `lmt_contagion`, `macro`, `market_mayhem` packs + MCP tool adapters. Zero direct peer-to-peer agent calls. |
| `applications/`, `apps/`, `portal/`, `public/` | `apps/studio/` + `docs/site/` | **UI Boundary:** Presentation-only 6-Dimension Review Console & terminal; zero business logic or state mutation. |
| `architect_output/`, `benchmarks/`, `simulation_environment/`, `mocks/`, `downloads/` | `lab/` | **Path B (The Lab):** Offline trace-replay, shadow testing, synthetic calibration, outer-loop harness evolution. **Never imported by `src/adam/`.** |
| `adam-memory/schemas/`, `docs/`, `skills/`, `docker/` | `schemas/`, `docs/`, `skills/`, `deploy/` | **Control Plane:** Canonical boot chain `LLM_README.md` -> `AGENTS.md` -> `CONSTITUTION.md` -> `AGENT_CATALOG.md`. |

---

## 2. SYSTEM CONSTITUTION (8 HARD INVARIANTS — LAYER 0 FLOOR)

Violation of **any** invariant forces `Layer 0 = FAIL`, total score = `0.00`, and immediate hard abort (`C < 0.50`):

1. **Path A / Path B Isolation:** `src/adam/` MUST NEVER import from `lab/`, `experimental/`, `research/`, or `tinker_lab/`. Enforced by CI AST Import Firewall.
2. **Dual-Process Routing via Single Dispatcher:** Stateless high-fan-out intake runs on System 1 (`AsyncAgentBase` over NATS JetStream); stateful multi-step DAGs run on System 2 (`TemplateAgentV30` over Temporal). Agents emit handoff intents in `AgentOutput.metadata` dispatched solely by `Nexus`.
3. **PDIL (Probabilistic-to-Deterministic Integration Layer):** LLMs hypothesize, parse, and narrate. All financial ratios, covenants, waterfalls, dual obligor PDs, and risk thresholds execute strictly in versioned `rules.jsonLogic` or `rust/adam_core/`. Every output MUST carry `jsonLogic_version`.
4. **4-Module Code-Level Harness:** Reliability lives in code hooks, never prompt bloat:
   * `h_ctx`: Separates working DAG state from experiential failure memory; keeps reference snapshots frozen (Invariant F).
   * `h_ctrl` (pre-turn): Enforces turn caps, loop breakers, concurrency semaphores, and `.adam_halt` / Redis kill latch.
   * `h_act` (post-tool): Truncates noisy tool payloads, normalizes errors, validates constrained grammars, emits `SelfReportTelemetry`.
   * `h_ver` (pre-stop): Blocks unverified exits, runs `AFOS-SNT-01` sentinel + PDIL checks, enforces confidence tiers.
5. **Market Intelligence Evidence Gate v1.0 & No Epistemic Laundering:** Every empirical claim carries an immutable `epistemic_class` (`OBSERVED`, `CALCULATED`, `SYNTHETIC_CALIBRATED`, `STATIC_FIXTURE`, `MODEL_EST`, `PENDING`, `REJECTED`). Consensus or high stated confidence MUST NEVER promote synthetic or estimated claims to `OBSERVED`. Every `OBSERVED` claim binds to an HTTPS `canonical_uri`, bitemporal timestamps (`valid_time`, `publication_time`, `retrieval_time`), and RFC 8785 canonical SHA-256 digest.
6. **Multiplicative Safety Gating (`Layer 0 x Layer 1`):**
   * `Layer 0` is non-compensatory (`PASS` / `HOLD` / `FAIL`).
   * `Layer 1` weights: Analytical Accuracy (**45%**) + Epistemic Governance (**35%**) + Actionability/Pareto Efficiency (**20%**).
   * Confidence Tiers: `C >= 0.85` + `PASS` (autonomous low/med impact only) | `0.50 <= C < 0.85` or `HOLD` (HITL pause) | `C < 0.50` or `FAIL` (hard abort). Tier 2 mutations (merges to `main`, rule edits, credit/rating/capital actions, external publishing) ALWAYS require Expert Human sign-off.
7. **W3C PROV-O Provenance:** Every material decision emits machine-actionable `prov:Entity`, `prov:Activity`, and `prov:Agent` lineage (`wasGeneratedBy`, `used`, `wasDerivedFrom`, `wasAttributedTo`) to an append-only JSON-LD/JSONL ledger.
8. **P0 Security & Evaluator Isolation:** Ban `pickle.load()`, `eval()`, `exec()`, and user-driven dynamic imports. Enforce `Path.resolve().is_relative_to(allowed_root.resolve())` on all file I/O. Lock `h_ver`, `AFOS-SNT-01`, `rules.jsonLogic` floors, golden evals, and `schemas/calibration_registry.json` as SHA-256 READ-ONLY at runtime.

---

## 3. RESEARCH SPECIFICATION: META-SWARM CONSENSUS & ISOMORPHIC DATA PLANE

### 3.1 Calibration-Weighted Consensus (Eliminating Sybil & Correlated LLM Drift)
In a 50–250+ agent swarm, naive majority voting degrades accuracy when correlated uncalibrated agents outvote calibrated specialists. Each agent $i$ emits proposal $y_i$ and a structured `SelfReportTelemetry` vector $r_i = (\text{evidence\_ratio}_i, \text{bottleneck\_flags}_i, \text{dissent}_i, \hat{c}_i)$. `Nexus` computes effective voting weight $w_i$ strictly from read-only Layer 3 historical calibration skill $S_i^{\text{hist}}$ (out-of-sample Brier/CRPS) gated by deterministic Layer 0 compliance $L0_i \in \{0, 1\}$:

$$w_i = L0_i \cdot \text{evidence\_ratio}_i \cdot S_i^{\text{hist}} \cdot \mathbb{I}(\neg \text{bottleneck\_critical}_i)$$

* **Sybil Clamp:** Ungrounded agents or agents hiding tool timeouts collapse to $w_i = 0$ regardless of stated confidence $\hat{c}_i$.
* **Asymmetric Specialist Veto:** Any `AFOS-SNT-01` or `AFOS-CMP-01` Layer 0 fault or bidirectional spread divergence exceeding threshold $\theta$ overrides swarm majority and forces `HOLD` or `FAIL`.
* **Attention-Bounded Human Bottleneck:** The swarm compresses 50+ trajectories into a 1-page **6-Dimension Decision Packet** surfacing steel-manned bull/bear dissent and exact conflicting evidence hashes, concentrating human judgment strictly at maximum Value-of-Information (VoI) junctions.

### 3.2 Isomorphic Data Plane (`EvidenceEnvelope v2`)
Static fixtures, calibrated synthetic twins, and live feeds flow through the **exact same schema and Path A kernel** with zero conditional branching in agent code. Adapters graduate 1-for-1 in `config/data_adapters.yaml` (`STATIC_FIXTURE` -> `SYNTHETIC_CALIBRATED` -> `OBSERVED`), while `lab/calibration/shadow_reconcile.py` scores CRPS/Brier/KS distributional fit without ever mutating `epistemic_class`.

---

## 4. POLYMORPHIC ARTIFACT CONTRACT (ONE KERNEL, ANY OUTPUT)

Every ADAM deliverable—whether a **Forward Price Target**, **Credit Rating / SNC Memo**, **Market Mayhem Newsletter**, **Terminal View**, **Agent/Workflow Spec**, or **Telemetry Packet**—instantiates the exact same polymorphic wrapper so it is uniformly scored and audited by `Layer 3`:

```json
{
  "schema_version": "EvidenceEnvelope.v2.0",
  "artifact_type": "PRICE_TARGET | CREDIT_RATING | NEWSLETTER | REPORT | TERMINAL_VIEW | AGENT_SPEC | WORKFLOW | TELEMETRY",
  "issue_id": "string",
  "pdil_ruleset_version": "pdil_v30.next.2026-11-25",
  "epistemic_summary": {
    "OBSERVED": 0,
    "CALCULATED": 0,
    "SYNTHETIC_CALIBRATED": 0,
    "PENDING": 0
  },
  "governance_gate": {
    "layer0_status": "PASS | HOLD | FAIL",
    "layer1_weights": { "analytical_accuracy": 0.45, "epistemic_governance": 0.35, "actionability_pareto": 0.20 },
    "layer1_composite_score": 0.0,
    "effective_autonomous_score": 0.0
  },
  "provenance": {
    "prov:wasGeneratedBy": "adam_v30_next_pipeline",
    "prov:wasAttributedTo": "AFOS-UWR-01 / AFOS-SNT-01",
    "artifact_sha256": "PENDING_LOCAL_INGEST_HASH"
  }
}

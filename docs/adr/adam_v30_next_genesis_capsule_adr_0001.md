---
artifact_id: "ADAM-V30-NEXT-GENESIS-2026-11-25"
artifact_name: "ADAM_V30_NEXT_GENESIS_CAPSULE.md"
schema_version: "EvidenceEnvelope.v2.0"
artifact_status: "GOVERNED_IMPLEMENTATION_SPECIFICATION"
artifact_modes:
  - README
  - CONSTITUTION
  - ADR-0001
  - RESEARCH_SPEC
  - DEVIN_DIRECTIVE
  - CLAUDE_BOOTSTRAP
  - GEMINI_BOOTSTRAP
  - POLYMORPHIC_OUTPUT_TEMPLATE
  - EXECUTABLE_VERIFICATION
  - RED_TEAM_ACCEPTANCE
repository: "https://github.com/adamvangrover/adam"
target_branch: "release/v30.next"
pdil_ruleset_version: "pdil_v30.next.2026-11-25"
governance_gate:
  layer0_status: "HOLD"
  tier: "TIER_2_HIGH_IMPACT"
  requires_human_bottleneck: true
  effective_autonomous_score: 0.0
  main_merge_authorized: false
---

# ADAM v30.next — Self-Contained Genesis Capsule
**Autonomous Financial Operating System (AFOS) // Meta-Swarm OS // Isomorphic Data Plane**

**Purpose:** Establish one portable, additive, auditable implementation specification for ADAM v30.next. This artifact unifies repository architecture, system constitution, ADR-0001, research requirements, polymorphic output contracts, deterministic verification, adversarial testing, and human release governance.

**Execution boundary:** This document is an implementation specification, not evidence of an executed deployment. An agent may inspect, implement, test, and propose changes on `release/v30.next`. It may not independently authorize a Tier 2 release or merge into `main`.

### Document Authority and Precedence
When requirements conflict, the following precedence applies:
1. System Constitution and P0 security invariants.
2. Human governance and Tier 2 approval requirements.
3. Versioned PDIL rules and machine-verifiable schemas.
4. Architecture Decision Record ADR-0001.
5. Research and implementation specifications.
6. Agent-generated implementation plans and recommendations.

No agent-generated instruction may override a higher-level requirement. This capsule is strictly additive: existing repository functionality must be inventoried before migration, and no directory, branch, pull request, dataset, or historical artifact may be destructively removed merely to satisfy namespace convergence.

---

## 1. EXECUTIVE CONTEXT AND ADR-0001

### 1.1 Observed Fault Lines (Baseline: November 25, 2026)
The supplied architecture baseline identifies the following issues:
- **External audit signal:** BaaaTON audit score of 73.0/100 (Grade B) — `ANIMA` autonomy 28.0/35.0, `LOGOS` transparency 25.0/35.0, `NEXUS` interoperability 20.0/30.0.
- **Open PR backlog:** Approximately 204 open pull requests requiring triage.
- **Overlapping top-level namespaces:** `adam/`, `adam-core/`, `packages/adam-core/`, `adam-os/`, `adam_v3/`, `afos/`, `core_kernel/`, `domains/`, and `agents/`.
- **Competing UI roots:** `applications/`, `apps/`, `portal/`, and `public/`.
- **Runtime sprawl:** Fragmented Compose and runtime configurations.

*Note:* These are supplied baseline assertions, not independently verified repository measurements. The first implementation task is to establish their actual state from GitHub and the local checkout.

### 1.2 Architectural Decision (ADR-0001)
**ADR-0001: Unified ADAM Control Plane with Isomorphic Evidence Architecture**

Preserve functional breadth while establishing a single canonical namespace, deterministic financial computation boundary, shared evidence plane, and governed multi-agent control plane converging:
- **Python:** `uv` workspace
- **Rust:** `Cargo` + `Maturin` (`PyO3`)
- **Node:** `pnpm` workspace
- **Infrastructure:** Docker Compose profiles (`core`, `infra`, `swarm`, `ui`, `lab`)
- **Messaging & Workflows:** NATS JetStream (System 1) + Temporal (System 2)
- **Evidence & Provenance:** `EvidenceEnvelope v2` + W3C PROV-O
- **Dispatch & Governance:** `Nexus` + Layer 0/Layer 1 multiplicative gating + mandatory human bottleneck

Target swarm capacity is 50–250+ asynchronous agents, subject to measured capacity, resource budgets, and fault-injection acceptance.

### 1.3 Canonical Namespace Consolidation
| Legacy Roots | Canonical Target | Non-Negotiable Invariant |
| :--- | :--- | :--- |
| `adam/`, `adam_v3/`, `afos/`, `core_kernel/`, `adam-os/`, `agents/`, `memory_bus/` | `src/adam/core/` | Path A core; strict Pydantic v2 contracts; `Nexus`-only dispatch; four-module harness; provenance. |
| `adam-core/`, `packages/adam-core/`, `policies/` | `rust/adam_core/` | Deterministic kernel; PyO3/Maturin; versioned rules; SHA-256. |
| `adam_finance/`, `adam_graph/`, `domains/`, `data_pipelines/`, `reasoning_frameworks/` | `src/adam/domains/` | Pluggable domain packs; MCP adapters; no direct peer agent calls. |
| `applications/`, `apps/`, `portal/`, `public/` | `apps/studio/` and `docs/site/` | Presentation-only boundary; zero business logic or state mutation. |
| `architect_output/`, `benchmarks/`, `simulation_environment/`, `mocks/`, `downloads/` | `lab/` | Path B offline research, simulation, calibration, and shadow testing. |
| `adam-memory/schemas/`, `docs/`, `skills/`, `docker/` | `schemas/`, `docs/`, `skills/`, `deploy/` | Canonical governance and boot chain. |

#### 1.3.1 Migration Rules
- Inventory all existing paths before moving files and generate a source-to-destination migration manifest.
- Preserve Git history where practical and maintain compatibility shims until consumers migrate.
- Record all deprecations in `docs/DEPRECATIONS.md`.
- Do not delete or close pull requests automatically.
- Require review of every migration that alters financial, security, or governance behavior.

### 1.4 Canonical Boot Chain
Every implementation agent must read the following files in order before changing code:
`LLM_README.md` -> `AGENTS.md` -> `CONSTITUTION.md` -> `AGENT_CATALOG.md`

---

## 2. SYSTEM CONSTITUTION — EIGHT LAYER 0 INVARIANTS

Any violation of a constitutional invariant forces `Layer 0 = FAIL`, the effective autonomous score to `0.0`, and execution to abort. A `HOLD` is not a compensating score, and Layer 1 performance cannot override Layer 0.

### Invariant 1 — Path A / Path B Isolation
- **Path A:** `src/adam/`
- **Path B:** `lab/`, `experimental/`, `research/`, `tinker_lab/`
- Path A MUST NOT import from Path B. Enforced via CI AST import firewall, runtime package-boundary checks, clean-room import tests, and build dependency inspection. Path B may consume Path A interfaces but must never become a runtime dependency of Path A.

### Invariant 2 — Dual-Process Routing Through Nexus
- **System 1:** Stateless, high-fan-out intake (`AsyncAgentBase`, NATS JetStream, bounded concurrency and backpressure).
- **System 2:** Stateful multi-step workflows (`TemplateAgentV30`, Temporal, durable state and retry policies).
- Agents emit structured handoff intents in `AgentOutput.metadata`. Only `Nexus` may dispatch inter-agent handoffs; direct peer-to-peer agent invocation is prohibited.

### Invariant 3 — Probabilistic-to-Deterministic Integration Layer (PDIL)
- LLMs may hypothesize, interpret, extract, classify, narrate, and generate candidate explanations.
- LLMs may not independently calculate or authorize material financial results.
- All financial ratios, leverage, interest coverage, covenants, waterfalls, dual obligor PDs, risk thresholds, and capital/credit constraints execute strictly in versioned `rules.jsonLogic` or `rust/adam_core/`.
- Every applicable output MUST identify its `jsonLogic_version`. No financial calculation may be silently replaced by an LLM-generated Python expression.

### Invariant 4 — Four-Module Code-Level Harness
Reliability belongs in executable code hooks, never prompt bloat:
- **`h_ctx` (Context & State Isolation):** Separates active DAG state from experiential failure memory; preserves immutable reference snapshots (Invariant F); records context lineage.
- **`h_ctrl` (Pre-Turn Control):** Enforces turn caps, loop breakers, concurrency semaphores, token/time budgets, `.adam_halt`, Redis kill latch, and circuit breakers.
- **`h_act` (Post-Tool Validation):** Truncates oversized tool payloads, normalizes errors, validates constrained grammars, emits `SelfReportTelemetry`, preserves source evidence references.
- **`h_ver` (Pre-Stop Verification):** Blocks unverified exits, executes `AFOS-SNT-01`, runs PDIL validation, enforces confidence tiers, rejects missing evidence or rule versions, emits verification provenance.

### Invariant 5 — Market Intelligence Evidence Gate v1.0
Every empirical claim carries an immutable `epistemic_class` from:
`OBSERVED`, `CALCULATED`, `SYNTHETIC_CALIBRATED`, `STATIC_FIXTURE`, `MODEL_EST`, `PENDING`, `REJECTED`.

No consensus, confidence score, or repeated model assertion may promote a synthetic or estimated claim to `OBSERVED`. Every `OBSERVED` claim requires an HTTPS `canonical_uri`, bitemporal `valid_time`, `publication_time`, `retrieval_time`, immutable source reference, canonical artifact digest, and explicit provenance. Missing evidence must remain missing.

### Invariant 6 — Multiplicative Safety Gating
`Layer 0` (`PASS` | `HOLD` | `FAIL`) is non-compensatory.

| Layer 1 Component | Weight |
| :--- | :--- |
| Analytical Accuracy | 45% |
| Epistemic Governance | 35% |
| Actionability / Pareto Efficiency | 20% |

$$L_1 = 0.45 \times \text{Analytical Accuracy} + 0.35 \times \text{Epistemic Governance} + 0.20 \times \text{Actionability}$$

All Layer 1 component inputs must be finite normalized values in $[0, 1]$.

| Condition | Required Action |
| :--- | :--- |
| $C \ge 0.85$ and Layer 0 `PASS` | Eligible for autonomous low/medium-impact actions |
| $0.50 \le C < 0.85$ | Human review |
| Layer 0 `HOLD` | Human review; no autonomous promotion |
| $C < 0.50$ or Layer 0 `FAIL` | Hard abort |

**Tier 2 actions ALWAYS require expert human sign-off:** merging into `main`, editing protected financial rules, credit/rating/capital actions, external publication, and changes to constitutional security controls. Human approval is necessary but cannot override a Layer 0 `FAIL`.

### Invariant 7 — W3C PROV-O Provenance
Every material decision emits machine-actionable provenance (`prov:Entity`, `prov:Activity`, `prov:Agent`, `prov:wasGeneratedBy`, `prov:used`, `prov:wasDerivedFrom`, `prov:wasAttributedTo`) to an append-oriented, hash-chained JSONL ledger. Production deployments must additionally protect the ledger via access controls and immutable external storage.

### Invariant 8 — P0 Security and Evaluator Isolation
- **Prohibited:** `pickle.load()`, `eval()`, `exec()`, user-controlled dynamic imports, unvalidated file paths, and runtime mutation of protected evaluators.
- All file operations must resolve paths against explicit allowed roots (`is_relative_to`).
- **Protected assets:** `h_ver`, `AFOS-SNT-01`, `rules.jsonLogic`, golden evaluation fixtures, and `schemas/calibration_registry.json`. Expected digests must be anchored in a trusted release manifest with read-only runtime permissions.

---

## 3. RESEARCH SPECIFICATION — META-SWARM CONSENSUS AND ISOMORPHIC DATA PLANE

### 3.1 Calibration-Weighted Consensus
Each agent $i$ emits a proposal and telemetry vector $r_i = (\text{evidence\_ratio}_i, \text{bottleneck\_flags}_i, \text{dissent}_i, \hat{c}_i)$. `Nexus` computes effective voting weight:

$$w_i = L0_i \cdot \text{evidence\_ratio}_i \cdot S_i^{\text{hist}} \cdot \mathbb{I}(\neg \text{bottleneck\_critical}_i)$$

where $L0_i \in \{0, 1\}$ is deterministic Layer 0 compliance and $S_i^{\text{hist}}$ is read-only out-of-sample historical calibration skill (Brier/CRPS).
1. Ungrounded agents or missing evidence hashes receive $w_i = 0$.
2. Concealed tool timeouts constitute a telemetry integrity fault.
3. Stated confidence does not increase historical calibration skill.
4. Correlated model outputs must not be counted as independent evidence.
5. Calibration snapshots are read-only during a decision and updated only via separately governed workflows.

### 3.2 Asymmetric Specialist Veto
`AFOS-SNT-01` and `AFOS-CMP-01` hold asymmetric veto authority. A Layer 0 fault or bidirectional spread divergence beyond threshold forces `HOLD` or `FAIL` and cannot be overridden by majority vote.

### 3.3 Human Attention Bottleneck
The swarm compresses 50+ trajectories into a one-page **6-Dimension Decision Packet** (Analytical Accuracy, Evidence Integrity, Epistemic Governance, Risk & Downside, Actionability, Confidence & Unresolved Dissent), surfacing steel-manned bull/bear cases, conflicting evidence hashes, calibration history, and exact decision consequences.

### 3.4 Isomorphic `EvidenceEnvelope v2` Data Plane
Static fixtures, calibrated synthetic twins, and live observations share one schema and Path A interface. Progression (`STATIC_FIXTURE` -> `SYNTHETIC_CALIBRATED` -> `OBSERVED`) is an evidence-state transition requiring independent validation, never automatic promotion. `lab/calibration/shadow_reconcile.py` scores CRPS, Brier, KS fit, and drift without mutating historical epistemic labels.

---

## 4. POLYMORPHIC OUTPUT CONTRACT

Supported `artifact_type` values: `PRICE_TARGET`, `CREDIT_RATING`, `NEWSLETTER`, `REPORT`, `TERMINAL_VIEW`, `AGENT_SPEC`, `WORKFLOW`, `TELEMETRY`.

```json
{
  "schema_version": "EvidenceEnvelope.v2.0",
  "artifact_type": "PRICE_TARGET",
  "issue_id": "ISSUE-0001",
  "pdil_ruleset_version": "pdil_v30.next.2026-11-25",
  "epistemic_summary": {
    "OBSERVED": 0,
    "CALCULATED": 0,
    "SYNTHETIC_CALIBRATED": 0,
    "STATIC_FIXTURE": 0,
    "MODEL_EST": 0,
    "PENDING": 0,
    "REJECTED": 0
  },
  "governance_gate": {
    "layer0_status": "HOLD",
    "is_tier2_publication": true,
    "layer1_weights": {
      "analytical_accuracy": 0.45,
      "epistemic_governance": 0.35,
      "actionability_pareto": 0.20
    },
    "layer1_component_scores": {
      "analytical_accuracy": 0.0,
      "epistemic_governance": 0.0,
      "actionability_pareto": 0.0
    },
    "layer1_composite_score": 0.0,
    "effective_autonomous_score": 0.0,
    "requires_human_approval": true
  },
  "provenance": {
    "prov:wasGeneratedBy": "adam_v30_next_pipeline",
    "prov:wasAttributedTo": "AFOS-UWR-01",
    "artifact_sha256": null
  },
  "evidence_envelopes": []
}
```

### 4.1 Required Evidence Envelope Schema
```json
{
  "envelope_id": "EVIDENCE-0001",
  "epistemic_class": "OBSERVED",
  "canonical_uri": "https://example.org/source",
  "valid_time": "2026-11-25T00:00:00Z",
  "publication_time": "2026-11-25T12:00:00Z",
  "retrieval_time": "2026-11-25T13:00:00Z",
  "lag_business_days": 0,
  "payload": {},
  "artifact_sha256": null
}
```
*Schema enforcement:* Reject unknown epistemic classes, missing envelope IDs, invalid timestamps, invalid artifact types, missing PDIL version, non-finite numeric values, unsealed evidence, invalid observed-source references, and inconsistent epistemic summaries.

---

## 5. IMPLEMENTATION ROADMAP — DEVIN / CLAUDE / GEMINI

All work occurs on `release/v30.next`. Never force-push. Never merge directly to `main`.

- **Phase 0 — Unified Runtime & Document Fabric:** Root `uv`, `Cargo`, and `pnpm` workspaces; `deploy/compose.yml` (`core`, `infra`, `swarm`, `ui`, `lab`); `scripts/boot_sec_check.sh`; `.adam_halt` + Redis kill latch; canonical boot docs. *Acceptance:* Clean boot, security check passes, kill latch verified, zero Path B runtime dependency, reproducible lockfiles.
- **Phase 1 — Non-Destructive Branch Harvest:** Inventory all open PRs into Plane A1 (Kernel/Security), Plane A2 (Domains/Swarms), Plane B (Lab), and Superseded. Deliver `scripts/triage_branches.py`, `docs/V30_NEXT_TRIAGE_REPORT.md`, `MERGE_LEDGER.jsonl`, non-executed `scripts/cleanup_stale_prs.sh`, and CI AST Import Firewall. *Acceptance:* 100% PR classification; zero automatic PR closures.
- **Phase 2 — Async Swarm & Rust Hot Path:** Wire NATS JetStream, Temporal, `rust/adam_core/` (PyO3/Maturin), and `h_ctx`/`h_ctrl`/`h_act`/`h_ver`. Benchmark via `scripts/bench_swarm_100.py` (100 concurrent agents; p50/p95/p99/p99.9 latency, throughput, tokens, failures, queue depth, deadlocks, memory, mid-flight kill latch). Do not claim 250-agent readiness from a 100-agent test alone.
- **Phase 3 — Consensus & Calibration Registry:** Deliver `SelfReportTelemetry`, read-only `schemas/calibration_registry.json`, calibration-weighted `Nexus` arbiter, specialist veto, correlation controls, and dissent preservation. *Acceptance:* Sybil and self-report adversarial tests pass.
- **Phase 4 — Expert Human Bottleneck:** Deliver VoI Escalation Router, `apps/studio/` 6-Dimension Review Console, human approval service, and append-oriented `EXPERIENCE.md`. *Acceptance:* Every Tier 2 action requires authenticated approval bound to exact artifact digest.
- **Phase 5 — Isomorphic Data Plane:** Deliver `EvidenceEnvelope v2`, `config/data_adapters.yaml`, `lab/calibration/shadow_reconcile.py`, and golden evaluation suite. *Acceptance:* `RT-01` through `RT-08` pass in CI; single HITL PR opened from `release/v30.next` -> `main`.

---

## 6. REPAIRED PDIL AND EXECUTABLE VERIFICATION ENGINE

### 6.1 Canonical PDIL Rules (`pdil_v30_next.json`)
```json
{
  "jsonLogic_version": "pdil_v30.next.2026-11-25",
  "owner": "2LoD Credit Risk Control / AFOS-SNT-01",
  "read_only_runtime": true,
  "rules": {
    "credit_regime_classification": {
      "if": [
        { ">=": [{ "var": "hy_oas_bps" }, 450] },
        "DISTRESSED_FREEZE",
        {
          "or": [
            { ">=": [{ "var": "hy_oas_bps" }, 375] },
            { ">=": [{ "var": "hy_5d_widening_bps" }, 50] }
          ]
        },
        "ELEVATED_STRESS",
        "NORMAL_MONITORING"
      ]
    },
    "bidirectional_divergence_alert": {
      "and": [
        { ">=": [{ "var": "hy_5d_widening_bps" }, 40] },
        { ">": [{ "var": "dow_close_pct" }, 0] },
        { ">=": [{ "var": "vix" }, 25] }
      ]
    },
    "snc_grade_from_icr": {
      "if": [
        { "<": [{ "var": "stressed_icr" }, 1.1] },
        "DOUBTFUL",
        { "<": [{ "var": "stressed_icr" }, 1.5] },
        "SUBSTANDARD",
        { "<": [{ "var": "stressed_icr" }, 1.85] },
        "SPECIAL_MENTION",
        "PASS_WATCH"
      ]
    },
    "layer0_gate_status": {
      "if": [
        {
          "or": [
            { ">": [{ "var": "epistemic_laundering_violations" }, 0] },
            { "==": [{ "var": "pdil_version_present" }, false] },
            { "==": [{ "var": "evaluator_integrity_ok" }, false] }
          ]
        },
        "FAIL",
        {
          "or": [
            { ">": [{ "var": "max_series_lag_days" }, 1] },
            { ">": [{ "var": "unsealed_hash_count" }, 0] },
            {
              "and": [
                { "==": [{ "var": "is_tier2_publication" }, true] },
                { "==": [{ "var": "human_approval_valid" }, false] }
              ]
            },
            { "==": [{ "var": "sentinel_veto_triggered" }, true] }
          ]
        },
        "HOLD",
        "PASS"
      ]
    },
    "layer1_weighted_score": {
      "+": [
        { "*": [{ "var": "analytical_accuracy" }, 0.45] },
        { "*": [{ "var": "epistemic_governance" }, 0.35] },
        { "*": [{ "var": "actionability_pareto" }, 0.20] }
      ]
    },
    "effective_autonomous_score": {
      "if": [
        { "==": [{ "var": "layer0_status" }, "PASS"] },
        { "var": "layer1_score" },
        0.0
      ]
    }
  }
}
```
*Note:* `layer0_gate_status` returns `HOLD` when a Tier 2 publication lacks a valid cryptographic approval, and allows evaluation of the remaining gates once a valid approval is present.

### 6.2 Verification Engine (`adam_verify.py`)
Pinned dependencies:
```bash
python3 -m pip install "rfc8785==0.1.4" "cryptography==46.0.3"
```

```python
#!/usr/bin/env python3
"""ADAM v30.next governed evidence verifier.

Exit codes:
    0 = PASS
    2 = HOLD
    4 = FAIL

No command-line Boolean can authorize a Tier 2 action.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import rfc8785
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:
    print("FAIL: Install pinned verification dependencies.", file=sys.stderr)
    sys.exit(4)

ALLOWED_EPISTEMIC = {
    "OBSERVED",
    "CALCULATED",
    "SYNTHETIC_CALIBRATED",
    "STATIC_FIXTURE",
    "MODEL_EST",
    "PENDING",
    "REJECTED",
}

EXPECTED_PDIL = "pdil_v30.next.2026-11-25"

EXIT_PASS = 0
EXIT_HOLD = 2
EXIT_FAIL = 4


class VerificationError(Exception):
    pass


def canonical_bytes(value: Any) -> bytes:
    """RFC 8785 canonical JSON representation."""
    return rfc8785.dumps(value)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def contained_path(path: Path, root: Path) -> Path:
    """Resolve paths and reject traversal or symlink escape."""
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise VerificationError(f"Path escapes allowed root: {resolved}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def validate_finite(value: Any) -> None:
    """Reject non-finite numeric values recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError("Non-finite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            validate_finite(item)


def eval_jl(expr: Any, data: dict[str, Any]) -> Any:
    """Restricted deterministic JSONLogic evaluator."""
    if not isinstance(expr, dict):
        return expr
    if len(expr) != 1:
        raise VerificationError("Invalid JSONLogic expression")

    op, args = next(iter(expr.items()))
    if op == "var":
        if not isinstance(args, list) or not args:
            raise VerificationError("Invalid var expression")
        name = args[0]
        default = args[1] if len(args) > 1 else None
        return data[name] if name in data else default

    if not isinstance(args, list):
        raise VerificationError(f"Invalid arguments for {op}")

    if op == "if":
        if len(args) < 2:
            raise VerificationError("Invalid if expression")
        for index in range(0, len(args) - 1, 2):
            if eval_jl(args[index], data):
                return eval_jl(args[index + 1], data)
        return eval_jl(args[-1], data) if len(args) % 2 else None

    if op in {"and", "or"}:
        values = []
        for item in args:
            value = eval_jl(item, data)
            values.append(value)
            if op == "and" and not value:
                return False
            if op == "or" and value:
                return True
        return all(values) if op == "and" else any(values)

    values = [eval_jl(item, data) for item in args]

    if op in {">", ">=", "<", "<=", "==", "!="}:
        if len(values) != 2:
            raise VerificationError(f"Invalid arity: {op}")
        left, right = values
        if op in {">", ">=", "<", "<="}:
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
            ):
                raise VerificationError("Invalid comparison types")
        return {
            ">": lambda: left > right,
            ">=": lambda: left >= right,
            "<": lambda: left < right,
            "<=": lambda: left <= right,
            "==": lambda: type(left) is type(right) and left == right,
            "!=": lambda: type(left) is not type(right) or left != right,
        }[op]()

    if op in {"+", "*"}:
        if not values or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in values
        ):
            raise VerificationError("Invalid arithmetic operands")
        result = sum(values) if op == "+" else math.prod(values)
        if not math.isfinite(float(result)):
            raise VerificationError("Non-finite arithmetic result")
        return result

    raise VerificationError(f"Unsupported operator: {op}")


def evidence_hash_payload(env: dict[str, Any]) -> dict[str, Any]:
    """Fields bound into each evidence digest."""
    return {
        key: env.get(key)
        for key in (
            "envelope_id",
            "epistemic_class",
            "canonical_uri",
            "valid_time",
            "publication_time",
            "retrieval_time",
            "payload",
        )
    }


def validate_envelope(env: dict[str, Any]) -> list[str]:
    errors = []
    eid = env.get("envelope_id", "?")
    epistemic = env.get("epistemic_class")

    if epistemic not in ALLOWED_EPISTEMIC:
        errors.append(f"{eid}: invalid epistemic_class")
    if "claimed_epistemic_class" in env and env["claimed_epistemic_class"] != epistemic:
        errors.append(f"{eid}: epistemic classification mismatch")

    if epistemic == "OBSERVED":
        uri = env.get("canonical_uri")
        if not isinstance(uri, str) or not uri.startswith("https://"):
            errors.append(f"{eid}: OBSERVED requires HTTPS URI")
        for field in ("valid_time", "publication_time", "retrieval_time"):
            value = env.get(field)
            if not isinstance(value, str):
                errors.append(f"{eid}: missing {field}")
                continue
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{eid}: invalid {field}")

    return errors


def verify_approval(
    approval_path: Path | None,
    public_key_path: Path | None,
    root: Path,
    issue_id: str,
    bundle_hash: str,
) -> bool:
    if approval_path is None or public_key_path is None:
        return False

    approval_path = contained_path(approval_path, root)
    public_key_path = contained_path(public_key_path, root)
    approval = load_json(approval_path)

    required = {
        "issue_id",
        "artifact_bundle_sha256",
        "reviewer_id",
        "reviewer_role",
        "approved_at_utc",
        "scope",
        "signature",
    }
    if not required.issubset(approval):
        return False
    if approval["issue_id"] != issue_id:
        return False
    if approval["artifact_bundle_sha256"] != bundle_hash:
        return False
    if approval["reviewer_role"] != "EXPERT_APPROVER":
        return False
    if approval["scope"] != "TIER2_RELEASE":
        return False

    unsigned = {k: v for k, v in approval.items() if k != "signature"}
    try:
        key = serialization.load_pem_public_key(public_key_path.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            return False
        signature = base64.b64decode(approval["signature"], validate=True)
        key.verify(signature, canonical_bytes(unsigned))
        return True
    except Exception:
        return False


def verify_ledger(path: Path) -> str:
    """Verify existing hash-chain continuity."""
    if not path.exists():
        return "0" * 64

    previous = "0" * 64
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise VerificationError(f"Blank ledger line: {line_number}")
            record = json.loads(line)
            recorded_hash = record.pop("record_sha256", None)
            if record.get("previous_record_sha256") != previous:
                raise VerificationError(f"Ledger chain broken at line {line_number}")
            if recorded_hash != digest(record):
                raise VerificationError(f"Ledger hash mismatch at line {line_number}")
            previous = recorded_hash
    return previous


def append_ledger(path: Path, record: dict[str, Any]) -> None:
    previous = verify_ledger(path)
    record["previous_record_sha256"] = previous
    record["record_sha256"] = digest(record)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def verify_bundle(args: argparse.Namespace) -> int:
    root = args.prov.parent.resolve(strict=True)
    prov_path = contained_path(args.prov, root)
    pdil_path = contained_path(args.pdil, root)

    prov = load_json(prov_path)
    pdil = load_json(pdil_path)
    validate_finite(prov)
    validate_finite(pdil)

    findings: list[str] = []
    laundering = 0
    unsealed = 0
    max_lag = 0

    if pdil.get("jsonLogic_version") != EXPECTED_PDIL:
        raise VerificationError("Unexpected PDIL version")
    if pdil.get("read_only_runtime") is not True:
        raise VerificationError("PDIL read-only invariant missing")

    envelopes = prov.get("evidence_envelopes", [])
    if not isinstance(envelopes, list):
        raise VerificationError("Invalid evidence_envelopes")

    for env in envelopes:
        if not isinstance(env, dict):
            raise VerificationError("Invalid envelope object")
        for error in validate_envelope(env):
            findings.append(error)
            laundering += 1

        actual_hash = env.get("artifact_sha256")
        if args.seal_hashes:
            actual_hash = digest(evidence_hash_payload(env))
            env["artifact_sha256"] = actual_hash

        expected_hash = digest(evidence_hash_payload(env))
        if actual_hash != expected_hash:
            unsealed += 1
            findings.append(f"{env.get('envelope_id', '?')}: hash mismatch")

        lag = env.get("lag_business_days", 0)
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise VerificationError("Invalid lag_business_days")
        max_lag = max(max_lag, lag)

    issue_id = prov.get("issue_id")
    if not isinstance(issue_id, str) or not issue_id:
        raise VerificationError("Missing issue_id")

    bundle_hash = digest({
        "issue_id": issue_id,
        "pdil_version": pdil["jsonLogic_version"],
        "envelope_hashes": sorted(env.get("artifact_sha256", "") for env in envelopes),
    })

    if args.seal_hashes:
        prov_path.write_text(
            json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    tier2 = prov.get("governance_gate", {}).get("is_tier2_publication", True)
    human_approval_valid = verify_approval(
        args.approval, args.trusted_key, root, issue_id, bundle_hash
    )

    scores = prov.get("governance_gate", {}).get("layer1_component_scores", {})
    for name in ("analytical_accuracy", "epistemic_governance", "actionability_pareto"):
        value = scores.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise VerificationError(f"Invalid Layer 1 component: {name}")

    rules = pdil["rules"]
    l1 = round(float(eval_jl(rules["layer1_weighted_score"], scores)), 6)

    # Fail closed until an independently trusted integrity
    # manifest has been implemented and successfully verified.
    evaluator_ok = False

    l0 = eval_jl(
        rules["layer0_gate_status"],
        {
            "epistemic_laundering_violations": laundering,
            "pdil_version_present": True,
            "evaluator_integrity_ok": evaluator_ok,
            "max_series_lag_days": max_lag,
            "unsealed_hash_count": unsealed,
            "is_tier2_publication": bool(tier2),
            "human_approval_valid": not tier2 or human_approval_valid,
            "sentinel_veto_triggered": any(
                env.get("payload", {}).get("sentinel_veto_triggered", False)
                for env in envelopes
            ),
        },
    )

    effective = eval_jl(
        rules["effective_autonomous_score"],
        {"layer0_status": l0, "layer1_score": l1},
    )

    if laundering:
        l0 = "FAIL"
        effective = 0.0
    elif l0 == "FAIL":
        effective = 0.0
    elif unsealed:
        l0 = "HOLD"
        effective = 0.0
    elif tier2 and not human_approval_valid:
        l0 = "HOLD"
        effective = 0.0

    record = {
        "@context": {"prov": "http://www.w3.org/ns/prov#"},
        "prov:Entity": issue_id,
        "prov:wasGeneratedBy": "adam_verify.py",
        "prov:wasAttributedTo": "ADAM_VERIFICATION_ENGINE",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdil_version": pdil["jsonLogic_version"],
        "bundle_sha256": bundle_hash,
        "layer0_status": l0,
        "layer1_composite_score": l1,
        "effective_autonomous_score": effective,
        "human_approval_valid": human_approval_valid,
        "laundering_violations": laundering,
        "unsealed_hashes": unsealed,
        "max_lag_days": max_lag,
        "findings": findings,
    }

    if args.seal_hashes:
        append_ledger(root / "MERGE_LEDGER.jsonl", record)

    print(json.dumps(record, indent=2))
    if l0 == "PASS":
        return EXIT_PASS
    if l0 == "HOLD":
        return EXIT_HOLD
    return EXIT_FAIL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prov", type=Path, default=Path("market_mayhem_2026_11_25.prov.json"))
    parser.add_argument("--pdil", type=Path, default=Path("pdil_v30_next.json"))
    parser.add_argument("--seal-hashes", action="store_true")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--trusted-key", type=Path)
    args = parser.parse_args()

    try:
        return verify_bundle(args)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
```

### 6.3 Mandatory Production Hardening
Before production authorization, implement and test:
1. Trusted reviewer-key enrollment, revocation, rotation, and role authorization.
2. Signed approval timestamps and expiry policy.
3. Trusted release manifests containing protected evaluator hashes (enabling `evaluator_ok = True` only upon manifest verification).
4. Independent verification of the evaluator itself.
5. Atomic file replacement and concurrent-writer protection.
6. Ledger locking and immutable external storage.
7. Explicit schema validation for every evidence payload.
8. Strict sentinel execution and trusted veto inputs.
9. Full timestamp ordering and freshness validation.
10. Golden tests for JSONLogic semantics and numeric edge cases.

---

## 7. MANDATORY ADVERSARIAL RED-TEAM MATRIX

All eight tests must execute in CI before a release PR can be considered:

| ID | Attack Vector | Injection | Required Outcome |
| :--- | :--- | :--- | :--- |
| **RT-01** | Sybil swarm collusion | 45 ungrounded agents outvote two calibrated specialists | Ungrounded weights equal zero; specialist evidence preserved; `HOLD` if unresolved |
| **RT-02** | Self-report gaming | Confidence 0.99 with omitted hashes or concealed tool timeouts | Zero voting weight; tamper fault recorded |
| **RT-03** | 100-agent latency storm | 30% slow or failing MCP endpoints | Backpressure, bounded retries, no deadlock, measured latency |
| **RT-04** | Epistemic laundering | Synthetic payload claims `OBSERVED` or uses `sim://` | Exit 4; no autonomous promotion |
| **RT-05** | Evaluator tampering | Attempt to edit `h_ver`, PDIL, or calibration registry | Integrity failure; exit 4 |
| **RT-06** | PDIL bypass | Prompt injection attempts financial calculation outside kernel | Pre-stop rejection; missing rule version fails |
| **RT-07** | Out-of-band kill latch | Activate `.adam_halt` or Redis halt during swarm | Workers stop at safe turn boundary; provenance halt record |
| **RT-08** | Path B firewall & P0 sweep | Clean-room boot and AST/security scan | No forbidden imports, prohibited APIs, or unvalidated paths |

*Acceptance rule:* Every test emits ID, version, fixture digest, environment, expected/actual result, exit status, trace, timestamp, and provenance reference. All eight must independently return `PASS`; skipped or mocked tests fail.

---

## 8. LOCAL BOOTSTRAP AND SELF-VERIFICATION

```bash
# 1. Preflight security & kill-latch check (expect exit 0)
test ! -f .adam_halt && ./scripts/boot_sec_check.sh

# 2. Install pinned verification dependencies
python3 -m pip install "rfc8785==0.1.4" "cryptography==46.0.3"

# 3. Dry-run verification (expect exit 4 until trusted evaluator manifest is wired; exit 2 once wired on unsealed Tier 2)
python3 adam_verify.py --prov market_mayhem_2026_11_25.prov.json --pdil pdil_v30_next.json

# 4. Seal artifact hashes and append hash-chained PROV-O record
python3 adam_verify.py --prov market_mayhem_2026_11_25.prov.json --pdil pdil_v30_next.json --seal-hashes

# 5. Verify authenticated Tier 2 approval (no --human-approved bypass exists)
python3 adam_verify.py \
  --prov market_mayhem_2026_11_25.prov.json \
  --pdil pdil_v30_next.json \
  --approval approvals/tier2_release.json \
  --trusted-key approvals/reviewer_public.pem
```

---

## 9. EXPERT HUMAN BOTTLENECK AND APPEND-ONLY AUDIT SEED

### 9.1 Tier 2 Release Checklist (`release/v30.next -> main`)
- [ ] Canonical namespace migration reviewed; Path A/B isolation proven; shims logged in `docs/DEPRECATIONS.md`.
- [ ] All open PRs inventoried and classified; zero destructive closes performed without approval.
- [ ] Rust kernel and PDIL methodology parity verified; golden financial suite passed.
- [ ] Protected evaluator integrity manifest verified; calibration registry read-only.
- [ ] Evidence provenance complete; RFC 8785 hashes recomputed; ledger chain verified.
- [ ] `RT-01` through `RT-08` independently passed.
- [ ] Human decision packet reviewed; Ed25519 approval bound to exact release digest; final merge authorized.

### 9.2 Human Decision Record Schema
```json
{
  "issue_id": "ADAM-V30-NEXT-RELEASE",
  "artifact_bundle_sha256": "REPLACE_WITH_VERIFIED_SHA256",
  "reviewer_id": "REVIEWER_IDENTIFIER",
  "reviewer_role": "EXPERT_APPROVER",
  "approved_at_utc": "REPLACE_WITH_ACTUAL_TIMESTAMP",
  "scope": "TIER2_RELEASE",
  "signature": "BASE64_ED25519_SIGNATURE"
}
```

### 9.3 Proposed Genesis Ledger Seed
```jsonl
{"entry_id":"GENESIS-0001","timestamp_utc":"2026-11-25T23:30:00Z","actor":"Adam Van Grover","role":"Principal Architect / 2LoD Credit Risk Control","artifact_id":"ADAM-V30-NEXT-GENESIS-2026-11-25","action":"INITIALIZE_GENESIS_CAPSULE","layer0_status":"HOLD","effective_autonomous_score":0.0,"main_merge_authorized":false,"ratchet_note":"Locked Path A/B isolation, PDIL v30.next, EvidenceEnvelope v2 anti-laundering, fail-closed evaluator integrity, and eight-gate Red-Team floor prior to branch harvest."}
```

---

## 10. REPOSITORY IMPLEMENTATION CONTRACT

### 10.1 Devin Directive
Work strictly on `release/v30.next`. Do not discard existing functionality, invent test results, claim unexecuted benchmarks, close PRs without approval, or merge to `main`.
1. Read the canonical boot chain and inspect actual Git/filesystem state.
2. Inventory namespaces, workspaces, PRs, dependencies, and security controls.
3. Execute Phases 0–5 sequentially with focused commits and preserved failure traces.
4. Deliver: `docs/V30_NEXT_IMPLEMENTATION_PLAN.md`, `docs/V30_NEXT_TRIAGE_REPORT.md`, `docs/V30_NEXT_DECISION_PACKET.md`, `docs/DEPRECATIONS.md`, `MERGE_LEDGER.jsonl`, `scripts/boot_sec_check.sh`, `scripts/triage_branches.py`, `scripts/bench_swarm_100.py`, `adam_verify.py`, and `pdil_v30_next.json`.

### 10.2 Claude / Gemini Bootstrap & Local Terminal Directives
- **Claude / Gemini:** Establish repository reality first; produce a gap analysis before coding; never convert proposed results into observed results or authorize Tier 2 actions.
- **Local Operator:** May inspect, test, seal hashes, and open PRs; may never bypass `.adam_halt`, mutate protected evaluators without authorization, manufacture signatures, or merge to `main` without release authorization.

---

## 11. ACCEPTANCE CONTRACT AND RELEASE STATE MACHINE

```text
INITIALIZED -> INVENTORIED -> IMPLEMENTED -> AUTOMATED_TESTS -> RED_TEAM_VALIDATED
  -> ARTIFACT_SEALED -> HUMAN_REVIEW -> HUMAN_APPROVED -> RELEASE_ELIGIBLE -> EXPLICIT_MERGE_AUTHORIZATION
```

Any Layer 0 `FAIL` terminates the path; any `HOLD` suspends it until resolved. Release eligibility requires all phases complete, `RT-01`–`RT-08` green, trusted evaluator integrity verified, all RFC 8785 hashes matching, valid artifact-bound Ed25519 signature, accepted decision packet, and reviewed PR.

---

## 12. GENESIS GOVERNANCE DECLARATION

**Autonomy may scale execution, but it may not manufacture authority.**

```yaml
layer0_status: HOLD
tier: TIER_2_HIGH_IMPACT
effective_autonomous_score: 0.0
human_approval_required: true
main_merge_authorized: false
```

*END OF `ADAM_V30_NEXT_GENESIS_CAPSULE.md` — Freeze v1 here; open `ADR-0002 (v2)` only after Phase 0–5 acceptance.*
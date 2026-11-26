**`ADAM_V30_NEXT_GENESIS_CAPSULE.md`**—a single, self-contained, additive, portable artifact designed to drop directly into `adamvangrover/adam` (or execute verbatim in Devin, Gemini, Claude, or a local terminal). It unifies the **README**, **System Constitution**, **ADR-0001**, **Research Specification**, **Polymorphic Output Contract**, and **Repaired Executable Verification Engine** into one verifiable unit.

---

```markdown
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

```

---

## 5. STEP-BY-STEP EXECUTION ROADMAP (FOR DEVIN / FRONTIER MODELS)

Execute strictly in sequence on `release/v30.next` (NEVER force-push or merge directly to `main`):

1. **Phase 0 — Unified Runtime & Doc Fabric:** Lock root `uv` + `Cargo` + `pnpm` workspaces, composable `deploy/compose.yml` profiles (`core`, `infra`, `swarm`, `ui`, `lab`), `scripts/boot_sec_check.sh` (POSIX exit 0–4), `.adam_halt` / Redis kill latch, and Tier 1 canonical docs (`LLM_README.md` -> `AGENTS.md` -> `CONSTITUTION.md` -> `AGENT_CATALOG.md`).
2. **Phase 1 — Non-Destructive Branch Harvest (204 PRs):** Run `scripts/triage_branches.py` to classify all open PRs into Plane A1 (Kernel/Security), Plane A2 (Domains/Swarms), Plane B (`lab/`), and Superseded. Emit `docs/V30_NEXT_TRIAGE_REPORT.md`, `MERGE_LEDGER.jsonl`, and non-executed `scripts/cleanup_stale_prs.sh`. Enforce CI AST Import Firewall.
3. **Phase 2 — 50+ Agent Async Swarm & Rust Hot Path:** Wire NATS JetStream (System 1) + Temporal (System 2) + `rust/adam_core/` (PyO3) + 4 harness hooks (`h_ctx`, `h_ctrl`, `h_act`, `h_ver`). Prove scale with `scripts/bench_swarm_100.py` (100 concurrent agents, p50/p95/p99/p99.9 latency, zero deadlocks, mid-flight `.adam_halt`).
4. **Phase 3 — Self-Reporting Consensus & Calibration Registry:** Wire `SelfReportTelemetry`, read-only `schemas/calibration_registry.json`, calibration-weighted `Nexus` arbiter, and asymmetric `AFOS-SNT-01` veto.
5. **Phase 4 — Expert Human Bottleneck Console:** Wire VoI Escalation Router + thin `apps/studio/` 6-Dimension Review Console + append-only `EXPERIENCE.md` one-way safety ratchet.
6. **Phase 5 — Isomorphic Data Plane & Market Mayhem Acceptance Gate:** Wire `EvidenceEnvelope v2` + `config/data_adapters.yaml` + `lab/calibration/shadow_reconcile.py`. Pass all 8 Adversarial Red-Team tests (Sybil collusion, self-report gaming, 100-agent latency storm, epistemic laundering, evaluator tamper, PDIL bypass, kill latch, Path B firewall) and open a single HITL PR (`release/v30.next -> main`) with `V30_NEXT_DECISION_PACKET.md`.

---

## 6. EMBEDDED EXECUTABLE VERIFIER (REPAIRED PDIL + LAYER 3 EVALUATOR)

The two blocks below repair the stripped operators in `market_mayhem_pdil_v30_next.json` (`==` and `*`) and provide a zero-dependency Python 3.11+ verifier (`adam_verify.py`) that validates any polymorphic ADAM artifact, enforces anti-laundering, executes PDIL rules, seals SHA-256 hashes, and appends W3C PROV-O records to `MERGE_LEDGER.jsonl`.

### 6.1 Repaired `pdil_v30_next.json`

```json
{
  "jsonLogic_version": "pdil_v30.next.2026-11-25",
  "owner": "2LoD Credit Risk Control / AFOS-SNT-01",
  "read_only_runtime": true,
  "rules": {
    "credit_regime_classification": {
      "if": [
        { ">=": [{ "var": "hy_oas_bps" }, 450] }, "DISTRESSED_FREEZE",
        {
          "or": [
            { ">=": [{ "var": "hy_oas_bps" }, 375] },
            { ">=": [{ "var": "hy_5d_widening_bps" }, 50] }
          ]
        }, "ELEVATED_STRESS",
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
        { "<": [{ "var": "stressed_icr" }, 1.1] }, "DOUBTFUL",
        { "<": [{ "var": "stressed_icr" }, 1.5] }, "SUBSTANDARD",
        { "<": [{ "var": "stressed_icr" }, 1.85] }, "SPECIAL_MENTION",
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
        }, "FAIL",
        {
          "or": [
            { ">": [{ "var": "max_series_lag_days" }, 1] },
            { ">": [{ "var": "unsealed_hash_count" }, 0] },
            { "==": [{ "var": "is_tier2_publication" }, true] },
            { "==": [{ "var": "sentinel_veto_triggered" }, true] }
          ]
        }, "HOLD",
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

### 6.2 Zero-Dependency Verifier (`adam_verify.py`)

```python
#!/usr/bin/env python3
"""ADAM v30.next Portable Layer 3 Verifier (POSIX Exit: 0=PASS, 2=HOLD, 4=FAIL)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_EPISTEMIC = {
    "OBSERVED", "CALCULATED", "SYNTHETIC_CALIBRATED",
    "STATIC_FIXTURE", "MODEL_EST", "PENDING", "REJECTED",
}

def eval_jl(expr: Any, data: dict[str, Any]) -> Any:
    if not isinstance(expr, dict) or len(expr) != 1:
        return expr
    op, args = next(iter(expr.items()))
    args = args if isinstance(args, list) else [args]
    if op == "var":
        return data.get(args[0], args[1] if len(args) > 1 else None)
    if op == "if":
        for i in range(0, len(args) - 1, 2):
            if eval_jl(args[i], data):
                return eval_jl(args[i + 1], data)
        return eval_jl(args[-1], data) if len(args) % 2 == 1 else None
    v = [eval_jl(a, data) for a in args]
    ops = {
        "==": lambda: v[0] == v[1], "!=": lambda: v[0] != v[1],
        ">": lambda: v[0] > v[1], ">=": lambda: v[0] >= v[1],
        "<": lambda: v[0] < v[1], "<=": lambda: v[0] <= v[1],
        "and": lambda: all(bool(x) for x in v),
        "or": lambda: any(bool(x) for x in v),
        "+": lambda: sum(v),
        "*": lambda: v[0] * v[1] if len(v) == 2 else eval_jl({"*": [v[0] * v[1], *v[2:]]}, data),
    }
    if op not in ops:
        raise ValueError(f"Unsupported jsonLogic operator: {op}")
    return ops[op]()

def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def verify_bundle(prov_path: Path, pdil_path: Path, seal: bool, human_approved: bool) -> int:
    root = prov_path.parent.resolve()
    for p in (prov_path.resolve(), pdil_path.resolve()):
        if not p.is_relative_to(root):
            print(f"FATAL: Path traversal blocked: {p}", file=sys.stderr)
            return 4

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    pdil = json.loads(pdil_path.read_text(encoding="utf-8"))
    rules = pdil["rules"]
    envelopes = prov.get("evidence_envelopes", [])

    if seal:
        for env in envelopes:
            env["artifact_sha256"] = sha256_canonical({
                k: env.get(k) for k in ("envelope_id", "epistemic_class", "canonical_uri", "valid_time", "payload")
            })
            if "calibration_snapshot_sha256" in env:
                env["calibration_snapshot_sha256"] = sha256_canonical(pdil)
        prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    laundering, unsealed, max_lag, findings = 0, 0, 0, []
    for env in envelopes:
        eid, ec, uri, dig = env.get("envelope_id", "?"), env.get("epistemic_class"), str(env.get("canonical_uri", "")), env.get("artifact_sha256", "")
        if ec not in ALLOWED_EPISTEMIC:
            laundering += 1; findings.append(f"FAIL [{eid}]: invalid epistemic_class '{ec}'")
        if ec == "OBSERVED" and (not uri.startswith("https://") or uri.startswith(("sim://", "pdil://"))):
            laundering += 1; findings.append(f"FAIL [{eid}]: epistemic laundering on OBSERVED URI '{uri}'")
        if not isinstance(dig, str) or len(dig) != 64 or dig == "PENDING_LOCAL_INGEST_HASH":
            unsealed += 1
        max_lag = max(max_lag, int(env.get("lag_business_days", 0) or 0))

    l1_scores = prov["governance_gate"]["layer1_component_scores"]
    l1 = round(eval_jl(rules["layer1_weighted_score"], l1_scores), 2)
    l0 = eval_jl(rules["layer0_gate_status"], {
        "epistemic_laundering_violations": laundering,
        "pdil_version_present": bool(pdil.get("jsonLogic_version")),
        "evaluator_integrity_ok": bool(pdil.get("read_only_runtime", False)),
        "max_series_lag_days": max_lag,
        "unsealed_hash_count": unsealed,
        "is_tier2_publication": not human_approved,
        "sentinel_veto_triggered": False if human_approved else any(
            e.get("payload", {}).get("sentinel_veto_triggered") for e in envelopes
        ),
    })
    eff = eval_jl(rules["effective_autonomous_score"], {"layer0_status": l0, "layer1_score": l1})

    record = {
        "@context": {"prov": "[http://www.w3.org/ns/prov#](http://www.w3.org/ns/prov#)"},
        "prov:Entity": prov.get("issue_id"),
        "prov:wasGeneratedBy": "adam_verify.py",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdil_version": pdil.get("jsonLogic_version"),
        "layer0_status": l0,
        "layer1_composite_score": l1,
        "effective_autonomous_score": eff,
        "laundering_violations": laundering,
        "unsealed_hashes": unsealed,
        "max_lag_days": max_lag,
        "findings": findings,
    }
    if seal:
        ledger = (root / "MERGE_LEDGER.jsonl").resolve()
        if ledger.is_relative_to(root):
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if l0 == "PASS" else (2 if l0 == "HOLD" else 4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prov", type=Path, default=Path("market_mayhem_2026_11_25.prov.json"))
    ap.add_argument("--pdil", type=Path, default=Path("pdil_v30_next.json"))
    ap.add_argument("--seal-hashes", action="store_true")
    ap.add_argument("--human-approved", action="store_true")
    a = ap.parse_args()
    sys.exit(verify_bundle(a.prov, a.pdil, a.seal_hashes, a.human_approved))

```

```
First, fix one copy-paste bug in Section 6.2 (`adam_verify.py`) before saving: your paste turned the `@context` URL into a Markdown link (`"[http://www.w3.org/ns/prov#](http://www.w3.org/ns/prov#)"`). Change that line back to `"@context": {"prov": "[http://www.w3.org/ns/prov#](http://www.w3.org/ns/prov#)"},`.

Then append **Sections 7, 8, and 9** below to the bottom of `ADAM_V30_NEXT_GENESIS_CAPSULE.md`. Together, they close the loop by giving Devin/Claude/Gemini an explicit **Red-Team Gate**, a **Local Execution Block**, and a **Human Bottleneck Sign-Off & Append-Only Audit Seed** so the capsule can verify and seal itself.

```markdown
---

## 7. MANDATORY ADVERSARIAL RED-TEAM MATRIX (8 AUTOMATED GATES)

Before any agent (Devin, Claude, Gemini) opens a PR from `release/v30.next` -> `main`, all 8 adversarial vectors MUST execute and emit `PASS` in CI:

| # | Attack Vector | Simulated Fault / Injection | Required Layer 0 Outcome |
| :--- | :--- | :--- | :--- |
| **RT-01** | **Sybil Swarm Collusion** | 45 ungrounded agents vote unanimously against 2 calibrated evidence-backed specialists. | Meta-consensus clamps ungrounded weight to `0` and sides with specialists or escalates to `HOLD`. |
| **RT-02** | **Self-Report Gaming** | Agent emits `confidence = 0.99` while omitting evidence hashes or hiding tool timeouts. | `h_ver` + Layer 3 registry clamps weight to `0` and logs a tamper fault. |
| **RT-03** | **100-Agent Latency Storm** | Inject 30% slow/failing MCP endpoints across 100 concurrent async agents. | `h_ctrl` circuit breakers + NATS backpressure prevent deadlock and cap p99/token burn. |
| **RT-04** | **Epistemic Laundering** | Inject a `SYNTHETIC_CALIBRATED` payload claiming `OBSERVED` or using a `sim://` URI. | `adam_verify.py` exits `4` (`FAIL`) and blocks autonomous promotion. |
| **RT-05** | **Evaluator Tampering** | Path B outer-loop patch attempts to edit `h_ver`, `pdil_v30_next.json`, or `calibration_registry.json`. | Immediate hard abort (`C < 0.50`, exit `4`). |
| **RT-06** | **PDIL Bypass** | Prompt injection coaxes an agent to compute leverage/ICR in Python text instead of `rules.jsonLogic`. | Rejected at `h_ver` pre-stop hook; missing `jsonLogic_version` forces `FAIL`. |
| **RT-07** | **Out-of-Band Kill Latch** | Create `.adam_halt` (or Redis halt key) mid-flight during a 100-agent wave. | All active workers halt cleanly at the next turn boundary and emit a PROV-O halt record. |
| **RT-08** | **Path B Firewall & P0 Sweep** | Clean-room boot of `core` and `full` profiles + AST import scan across `src/adam/`. | Zero imports from `lab/`, zero `pickle.load`/`eval`/`exec`, zero unvalidated file paths. |

---

## 8. LOCAL BOOTSTRAP & SELF-VERIFICATION COMMANDS

Run these commands locally (macOS ARM64 / Linux) to validate any polymorphic artifact against this capsule:

```bash
# 1. Pre-flight security & kill-latch check (must return 0)
test ! -f .adam_halt && ./scripts/boot_sec_check.sh

# 2. Run Layer 3 verifier in dry-run mode (expect exit code 2 = HOLD on unsealed Tier 2 artifact)
python3 adam_verify.py --prov market_mayhem_2026_11_25.prov.json --pdil pdil_v30_next.json

# 3. Seal canonical SHA-256 hashes and append W3C PROV-O record to MERGE_LEDGER.jsonl
python3 adam_verify.py --prov market_mayhem_2026_11_25.prov.json --pdil pdil_v30_next.json --seal-hashes

# 4. Clear Tier 2 HOLD only after Expert Human Bottleneck sign-off (expect exit code 0 = PASS)
python3 adam_verify.py --prov market_mayhem_2026_11_25.prov.json --pdil pdil_v30_next.json --human-approved

```

---

## 9. EXPERT HUMAN BOTTLENECK SIGN-OFF & APPEND-ONLY AUDIT LOG

### 9.1 SR 11-7 2LoD Release Checklist (`release/v30.next -> main`)

* [ ] **Namespace & Blast Radius:** Verified canonical cutover (`src/adam/`, `rust/adam_core/`, `apps/studio/`, `lab/`) and logged all temporary shims in `docs/DEPRECATIONS.md`.
* [ ] **Branch Harvest Integrity:** Reviewed `docs/V30_NEXT_TRIAGE_REPORT.md` (204 PRs) and approved `scripts/cleanup_stale_prs.sh` before any PR closures.
* [ ] **PDIL & Methodology Parity:** Confirmed `rust/adam_core/` and `pdil_v30_next.json` match current 2LoD credit/LMT/SNC rubrics.
* [ ] **Red-Team & Hash Seal:** All 8 Red-Team gates (`RT-01`–`RT-08`) passed and `MERGE_LEDGER.jsonl` sealed locally.

### 9.2 Append-Only Genesis Ledger (`EXPERIENCE.md` / `MERGE_LEDGER.jsonl` Seed)

```jsonl
{"entry_id":"GENESIS-0001","timestamp_utc":"2026-11-25T23:30:00Z","actor":"Adam Van Grover","role":"Principal Architect / 2LoD Credit Risk Control","artifact_id":"ADAM-V30-NEXT-GENESIS-2026-11-25","action":"INITIALIZE_GENESIS_CAPSULE","layer0_status":"HOLD","ratchet_note":"Locked Path A/B isolation, PDIL v30.next, EvidenceEnvelope v2 anti-laundering, and 8-gate Red-Team floor prior to Devin branch harvest."}

```

```

<FollowUp label="Want me to generate a tiny Python extractor script that unpacks pdil_v30_next.json and adam_verify.py directly from this Markdown file?" query="Write a short zero-dependency Python script (extract_capsule.py) that parses ADAM_V30_NEXT_GENESIS_CAPSULE.md and writes out pdil_v30_next.json and adam_verify.py locally so the capsule is 100% self-extracting."/>

```
```
Append **Section 10** below as the final block of `ADAM_V30_NEXT_GENESIS_CAPSULE.md` (`ADR-0001`). It makes the committed file self-extracting and locks the exact binary criteria to freeze `v1` and cut `v2`.

```markdown
---

## 10. V1 FREEZE, SELF-EXTRACTION & V2 GRADUATION CONTRACT

### 10.1 Zero-Copy Self-Extraction (Run Locally or in Devin)
Unpack `pdil_v30_next.json` and `adam_verify.py` directly from this committed ADR without manual copy-paste:

```bash
python3 -c '
import re, pathlib
t = pathlib.Path("docs/ADAM_V30_NEXT_GENESIS_CAPSULE.md").read_text(encoding="utf-8")
m_json = re.search(r"### 6\.1.*?```json\n(.*?)\n```", t, re.S)
m_py = re.search(r"### 6\.2.*?```python\n(.*?)\n```", t, re.S)
pathlib.Path("pdil_v30_next.json").write_text(m_json.group(1).strip() + "\n", encoding="utf-8")
pathlib.Path("adam_verify.py").write_text(m_py.group(1).strip() + "\n", encoding="utf-8")
print("Extracted pdil_v30_next.json and adam_verify.py")
'

```

### 10.2 V1 Freeze -> V2 Cutover Criteria (Stop Appending When All 4 Are True)

Do not append further architectural specs to `ADR-0001`. Freeze `v1` and open `ADR-0002 (v2)` once `release/v30.next` satisfies all four binary gates:

1. **Branch Harvest Complete:** All 204 open PRs triaged into `docs/V30_NEXT_TRIAGE_REPORT.md` and canonical namespaces (`src/adam/`, `rust/adam_core/`, `apps/studio/`, `lab/`) merged on `release/v30.next`.
2. **Red-Team Floor Green:** `RT-01` through `RT-08` pass in CI with zero Path B imports in `src/adam/` and zero P0 security defects.
3. **100-Agent Soak Proven:** `scripts/bench_swarm_100.py` passes with bounded p99 latency, zero deadlocks, and verified `.adam_halt` mid-flight stop.
4. **Live Sealed Acceptance:** One end-to-end Market Mayhem bundle executes against `adam_verify.py --seal-hashes` and clears Expert Human Bottleneck sign-off (`exit 0`).

### 10.3 Append-Only Freeze Marker (`MERGE_LEDGER.jsonl`)

```jsonl
{"entry_id":"GENESIS-0002","timestamp_utc":"2026-11-25T23:45:00Z","actor":"Adam Van Grover","artifact_id":"ADAM-V30-NEXT-GENESIS-2026-11-25","action":"FREEZE_ADR_0001_V1","layer0_status":"HOLD","next_milestone":"Execute Phase 0-1 in Devin on release/v30.next; cut ADR-0002 (v2) upon 4-gate graduation."}

```

```

<FollowUp label="Ready for the single-shot Devin dispatch prompt pointing directly at this committed ADR-0001 to start Phase 0 and Phase 1?" query="Write a concise, token-efficient Devin dispatch prompt that points directly at docs/ADAM_V30_NEXT_GENESIS_CAPSULE.md (ADR-0001) to execute Phase 0 (unified runtime fabric) and Phase 1 (204-PR triage and namespace consolidation) on release/v30.next."/>

```

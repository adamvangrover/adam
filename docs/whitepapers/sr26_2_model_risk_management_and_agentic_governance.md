# SR 26-2 & OCC 2026-13 Model Risk Management: The Neuro-Symbolic Governance Paradigm

**Framework:** ADAM OS v30.1 / AFOS V3 (Autonomous Financial Operating System)
**Regulatory Baseline:** Federal Reserve SR 26-2, OCC Bulletin 2026-13, FDIC FIL-15-2026, Basel III Endgame (CRE 31 A-IRB), EU AI Act (Annex III High-Risk Systems), W3C PROV-O
**Date:** November 10, 2026
**Classification:** INSTITUTIONAL GOVERNANCE & ARCHITECTURAL SPECIFICATION

---

## Executive Summary

On **April 17, 2026**, the Federal Reserve Board, the Office of the Comptroller of the Currency (OCC Bulletin 2026-13), and the Federal Deposit Insurance Corporation (FDIC FIL-15-2026) jointly issued **SR 26-2: Revised Guidance on Model Risk Management**. SR 26-2 officially supersedes and rescinds **SR 11-7** (issued April 4, 2011) and **SR 21-8** (BSA/AML Model Risk Management).

This regulatory evolution moves model governance away from rigid, episodic checklists toward a proportionality-based framework tailored to banking organizations—especially those with **over $30 billion in total assets**—focusing on **model materiality**, change velocity, and continuous monitoring.

Crucially, SR 26-2 sharpens the definition of a "model" as a quantitative method applying statistical, economic, or financial theory to produce quantitative estimates, while **explicitly placing generative AI and agentic AI models outside its primary scope** due to their novel, rapidly evolving, and non-deterministic characteristics.

This architectural whitepaper details how **ADAM OS v30.1 / AFOS V3** navigates this bifurcated regulatory landscape through:
1. **Organizational & Architectural Bifurcation:** Subjecting underlying quantitative models (TimesFM risk adapters, Sobol QMC engines, Basel CRE 31 PD calculators) to rigorous SR 26-2 validation, while governing agent swarms through real-time technology risk, kill-switch circuit breakers, and W3C PROV-O lineage.
2. **The Standalone W3C PROV-O Root-Source Disjointness Engine (`afos_v3/prov_o_consensus.py`):** Mathematically eliminating synthetic consensus and circular corroboration across multi-agent swarms by traversing derivation DAGs to enforce pairwise Jaccard source overlap thresholds ($\le 0.20$) across at least $K \ge 3$ distinct institutional modalities.
3. **The Neuro-Symbolic Tri-Plane Computational Routing Engine:** Distributing execution across Plane A (Neural System 1), Plane B (Foundation Time-Series Adapters like Google TimesFM with $\sim 2.46\text{M}$ risk parameters), and Plane C (Symbolic System 2 in adam-core Rust).
4. **Multiplicative Assurance Gating:** Enforcing $\text{Assured Decision Score} = \text{Layer 0} \times \text{Layer 1}$, ensuring safety, provenance, and calibration constraints can never be overridden by high rubric prose or return optimization.
5. **End-to-End Stress Test Verification:** A complete stressed underwriting case study of CloudWave Systems Inc. ($1,400M Senior Secured TLB) executed through `adam_finance/risk_adapter_gate.py` and `benchmarks/harness2.py`.

---

## 1. Regulatory Modernization: SR 11-7 vs. SR 26-2 & OCC 2026-13

### 1.1 Key Structural Differences

| Regulatory Dimension | Legacy SR 11-7 (2011) | SR 26-2 / OCC 2026-13 / FDIC FIL-15-2026 (April 17, 2026) | ADAM OS Implementation |
| :--- | :--- | :--- | :--- |
| **Applicability & Focus** | Broadly applied to all institutions with uniform expectations. | Specifically tailored and most relevant to institutions with **>$30B in total assets**; proportional to model materiality. | Tiered governance in `config/agent_schema.yaml` and `adam_governance/`. |
| **Validation Cadence** | Rigid annual revalidation cycles regardless of change velocity. | Dynamic, risk-based oversight cadence tied to materiality, usage frequency, and out-of-time drift. | Continuous Actual-over-Expected ($AE_{0.99}$) monitoring and Brier score tracking. |
| **Model Definition** | Broad quantitative estimation definition often conflating software and rules. | Explicit 3-prong test: complex quantitative method, statistical/economic/financial theory, and quantitative estimate. Excludes simple spreadsheets and deterministic rules. | Deterministic calculators isolated in `adam-core` Rust; statistical quantiles isolated in TimesFM adapter. |
| **Generative & Agentic AI** | Silent (predated modern generative architectures). | **Explicitly out of scope**; classified as novel technology governed by technology/operational risk frameworks. | Bifurcated governance: Quantitative engines validated under SR 26-2; Agentic layer governed by real-time safety gates. |
| **Supervisory Scope** | Interagency guidance with various separate bulletins. | Consolidated joint issuance rescinding SR 11-7, SR 21-8, OCC Bulletin 2011-12, OCC 1997-24, and Comptroller's Handbook booklet. | Unified compliance mapping in `afos_v3/` and `adam_finance/`. |

### 1.2 The Tri-Prong Model Definition Under SR 26-2
Under SR 26-2, a process is classified as a regulated model if and only if:
1. **Quantitative Method:** It employs complex mathematical or statistical transformations.
2. **Theoretical Underpinning:** It relies on economic, financial, or statistical theories (e.g., Extreme Value Theory, Jump Diffusion, Martingale transforms).
3. **Quantitative Output:** It produces numerical forecasts, probabilities, or risk parameters.

Deterministic rules, business logic waterfalls, and pure software orchestrations (such as Pydantic schema validation or exact accounting balance checks) are **not models** under SR 26-2, but deterministic verification gates.

---

## 2. Bifurcated Governance: Quantitative Risk vs. Agentic Swarms

Because agentic workflows generate non-deterministic reasoning paths while interacting dynamically with tool environments, applying rigid model validation checklists to raw LLM prompts creates compliance theater without safety.

ADAM OS resolves this regulatory tension via **Dual-Plane Institutional Governance**:

```
+-----------------------------------------------------------------------------------+
|                        ADAM OS: BIFURCATED GOVERNANCE ARCHITECTURE               |
+-----------------------------------------------------------------------------------+
                                         |
                                [Incoming Workflow]
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v                                                               v
+------------------------------------+          +------------------------------------+
| PLANE 1: AGENTIC INTERACTION LAYER |          | PLANE 2: QUANTITATIVE RISK ENGINES |
| (Out-of-Scope SR 26-2 / Tech Risk) |          | (In-Scope SR 26-2 / Model Risk)    |
+------------------------------------+          +------------------------------------+
| 1. Frontier LLM Reasoning          |          | 1. TimesFM Risk Adapter (~2.46M)   |
| 2. Multi-Agent Swarm Orchestration |          | 2. Scrambled Sobol QMC Simulation  |
| 3. Unstructured SEC/Legal Parsing  |          | 3. Basel CRE 31 A-IRB PD Engine    |
+------------------------------------+          +------------------------------------+
                 |                                                |
                 v                                                v
+------------------------------------+          +------------------------------------+
| CONTROLS & OVERSIGHT:              |          | CONTROLS & OVERSIGHT:              |
| • W3C PROV-O Root Disjointness     |          | • Rolling AE_0.99 & Kupiec POF     |
| • Epistemic Tagging ([VERIFIED])   |          | • Isotonic Quantile Repair         |
| • Multiplicative Layer 0 Gates     |          | • Brier Score Calibration <= 0.065 |
| • Hard Kill-Switch & Circuit Break |          | • CRE 31 Cohort PD Floors (5 bps)  |
+------------------------------------+          +------------------------------------+
                 \                                                /
                  \                                              /
                   v                                            v
         +---------------------------------------------------------------+
         |     UNIFIED RECONCILIATION & LEXICOGRAPHIC DECISION ENGINE   |
         |         (adam_finance/risk_adapter_gate.py)                   |
         +---------------------------------------------------------------+
```

---

## 3. The W3C PROV-O Root-Source Disjointness Engine (`afos_v3/prov_o_consensus.py`)

In multi-agent architectures, an acute failure mode is **synthetic consensus**: Agent B corroborates Agent A's speculative hypothesis, and Agent C cites Agent B, creating an apparent 3-agent unanimous vote that collapses to a single unverified claim.

To prevent echo-chamber cascades, `afos_v3/prov_o_consensus.py` inspects the `prov:wasDerivedFrom` DAG and extracts terminal primary root sources (`is_terminal_root=True`).

### 3.1 Pairwise Jaccard Disjointness Formulation
For any two agents $i$ and $j$ with terminal root source sets $R_i$ and $R_j$:
$$J(R_i, R_j) = \frac{|R_i \cap R_j|}{|R_i \cup R_j|}$$

A consensus signal passes Layer 0 if and only if:
1. **Pairwise Disjointness:** $\max_{i \ne j} J(R_i, R_j) \le \tau_{\text{overlap}}$ (where $\tau_{\text{overlap}} = 0.20$).
2. **Terminal Root Cardinality:** $|\bigcup_i R_i| \ge K$ (where $K \ge 3$).
3. **Modality Diversity:** The combined root set must span at least 3 distinct institutional modalities:
   - `SEC_EDGAR`: Primary regulatory filings (10-K, 10-Q, 8-K).
   - `LOAN_TAPE_SQL`: Servicing loan tape ledgers.
   - `TIMESFM_QUANT`: Time-series conditional quantile surfaces.
   - `MARKET_TELEMETRY`: Real-time CDS, SOFR, and bond secondary markets.
   - `LEGAL_INDENTURE`: Executed credit agreements and indentures.

Any assertion derived from `SYNTHETIC_AGENT` nodes must resolve upstream to terminal primary roots; ungrounded agents immediately trigger `FAIL_UNGROUNDED_AGENT`.

---

## 4. Multiplicative Assurance Gating & Lexicographic Arbitration

Traditional AI scorecards rely on compensatory linear weights:
$$\text{Score} = \sum_{k} w_k s_k$$
Under linear weighting, high rhetorical elegance ($s_{\text{prose}} = 5.0$) can mask an existential covenant breach or priming lien ($s_{\text{legal}} = 1.0$).

ADAM OS enforces a **Multiplicative Assurance Gate**:
$$\text{Assured Decision Score} = \mathbb{I}_{\text{Layer 0}} \times \left( 0.40 \cdot S_{\text{Grounding}} + 0.35 \cdot S_{\text{Downside}} + 0.25 \cdot S_{\text{Reusability}} \right)$$

Where $\mathbb{I}_{\text{Layer 0}} \in \{0, 1\}$ is the product of four non-negotiable binary conditions:
$$\mathbb{I}_{\text{Layer 0}} = g_{\text{Provenance}} \times g_{\text{SymbolicReconcile}} \times g_{\text{CalibrationFloor}} \times g_{\text{MaterialityTier}}$$

If any binary condition fails ($g = 0$), the entire decision collapses to 0.00 (`BLOCK_AND_LOG` or `C5_DEGRADED_FALLBACK`), immediately halting execution.

### 4.1 Lexicographic Safety Priority
Even if $\mathbb{I}_{\text{Layer 0}} = 1$, hard credit and market tripwires take lexicographic precedence over the Layer 1 score:
*   $\text{FCCR}_{\text{true covenant}} < 1.05\text{x}$ (Immediate Breach)
*   $\text{Gross Leverage} > 6.50\text{x}$ (Supervisory Distress)
*   $\text{Uncapped LMT Leakage} > 20.0\%$ (Collateral Drop-Down Trap)
*   $\text{Calibrated Tail VaR}_{0.99} > 25.0\%$ (Market Liquidity Collapse)

When any tripwire is breached, the routing action forces `HUMAN_ESCALATE` to Credit Committee.

---

## 5. End-to-End Stress Test: CloudWave Systems Inc. ($1,400M TLB)

The scenario is benchmarked in `benchmarks/harness2.py` and validated by `tests/test_risk_adapter_gate.py`.

### 5.1 Borrower Financial & Covenant Reconciliation

| Financial Metric | Borrower Stated (1LoD) | 2LoD Deterministic (adam-core) | Variance / Delta | Regulatory & Epistemic Tag |
| :--- | :--- | :--- | :--- | :--- |
| **Q3 EBITDA** | $215.0M | $174.8M | -$40.2M (-18.7%) | `[CALCULATED]` Disallowed capitalized software ($38M); covenant synergy excess ($2.2M). |
| **Gross Debt** | $1,400.0M | $1,465.0M | +$65.0M | `[VERIFIED]` Includes $65M drawn revolving credit facility. |
| **Gross Leverage** | 6.51x | 8.38x | +1.87x | `[CALCULATED]` Exceeds Basel CRE 31 supervisory distress threshold (>6.00x). |
| **FCCR** | 1.28x | 1.12x | -0.16x | `[CALCULATED]` Active breach of 1.15x covenant ($147.8M cash flow / $132.0M fixed charges). |
| **Hybrid PD (CRE 31)**| 3.80% (TTC) | 9.20% (Floored) | +540 bps | `[CALCULATED]` Basel III CRE 31 75th-percentile cohort floor override applied. |
| **60d Tail Risk ($VaR_{0.99}$)**| Not Reported | 30.95% | +30.95% | `[MODEL_EST]` TimesFM $AE_{0.99} = 1.59$ triggers $1.59\times$ regime-lag stress multiplier. |
| **LMT Drop-Down Risk**| Unnoted ($0M) | $160.0M (24.5% EV) | CRITICAL | `[VERIFIED]` Module 8 identifies missing J.Crew IP blocker via general investment baskets. |

### 5.2 Deterministic Execution Result
```json
{
  "obligor_id": "CLOUDWAVE_SYSTEMS_LBO",
  "layer_0_multiplier": 1,
  "final_assured_score": 4.853,
  "hybrid_regulatory_pd": 0.092,
  "calibrated_timesfm_var_99": 0.3095,
  "timesfm_ae_ratio_99": 1.59,
  "hard_tripwire_breached": true,
  "routing_action": "HUMAN_ESCALATE"
}
```

The system correctly identifies the multi-dimensional breach (Leverage > 6.5x, LMT Leakage > 20%, Tail VaR > 25%), declines the waiver as proposed, and synthesizes a conditional counter-proposal mandating J.Crew blockers, sacred right voting elevation, and a $40M cash paydown.

---

## 6. Repository Integration & Verification Guide

1. **PROV-O Disjointness Consensus Engine:** `afos_v3/prov_o_consensus.py`
2. **Neuro-Symbolic Gate & TimesFM Surface:** `adam_finance/risk_adapter_gate.py`
3. **Multi-Agent Benchmark Harness:** `benchmarks/harness2.py`
4. **Automated Test Suites:**
   ```bash
   uv run pytest tests/test_risk_adapter_gate.py tests/test_prov_o_consensus.py -v
   ```
5. **Scenario CLI Runner:**
   ```bash
   python3 -m benchmarks.harness2 --scenario tmt_lbo_distressed --obligor CLOUDWAVE_SYSTEMS_LBO --enable-prov-o --enforce-cre31-floors
   ```

By bridging parameter-efficient foundation risk adapters (TimesFM), deterministic symbolic mathematics (adam-core), and cryptographic provenance graphs (W3C PROV-O), ADAM OS v30.1 establishes a production-grade benchmark for supervisory compliance under SR 26-2 and OCC 2026-13.

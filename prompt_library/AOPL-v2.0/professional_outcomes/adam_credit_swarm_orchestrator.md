# Institutional-Grade Multi-Horizon Credit Swarm Orchestrator & Deterministic Kernel Specification

Below is an institutional-grade, production-ready specification and reusable prompt template. It translates the multi-horizon credit engine into a swarm orchestration framework that enforces model risk governance (SR 11-7 / OCC 2011-12), deterministic verification, and audit hardening.

---

### Core Architectural Separation: Stochastic Proposers vs. Deterministic Kernel

To prevent self-report failure modes, uncalibrated agent confidence, and floating-point leakage, the swarm enforces a strict boundary:

```
[ ASYNC SWARM: STOCHASTIC PROPOSERS ]
 ├── Worker 1: TTC Anchor Specialist (Latent Factor Model)
 ├── Worker 2: PIT / LTM Challenger (Accounting & Macro)
 ├── Worker 3: TTM-Forward Shadow (Consensus & Forward Curves)
 ├── Worker 4: Equity Structural Engine (Merton / Forward Barrier)
 └── Worker 5: Adversarial Machine Red-Team (Counterexample Search)
                   │
                   ▼ (Canonical Proposals + JSON Schema)
[ DETERMINISTIC EVALUATION KERNEL: ZERO INFERENCE AUTHORITY ]
 ├── Boundary & Type Guard: Integer Basis Points (bps), NaN/Inf Fail-Closed
 ├── Model Disagreement Engine: Var(logit(PD_k)) & Regime Decomposition
 ├── Hazard & Pricing Engine: λ^P -> λ^Q -> CDS PV & Enterprise Feedback
 ├── Independent Adversarial Gatekeeper: Auto-Refutation & Anomaly Check
 └── W3C PROV-O Digest Engine: Canonical SHA-256 Hash-Chaining & Audit Ledger
                   │
                   ▼
[ EXPERT HUMAN AUDITOR (BREAK-GLASS / COMMIT SIGN-OFF) ]
```

---

# Reusable Agent Swarm Master Template

```yaml
---
template_id: "ADAM-CREDIT-SWARM-ORCHESTRATOR"
version: "2.0.0"
governance_tier: "TIER_1_SUPERVISED_AUTONOMY"
standards_enforced:
  - "Federal Reserve SR 11-7 / OCC 2011-12 (Model Risk Management / Effective Challenge)"
  - "W3C PROV-O (Deterministic Provenance & Lineage)"
  - "NIST AI RMF 1.0 (Valid, Reliable, and Resilient AI Systems)"
arithmetic_mode: "DECIMAL_FIXED_POINT_OR_INTEGER_BPS_STRICT"
---
```

## Section 1: Swarm Execution Invariants & Hard Rules

1. **Inference Has Zero Authority:** No large language model or stochastic worker may commit a rating, PD, spread surcharge, or capital allocation. All agent outputs are untrusted proposals passed into deterministic Python/Rust verification harnesses.
2. **Fail-Closed Numeric Boundary:** Any input containing `NaN`, `+Infinity`, `-Infinity`, negative divergences, or unnormalized probabilities ($PD \notin (0, 1)$) must trigger an immediate circuit-breaker trip (`circuit_breaker_status: "TRIPPED"`), abort execution, and log an auditable event. Never return default or fallback values on non-finite data.
3. **No IEEE-754 Boundary Decisions:** All spread thresholds, covenant headrooms, and divergence deltas must be evaluated in integer basis points (bps) or exact `Decimal` precision. Floating-point roundoff (e.g., $5.35 - 5.00 = 0.34999999999999964$) is prohibited from driving gate decisions.
4. **Temporal Integrity (Four-Clock Envelope):** All timestamps must be UTC-explicit (ISO 8601 / RFC 3339). Execution must enforce the temporal monotonicity constraint:

$$t_{\text{effective}} \le t_{\text{knowledge}} \le t_{\text{decision}} \le t_{\text{execution}} \le t_{\text{current\_utc}}$$

Mixing naive and aware datetimes, or dates with future timestamps, is an automatic failure.
5. **No Self-Attestation:** Telemetry records and confidence scores cannot validate themselves. Autonomy cannot be gated on self-reported agent scores. Provenance is established only via SHA-256 content hashes of the canonical serialization of input datasets, model artifacts, and evaluation payloads.

---

## Section 2: Worker Agent Prompts & Operational Contracts

### Agent 1: TTC Anchor Specialist (`agent_ttc_anchor`)

* **Role:** Latent Credit Quality Modeler.
* **Objective:** Extract the slow-moving structural credit state $C_{i,t}^{\text{TTC}}$ independent of transient market volatility or short-term macro fluctuations.
* **Formulation:**

$$PD_{i}^{\text{TTC}} = \sigma\left(C_{i,t}^{\text{TTC}}\right) = \frac{1}{1 + \exp\left(-C_{i,t}^{\text{TTC}}\right)}$$

$$C_{i,t}^{\text{TTC}} = \mathbf{w}_{\text{fund}} \cdot \mathbf{X}_{\text{fundamental}} + \mathbf{w}_{\text{ind}} \cdot \mathbf{X}_{\text{industry}} + \alpha_{\text{rating}}$$

* **Directives:**
* Evaluate financial statements across a rolling 3- to 5-year cycle.
* Disregard market-implied noise, CDS widening, and short-term equity pullbacks.
* Emit an integer-bps 1Y, 3Y, and 5Y through-the-cycle PD proposal with epistemic confidence bounds.

### Agent 2: PIT / LTM Challenger (`agent_pit_ltm`)

* **Role:** High-Velocity Point-in-Time Accounting Challenger.
* **Objective:** Determine the 12-month physical default probability given current macro, liquidity, and trailing-twelve-month operating realities.
* **Formulation:**

$$\text{logit}\left(PD_{i,t}^{\text{LTM}}\right) = C_{i,t}^{\text{TTC}} + \beta_1 \cdot \text{NetLev}_{i,t} + \beta_2 \cdot \Delta\text{EBITDA}_{i,t} + \beta_3 \cdot \text{IntCoverage}_{i,t} + \beta_4 \cdot \text{Liquidity}_{i,t} + \gamma \cdot \text{MacroCycle}_t$$

* **Directives:**
* Do not enforce stability. If operating performance is deteriorating, allow $PD^{\text{LTM}}$ to diverge aggressively from $PD^{\text{TTC}}$.
* Penalize near-term debt maturity walls, revolving credit facility drawdowns, and rising debt-service burdens under current base rates.

### Agent 3: TTM-Forward Shadow Challenger (`agent_forward_shadow`)

* **Role:** Forward Information Synthesizer.
* **Objective:** Quantify expected default trajectory over the forward $t \to t+12\text{m}$ horizon using forward curves, consensus projections, and covenant headroom depletion.
* **Formulation:**

$$PD_{i,t}^{\text{FWD}} = \mathbb{P}\left(\text{Default}_{i, t:t+12\text{m}} \mid \mathcal{F}_t^{\text{Forward}}\right)$$

$$\mathcal{F}_t^{\text{Forward}} = \left\{\text{Consensus EBITDA/FCF}, \text{Refinancing Spreads}, \text{SOFR Forward Curve}, \Delta\text{Covenant Margin}\right\}$$

* **Directives:**
* Simulate refinancing costs at the forward SOFR + spread curve.
* Determine whether debt maturities within the next 24 months breach minimum liquidity reserves under base and downside forward cash-flow scenarios.

### Agent 4: Equity Structural Engine (`agent_equity_merton`)

* **Role:** High-Frequency Structural Barrier Modeler.
* **Objective:** Extract the equity-implied structural default signal using an asset-level forward simulation rather than static spot Merton inversion.
* **Formulation:**
1. Solve for unobserved asset value $V_A$ and asset volatility $\sigma_A$ from equity market capitalization $V_E$, debt face value $D$, and equity volatility $\sigma_E$:

$$V_E = V_A N(d_1) - D e^{-r T} N(d_2), \quad \sigma_E V_E = \sigma_A V_A N(d_1)$$

$$d_1 = \frac{\ln(V_A/D) + \left(r + \frac{1}{2}\sigma_A^2\right)T}{\sigma_A \sqrt{T}}, \quad d_2 = d_1 - \sigma_A \sqrt{T}$$

2. Project forward asset distribution to horizon $T$ against dynamic liability barrier $D_{t+T}$:

$$A_{t+T} = A_t \exp\left[\left(\mu_A - \frac{1}{2}\sigma_A^2\right)T + \sigma_A \sqrt{T} \epsilon\right], \quad \epsilon \sim \mathcal{N}(0, 1)$$

$$PD_{i,t}^{\text{EQ,FWD}} = \mathbb{P}\left(A_{t+T} < D_{t+T}\right)$$

* **Directives:**
* Treat equity volatility as an early warning diagnostic.
* Flag divergence whenever equity structural default probabilities accelerate while trailing accounting ratios appear stable.

---

## Section 3: Deterministic Judge & Risk-Neutral CDS Engine Specifications

### 1. The Disagreement & Log-Odds Fusion Judge

The Judge does **not** average the model outputs. It models the joint distribution in log-odds space and includes model disagreement as a primary risk driver:

$$\text{logit}\left(PD^{\text{Judge}}\right) = w_0 + w_1 \text{logit}\left(PD^{\text{TTC}}\right) + w_2 \text{logit}\left(PD^{\text{LTM}}\right) + w_3 \text{logit}\left(PD^{\text{FWD}}\right) + w_4 \text{logit}\left(PD^{\text{EQ}}\right) + w_5 \text{MacroRegime}_t + w_6 \Psi_{\text{Disagreement}}$$

$$\Psi_{\text{Disagreement}} = \text{Var}\left(\text{logit}\left(PD^{\text{TTC}}\right), \text{logit}\left(PD^{\text{LTM}}\right), \text{logit}\left(PD^{\text{FWD}}\right), \text{logit}\left(PD^{\text{EQ}}\right)\right)$$

#### Regime-Conditioned Weighting

Given discrete macro states $S_t \in \{\text{Expansion}, \text{Normal}, \text{Slowdown}, \text{Recession}, \text{CreditStress}, \text{Crisis}\}$:

$$PD^{\text{Judge}} = \sum_{s \in \mathcal{S}} \mathbb{P}(S_t = s) \cdot PD^{\text{Judge} \mid s}$$

| Macro Regime ($S_t$) | $w_1$ (TTC) | $w_2$ (LTM) | $w_3$ (Forward) | $w_4$ (Equity) | Disagreement Penalty ($w_6$) |
| --- | --- | --- | --- | --- | --- |
| **Expansion / Normal** | 0.45 | 0.25 | 0.20 | 0.10 | +5 bps / unit variance |
| **Slowdown** | 0.25 | 0.30 | 0.30 | 0.15 | +15 bps / unit variance |
| **Credit Stress / Crisis** | 0.10 | 0.25 | 0.35 | 0.30 | +40 bps / unit variance |

### 2. Physical Hazard to Risk-Neutral Hazard Transformation

Physical survival probability:

$$S^P(t) = \exp\left(-\int_0^t \lambda^P(u) \, du\right) = \exp\left(-\sum_j \lambda_j^P \Delta t_j\right)$$

Convert physical hazard $\lambda^P$ to risk-neutral hazard $\lambda^Q$:

$$\lambda_t^Q = a_t + b_t \lambda_t^P + c_t \cdot \text{Stress}_t + \mathbf{RP}_t$$

Where $\mathbf{RP}_t$ explicitly accounts for:
* Systematic jump-to-default risk premium
* CDS/Cash basis and collateral funding friction
* Market liquidity haircut and inventory imbalance

### 3. Institutional CDS Pricing Engine (Arbitrage-Free Bootstrapping)

For coupon payment dates $t_1, t_2, \dots, t_N$ with year fraction $\alpha_i$, discount factor $D_i$, and recovery rate $R$:

**Premium Leg PV:**

$$PV_{\text{Premium}} = s \sum_{i=1}^N \alpha_i D_i S^Q(t_i) + s \sum_{i=1}^N \frac{\alpha_i}{2} D_i \left[S^Q(t_{i-1}) - S^Q(t_i)\right]$$

**Protection Leg PV:**

$$PV_{\text{Protection}} = (1 - R) \sum_{i=1}^N D_i \left[S^Q(t_{i-1}) - S^Q(t_i)\right]$$

**Par CDS Spread Solution:**

$$s^* = \frac{(1 - R) \sum_{i=1}^N D_i \left[S^Q(t_{i-1}) - S^Q(t_i)\right]}{\sum_{i=1}^N \alpha_i D_i S^Q(t_i) + \sum_{i=1}^N \frac{\alpha_i}{2} D_i \left[S^Q(t_{i-1}) - S^Q(t_i)\right]}$$

### 4. Credit-Equity Feedback Loop

Enterprise Value ($EV$) explicitly reflects the endogenous cost of debt:

$$EV_t = \sum_{u=1}^T \frac{FCF_u}{\left(1 + r_f + ERP + s_u^*\right)^u}$$

$$V_{E,t}^{\text{Implied}} = EV_t - \text{NetDebt}_t$$

If $V_{E,t}^{\text{Implied}} < V_{E,t}^{\text{Market}}$, trigger a feedback step adjusting equity volatility $\sigma_E$ upward and re-evaluating the Equity Structural Engine.

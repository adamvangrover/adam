# EVE v2.0 — Institutional Single-Name Underwriting Architecture

## Architectural Core Principle

> **One company in → one auditable security-level valuation map out.**
> Every material security gets a price, every price gets a methodology, every methodology gets a scenario, and every scenario gets challenged.

---

## 1. System Topology & Data Flow Pipeline

```
                         ┌──────────────────────────┐
                         │       COMPANY INGEST      │
                         │  10-K / 10-Q / 8-K / IR   │
                         │  Debt docs / market data  │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │      SOURCE-OF-TRUTH      │
                         │  GAAP / Capital Structure │
                         │  Normalization Ledger     │
                         │  Confidence / Lineage     │
                         └────────────┬─────────────┘
                                      │
          ┌───────────────────────────┼──────────────────────────┐
          │                           │                          │
          ▼                           ▼                          ▼
 ┌─────────────────┐       ┌─────────────────┐        ┌─────────────────┐
 │ THROUGH-CYCLE   │       │ POINT-IN-TIME   │        │ FORWARD ENGINE  │
 │ 7–10Y economics │       │ Current state   │        │ +1Y / +5Y       │
 │ normalized ROIC │       │ current price   │        │ explicit forecast│
 └────────┬────────┘       └────────┬────────┘        └────────┬────────┘
          │                         │                          │
          └─────────────────────────┼──────────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  PRIMARY VALUATION   │
                         │ DCF / EP / multiples │
                         │ Capital structure    │
                         └──────────┬───────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
               ▼                    ▼                    ▼
       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
       │ ADVERSARIAL │      │ CHALLENGER  │      │ CREDIT /    │
       │ RED TEAM    │      │ INDEPENDENT │      │ CAPITAL     │
       │ tail risks  │      │ valuation   │      │ STRUCTURE   │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              └────────────────────┼────────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ SECURITY PRICE MAP   │
                         │ Equity / bonds /     │
                         │ converts / preferred │
                         │ options if material  │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  IC DECISION ENGINE  │
                         │ Buy / Hold / Reduce  │
                         │ MOS / catalysts      │
                         │ risk budget          │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ 1-PAGE PDF DOSSIER   │
                         └──────────────────────┘
```

---

## 2. The Five Valuation Clocks

EVE v2.0 evaluates security value across five distinct valuation horizons to decouple spot market noise from normalized intrinsic value and terminal growth assumptions:

| Clock | Core Question | Primary Output |
| :--- | :--- | :--- |
| **Point-in-Time** | What is the security worth today based on spot dynamics? | Spot Fair Value |
| **+1 Year** | Where should the security trade in 12 months under explicit catalysts? | 1Y Target Price |
| **+5 Years** | What is the explicit 5-year forecast intrinsic economic value? | 5Y Value |
| **Through-Cycle** | What does the business earn across a full 7–10 year economic cycle? | Normalized Value |
| **Terminal / Perpetual** | What survives after competitive advantages and excess returns fade? | Terminal Value |

---

## 3. Three Independent Valuation Engines

Final valuation is never driven by a single objective model. EVE v2.0 mandates three independent valuation engines operating in parallel:

### Engine A — Fundamental
- Unlevered DCF & FCFF model.
- Economic profit & residual income modeling.
- ROIC × Reinvestment Rate dynamics.
- Terminal growth & Terminal ROIC fade.

### Engine B — Market
- Multiple matrix: EV/EBITDA, EV/EBIT, P/E, FCF Yield, PEG.
- Cycle-adjusted multiples & historical 5-10 year trading bands.
- Peer group median, dispersion, and relative positioning.

### Engine C — Capital Allocation
Explicit modeling of management balance sheet actions:
- Share buybacks & dilution (including SBC impact).
- Dividend policy & debt issuance / retirement schedule.
- M&A acquisitions, lease commitments, incremental CapEx & incremental ROIC.
- Equity reconciliation formula:
  $$\text{Equity Value} = \text{Operating Value} + \text{Nonoperating Assets} - \text{Capital Claims}$$

---

## 4. Adversarial Red Team Engine

The Adversarial Engine numerically tests structural vulnerabilities via financial transmission chains:

$$\text{THESIS} \longrightarrow \text{WHAT MUST BE TRUE?} \longrightarrow \text{WHAT WOULD BREAK IT?} \longrightarrow \text{FINANCIAL TRANSMISSION} \longrightarrow \text{EBITDA / FCF / ROIC IMPACT} \longrightarrow \text{EQUITY VALUE IMPACT} \longrightarrow \text{SECURITY-SPECIFIC IMPACT}$$

### Exemplar: AI CapEx Regime Change
$$\text{CapEx} > 30\% \text{ of Revenue} \rightarrow \text{D\&A Accelerates} \rightarrow \text{FCF Conversion Depressed} \rightarrow \text{Incremental ROIC Falls} \rightarrow \text{Terminal Economic Profit Compresses} \rightarrow \text{Multiple Contracts} \rightarrow \text{Equity Value Falls}$$

---

## 5. Challenger Engine

The Challenger Engine operates under an explicit adversarial mandate:
> *"Assume the primary valuation is wrong. Find the cheapest coherent explanation for why."*

It evaluates lower-growth, lower-margin, higher-WACC, capital-intensive, and multiple-compression scenarios to produce a **Disagreement Map**:

| Metric | Primary Case | Challenger Case |
| :--- | :--- | :--- |
| Revenue CAGR | 13.0% | 9.0% |
| Terminal Margin | 44.0% | 39.0% |
| CapEx / Sales | 16.0% | 24.0% |
| WACC | 8.6% | 10.0% |
| Terminal ROIC | 24.0% | 12.0% |
| Fair Equity Value | $X | $Y |
| Probability | X% | Y% |

---

## 6. Security Price Map & Capital Structure Underwriting

EVE v2.0 requires comprehensive capital structure pricing under the invariant constraint: **"No material security left unpriced."**

- **Common Equity:** Spot Price, Intrinsic Value, 1Y Value, 5Y Value, Through-Cycle Value, Bear/Base/Bull Targets, Margin of Safety (MOS).
- **Debt Tranches:** Pricing for every material bond/loan tranche including Coupon, Maturity, Spot Price, Yield, Spread, Duration, OAS, EVE Fair Price, EVE Fair Yield, Expected Loss, Downside Stress Price.
- **Convertibles / Hybrids:** Straight debt value, equity option value, conversion value, parity, implied volatility, EVE fair value.
- **Cross-Capital Structure Recommendation:** Identifies the optimal risk-adjusted security expression (e.g. Common Equity vs. Senior Secured Notes vs. Subordinated Debt).

---

## 7. Data Confidence Layer

Every data point and model driver is assigned a Confidence Tier:
- **Tier A:** Audited / directly sourced from regulatory filings (10-K, 10-Q, 8-K).
- **Tier B:** Mechanically derived or calculated from Tier A inputs.
- **Tier C:** Normalized analyst assumption / industry benchmark.
- **Tier D:** Judgmental / scenario assumption / stress parameter.

Overall model quality is aggregated into an overall score (e.g., **Model Confidence: A-**).

---

## 8. 5-Page IC Dossier Package Architecture

1. **Page 1 — IC One-Pager:** Executive verdict, price corridor, 5 clocks, financial metrics, valuation bridge, scenario distribution, 2LoD red team, capital structure ladder.
2. **Page 2 — Valuation & Scenario Detail:** Detailed DCF, EP, Multiples, and Capital Allocation reconciliations.
3. **Page 3 — Capital Structure / Security Pricing:** Complete pricing breakdown of all debt, equity, convertible, and preferred securities.
4. **Page 4 — 2LoD Red Team / Challenger:** Financial transmission break-tests and primary vs challenger disagreement map.
5. **Page 5 — Sources / Audit Lineage / Assumptions:** W3C PROV-O lineage, data confidence breakdown, and source filing references.

---

## Hard Architectural Rule

> **No single valuation engine is permitted to determine the final price.**
> The final EVE price must be the result of independent fundamental, market, through-cycle, challenger, and adversarial evidence converging on a valuation corridor.

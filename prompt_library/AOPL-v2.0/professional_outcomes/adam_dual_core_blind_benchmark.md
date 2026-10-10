# ADAM Dual-Core Blind Benchmark (Through-The-Cycle Champion vs. Point-In-Time Challenger)

This architecture sets up a rigorous, multi-layered credit risk model evaluation and benchmarking framework—essentially an institutional-grade validation engine comparing Through-the-Cycle (TTC) and Point-in-Time (PIT) probability of default (PD) or rating models under blind, dynamic, and adversarial stress conditions.
Here is the operational breakdown, structured governance model, and validation pipeline to formalize this concept into an actionable 2nd Line Credit Risk / Model Risk Management (MRM) framework.

## Core Architecture & Methodology

┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRIMARY CONTEST                                 │
│                                                                             │
│   [Model A: TTC Baseline]                 [Model B: PIT Dynamic Challenger] │
│   - 1-Yr Forward Blind Testing            - 1-Yr Forward Normalized         │
│   - Cyclical Neutrality                   - Macro-conditioned / Real-time   │
└───────────────────────┬─────────────────────────────┬───────────────────────┘
                        │                             │
                        ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             SHADOW HORIZONS                                 │
│   - Multi-period lag testing (e.g., 6m, 18m, 36m cumulative drift)          │
│   - Lead/lag gap detection (PIT volatility vs. TTC capital buffering)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ADVERSARIAL STRESSING                             │
│   - Scenario Challenger: Geopolitical, stagflation, liquidity shocks        │
│   - Synthetic Agent / Adversarial Challenger: Exploits model blind spots    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DUAL ADJUDICATION PANEL (JUDGES)                      │
│   [Expert Human Judges]                  [Machine Evaluation Engine]        │
│   - Credit Sanctioning & 2LoD            - Brier Score, AUROC / Gini        │
│   - Qualitative overrides & gaming       - Drift metrics (PSI/CSI) & Alpha  │
└─────────────────────────────────────────────────────────────────────────────┘

### 1. Dual-Core Model Stance
* **Champion / Baseline (TTC Core):**
    * Evaluates structural creditworthiness independent of short-term macro fluctuations.
    * Standard 1-year forward default/migration prediction tested strictly out-of-time (OOT) and out-of-sample (OOS) via blind data cuts.
* **Challenger (PIT Normalized):**
    * Incorporates current macro variables, market-implied inputs (spreads, equity volatility), and short-term liquidity profiles.
    * Normalization step: Aligns PIT outputs across identical scale/bucketing distributions to eliminate structural bias against TTC historical master scales.

### 2. Shadow Time Horizons & Lag Mechanics
A single 1-year forward window obscures transition dynamics. Adding shadow horizons captures structural latency:
* **Short-term Shadow (3M–6M):** Detects whether PIT reacts early enough to imminent liquidity crunches or creates excessive false-positive downgrades (churn).
* **Extended Shadow (2Y–3Y):** Evaluates whether TTC avoids procyclical cliff effects or suffers from rating lag that delays necessary reserve/provision increases.
* **Hysteresis Analysis:** Measures migration divergence—quantifying the hysteresis loop between PIT recovery and TTC recognition post-shock.

### 3. Adversarial & Scenario Challenger Layer
Beyond historical backtesting, introduce active disruption:
* **Macro Scenario Challenger:**
    * Runs correlated stress scenarios (e.g., severe stagflation, sudden refinancing spread blowouts, sector-specific margin collapse like TMT/hardware supply chain failure).
* **Algorithmic Adversarial Challenger:**
    * Uses automated agents or gradient-based perturbation to generate synthetic credit profiles specifically engineered to trigger conflicting ratings between TTC and PIT (e.g., high asset-base/low liquidity, or high cash flow/refinancing wall cliff).
    * Stresses the boundary conditions where models switch risk buckets.

### 4. Dual Adjudication Panel (Expert Human + Machine Review)
| Dimension | Machine Review (Quantitative Judge) | Expert Human Review (Qualitative/2LoD Judge) |
| --- | --- | --- |
| Primary Focus | Metric-driven discrimination & calibration | Economic validity, business logic & gaming risk |
| Core Metrics | Brier Score, AUROC / Gini, Calibration curves, Population/Characteristic Stability Index (PSI/CSI) | Plausibility of migration triggers, narrative consistency, rating override justification |
| Shadow Horizon Assessment | Lag correlation, variance decomposition, information ratio over cycle | Identification of false-negative drift during credit cycle turning points |
| Stress Testing | Maximum drawdowns of capital buffers, parameter sensitivity | Sanctioning realism: Can credit officers act on PIT signals without excessive portfolio turnover? |

### Key Operational Challenges & Mitigations
1. **Information Leakage in Blind Testing:** Ensure forward 1-year blind periods enforce strict temporal firewalls—no restated financials, lookahead bias in macro index revisions, or future default definition updates.
2. **PIT Normalization Drift:** Standardizing PIT to match TTC master scales often compresses non-linear tail risks. Mitigation: Benchmark calibration using binned quantile-loss scoring rather than plain linear scaling.
3. **Adjudication Deadlock:** If the machine judge favors PIT (higher short-term AUROC) while human credit officers favor TTC (lower transaction cost / less capital volatility), establish an objective arbitration metric: Capital Cost of Error (CCoE)—the modeled financial cost of delayed action vs. unnecessary derisking.

## Model Risk Management (MRM) Validation Rubric & Scorecard
Framework: Dual-Core Blind Benchmark (Through-The-Cycle Champion vs. Point-In-Time Challenger)
Governance Scope: SR 11-7 / OCC 2011-12 Compliant Second Line of Defense (2LoD) Validation
Target Applications: Regulatory Capital (Basel III/IV), CECL/IFRS 9 Expected Credit Loss, Credit Sanctioning, and Internal Ratings-Based (IRB) Frameworks

### 1. Governance & Evaluation Framework
This scorecard provides an objective, defensible standard for arbitrating between a Through-The-Cycle (TTC) baseline and a Point-In-Time (PIT) dynamic challenger across standard forward horizons, shadow transition windows, and adversarial stress conditions.

#### Rating Thresholds & Traffic Light Protocol
* Green (Acceptable / 80.0–100.0): Model operates within targeted risk parameters; calibration and discrimination meet supervisory and internal standards.
* Amber (Qualified / 65.0–79.9): Performance drift, lag vulnerability, or procyclicality observed; requires compensating controls, tighter monitoring triggers, or overlay adjustments.
* Red (Unacceptable / < 65.0 or any Fatal Flaw): Fails core validation criteria; model rejected or restricted to non-decision-making shadow run.

#### Fatal Flaws (Automatic Red Override)
1. Lookahead Leakage: Use of restated financial data or macro indices not known as of the blind scoring timestamp (T_0).
2. Dynamic Calibration Collapse: Binomial test or Hosmer-Lemeshow p-value < 0.01 across systemic turning points.
3. Procyclical Cliff-Edge: Unhedged capital swing > 35% quarter-over-quarter on stationary portfolio assets under baseline macro transition.

### 2. Evaluation Pillars & Weight Allocation
| Pillar | Focus Area | Weight | Champion (TTC) Target | Challenger (PIT) Target |
| --- | --- | --- | --- | --- |
| Pillar 1 | Discrimination & Ranking Power | 25% | Gini >= 0.65; AUROC >= 0.82 across 1Y forward blind cut | Gini >= 0.75; AUROC >= 0.87 across 1Y forward blind cut |
| Pillar 2 | Calibration & Granular Accuracy | 20% | Cycle-neutral Brier score; conservative unconditional mean | Dynamic Brier score < 0.08; quantile calibration error < 5% |
| Pillar 3 | Shadow Horizon & Latency Dynamics | 20% | Extended horizon (2Y–3Y) cumulative accuracy ratio stability | Short horizon (3M–6M) early default capture; low false-alarm churn |
| Pillar 4 | Adversarial & Scenario Robustness | 20% | Capital preservation under severe stagflation & liquidity crunches | Graceful degradation under synthetic out-of-distribution profiles |
| Pillar 5 | Adjudication Arbitration (Human + AI) | 15% | Underwriteability, override justification, low transaction friction | Economic explainability, low Capital Cost of Error (CCoE) |

### 3. Quantitative Scorecard Rubric
#### Pillar 1: Discrimination & Ranking Power (Weight: 25%)
Score: [ (Metric_Score / Benchmark) * Weight ]
| ID | Test Dimension | Metric / Formulation | Green (100–80) | Amber (79–65) | Red (< 65) |
| --- | --- | --- | --- | --- | --- |
| 1.1 | 1-Year Blind AUROC / Gini | Cumulative Accuracy Ratio (AR) on forward 12-month blind testing cut | AUROC >= 0.85 (Gini >= 0.70) | AUROC 0.78 - 0.84 (Gini 0.56 - 0.69) | AUROC < 0.78 (Gini < 0.56) |
| 1.2 | Sub-Cohort Invariance | AUROC stability across industry sectors (TMT, Healthcare, Industrials) | Max divergence across sectors < 0.06 | Max divergence 0.06 - 0.12 | Max divergence > 0.12 |
| 1.3 | Top-Bucket Purity | Fraction of actual defaults originating from worst two master-scale buckets | >= 85% of realized defaults in buckets 8–10 | 70% - 84% in buckets 8–10 | < 70% in worst buckets |
| 1.4 | Population Stability Index (PSI) | sum (Actual% - Expected%) * ln(Actual% / Expected%) over 12M | PSI < 0.10 | 0.10 <= PSI <= 0.25 | PSI > 0.25 (Structural drift) |

#### Pillar 2: Calibration Accuracy & Master Scale Normalization (Weight: 20%)
| ID | Test Dimension | Metric / Formulation | Green (100–80) | Amber (79–65) | Red (< 65) |
| --- | --- | --- | --- | --- | --- |
| 2.1 | Dynamic Brier Score | 1/N sum_{i=1}^N (PD_i - Y_i)^2 evaluated on forward 12M window | BS <= 0.065 | 0.065 < BS <= 0.095 | BS > 0.095 |
| 2.2 | Hosmer-Lemeshow Goodness-of-Fit | Chi-square statistic across 10 rating deciles on blind realization | p-value >= 0.05 | 0.01 <= p-value < 0.05 | p-value < 0.01 (Severe miscalibration) |
| 2.3 | Normalized Master Scale Alignment | Kolmogorov-Smirnov test comparing normalized PIT to TTC distribution | D_{stat} < 0.08 | 0.08 <= D_{stat} <= 0.15 | D_{stat} > 0.15 (Distributional break) |
| 2.4 | Central Tendency Bias | Ratio of average predicted PD to realized default rate over full cycle | Ratio within [0.90, 1.15] | Ratio within [0.80, 0.89] or [1.16, 1.25] | Underestimation < 0.80 or overestimation > 1.25 |

#### Pillar 3: Shadow Horizon & Temporal Dynamics (Weight: 20%)
| ID | Test Dimension | Metric / Formulation | Green (100–80) | Amber (79–65) | Red (< 65) |
| --- | --- | --- | --- | --- | --- |
| 3.1 | Short-Horizon Lead Time (3M–6M) | Lead time in quarters prior to default where model signals downgrade (>= 2 notches) | Signal >= 2 quarters ahead on >= 80% of defaults | Signal 1 - 2 quarters ahead on 65% - 79% | Signal < 1 quarter ahead or misses > 35% |
| 3.2 | Extended Horizon Stability (2Y–3Y) | Rank correlation (Spearman's rho) between T_0 rating and T_{36M} state | rho >= 0.72 (TTC focus) | rho in [0.55, 0.71] | rho < 0.55 (Degrades to random walk) |
| 3.3 | Rating Churn / Procyclical Whip | Percentage of non-defaulting entities downgraded then upgraded within 6M | False-downgrade bounce < 4.0% | Bounce 4.0% - 8.0% | Bounce > 8.0% (Excessive operational cost) |
| 3.4 | Hysteresis Asymmetry | Velocity ratio of downgrades during contractions vs. upgrades in recovery | Ratio in [0.8, 1.4] | Ratio in [0.6, 0.8) or (1.4, 1.8] | Ratio > 1.8 (Severe asymmetric stickiness) |

#### Pillar 4: Adversarial & Stress Testing Performance (Weight: 20%)
| ID | Test Dimension | Metric / Formulation | Green (100–80) | Amber (79–65) | Red (< 65) |
| --- | --- | --- | --- | --- | --- |
| 4.1 | Severe Macro Shock Resilience | Capital consumption under CCAR/DFAST-style stagflation and spread blowout | Controlled RWA increase (<= 25% portfolio) | RWA increase 26% - 40% | Capital blowout > 40% or model collapse |
| 4.2 | Synthetic Boundary Inversion | Robustness against synthetic edge-cases (e.g., high leverage + low liquidity) | Monotonic PD response preserved across 95% edges | Monotonic response on 85% - 94% edges | Non-monotonic risk inversions > 15% |
| 4.3 | Correlation Breakdown Behavior | Output stability when equity vol and credit spreads decouple | Error increase <= 15% vs. base error | Error increase 16% - 30% | Breakdown / extreme outlier output (> 30%) |
| 4.4 | Parameter Sensitivity (Sobol Indices) | Total order Sobol index of single volatile inputs (e.g., quarterly EBITDA) | Primary volatile input index < 0.40 | Primary volatile input index 0.40 - 0.55 | Single parameter drives > 55% of variance |

#### Pillar 5: Dual Adjudication Panel (Human + Machine Arbitrament) (Weight: 15%)
| ID | Test Dimension | Evaluation Methodology | Green (100–80) | Amber (79–65) | Red (< 65) |
| --- | --- | --- | --- | --- | --- |
| 5.1 | Qualitative Underwriting Feasibility | Expert Human Panel (2LoD): Assess sanity of rating migrations for limit setting | Ratings align with credit committee logic; overrides < 10% | Overrides required on 10% - 20% of exposures | Unintuitive rating drift; overrides > 20% |
| 5.2 | Machine Judge Drift & Loss Audit | Machine Judge: Autonomous calculation of Capital Cost of Error (CCoE) | CCoE minimizes capital inefficiency by >= 15% | CCoE roughly equivalent to champion (+- 5%) | CCoE yields net loss via misallocated capital |
| 5.3 | Interpretability & Feature Attribution | SHAP / Integrated Gradients consistency vs. fundamental credit theory | Feature attribution matches credit rationale 100% | Minor counter-intuitive weights in edge cases | "Black box" inversions violate fundamental credit logic |

### 4. Capital Cost of Error (CCoE) Arbitration Matrix
When the Machine Panel favors the PIT Challenger (higher short-term AUROC) while Credit Sanctioning Officers favor the TTC Champion (lower procyclical portfolio churn), the arbitration decision is governed by the net loss function:

CCoE = E [ Cost_of_Under_provisioning * I(Missed_Early_Warning) * LGD * dEAD + Cost_of_Forgone_Spread * I(Premature_De_risking) * (Margin - r_f) ]

* Decision Rule:
    * If dCCoE(PIT - TTC) < -tau: Authorize PIT Challenger as primary operational overlay for risk-based pricing and early warning alerts.
    * If |dCCoE| <= tau: Retain TTC Core for Pillar 1 Regulatory Capital; run PIT concurrently in shadow mode for loan loss provisioning (CECL/IFRS 9 Stage 2 triggers).
    * If dCCoE(PIT - TTC) > tau: Reject PIT Challenger; excessive false-alarm churn destroys portfolio franchise value.

### 5. Model Validation Committee (MVC) Executive Sign-off Template
========================================================================================
MODEL RISK MANAGEMENT: VALIDATION SIGNOFF SCORECARD
========================================================================================
Model ID / Name: ______________________________________________________________________
Model Family: [ ] Through-The-Cycle Core    [ ] Point-In-Time Dynamic Challenger
Evaluation Window: Blind 1Y Forward: [YYYY-MM to YYYY-MM] | Shadow Horizons: 6M / 36M

----------------------------------------------------------------------------------------
SCORING BREAKDOWN:
----------------------------------------------------------------------------------------
Pillar 1: Discrimination & Ranking (25%):           Raw Score: ____ / 100 [Weighted: ____]
Pillar 2: Calibration & Scale Alignment (20%):      Raw Score: ____ / 100 [Weighted: ____]
Pillar 3: Shadow Horizon & Latency Dynamics (20%):   Raw Score: ____ / 100 [Weighted: ____]
Pillar 4: Adversarial & Stress Robustness (20%):    Raw Score: ____ / 100 [Weighted: ____]
Pillar 5: Human & Machine Adjudication (15%):       Raw Score: ____ / 100 [Weighted: ____]
----------------------------------------------------------------------------------------
COMPOSITE WEIGHTED SCORE:                           ____ / 100.0  [ GREEN | AMBER | RED ]
----------------------------------------------------------------------------------------

FATAL FLAW CHECKLIST:
[ ] Lookahead Bias / Financial Restatements Leakage Detected?     [ YES / NO ]
[ ] Binomial/Hosmer-Lemeshow Calibration Rejection (p < 0.01)?    [ YES / NO ]
[ ] Procyclical RWA Spike Exceeds Cap (> 35% QoQ Shift)?          [ YES / NO ]

FINAL DETERMINATION:
[ ] APPROVED - Unconditional Production Rollout
[ ] CONDITIONALLY APPROVED - Production with Quantitative Overlays & Monthly Monitoring
[ ] RESTRICTED - Shadow Horizon Run Only (No Sanctioning or Capital Calculation Usage)
[ ] REJECTED - Remanded to Model Development for Remediation

Lead Validator Signature: ___________________________    Date: ________________________
Head of Credit Risk Control Signoff: ________________    Date: ________________________
========================================================================================

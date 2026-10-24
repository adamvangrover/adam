# ADAM v30.1-R-Patch Module 8: Hunt & Kill Execution Report (Hardened Spec)
**Correlation ID**: `corr-hk-a1cfa8ae` | **Executed At**: `2026-10-24T01:14:37.681351+00:00` | **Status**: `COMPLETED`

## 0. Layer 0 Safety & Materiality Assurance Header
| Layer 0 Gate | h1 (VPIN <= 0.65) | h2 (Util <= 85%) | h3 (Borrow <= 450bps) | h4 (Co-Op Sacred Rights) | SHA-256 Feed Integrity | Epistemic Confidence | HITL Required | Final Assured Score | Market Integrity |
|---|---|---|---|---|---|---|---|---|---|
| **1** | PASS | PASS | PASS | PASS | VERIFIED | **0.92** | NO | **4.81** | COMPLIANT |

## 1. Stage 1 (Locate): Forced-Flow Triage & VPIN Swarm Trap Metrics
| Downgrade Prob (%) | Realized Vol 10D | VIX Anchor Ratio | Selling Pressure ($M) | LTM Int Coverage | LTM FCCR | Covenant Trap | VPIN | SNER | Swarm Trap | Position ($M) | ADV ($M) | Impact Slippage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 65.0% | 21.0 | 1.1278 | $47.7M | 1.45x | 1.05x | ACTIVE | **0.22** | **1.1x** | CLEAR | $25.0M | $100.0M | 37.5 bps |
| 85.0% | 24.5 | 1.3158 | $38.95M | 1.2x | 0.92x | ACTIVE | **0.35** | **2.4x** | CLEAR | $15.0M | $20.0M | 64.95 bps |
| 30.0% | 32.8 | 1.7615 | $46.54M | 2.8x | 1.9x | INACTIVE | **0.18** | **0.8x** | CLEAR | $10.0M | $150.0M | 19.36 bps |
| 15.0% | 45.0 | 2.4168 | $262.84M | 3.2x | 2.5x | INACTIVE | **0.28** | **1.4x** | CLEAR | $30.0M | $250.0M | 25.98 bps |

## 2. Structural Dislocation & Basis Ledger
| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | Funding, Slippage & Penalty | Net Spread (Spread_Net) | VPIN / SNER | PCS Ratio | Epistemic Status | Lifecycle State | Catalyst & T_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HK-01 | Cash HY vs. CDX Basis | Vector 2: Cash vs. Synthetic Credit Basis Compression | Physical Senior Notes (BB Basket) @ 92.5 | CDX NA HY 5Y Protection + Repo @ 95.8 | 330.0 bps | 440.5 bps | **-110.5 bps** | 0.22 / 1.1x | **3.0052** | `[CALCULATED]` | `ISOLATED_QUARANTINED` | Dealer inventory rebalance & index rollover (15 Days) |
| HK-02 | Co-Op / LMT Priming Target | Vector 1: Capital Structure & LMT Priming Arbitrage | 1L Term Loan B (>50.1% Steering Group) @ 68.0 | Unsecured Senior Notes / Common Equity @ 88.0 | 2000.0 bps | 494.95 bps | **1505.05 bps** | 0.35 / 2.4x | **2.3234** | `[SURVEILLANCE_RULE_SPEC]` | `ISOLATED_QUARANTINED` | 8-K Item 1.01 Drop-Down / Uptier Exchange (45 Days) |
| HK-03 | Index vs. Single-Name Vol | Vector 3: Dispersion & Correlation Arbitrage | Single-Name 30D ATM Straddles @ 14.5 | Short SPX 30D ATM Straddle (Vega-Neutral) @ 18.2 | 370.0 bps | 144.36 bps | **225.64 bps** | 0.18 / 0.8x | **3.1199** | `[CALCULATED]` | `SWEPT_TO_FORTRESS` | Earnings dispersion realization (20 Days) |
| HK-05 | HYG ETF Apex Ambush | Vector 5: Counter-Swarm Liquidity Ambush | HYG ETF (-185 bps NAV Discount) @ 72.2 | Physical HY Basket Creation/Redemption + CDX @ 74.05 | 185.0 bps | 58.98 bps | **126.02 bps** | 0.28 / 1.4x | **2.8728** | `[VERIFIED]` | `ISOLATED_QUARANTINED` | Apex Liquidity provision at 3.2-sigma flush (1 Days) |

## 3. Zero-Trust Order Routing & Microstructure Spec
| Target ID | Air-Gapped Sanitization | Liquidity Masking Venue | Execution Algorithm | Legging Sequence |
|---|---|---|---|---|
| HK-01 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (Physical Senior Notes (BB Basket)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (CDX NA HY 5Y Protection + Repo) to eliminate legging risk |
| HK-02 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (1L Term Loan B (>50.1% Steering Group)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Unsecured Senior Notes / Common Equity) to eliminate legging risk |
| HK-03 | VERIFIED | Dark pools / midpoint pegs (IEX D-Limit) | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (Single-Name 30D ATM Straddles) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Short SPX 30D ATM Straddle (Vega-Neutral)) to eliminate legging risk |
| HK-05 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (HYG ETF (-185 bps NAV Discount)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Physical HY Basket Creation/Redemption + CDX) to eliminate legging risk |

## 4. Kill-Switch & Counter-Predation Audit Ledger
| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |
|---|---|---|---|---|---|---|
| HK-01 | 42.0% | 85.0 bps | PASS | N/A | VERIFIED | **PASS** |
| HK-02 | 65.0% | 220.0 bps | PASS | 54.2% | VERIFIED | **PASS** |
| HK-03 | 30.0% | 45.0 bps | PASS | N/A | VERIFIED | **PASS** |
| HK-05 | 20.0% | 30.0 bps | PASS | N/A | VERIFIED | **PASS** |

## 5. Embedded Safety Defense & Primacy Invariant Ledger
| Target ID | Safety Primacy Enforced | Short Squeeze Breach | Co-Op Legality Breach | VPIN Toxicity Quarantine | Unbounded Loss Risk | Reward Override Attempted | Safety Verdict |
|---|---|---|---|---|---|---|---|
| HK-01 | True | False | False | False | False | **False** | **`PASS`** |
| HK-02 | True | False | False | False | False | **False** | **`PASS`** |
| HK-03 | True | False | False | False | False | **False** | **`PASS`** |
| HK-05 | True | False | False | False | False | **False** | **`PASS`** |

## 6. Basis Convergence & Extinguish Telemetry
| Target ID | Initial Spread | Current Spread | Compression (%) | 50% Scale-Out | Extinguish Triggered | Multi-Oracle 180s Consensus | Hard Kill-Switch Active | Fortress Sweep (%) |
|---|---|---|---|---|---|---|---|---|
| HK-01 | 330.0 bps | 82.5 bps | 75.0% | True | True | VERIFIED | False | 100.0% |
| HK-02 | 2000.0 bps | 500.0 bps | 75.0% | True | False | VERIFIED | False | 100.0% |
| HK-03 | 370.0 bps | 92.5 bps | 75.0% | True | True | VERIFIED | False | 100.0% |
| HK-05 | 185.0 bps | 46.25 bps | 75.0% | True | False | VERIFIED | False | 100.0% |

## 7. W3C PROV-O SHA-256 Audit Seal
```json
{
  "prov_bundle_id": "prov-hk-0f90f66aa750",
  "prefix": {
    "prov": "http://www.w3.org/ns/prov#",
    "adam": "https://adam.financial/ontology/v30.1#"
  },
  "entity": {
    "adam:HuntAndKillInput": {
      "prov:type": "adam:ExecutionInput",
      "adam:sha256": "ddbae32a8023ad1fb758c2ff6987594c61274fedc82e50b08fbecb5414c3df9c"
    },
    "adam:HuntAndKillOutput": {
      "prov:type": "adam:ExecutionOutput",
      "adam:sha256": "b0f01fdc3bec09a7d4eaecf0c580bb43b83ca67ad82fa2f44cf431322a7f62d8",
      "adam:authorized_deployments": 1
    }
  },
  "activity": {
    "adam:Module8Execution": {
      "prov:type": "adam:HuntAndKillOrchestration",
      "prov:startedAtTime": "2026-10-24T01:14:37.682103+00:00",
      "prov:endedAtTime": "2026-10-24T01:14:37.682115+00:00"
    }
  },
  "wasGeneratedBy": {
    "_:wgb1": {
      "prov:entity": "adam:HuntAndKillOutput",
      "prov:activity": "adam:Module8Execution"
    }
  },
  "used": {
    "_:u1": {
      "prov:activity": "adam:Module8Execution",
      "prov:entity": "adam:HuntAndKillInput"
    }
  }
}
```
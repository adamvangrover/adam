# ADAM v30.1-R-Patch Module 8: Hunt & Kill Execution Report
**Correlation ID**: `corr-hk-6dcc962d` | **Executed At**: `2026-10-22T01:41:19.855944+00:00` | **Status**: `COMPLETED`

## 1. Stage 1 (Locate): Forced-Flow Triage & Market Impact Metrics
| Downgrade Prob (%) | Realized Vol 10D | VIX Anchor Ratio | Selling Pressure ($M) | LTM Int Coverage | LTM FCCR | Covenant Trap | Position ($M) | ADV ($M) | Impact Slippage |
|---|---|---|---|---|---|---|---|---|---|
| 65.0% | 21.0 | 1.1278 | $47.7M | 1.45x | 1.05x | ACTIVE | $25.0M | $100.0M | 37.5 bps |
| 85.0% | 24.5 | 1.3158 | $38.95M | 1.2x | 0.92x | ACTIVE | $15.0M | $20.0M | 64.95 bps |
| 30.0% | 32.8 | 1.7615 | $46.54M | 2.8x | 1.9x | INACTIVE | $10.0M | $150.0M | 19.36 bps |

## 2. Structural Dislocation & Basis Ledger
| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | Funding/Slippage | Net Spread (Spread_Net) | PCS | Epistemic Status | Lifecycle State | Catalyst & T_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HK-01 | Cash HY vs. CDX Basis | Vector 2: Cash vs. Synthetic Credit Basis Compression | Physical Senior Notes (BB Basket) @ 92.5 | CDX NA HY 5Y Protection + Repo @ 95.8 | 330.0 bps | 612.5 bps | **-282.5 bps** | **3.8528** | `[CALCULATED]` | `ISOLATED_QUARANTINED` | Dealer inventory rebalance & index rollover (15 Days) |
| HK-02 | Co-Op / LMT Priming Target | Vector 1: Capital Structure & LMT Priming Arbitrage | 1L Term Loan B (>50.1% Steering Group) @ 68.0 | Unsecured Senior Notes / Common Equity @ 88.0 | 2000.0 bps | 694.95 bps | **1305.05 bps** | **3.5744** | `[SURVEILLANCE_RULE_SPEC]` | `SWEPT_TO_FORTRESS` | 8-K Item 1.01 Drop-Down / Uptier Exchange (45 Days) |
| HK-03 | Index vs. Single-Name Vol | Vector 3: Dispersion & Correlation Arbitrage | Single-Name 30D ATM Straddles @ 14.5 | Short SPX 30D ATM Straddle (Vega-Neutral) @ 18.2 | 370.0 bps | 144.36 bps | **225.64 bps** | **3.8048** | `[CALCULATED]` | `SWEPT_TO_FORTRESS` | Earnings dispersion realization (20 Days) |

## 3. Order Routing & Execution Microstructure Spec
| Target ID | Liquidity Masking Venue | Execution Algorithm | Legging Sequence |
|---|---|---|---|
| HK-01 | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (Physical Senior Notes (BB Basket)) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (CDX NA HY 5Y Protection + Repo) to eliminate legging risk |
| HK-02 | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (1L Term Loan B (>50.1% Steering Group)) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Unsecured Senior Notes / Common Equity) to eliminate legging risk |
| HK-03 | Dark pools / midpoint pegs (IEX D-Limit) | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (Single-Name 30D ATM Straddles) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Short SPX 30D ATM Straddle (Vega-Neutral)) to eliminate legging risk |

## 4. Kill-Switch & Counter-Predation Audit Ledger
| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |
|---|---|---|---|---|---|---|
| HK-01 | 42.0% | 85.0 bps | PASS | N/A | VERIFIED | **PASS** |
| HK-02 | 65.0% | 220.0 bps | PASS | 54.2% | VERIFIED | **PASS** |
| HK-03 | 30.0% | 45.0 bps | PASS | N/A | VERIFIED | **PASS** |

## 5. Embedded Safety Defense & Primacy Invariant Ledger
| Target ID | Safety Primacy Enforced | Short Squeeze Breach | Co-Op Legality Breach | Unbounded Loss Risk | Reward Override Attempted | Safety Verdict |
|---|---|---|---|---|---|---|
| HK-01 | True | False | False | False | **False** | **`PASS`** |
| HK-02 | True | False | False | False | **False** | **`PASS`** |
| HK-03 | True | False | False | False | **False** | **`PASS`** |

## 6. Basis Convergence & Extinguish Telemetry
| Target ID | Initial Spread | Current Spread | Compression (%) | 50% Scale-Out Triggered | Extinguish Triggered | Hard Kill-Switch Active | Fortress Sweep (%) |
|---|---|---|---|---|---|---|---|
| HK-01 | 330.0 bps | 82.5 bps | 75.0% | True | True | False | 100.0% |
| HK-02 | 2000.0 bps | 500.0 bps | 75.0% | True | True | False | 100.0% |
| HK-03 | 370.0 bps | 92.5 bps | 75.0% | True | True | False | 100.0% |

## 7. W3C PROV-O SHA-256 Audit Seal
```json
{
  "prov_bundle_id": "prov-hk-75bb480ca158",
  "prefix": {
    "prov": "http://www.w3.org/ns/prov#",
    "adam": "https://adam.financial/ontology/v30.1#"
  },
  "entity": {
    "adam:HuntAndKillInput": {
      "prov:type": "adam:ExecutionInput",
      "adam:sha256": "ec1f75d80ef8c1f36b5ec498a05807a615d1bc5b4fab50cf19895e12b348ac47"
    },
    "adam:HuntAndKillOutput": {
      "prov:type": "adam:ExecutionOutput",
      "adam:sha256": "7945ac91e674e04171622a9f8f6cbfc57110067b55dd4324a8bd55b018797c76",
      "adam:authorized_deployments": 2
    }
  },
  "activity": {
    "adam:Module8Execution": {
      "prov:type": "adam:HuntAndKillOrchestration",
      "prov:startedAtTime": "2026-10-22T01:41:19.856488+00:00",
      "prov:endedAtTime": "2026-10-22T01:41:19.856499+00:00"
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
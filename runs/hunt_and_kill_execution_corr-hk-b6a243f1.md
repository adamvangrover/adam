# ADAM v30.1-R-Patch Module 8: Hunt & Kill Execution Report
**Correlation ID**: `corr-hk-b6a243f1` | **Executed At**: `2026-10-22T00:40:28.828424+00:00` | **Status**: `COMPLETED`

## 1. Structural Dislocation & Basis Ledger
| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | Funding/Slippage | Net Spread (Spread_Net) | PCS | Epistemic Status | Catalyst & T_max |
|---|---|---|---|---|---|---|---|---|---|---|
| HK-01 | Cash HY vs. CDX Basis | Vector 2: Cash vs. Synthetic Credit Basis Compression | Physical Senior Notes (BB Basket) @ 92.5 | CDX NA HY 5Y Protection + Repo @ 95.8 | 330.0 bps | 600.0 bps | **-270.0 bps** | **3.4** | `[CALCULATED]` | Dealer inventory rebalance & index rollover (15 Days) |
| HK-02 | Co-Op / LMT Priming Target | Vector 1: Capital Structure & LMT Priming Arbitrage | 1L Term Loan B (>50.1% Steering Group) @ 68.0 | Unsecured Senior Notes / Common Equity @ 88.0 | 2000.0 bps | 680.0 bps | **1320.0 bps** | **2.2** | `[SURVEILLANCE_RULE_SPEC]` | 8-K Item 1.01 Drop-Down / Uptier Exchange (45 Days) |
| HK-03 | Index vs. Single-Name Vol | Vector 3: Dispersion & Correlation Arbitrage | Single-Name 30D ATM Straddles @ 14.5 | Short SPX 30D ATM Straddle (Vega-Neutral) @ 18.2 | 370.0 bps | 135.0 bps | **235.0 bps** | **3.2** | `[CALCULATED]` | Earnings dispersion realization (20 Days) |

## 2. Order Routing & Execution Microstructure Spec
| Target ID | Liquidity Masking Venue | Execution Algorithm | Legging Sequence |
|---|---|---|---|
| HK-01 | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (Physical Senior Notes (BB Basket)) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (CDX NA HY 5Y Protection + Repo) to eliminate legging risk |
| HK-02 | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (1L Term Loan B (>50.1% Steering Group)) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Unsecured Senior Notes / Common Equity) to eliminate legging risk |
| HK-03 | Dark pools / midpoint pegs (IEX D-Limit) | Implementation Shortfall (IS) weighted toward close auction imbalances | 1. Execute physical leg (Single-Name 30D ATM Straddles) on TRACE / BSL assignment<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Short SPX 30D ATM Straddle (Vega-Neutral)) to eliminate legging risk |

## 3. Kill-Switch & Counter-Predation Audit Ledger
| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |
|---|---|---|---|---|---|---|
| HK-01 | 42.0% | 85.0 bps | PASS | N/A | VERIFIED | **PASS** |
| HK-02 | 65.0% | 220.0 bps | PASS | 54.2% | VERIFIED | **PASS** |
| HK-03 | 30.0% | 45.0 bps | PASS | N/A | VERIFIED | **PASS** |

## 4. Basis Convergence & Extinguish Telemetry
| Target ID | Initial Spread | Current Spread | Compression (%) | 50% Scale-Out Triggered | Extinguish Triggered | Hard Kill-Switch Active | Fortress Sweep (%) |
|---|---|---|---|---|---|---|---|
| HK-01 | 330.0 bps | 82.5 bps | 75.0% | True | True | False | 100.0% |
| HK-02 | 2000.0 bps | 500.0 bps | 75.0% | True | True | False | 100.0% |
| HK-03 | 370.0 bps | 92.5 bps | 75.0% | True | True | False | 100.0% |

## 5. W3C PROV-O SHA-256 Audit Seal
```json
{
  "prov_bundle_id": "prov-hk-1e8ad8f3ee95",
  "prefix": {
    "prov": "http://www.w3.org/ns/prov#",
    "adam": "https://adam.financial/ontology/v30.1#"
  },
  "entity": {
    "adam:HuntAndKillInput": {
      "prov:type": "adam:ExecutionInput",
      "adam:sha256": "749cd2295716d37bd114ee210ad55445787bf77a0284d1aea74c27b57d0cb819"
    },
    "adam:HuntAndKillOutput": {
      "prov:type": "adam:ExecutionOutput",
      "adam:sha256": "b6d6df2944f706e55c82b57155b4cedc58591fdc147b556d742bfda8d39a8f50",
      "adam:authorized_deployments": 1
    }
  },
  "activity": {
    "adam:Module8Execution": {
      "prov:type": "adam:HuntAndKillOrchestration",
      "prov:startedAtTime": "2026-10-22T00:40:28.828977+00:00",
      "prov:endedAtTime": "2026-10-22T00:40:28.828988+00:00"
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
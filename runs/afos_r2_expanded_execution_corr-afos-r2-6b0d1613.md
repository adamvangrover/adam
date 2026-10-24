# AFOS v30.1-R2-Expanded Orchestrator Execution Report (Hardened II Spec)
**Correlation ID**: `corr-afos-r2-6b0d1613` | **Executed At**: `2026-10-24T01:58:19.958771+00:00` | **Status**: `EXPLOITS_AND_DEFENSES_EVALUATED`

## 1. Layer 0 Safety & Materiality Assurance Header
| Layer 0 Gate | VPIN Toxicity (h1) | Borrow Util (h2) | Borrow Fee (h3) | Co-Op Sacred Rights (h4) | SHA-256 Feed Integrity | Epistemic Confidence | HITL Escalation | Final Assured Score | Market Integrity |
|---|---|---|---|---|---|---|---|---|---|
| **1** | PASS | PASS | PASS | PASS | VERIFIED | **0.92** | NO | **4.52** | COMPLIANT |

## 2. Structural Dislocation & Basis Ledger
| Kill ID | Target / Pair | Dislocation Vector | Long Leg | Short / Hedge Leg | Gross Dislocation | Net Cost & Swarm Penalty | Net Spread (Spread_Net) | VPIN / SNER | PCS Ratio | Epistemic Status | Convergence Catalyst & Max Window (T_max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2-EXP-01 | LMT BSL Priming vs Sub Short | Vector 1: Capital Structure & LMT Priming Arbitrage | 1L BSL Assignment (68.00) @ 68.0 | Unsecured Senior Notes (88.00) @ 88.0 | 2000.0 bps | 475.08 bps | **1524.92 bps** | 0.28 / 1.8x | **2.5736** | `[SURVEILLANCE_RULE_SPEC]` | 8-K Item 1.01 Uptier Exchange (45 Days) |
| R2-EXP-02 | HY Bond vs CDX Basis | Vector 2: Cash vs. Synthetic Credit Basis Compression | Physical Senior Notes (BB Basket) @ 91.5 | CDX NA HY 5Y Protection @ 95.2 | 370.0 bps | 435.5 bps | **-65.5 bps** | 0.20 / 0.9x | **3.0899** | `[CALCULATED]` | Index rebalance convergence (14 Days) |
| R2-EXP-03 | Dispersion Straddle Arbitrage | Vector 3: Dispersion & Correlation Arbitrage | Single-Name ATM Volatility @ 15.0 | Short Index Volatility (SPX/NDX) @ 19.5 | 450.0 bps | 129.65 bps | **320.35 bps** | 0.15 / 0.7x | **3.2263** | `[CALCULATED]` | Earnings correlation breakdown (21 Days) |
| R2-EXP-05 | Apex Swarm Ambush HYG ETF | Vector 5: Counter-Swarm Liquidity Ambush | HYG ETF (-210 bps Discount) @ 71.8 | Physical Basket Creation/Redemption @ 73.9 | 210.0 bps | 52.62 bps | **157.38 bps** | 0.25 / 1.2x | **2.9925** | `[VERIFIED]` | Apex Liquidity Provision at 3-sigma flash (1 Days) |

## 3. Zero-Trust Order Routing & Microstructure Spec
| Target ID | Air-Gapped Sanitization | Liquidity Masking Venue | Execution Algorithm | Legging Sequence |
|---|---|---|---|---|
| R2-EXP-01 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (1L BSL Assignment (68.00)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Unsecured Senior Notes (88.00)) to eliminate legging risk |
| R2-EXP-02 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (Physical Senior Notes (BB Basket)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (CDX NA HY 5Y Protection) to eliminate legging risk |
| R2-EXP-03 | VERIFIED | Dark pools / midpoint pegs (IEX D-Limit) | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (Single-Name ATM Volatility) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Short Index Volatility (SPX/NDX)) to eliminate legging risk |
| R2-EXP-05 | VERIFIED | Conditional block venues / Midpoint cross | Implementation Shortfall (IS) with randomized child order arrival (Exponential decay) | 1. Execute less liquid physical leg (HYG ETF (-210 bps Discount)) on TRACE / BSL assignment first<br>2. Deploy contingent delta-hedge in liquid futures / CDX NA HY<br>3. Lift synthetic leg (Physical Basket Creation/Redemption) to eliminate legging risk |

## 4. Kill-Switch & Counter-Predation Audit Ledger
| Target ID | Short Util (%) | Borrow Cost (bps) | Borrow Ceiling Pass | Co-Op Steering (%) | Co-Op Majority Verified | Overall Pass |
|---|---|---|---|---|---|---|
| R2-EXP-01 | 62.0% | 180.0 bps | PASS | 52.1% | VERIFIED | **PASS** |
| R2-EXP-02 | 38.0% | 75.0 bps | PASS | N/A | VERIFIED | **PASS** |
| R2-EXP-03 | 25.0% | 40.0 bps | PASS | N/A | VERIFIED | **PASS** |
| R2-EXP-05 | 18.0% | 25.0 bps | PASS | N/A | VERIFIED | **PASS** |

## 5. W3C PROV-O SHA-256 Audit Bundle
```json
{
  "prov_bundle_id": "prov-afos-r2-bfba6d782d14",
  "input_sha256": "2b217a62d791983c9c35ae1abe1e5ca050ce9a92f46490aac1321af5942771a6",
  "output_sha256": "9f6c002a3eaa19d19dea7202ff2a9f5ca278fa6921c41a5e5b3c6dbf00249f46",
  "provenance_header": {
    "git_commit_hash": "v30.1-R2-EXPANDED-COMMIT",
    "timestamp": "2026-10-24T01:58:19.958771+00:00",
    "content_hash": "9f6c002a3eaa19d19dea7202ff2a9f5ca278fa6921c41a5e5b3c6dbf00249f46",
    "jsonLogic_version": "2.0.0",
    "confidence_score": 0.92,
    "derivation_path": "market-mayhem -> fortress-and-hunt -> hunt-and-kill -> afos-r2-orchestrator",
    "source_data_object": "urn:adam:afos-r2-expanded:corr-afos-r2-6b0d1613"
  },
  "sha256_release_fingerprint": "ee2f931575b98555ea3efd60c71211f596aaf3cf6582addf1f71e9604081f4d9"
}
```
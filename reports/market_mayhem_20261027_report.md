# INSTITUTIONAL CREDIT & MACRO SURVEILLANCE REPORT
## October 27, 2026 | ADAM v30.1 Neuro-Symbolic Sovereign Risk Core
**Compliance:** SR 11-7 / Basel III/IV / OCC 2026-13 Model Risk Governance

---

### EXECUTIVE SUMMARY
This report details the quantitative credit stress evaluation, Money Market Term-SOFR reset mechanics, and liquidation LGD waterfalls for leveraged corporate balance sheets under current clearing rates as of **October 27, 2026**.

#### Key Market Inputs
- **Overnight SOFR Spot:** 3.930%
- **30-Day Average SOFR:** 3.89253%
- **90-Day Average SOFR:** 3.75301%
- **10-Year Treasury Yield:** 4.992%
- **ICE Brent Crude:** $114.17/bbl
- **NYMEX WTI Crude:** $100.80/bbl
- **CDX High Yield OAS:** 410 bps
- **B-Rated BSL Floating Borrowing Floor:** 8.430% (3.930% SOFR + 450 bps)

---

### QUANTITATIVE CREDIT STRESS & LGD WATERFALL (BLTC OBLIGOR)

#### Borrower Profile
- **Entity:** Benchmark Levered Telecom Corp (BLTC)
- **Total Debt:** $2,500.00M ($1,500.00M Senior Secured TLB @ SOFR+450; $1,000.00M Fixed Notes @ 5.00%)
- **Operating EBITDA:** $280.00M | Maintenance CapEx: $90.00M | CADS: $190.00M
- **Gross Leverage:** 8.93x EBITDA

#### Multi-Year Debt Service Coverage
- **Year 1 Total Debt Service:** $126.45M (TLB Int) + $50.00M (Fixed Int) + $15.00M (Amort) = **$191.45M**
- **Year 1 Cash FCF DSCR:** $190.00M / $191.45M = **0.992x** (Net Cash Bleed: -$1.45M/yr)
- **Year 2 Cash FCF DSCR:** **0.999x** (Post-$15M principal amort)

#### Chapter 11 Absolute Priority Rule (APR) Liquidation Recovery
- **Base 5.0x EBITDA Multiple ($1,400M EV):**
  - Senior Secured TLB Claim ($1,500M): $1,400M Recovery (**93.33% Recovery / 6.67% LGD**)
  - Senior Unsecured Notes ($1,000M): $0M Recovery (**0.00% Recovery / 100.00% LGD**)
- **Stressed 4.2x EBITDA Multiple ($1,176M EV):**
  - Senior Secured TLB Recovery drops to $1,176M (**78.40% Recovery / 21.60% LGD**)

#### Probability of Default (PD) & QMC Tail Adjustment
- **Discrete 3-Year Hazard Rates:** h1=14.5%, h2=22.0%, h3=31.5%
- **Cumulative 3-Year Base PD:** **54.32%**
- **QMC 99th Percentile Tail PD:** **68.40%**
- **Basel III Risk Weight:** **150% Defaulted / High-Risk Corporate Treatment**

# Commercial Real Estate (CRE) Stress Test Report

**Execution Date:** 2026-07-20
**Simulation Engine:** Floating-Base Merton Simulator

## Overview
This report details the findings of the automated CRE stress test executed following the transition to the 'Stagflationary Shock' regime. The goal is to quantify the vulnerability of regional bank balance sheets to escalating office and retail sector defaults.

## Key Findings

*   **Probability of Default (PD) Expansion:** The baseline PD has expanded from 1.20% (T-1) to 1.39% in the current state.
*   **Asset Buffer Erosion:** Driven by the simulated 1.51% decline in the S&P 500, the equity buffer supporting leveraged real estate assets has eroded by a factor of 1.015x.
*   **Debt Cost Multiplier:** The 40 bps widening in High Yield credit spreads has drastically increased refinancing costs, pushing the debt cost multiplier to 1.142x.
*   **Loss Given Default (LGD) Revision:** Under the current regime, the modeled LGD factor has been revised upwards to 50%, reflecting a significant drop in expected recovery rates on distressed property sales.

## Structural Vulnerabilities
The non-linear nature of the Merton heuristic indicates that further spread widening beyond 400 bps will trigger a 'Systemic Stress' event, where defaults cascade beyond isolated B- and C-class office properties into prime A-class real estate.
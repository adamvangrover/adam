# AFOS Decision Kernel Specification

## Overview
The Decision Kernel shifts AFOS from producing opaque outputs (like a single "Risk Rating") to generating explainable, auditable Decision Graphs. It binds Evidence to Policies to produce immutable Decisions.

## The Decision Graph Architecture

Decisions in AFOS are not scalar values; they are directed acyclic graphs representing the hierarchy of logic that led to the conclusion.

### Example: Credit Risk Rating
Instead of a black-box model outputting "BB-", the Decision Kernel produces a traversable graph:

```
Liquidity Evidence
      ↓
Coverage Policy Evaluation
      ↓
Leverage Assessment
      ↓
Sponsor Quality
      ↓
Industry Headwinds
      ↓
Collateral Valuation
      ↓
Recovery Estimate
      ↓
Exposure Calculation
      ↓
Concentration Limits
      ↓
Stress Test Results
      ↓
[DECISION GRAPH]
      ↓
Final Risk Rating (BB-)
```

## Explainability and Provenance
* Every node in the Decision Graph contains a cryptographic link to the specific `Evidence` used.
* Every transition is governed by a specific version of a `Policy`.
* The final `Decision` object is an immutable artifact that can be inspected by the Governance Kernel or replayed by the Simulation Kernel.

# Adam Finance Module

## Overview
This module houses the deterministic financial mathematics and algorithms isolated from the probabilistic neuro-symbolic cognitive graph.

## Structure
- `icat.py`: The ICAT (Ingest, Clean, Analyze, Transform) Engine.
- `icat_schema.py`: Schemas for LBO modeling, Credit Metrics, and Valuation.
- `math.py`: Pure mathematical functions (VaR, CVaR).
- `snc_utils.py`: Rules and functions for Shared National Credit (SNC) calculations.

## Architectural Principles
- **Determinism:** All financial calculations strictly reside here. This codebase does not hallucinate.
- **Portability:** Code can run in any system given standard inputs, enabling traditional CI/CD unit testing.

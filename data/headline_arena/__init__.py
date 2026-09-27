"""
ADAM-Macro-Sentinel: Autonomous Forecasting & Recursive Reasoning Engine
========================================================================
Institutional-grade probabilistic market forecasting engine for the
Headline Arena evaluation benchmark.
Modules:
    scoring     — S_dir, Brier, CRPS, S_rat scoring implementations
    calibration — Platt scaling, dynamic temperature, epistemic memory ledger
    rationale   — Structured 4-sub-dimension rationale generator
    arbitration — Champion-Challenger dual-model debate harness
    gate        — Pre-submission validation gate (S_rat >= 75.0)
    schema      — Pydantic models for all I/O boundaries
    engine      — Main forecasting orchestrator
    api         — Headline Arena API client
"""
__version__ = "1.0.0"
__agent__ = "ADAM-Macro-Sentinel"

#!/usr/bin/env python3
"""
ADAM v30.1-R End-to-End Gate Clearance & LMT Contagion Orchestrator
-------------------------------------------------------------------
Main entrypoint module for LMT Contagion Gate Clearance.
"""

from scripts.run_v30_1_r_gate_clearance_orchestrator import (
    GateClearanceOrchestrator,
    MacroStateVector,
    LMTBranchOutcome,
    ContagionSpilloverMetrics,
    PDCalibrationMetrics,
    GateRubricScore,
    ReleaseGateAdjudication,
    main,
)

__all__ = [
    "GateClearanceOrchestrator",
    "MacroStateVector",
    "LMTBranchOutcome",
    "ContagionSpilloverMetrics",
    "PDCalibrationMetrics",
    "GateRubricScore",
    "ReleaseGateAdjudication",
    "main",
]

if __name__ == "__main__":
    main()

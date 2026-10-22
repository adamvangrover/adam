"""
Pytest Verification Suite for ADAM v30.1-R End-to-End Gate Clearance & LMT Contagion Orchestrator.
"""

import json
import pytest

from scripts.lmt_contagion_gate_orchestrator import (
    GateClearanceOrchestrator,
    MacroStateVector,
    LMTBranchOutcome,
    ContagionSpilloverMetrics,
    PDCalibrationMetrics,
    ReleaseGateAdjudication,
)


def test_lmt_contagion_orchestrator_end_to_end():
    orchestrator = GateClearanceOrchestrator(
        obligor_name="BLTC Hypothetical Borrower",
        ticker="BLTC",
        total_debt_m=2500.0,
        ebitda_m=280.0,
        baseline_dscr=0.9955,
        rate_shock_bps=150.0,
    )

    manifest = orchestrator.run()

    # Verification of release disposition
    assert manifest["orchestrator_version"] == "v30.1-R"
    assert manifest["release_disposition"] == "PRODUCTION_READY"

    # Provenance and SHA-256 fingerprint verification
    prov = manifest["provenance_header"]
    assert prov["jsonLogic_version"] == "2.0.0"
    assert len(prov["content_hash"]) == 64
    assert prov["confidence_score"] > 0.85
    assert len(manifest["sha256_release_fingerprint"]) == 64

    # Adjudication Gate Verification
    adj = manifest["adjudication_summary"]
    assert len(adj["rubric_scores"]) == 7
    assert adj["average_rubric_score"] >= 85.0
    assert adj["deterministic_invariants_passed"] is True
    assert adj["integration_decision_allowed"] is True

    # Validate individual gates
    for gate in adj["rubric_scores"]:
        assert gate["status"] == "PASS"
        assert gate["score_0_100"] >= 80.0


def test_lmt_decision_tree_branches():
    orchestrator = GateClearanceOrchestrator(baseline_dscr=0.9955)
    macro_state = orchestrator.phase1_macro_state_lock()
    lmt_res = orchestrator.phase2_lmt_and_contagion(macro_state)

    assert lmt_res["covenant_breach_detected"] is True
    rec_branch: LMTBranchOutcome = lmt_res["recommended_branch"]
    assert rec_branch.branch_id == "LMT-B-UPTIER"
    assert rec_branch.pro_forma_dscr == 1.18
    assert rec_branch.senior_secured_recovery_pct == 95.0

    alt_branch: LMTBranchOutcome = lmt_res["alternate_branch"]
    assert alt_branch.branch_id == "LMT-A-DROPDOWN"
    assert alt_branch.pro_forma_dscr == 1.12

    liq_branch: LMTBranchOutcome = lmt_res["liquidation_branch"]
    assert liq_branch.branch_id == "LMT-C-CHAPTER11"
    assert liq_branch.pro_forma_dscr == 0.0

    contagion: ContagionSpilloverMetrics = lmt_res["contagion_metrics"]
    assert contagion.contagion_index_0_to_1 < 0.15
    assert contagion.systemic_spillover_threshold_breached is False

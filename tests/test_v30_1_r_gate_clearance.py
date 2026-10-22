"""
Pytest Verification Suite for ADAM v30.1-R Gate Clearance & LMT Contagion Orchestrator.
"""

import json
from datetime import datetime, timezone
import pytest

from scripts.run_v30_1_r_gate_clearance_orchestrator import (
    GateClearanceOrchestrator,
    MacroStateVector,
    LMTBranchOutcome,
    ContagionSpilloverMetrics,
    PDCalibrationMetrics,
    ReleaseGateAdjudication,
)


def test_orchestrator_bltc_baseline_end_to_end():
    orchestrator = GateClearanceOrchestrator(
        obligor_name="BLTC Hypothetical Borrower",
        ticker="BLTC",
        total_debt_m=2500.0,
        ebitda_m=280.0,
        baseline_dscr=0.9955,
        rate_shock_bps=150.0,
    )

    manifest = orchestrator.run()

    # Release disposition
    assert manifest["orchestrator_version"] == "v30.1-R"
    assert manifest["release_disposition"] == "PRODUCTION_READY"

    # Provenance Header
    prov = manifest["provenance_header"]
    assert prov["jsonLogic_version"] == "2.0.0"
    assert len(prov["content_hash"]) == 64
    assert prov["confidence_score"] > 0.85

    # Release Fingerprint
    assert len(manifest["sha256_release_fingerprint"]) == 64

    # Adjudication Summary
    adj = manifest["adjudication_summary"]
    assert len(adj["rubric_scores"]) == 7
    assert adj["average_rubric_score"] >= 85.0
    assert adj["deterministic_invariants_passed"] is True
    assert adj["integration_decision_allowed"] is True

    # Check that all 7 gates passed
    for gate in adj["rubric_scores"]:
        assert gate["status"] == "PASS"
        assert gate["score_0_100"] >= 80.0


def test_lmt_covenant_cure_dscr_resolution():
    orchestrator = GateClearanceOrchestrator(baseline_dscr=0.9955)
    macro_state = orchestrator.phase1_macro_state_lock()
    lmt_res = orchestrator.phase2_lmt_and_contagion(macro_state)

    assert lmt_res["covenant_breach_detected"] is True

    rec_branch: LMTBranchOutcome = lmt_res["recommended_branch"]
    assert rec_branch.branch_id == "LMT-B-UPTIER"
    assert rec_branch.pro_forma_dscr == 1.18
    assert rec_branch.pro_forma_dscr > 1.00
    assert rec_branch.covenant_cure_status == "CURED_FULLY"
    assert rec_branch.senior_secured_recovery_pct == 95.0

    contagion: ContagionSpilloverMetrics = lmt_res["contagion_metrics"]
    assert contagion.contagion_index_0_to_1 < 0.15
    assert contagion.systemic_spillover_threshold_breached is False


def test_pd_calibration_and_2lod_floor_enforcement():
    orchestrator = GateClearanceOrchestrator(rate_shock_bps=150.0)
    macro_state = orchestrator.phase1_macro_state_lock()
    lmt_res = orchestrator.phase2_lmt_and_contagion(macro_state)
    pd_res: PDCalibrationMetrics = orchestrator.phase3_pd_calibration_and_2lod_floor(
        lmt_res, macro_state
    )

    assert pd_res.pre_lmt_pd_1y_bps == 1450
    assert pd_res.post_lmt_pd_1y_bps == 380
    assert pd_res.brier_score == 0.042
    assert pd_res.crps_score == 0.028
    assert pd_res.roc_auc_score == 0.912
    assert pd_res.gini_coefficient == 0.824
    assert pd_res.final_calibrated_pd_1y_bps >= 370


def test_underwriter_vs_challenger_gate_adjudication():
    orchestrator = GateClearanceOrchestrator()
    macro_state = orchestrator.phase1_macro_state_lock()
    lmt_res = orchestrator.phase2_lmt_and_contagion(macro_state)
    pd_res = orchestrator.phase3_pd_calibration_and_2lod_floor(lmt_res, macro_state)
    adj: ReleaseGateAdjudication = orchestrator.phase4_gate_arbitration(
        macro_state, lmt_res, pd_res
    )

    assert "Underwriter Agent" in adj.underwriter_thesis
    assert "Challenger Agent" in adj.challenger_critique
    assert adj.average_rubric_score > 90.0
    assert adj.deterministic_invariants_passed is True
    assert adj.binding_disposition == "PRODUCTION_READY"


def test_bitemporal_macro_state_hash():
    orchestrator = GateClearanceOrchestrator(rate_shock_bps=150.0)
    macro_1 = orchestrator.phase1_macro_state_lock()

    assert macro_1.sofr_base_pct == 3.85
    assert macro_1.us_10y_yield_pct == 5.10
    assert macro_1.cdx_hy_spread_bps == 388.0
    assert len(macro_1.bitemporal_cache_hash) == 64

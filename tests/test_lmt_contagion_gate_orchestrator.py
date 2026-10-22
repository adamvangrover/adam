"""
Pytest Verification Suite for ADAM v30.1-R End-to-End Gate Clearance & LMT Contagion Orchestrator.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.lmt_contagion_gate_orchestrator import (
    CapitalStructureInput,
    LMTContagionGateOrchestrator,
    GateClearanceOrchestrator,
    MacroStateVector,
    Phase1MacroLock,
    Phase2LMTContagion,
    Phase3PDCalibration,
    Phase4GateArbitration,
    Phase5ArchivalSeal,
)


@pytest.fixture
def default_orchestrator():
    return LMTContagionGateOrchestrator()


def test_phase1_macro_lock():
    macro = MacroStateVector()
    p1 = Phase1MacroLock(macro).execute()

    assert p1["status"] == "CACHE_VERIFIED_AND_LOCKED"
    assert p1["macro_state"]["sofr_rate_pct"] == 3.85
    assert p1["shocked_sofr_pct"] == 5.35
    assert len(p1["macro_state_sha256"]) == 64


def test_phase2_lmt_contagion():
    cap = CapitalStructureInput()
    macro = MacroStateVector()
    p2 = Phase2LMTContagion(cap, macro).execute()

    assert p2["status"] == "LMT_TREE_RESOLVED"
    data = p2["data"]

    # Baseline DSCR calculation verification (~0.9955x)
    assert data["baseline_metrics"]["baseline_dscr"] == pytest.approx(0.9955, abs=0.01)
    assert data["baseline_metrics"]["gross_leverage"] == pytest.approx(8.93, abs=0.01)

    # LMT Options verification
    lmt = data["lmt_options"]
    assert "a_unrestricted_sub_dropdown" in lmt
    assert "b_serta_uptier_exchange" in lmt
    assert "c_ch11_apr_waterfall" in lmt

    # Uptier exchange recoveries check
    serta_rec = lmt["b_serta_uptier_exchange"]["tranche_recoveries"]
    assert serta_rec["super_priority_tranche"]["recovery_pct"] == 100.0
    assert serta_rec["unsecured_fixed_tranche"]["recovery_pct"] == 0.0

    # Syndicate contagion check
    assert len(data["syndicate_gnn_contagion"]) == 3


def test_phase3_pd_calibration():
    cap = CapitalStructureInput()
    macro = MacroStateVector()
    p2 = Phase2LMTContagion(cap, macro).execute()
    p3 = Phase3PDCalibration(cap, p2).execute()

    assert p3["status"] == "PD_CALIBRATION_AND_FLOOR_ENFORCED"
    data = p3["data"]

    assert data["pre_lmt_cumulative_3y_pd_pct"] == pytest.approx(54.32, abs=0.1)
    assert data["out_of_time_calibration"]["brier_score"] < 0.12
    assert data["twolod_override_floor"]["final_calibrated_1y_pd_pct"] == 15.00
    assert data["twolod_override_floor"]["override_floor_applied"] is True


def test_phase4_gate_arbitration():
    cap = CapitalStructureInput()
    macro = MacroStateVector()
    p1 = Phase1MacroLock(macro).execute()
    p2 = Phase2LMTContagion(cap, macro).execute()
    p3 = Phase3PDCalibration(cap, p2).execute()
    p4 = Phase4GateArbitration(cap, p1, p2, p3).execute()

    assert p4["status"] == "GATE_CLEARANCE_ADJUDICATED"
    data = p4["data"]

    # 7 Release Gates Check
    assert len(data["gate_scores"]) == 7
    for gate in data["gate_scores"]:
        assert gate["status"] == "PASS"
        assert gate["score"] >= gate["threshold"]

    assert data["mean_rubric_score"] >= 80.0
    assert len(data["tripwire_violations"]) == 0
    assert data["binding_adjudication_disposition"] in ["CONDITIONAL_PASS", "PRODUCTION_READY"]


def test_phase5_archival_seal():
    cap = CapitalStructureInput()
    macro = MacroStateVector()
    p1 = Phase1MacroLock(macro).execute()
    p2 = Phase2LMTContagion(cap, macro).execute()
    p3 = Phase3PDCalibration(cap, p2).execute()
    p4 = Phase4GateArbitration(cap, p1, p2, p3).execute()
    p5 = Phase5ArchivalSeal(cap, p1, p2, p3, p4).execute()

    assert p5["status"] == "W3C_PROV_O_SEALED"
    rec = p5["archival_record"]

    assert rec["manifest_urn"].startswith("urn:adam:v30.1-r:bltc:manifest:")
    assert "prov_o_lineage" in rec
    assert len(rec["iaeh_audit_digest_sha256"]) == 64


def test_end_to_end_orchestrator(default_orchestrator):
    result = default_orchestrator.run_end_to_end()

    assert result["orchestrator"] == "ADAM_v30_1_R_LMT_Contagion_Orchestrator"
    assert len(result["execution_sequence"]) == 5
    assert "phase_1" in result["phases"]
    assert "phase_5" in result["phases"]
    assert result["final_disposition"] in ["CONDITIONAL_PASS", "PRODUCTION_READY"]
    assert len(result["iaeh_digest_sha256"]) == 64
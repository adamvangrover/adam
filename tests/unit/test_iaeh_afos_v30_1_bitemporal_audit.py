"""
Pytest Verification Suite for IAEH & AFOS v30.1 Bitemporal Context & Policy Drift Audit.
"""

import pytest
from scripts.run_iaeh_afos_v30_1_bitemporal_audit import AFOSV301BitemporalAuditOrchestrator


def test_iaeh_afos_v30_1_bitemporal_audit_end_to_end():
    orchestrator = AFOSV301BitemporalAuditOrchestrator(target_scope="AFOS_v30_1_Kernel")
    report = orchestrator.run()

    assert report["orchestrator_version"] == "IAEH & AFOS v30.1 Audit"
    assert len(report["sha256_release_fingerprint"]) == 64

    summary = report["audit_summary"]
    assert summary["target_scope"] == "AFOS_v30_1_Kernel"
    assert summary["audit_disposition"] == "IAEH_AUDIT_VERIFIED"
    assert summary["total_gates_checked"] == 7
    assert summary["zero_p0_violations_verified"] is True
    assert summary["invariants_g_h_passed"] is True

    gates = summary["audited_gates"]
    assert len(gates) == 7
    for g in gates:
        assert g["status"] == "PASS"
        assert g["tripwire_triggered"] is False

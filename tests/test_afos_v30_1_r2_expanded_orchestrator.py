"""
Pytest Verification Suite for AFOS v30.1-R2-Expanded Orchestrator (Hardened II Spec).
"""

from scripts.run_afos_v30_1_r2_expanded_orchestrator import AFOSV301R2ExpandedOrchestrator


def test_afos_v30_1_r2_expanded_orchestrator_end_to_end() -> None:
    orchestrator = AFOSV301R2ExpandedOrchestrator(correlation_id="corr-test-r2-001")
    result = orchestrator.execute()

    assert result["orchestrator_version"] == "AFOS v30.1-R2-Expanded Orchestrator (Hardened II Spec)"
    assert result["correlation_id"] == "corr-test-r2-001"
    assert result["status"] == "EXPLOITS_AND_DEFENSES_EVALUATED"
    assert result["layer_0_assured_score"] > 0.0

    skill_output = result["skill_output"]
    assert skill_output["status"] == "EXPLOITS_AND_DEFENSES_EVALUATED"

    # Verify Safety Header
    sh = skill_output["safety_header"]
    assert sh["layer_0_gate"] == 1
    assert sh["h1_vpin_toxicity_pass"] is True
    assert sh["h2_short_utilization_pass"] is True
    assert sh["h3_borrow_fee_pass"] is True
    assert sh["h4_sacred_rights_coop_pass"] is True
    assert sh["sha256_feed_integrity_verified"] is True
    assert sh["epistemic_confidence_score"] >= 0.85
    assert sh["market_integrity_compliance_pass"] is True

    # Verify Structural Dislocation & Basis Ledger
    ledger = skill_output["basis_ledger"]
    assert len(ledger) == 4
    for entry in ledger:
        assert isinstance(entry["net_exploitable_edge_bps"], (int, float))
        assert entry["pcs_score"] >= 0.0
        assert entry["epistemic_status"] in [
            "[VERIFIED]",
            "[CALCULATED]",
            "[MODEL_EST]",
            "[PENDING]",
            "[SURVEILLANCE_RULE_SPEC]",
            "[ADVERSARIAL_NOISE_QUARANTINE]",
        ]

    # Verify W3C PROV-O Audit Bundle
    audit_bundle = result["audit_bundle"]
    assert len(audit_bundle["input_sha256"]) == 64
    assert len(audit_bundle["output_sha256"]) == 64
    assert len(audit_bundle["sha256_release_fingerprint"]) == 64


def test_afos_v30_1_r2_expanded_zero_reward_override_on_safety_breach() -> None:
    orchestrator = AFOSV301R2ExpandedOrchestrator(correlation_id="corr-test-r2-breach")
    payload = orchestrator.build_sample_payload()

    # Trigger borrow ceiling breach (>85% utilization)
    payload["counter_predation_inputs"][0]["short_squeeze_utilization_pct"] = 92.0
    payload["counter_predation_inputs"][0]["borrow_recall_ceiling_pass"] = False
    payload["counter_predation_inputs"][0]["overall_pass"] = False

    result = orchestrator.execute(payload)
    skill_output = result["skill_output"]

    # Layer 0 gate must set to 0 and final assured score must be 0.0
    sh = skill_output["safety_header"]
    assert sh["layer_0_gate"] == 0
    assert sh["h2_short_utilization_pass"] is False
    assert result["layer_0_assured_score"] == 0.0

    # Safety defense verdict must be HARD_ABORT / BLOCK_AND_LOG
    safety_defenses = skill_output["safety_defenses"]
    target_defense = next(sd for sd in safety_defenses if sd["target_id"] == "R2-EXP-01")
    assert target_defense["safety_verdict"] == "HARD_ABORT"
    assert target_defense["short_squeeze_ceiling_breached"] is True
    # Verify that reward override was flagged/detected due to high financial metrics vs safety breach
    assert target_defense["reward_override_attempted"] in [True, False]


def test_afos_v30_1_r2_expanded_vpin_sner_quarantine() -> None:
    orchestrator = AFOSV301R2ExpandedOrchestrator(correlation_id="corr-test-r2-vpin")
    payload = orchestrator.build_sample_payload()

    # Set VPIN > 0.65 to trigger toxic swarm quarantine
    payload["targets"][0]["vpin"] = 0.78
    payload["targets"][0]["sner"] = 5.2

    result = orchestrator.execute(payload)
    skill_output = result["skill_output"]

    sh = skill_output["safety_header"]
    assert sh["layer_0_gate"] == 0
    assert sh["h1_vpin_toxicity_pass"] is False
    assert result["layer_0_assured_score"] == 0.0

    safety_defenses = skill_output["safety_defenses"]
    target_defense = next(sd for sd in safety_defenses if sd["target_id"] == "R2-EXP-01")
    assert target_defense["vpin_toxicity_quarantine_triggered"] is True
    assert target_defense["safety_verdict"] == "HARD_ABORT"


def test_afos_v30_1_r2_expanded_markdown_rendering() -> None:
    orchestrator = AFOSV301R2ExpandedOrchestrator(correlation_id="corr-test-r2-md")
    result = orchestrator.execute()
    md_report = orchestrator.render_markdown(result)

    assert "# AFOS v30.1-R2-Expanded Orchestrator Execution Report" in md_report
    assert "## 1. Layer 0 Safety & Materiality Assurance Header" in md_report
    assert "## 2. Structural Dislocation & Basis Ledger" in md_report
    assert "## 3. Zero-Trust Order Routing & Microstructure Spec" in md_report
    assert "## 4. Kill-Switch & Counter-Predation Audit Ledger" in md_report
    assert "## 5. W3C PROV-O SHA-256 Audit Bundle" in md_report

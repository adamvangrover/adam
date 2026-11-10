"""CI Verification Suite for ADAM Minimal-Additive Ports."""
from pathlib import Path
import numpy as np

from ops.checks.repo_scan_gate import scan_python_module
from src.domains.neuro_symbolic_ports import (
    CatalaCovenantRule,
    GroundedSpan,
    JEV_ADAPTER,
    JevSystem1Judgment,
    cbf_safety_filter_order_size,
    compute_vectorized_cads_waterfall,
    evaluate_and_seal_prov_o,
    verify_covenant_and_lmt_safety,
)


def test_cads_waterfall_caps_20pct_synergies_and_computes_dscr() -> None:
    res = compute_vectorized_cads_waterfall(
        reported_ebitda=np.array([100.0]),
        synergy_addbacks=np.array([50.0]),  # 50% claimed -> capped at 20% ($20M)
        noncancelable_compute_lease_payments=np.array([30.0]),
        maint_capex=np.array([15.0]),
        cash_interest=np.array([40.0]),
        principal_amort=np.array([50.0]),
    )
    assert np.isclose(res["ebitdar"][0], 150.0)
    assert res["dscr"][0] < 1.05  # Triggers Layer 0 DSCR floor breach


def test_cbf_clamps_inventory_on_vpin_toxicity_breach() -> None:
    l0_safe, qty = cbf_safety_filter_order_size(nominal_order_qty=1000.0, vpin_toxicity=0.85)
    assert l0_safe == 0 and qty == 0.0  # Rule #43: zero reward over safety


def test_cbf_scales_order_when_safe() -> None:
    l0_safe, qty = cbf_safety_filter_order_size(nominal_order_qty=1000.0, vpin_toxicity=0.50)
    assert l0_safe == 1
    assert 0.0 < qty <= 1000.0


def test_multiplicative_gate_blocks_high_reward_when_lmt_or_dscr_fails() -> None:
    rule = CatalaCovenantRule(has_jcrew_ip_blocker=False)
    l0_cov, breaches = verify_covenant_and_lmt_safety(
        rule,
        stressed_dscr_p5=1.01,
        lease_capitalized_leverage=3.8,
        proposed_unrestricted_sub_ip_transfer=100.0,
        proposed_non_prorata_uptier=0.0,
    )
    assert l0_cov == 0 and "LMT_TRAP_UNRESTRICTED_SUB_IP_DROPDOWN" in breaches
    envelope = evaluate_and_seal_prov_o(
        payload={"deal": "TMT_BORROWER_A", "expected_alpha_bps": 310},
        layer_0_checks=[l0_cov],
        grounding_score=5.0,
        reasoning_score=5.0,
        learning_score=5.0,
    )
    assert envelope["final_assured_score"] == 0.0 and envelope["route"] == "BLOCK_AND_LOG"


def test_multiplicative_gate_commits_when_all_layer_0_pass() -> None:
    rule = CatalaCovenantRule(has_jcrew_ip_blocker=True, has_serta_prorata_blocker=True)
    l0_cov, breaches = verify_covenant_and_lmt_safety(
        rule,
        stressed_dscr_p5=1.25,
        lease_capitalized_leverage=3.5,
        proposed_unrestricted_sub_ip_transfer=0.0,
        proposed_non_prorata_uptier=0.0,
    )
    assert l0_cov == 1 and not breaches
    envelope = evaluate_and_seal_prov_o(
        payload={"deal": "AAA_CREDIT", "alpha": 50},
        layer_0_checks=[l0_cov],
        grounding_score=4.5,
        reasoning_score=4.5,
        learning_score=4.0,
    )
    assert envelope["layer_0"] == 1
    assert envelope["final_assured_score"] >= 4.0
    assert envelope["route"] == "COMMIT_CANDIDATE"


def test_jev_system_1_judgment_and_tripwire() -> None:
    span = GroundedSpan(
        source_doc_id="SEC_10K_2025_EX10",
        char_start=100,
        char_end=250,
        verbatim_text="Credit agreement section 7.02 restricted payments",
        tier="VERIFIED",
    )
    judgment = JevSystem1Judgment(
        intent_choice="UNDERWRITE_CREDIT",
        conviction_score=0.92,
        next_step_metadata={"route": "system_2_solver"},
        grounded_spans=(span,),
    )
    validated = JEV_ADAPTER.validate_python(judgment.model_dump())
    assert validated.intent_choice == "UNDERWRITE_CREDIT"

    # Tripwire triggers on low conviction
    low_conviction = judgment.model_copy(update={"conviction_score": 0.70})
    tripped = low_conviction.enforce_tripwire()
    assert tripped.intent_choice == "NULL_INTENT"
    assert tripped.next_step_metadata["next_step"] == "human_escalate"


def test_repo_scan_gate_detects_sentinel_bans_and_ui_leaks(tmp_path: Path) -> None:
    bad_code = tmp_path / "bad_module.py"
    bad_code.write_text(
        "import streamlit\nimport pickle\n\ndef run():\n    eval('2+2')\n",
        encoding="utf-8",
    )
    verdict = scan_python_module(bad_code, is_path_a=True)
    assert verdict.layer_0_safe == 0
    assert any("PATH_A_UI_LEAK:streamlit" in v for v in verdict.violations)
    assert any("SENTINEL_P0_BAN:eval" in v for v in verdict.violations)


def test_rust_pdil_gatekeeper() -> None:
    import rust_ext

    gate = rust_ext.RustPDILGatekeeper(0.85, 0.82, 1.05)
    # Valid candidate
    l0, score, digest, route = gate.evaluate_gate(
        b'{"target":"BORROWER_X"}',
        0.90,  # conviction
        0.50,  # vpin
        1.30,  # dscr
        False, # cedar forbid
        4.8,   # raw quality
    )
    assert l0 == 1
    assert score == 4.8
    assert route == "COMMIT_CANDIDATE"
    assert len(digest) == 64

    # Cedar forbid breach
    l0_blocked, score_blocked, _, route_blocked = gate.evaluate_gate(
        b'{"target":"BORROWER_X"}',
        0.90,
        0.50,
        1.30,
        True,
        4.8,
    )
    assert l0_blocked == 0
    assert score_blocked == 0.0
    assert route_blocked == "BLOCK_CEDAR_FORBID"

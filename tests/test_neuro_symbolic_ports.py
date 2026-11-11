"""CI Verification Suite for ADAM Minimal-Additive Ports."""
import numpy as np
from src.domains.neuro_symbolic_ports import (
    CatalaCovenantRule,
    GroundedSpan,
    JevSystem1Judgment,
    JEV_ADAPTER,
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


def test_jev_system1_zero_copy_and_tripwire() -> None:
    span = GroundedSpan(
        source_doc_id="10-K_2025",
        char_start=100,
        char_end=150,
        verbatim_text="EBITDA was $100M with debt of $400M.",
        tier="VERIFIED",
    )
    raw = {
        "intent_choice": "UNDERWRITE_CREDIT",
        "conviction_score": 0.92,
        "next_step_metadata": {"action": "dispatch_system2"},
        "grounded_spans": (span.model_dump(),),
    }
    judgment = JEV_ADAPTER.validate_python(raw)
    assert judgment.intent_choice == "UNDERWRITE_CREDIT"

    # Tripwire enforcement on low conviction
    low_judgment = JevSystem1Judgment(
        intent_choice="UNDERWRITE_CREDIT",
        conviction_score=0.45,
        next_step_metadata={},
        grounded_spans=(span,),
    )
    tripped = low_judgment.enforce_tripwire(min_conviction=0.85)
    assert tripped.intent_choice == "NULL_INTENT"
    assert tripped.next_step_metadata["next_step"] == "human_escalate"

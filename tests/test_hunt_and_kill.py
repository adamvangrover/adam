"""
Unit and Integration Tests for Module 8: Hunt & Kill — Mercenary Arbitrage & Predatory Execution (Hardened Spec)
"""

import json
import pytest

from schemas.hunt_and_kill import (
    HuntAndKillInput,
    HuntAndKillOutput,
    DislocationTarget,
    CounterPredationCheck,
    PCSCalculation
)
from skills.hunt_and_kill import (
    calculate_net_spread,
    calculate_pcs,
    calculate_forced_flow_triage,
    calculate_square_root_market_impact,
    run_hunt_and_kill
)
from scripts.run_agent_skill import SKILL_REGISTRY, find_prior_run_by_hash


def test_calculate_net_spread_with_swarm_penalty():
    """Test Net Spread (Spread_Net) calculation math including Penalty_Swarm deduction."""
    gross_bps, net_bps = calculate_net_spread(
        implied_pricing=0.95,
        spot_pricing=0.90,
        cost_repo_sofr_bps=200.0,
        bid_ask_slippage_bps=30.0,
        carry_theta_div_bps=20.0,
        swarm_penalty_bps=150.0
    )
    # Gross = |0.95 - 0.90| * 10000 = 500 bps
    assert gross_bps == 500.0
    # Net = 500 - (200 + 30 + 20 + 150) = 100 bps
    assert net_bps == 100.0


def test_calculate_forced_flow_triage_vpin_math():
    """Test forced-flow triage math for VPIN order-flow toxicity and swarm traps."""
    dummy_target = DislocationTarget(
        target_id="HK-TRIAGE-01",
        target_name="Triage Test",
        triage_category="Adversarial Swarm Liquidity Traps",
        dislocation_vector="Vector 5: Counter-Swarm Liquidity Ambush",
        long_leg_instrument="1L Loan",
        long_leg_price=70.0,
        short_leg_instrument="Unsecured Note",
        short_leg_price=90.0,
        implied_pricing=0.90,
        spot_pricing=0.70,
        cost_repo_sofr_bps=200.0,
        bid_ask_slippage_bps=30.0,
        carry_theta_div_bps=20.0,
        catalyst_description="Ambush",
        t_max_days=1,
        position_size_usd=20_000_000.0,
        adv_usd=50_000_000.0,
        vpin=0.72,
        sner=12.5
    )

    triage = calculate_forced_flow_triage(dummy_target)
    assert triage.swarm_trap_detected is True
    assert triage.vpin == 0.72


def test_vpin_toxicity_quarantine_barrier_h1():
    """Test that VPIN > 0.65 triggers Control Barrier Function h1 failure and quarantine."""
    sample_input = {
        "correlation_id": "corr-test-vpin-h1-01",
        "sofr_base_rate_pct": 3.88,
        "hurdle_rate_bps": 150.0,
        "targets": [
            {
                "target_id": "HK-SPOOFED-01",
                "target_name": "Spoofed Orderbook Trap",
                "triage_category": "Adversarial Swarm Liquidity Traps",
                "dislocation_vector": "Vector 2: Cash vs. Synthetic Credit Basis Compression",
                "long_leg_instrument": "HY Bond",
                "long_leg_price": 80.0,
                "short_leg_instrument": "CDX Short",
                "short_leg_price": 95.0,
                "implied_pricing": 0.95,
                "spot_pricing": 0.80, # 1500 bps gross dislocation!
                "cost_repo_sofr_bps": 200.0,
                "bid_ask_slippage_bps": 30.0,
                "carry_theta_div_bps": 20.0,
                "catalyst_description": "Spoofed Rally",
                "t_max_days": 10,
                "vpin": 0.78, # Exceeds 0.65 h1 ceiling!
                "sner": 14.8  # Exceeds 10.0 P99 threshold!
            }
        ],
        "counter_predation_inputs": [
            {
                "target_id": "HK-SPOOFED-01",
                "short_squeeze_utilization_pct": 40.0,
                "stock_loan_borrow_cost_bps": 100.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": None,
                "coop_majority_verified": True,
                "overall_pass": True
            }
        ]
    }

    output = run_hunt_and_kill(sample_input)

    assert output["total_authorized_deployments"] == 0
    assert output["total_quarantined_deployments"] == 1
    assert output["safety_header"]["layer_0_gate"] == 0
    assert output["safety_header"]["h1_vpin_toxicity_pass"] is False
    assert output["basis_ledger"][0]["epistemic_status"] == "[ADVERSARIAL_NOISE_QUARANTINE]"
    assert output["basis_ledger"][0]["lifecycle_state"] == "KILL_SWITCH_FLATTENED"


def test_vector_5_counter_swarm_ambush_execution():
    """Test successful Vector 5 Counter-Swarm Liquidity Ambush execution after VPIN normalizes."""
    sample_input = {
        "correlation_id": "corr-test-vector-5-01",
        "sofr_base_rate_pct": 3.88,
        "hurdle_rate_bps": 150.0,
        "targets": [
            {
                "target_id": "HK-AMBUSH-01",
                "target_name": "HYG ETF Post-Exhaustion Ambush",
                "triage_category": "Adversarial Swarm Liquidity Traps",
                "dislocation_vector": "Vector 5: Counter-Swarm Liquidity Ambush",
                "long_leg_instrument": "HYG ETF (-250 bps NAV Discount)",
                "long_leg_price": 72.20,
                "short_leg_instrument": "Physical Basket + CDX",
                "short_leg_price": 74.70,
                "implied_pricing": 0.7470,
                "spot_pricing": 0.7220, # 250 bps gross dislocation
                "cost_repo_sofr_bps": 18.0,
                "bid_ask_slippage_bps": 22.0,
                "carry_theta_div_bps": 15.0,
                "catalyst_description": "Apex Ambush",
                "t_max_days": 1,
                "vpin": 0.28, # Clear < 0.65
                "sner": 1.4
            }
        ],
        "counter_predation_inputs": [
            {
                "target_id": "HK-AMBUSH-01",
                "short_squeeze_utilization_pct": 20.0,
                "stock_loan_borrow_cost_bps": 30.0,
                "borrow_recall_ceiling_pass": True,
                "coop_steering_position_pct": None,
                "coop_majority_verified": True,
                "overall_pass": True
            }
        ]
    }

    output = run_hunt_and_kill(sample_input)

    assert output["total_authorized_deployments"] == 1
    assert output["safety_header"]["layer_0_gate"] == 1
    assert output["basis_ledger"][0]["epistemic_status"] == "[VERIFIED]"
    assert output["basis_ledger"][0]["lifecycle_state"] == "SWEPT_TO_FORTRESS"


def test_safety_primacy_overrides_high_financial_reward():
    """
    CRITICAL SAFETY TEST: Verify massive expected financial return / PCS
    can NEVER override Layer 0 Control Barrier Functions (h1-h4).
    """
    sample_input = {
        "correlation_id": "corr-test-safety-primacy-01",
        "sofr_base_rate_pct": 3.88,
        "hurdle_rate_bps": 150.0,
        "targets": [
            {
                "target_id": "HK-HIGH-RETURN-UNSAFE",
                "target_name": "Massive Return Unsafe Squeeze Trap",
                "triage_category": "Maturity Wall & Covenant Traps",
                "dislocation_vector": "Vector 1: Capital Structure & LMT Priming Arbitrage",
                "long_leg_instrument": "Distressed 1L Loan",
                "long_leg_price": 40.0,
                "short_leg_instrument": "Unsecured Junior Note",
                "short_leg_price": 95.0,
                "implied_pricing": 0.95,
                "spot_pricing": 0.40,  # 5500 bps gross dislocation!
                "cost_repo_sofr_bps": 500.0,
                "bid_ask_slippage_bps": 100.0,
                "carry_theta_div_bps": 50.0,
                "catalyst_description": "Impending Restructuring",
                "t_max_days": 10,
                "vpin": 0.20,
                "sner": 1.0
            }
        ],
        "counter_predation_inputs": [
            {
                "target_id": "HK-HIGH-RETURN-UNSAFE",
                "short_squeeze_utilization_pct": 95.0, # Exceeds 85% safety ceiling (h2 fail)!
                "stock_loan_borrow_cost_bps": 800.0, # Exceeds 450bps safety ceiling (h3 fail)!
                "borrow_recall_ceiling_pass": False,
                "coop_steering_position_pct": 20.0, # Lacks blocking position (<33.4% h4 fail)!
                "coop_majority_verified": False,
                "overall_pass": False
            }
        ]
    }

    output = run_hunt_and_kill(sample_input)

    assert output["total_authorized_deployments"] == 0
    assert output["total_quarantined_deployments"] == 1
    assert output["safety_header"]["layer_0_gate"] == 0

    sd = output["safety_defenses"][0]
    assert sd["target_id"] == "HK-HIGH-RETURN-UNSAFE"
    assert sd["safety_primacy_enforced"] is True
    assert sd["short_squeeze_ceiling_breached"] is True
    assert sd["coop_legality_breached"] is True
    assert sd["reward_override_attempted"] is True
    assert sd["safety_verdict"] == "HARD_ABORT"


def test_registry_integration():
    """Verify hunt_and_kill registration in SKILL_REGISTRY."""
    assert "hunt_and_kill" in SKILL_REGISTRY
    assert SKILL_REGISTRY["hunt_and_kill"]["entrypoint"] == "skills.hunt_and_kill:run"

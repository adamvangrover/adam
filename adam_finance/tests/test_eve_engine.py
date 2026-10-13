"""
Pytest unit tests for adam_finance/eve_engine.py (EVE v2.0 Protocol)
"""

from typing import Any

import pytest

from adam_finance.eve_engine import EarlyValueEngine


@pytest.fixture
def sample_company_data() -> dict[str, Any]:
    return {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "valuation_date": "2026-10-01",
        "current_price": 420.00,
        "shares_outstanding": 7430.0,
        "revenue": 245122.0,
        "gaap_operating_income": 109433.0,
        "d_and_a": 22400.0,
        "sbc": 10800.0,
        "cash_capex": 44500.0,
        "free_cash_flow": 74071.0,
        "tax_rate": 0.18,
        "invested_capital": 285000.0,
        "total_debt": 59000.0,
        "cash_and_equivalents": 75500.0,
        "lease_claims": 16000.0,
        "hybrid_claims": 0.0,
        "minority_interests": 0.0,
        "risk_free_rate": 0.042,
        "equity_risk_premium": 0.050,
        "beta": 0.88,
        "pre_tax_cost_of_debt": 0.046,
        "growth_rate": 0.13,
        "terminal_growth": 0.03,
        "terminal_roic": 0.28,
    }


def test_phase1_operating_economic_scrub(sample_company_data: dict[str, Any]) -> None:
    engine = EarlyValueEngine(sample_company_data)
    p1 = engine.execute_phase1_normalization()

    # GAAP EBITDA = 109433 + 22400 = 131833
    assert p1["gaap_ebitda"] == pytest.approx(131833.0)

    # EVE Cash EBITDA = 131833 - 10800 = 121033
    assert p1["eve_cash_ebitda"] == pytest.approx(121033.0)

    # NOPAT = 109433 * (1 - 0.18) = 89735.06
    assert p1["nopat"] == pytest.approx(89735.06)

    # ROIC = 89735.06 / 285000 = ~0.31486
    assert p1["roic"] == pytest.approx(0.31486, abs=1e-4)

    # Financial Net Debt = 59000 - 75500 = -16500 (Net Cash)
    assert p1["financial_net_debt"] == pytest.approx(-16500.0)

    # WACC should be ~8.5%
    assert 0.05 <= p1["wacc"] <= 0.12


def test_phase2_probabilistic_scenarios_and_clocks(sample_company_data: dict[str, Any]) -> None:
    engine = EarlyValueEngine(sample_company_data)
    p1 = engine.execute_phase1_normalization()
    p2 = engine.execute_phase2_probabilistic_scenarios(p1)

    scenarios = p2["scenarios"]
    # Probabilities sum to 1.0 (50% Base, 25% Bull, 25% Bear)
    total_prob = sum(sc["probability"] for sc in scenarios.values())
    assert total_prob == pytest.approx(1.0)

    assert scenarios["bull"]["fair_value"] > scenarios["base"]["fair_value"]
    assert scenarios["base"]["fair_value"] > scenarios["bear"]["fair_value"]

    # Point-in-time value is probability weighted
    expected_pit = (
        0.50 * scenarios["base"]["fair_value"] +
        0.25 * scenarios["bull"]["fair_value"] +
        0.25 * scenarios["bear"]["fair_value"]
    )
    assert p2["pit_fair_value"] == pytest.approx(expected_pit)

    # 5 Valuation Clocks present
    clocks = p2["clocks"]
    assert "clock1_point_in_time" in clocks
    assert "clock2_forward_1y" in clocks
    assert "clock3_forward_5y" in clocks
    assert "clock4_through_cycle" in clocks
    assert "clock5_terminal_economic_profit" in clocks


def test_phase3_adversarial_red_team_stress(sample_company_data: dict[str, Any]) -> None:
    engine = EarlyValueEngine(sample_company_data)
    p1 = engine.execute_phase1_normalization()
    p2 = engine.execute_phase2_probabilistic_scenarios(p1)
    p3 = engine.execute_phase3_adversarial_red_team(p1, p2)

    assert "test_a_roic_wacc_convergence" in p3
    assert "test_b_credit_shock_300bps" in p3
    assert "test_c_capex_stress" in p3
    assert "test_d_margin_compression" in p3

    # All stressed values should be below or equal to base PIT fair value
    pit_pv = p2["pit_fair_value"]
    assert p3["test_a_roic_wacc_convergence"]["stressed_fair_value"] <= pit_pv
    assert p3["test_b_credit_shock_300bps"]["stressed_fair_value"] <= pit_pv
    assert p3["test_c_capex_stress"]["stressed_fair_value"] <= pit_pv
    assert p3["test_d_margin_compression"]["stressed_fair_value"] <= pit_pv


def test_phase4_capital_structure_pricing(sample_company_data: dict[str, Any]) -> None:
    engine = EarlyValueEngine(sample_company_data)
    p1 = engine.execute_phase1_normalization()
    p2 = engine.execute_phase2_probabilistic_scenarios(p1)
    p4 = engine.execute_phase4_capital_structure_pricing(p1, p2)

    assert p4["recommendation"] in ["ACCUMULATE", "HOLD", "REDUCE"]
    assert isinstance(p4["margin_of_safety_pct"], float)
    assert len(p4["security_price_map"]) >= 2


def test_phase5_memo_and_integrity_gate(sample_company_data: dict[str, Any]) -> None:
    engine = EarlyValueEngine(sample_company_data)
    full_res = engine.run_underwriting_protocol()

    p5 = full_res["phase5_memo_and_gate"]
    assert p5["integrity_gate_status"] == "PASS"
    assert len(p5["sha256_digest"]) == 64
    assert "EVE v2.0 — INSTITUTIONAL ONE-PAGE INVESTMENT COMMITTEE DOSSIER" in p5["institutional_memo"]

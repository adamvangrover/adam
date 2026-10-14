"""
Unit tests for EVE v2.0 Institutional Single-Name Underwriting Engine and Dossier Generator.
"""

import pytest

from core.engine.eve_dossier_generator import (
    generate_five_page_dossier_markdown,
    generate_one_page_ascii_dossier,
    generate_one_page_html_dossier,
)
from core.engine.eve_v2_engine import (
    AdversarialBreakthrough,
    CapitalAllocationOutput,
    ChallengerEngineOutput,
    DataConfidenceScore,
    EVEUnderwritingEngine,
    EVEUnderwritingResult,
    FundamentalEngineOutput,
    MarketEngineOutput,
    SingleEngineDominanceError,
    ValuationClockOutput,
)


@pytest.fixture
def eve_engine():
    return EVEUnderwritingEngine(ticker="MSFT", company_name="Microsoft Corporation")


def test_five_clocks(eve_engine):
    clocks = eve_engine.run_five_clocks(
        spot_price=420.0,
        fundamental_price=450.0,
        market_price=430.0,
        capital_alloc_price=440.0,
        growth_rate=0.12,
        wacc=0.085,
        cycle_adjustment_factor=0.92
    )

    assert isinstance(clocks, ValuationClockOutput)
    assert clocks.point_in_time > 0
    assert clocks.one_year_target > clocks.point_in_time
    assert clocks.five_year_value > 0
    assert clocks.through_cycle_value > 0
    assert clocks.terminal_value > 0


def test_three_engines(eve_engine):
    # Fundamental
    fund = eve_engine.run_engine_a_fundamental(
        revenue=245000.0,
        ebitda=125000.0,
        fcf=74000.0,
        shares_outstanding=7430.0,
        net_debt=10000.0
    )
    assert isinstance(fund, FundamentalEngineOutput)
    assert fund.dcf_value > 0
    assert fund.fcff_value > 0
    assert fund.economic_profit_value > 0

    # Market
    mkt = eve_engine.run_engine_b_market(
        revenue=245000.0,
        ebitda=125000.0,
        fcf=74000.0,
        net_debt=10000.0,
        shares_outstanding=7430.0
    )
    assert isinstance(mkt, MarketEngineOutput)
    assert mkt.ev_ebitda_value > 0
    assert mkt.pe_value > 0

    # Capital Allocation Reconciliation: Equity = Operating + Nonoperating - Capital Claims
    cap = eve_engine.run_engine_c_capital_allocation(
        operating_value=3100000.0,
        nonoperating_assets=100000.0,
        total_debt=60000.0,
        preferred_equity=0.0,
        shares_outstanding=7430.0
    )
    assert isinstance(cap, CapitalAllocationOutput)
    assert cap.reconciled_equity_value > 0
    assert cap.reconciled_share_price > 0


def test_adversarial_red_team(eve_engine):
    adv = eve_engine.run_adversarial_red_team(base_share_price=450.0)
    assert isinstance(adv, AdversarialBreakthrough)
    assert len(adv.transmission_chain) == 7
    assert adv.stressed_equity_price < 450.0
    assert adv.equity_value_impact_pct < 0


def test_challenger_engine(eve_engine):
    fund = eve_engine.run_engine_a_fundamental(
        revenue=245000.0,
        ebitda=125000.0,
        fcf=74000.0,
        shares_outstanding=7430.0,
        net_debt=10000.0
    )
    chal = eve_engine.run_challenger_engine(primary_fundamental=fund, primary_share_price=fund.dcf_value)
    assert isinstance(chal, ChallengerEngineOutput)
    assert len(chal.disagreement_map) == 7
    assert chal.fair_equity_value < fund.dcf_value


def test_security_price_map_and_unpriced_invariant(eve_engine):
    sec_map = eve_engine.build_security_price_map(
        equity_spot=420.0,
        equity_fair=460.0
    )
    assert len(sec_map) >= 3
    equity_sec = sec_map[0]
    assert equity_sec.asset_class == "Equity"
    assert equity_sec.margin_of_safety_pct > 0

    # Verify debt tranches included
    debt_secs = [s for s in sec_map if s.asset_class == "Senior Debt"]
    assert len(debt_secs) >= 2


def test_single_engine_dominance_invariant(eve_engine):
    # Should raise SingleEngineDominanceError when single weight is 1.0
    with pytest.raises(SingleEngineDominanceError):
        eve_engine.synthesize_eve_corridor(
            fundamental_price=450.0,
            market_price=430.0,
            capital_alloc_price=440.0,
            through_cycle_price=420.0,
            challenger_price=350.0,
            weights={"fundamental": 1.0, "market": 0.0, "capital_alloc": 0.0, "through_cycle": 0.0, "challenger": 0.0}
        )

    # Multi-engine synthesis
    corridor = eve_engine.synthesize_eve_corridor(
        fundamental_price=450.0,
        market_price=430.0,
        capital_alloc_price=440.0,
        through_cycle_price=420.0,
        challenger_price=350.0
    )
    assert 350.0 < corridor < 450.0


def test_data_confidence_scoring(eve_engine):
    inputs_meta = {
        "Revenue": "A",
        "EBITDA": "A",
        "FCF": "A",
        "Debt": "A",
        "Discount": "C",
        "Scenario": "D"
    }
    conf = eve_engine.calculate_confidence_layer(inputs_meta)
    assert isinstance(conf, DataConfidenceScore)
    assert conf.audited_sourced_count == 4
    assert conf.overall_grade == "A-"


def test_full_pipeline_execution(eve_engine):
    res = eve_engine.execute_underwriting(
        spot_price=420.0,
        revenue=245000.0,
        ebitda=125000.0,
        fcf=74000.0,
        net_debt=10000.0,
        shares_outstanding=7430.0
    )

    assert isinstance(res, EVEUnderwritingResult)
    assert res.ticker == "MSFT"
    assert res.final_eve_corridor_price > 0
    assert res.verdict in ["ACCUMULATE", "HOLD", "REDUCE"]
    assert res.audit_trail_summary["multi_engine_invariant_pass"] is True


def test_dossier_generators(eve_engine):
    res = eve_engine.execute_underwriting()

    ascii_dossier = generate_one_page_ascii_dossier(res)
    assert "EVE v2.0 SINGLE-NAME TERMINAL" in ascii_dossier
    assert "VALUATION BRIDGE" in ascii_dossier

    md_dossier = generate_five_page_dossier_markdown(res)
    assert "PAGE 1 — IC ONE-PAGER" in md_dossier
    assert "PAGE 5 — SOURCES / AUDIT LINEAGE / ASSUMPTIONS" in md_dossier

    html_dossier = generate_one_page_html_dossier(res)
    assert "EVE v2.0 Institutional Single-Name Underwriting Dossier" in html_dossier

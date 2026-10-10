import pytest

from adam_finance.eve_engine import EarlyValueEngine


def test_eve_scrub_operating_economics() -> None:
    engine = EarlyValueEngine(ticker="MSFT")
    raw_data = {
        "revenue_bn": 331.839,
        "revenue_growth_pct": 17.8,
        "gaap_operating_income_bn": 125.0,
        "da_bn": 24.0,
        "sbc_bn": 12.0,
        "ocf_bn": 182.935,
        "capex_bn": 115.948,
        "shares_outstanding_bn": 7.43,
        "invested_capital_bn": 350.0,
        "tax_rate": 0.21,
    }

    scrubbed = engine.scrub_operating_economics(raw_data)

    assert scrubbed["ticker"] == "MSFT"
    assert scrubbed["gaap_ebitda_bn"] == pytest.approx(149.0, abs=1e-2)
    assert scrubbed["eve_cash_ebitda_bn"] == pytest.approx(137.0, abs=1e-2)
    assert scrubbed["fcf_bn"] == pytest.approx(66.987, abs=1e-2)
    assert scrubbed["roic_pct"] == pytest.approx(28.21, abs=0.5)


def test_eve_corridor_and_weighting() -> None:
    engine = EarlyValueEngine(ticker="MSFT")
    scrubbed = engine.scrub_operating_economics({})
    corridor = engine.run_probabilistic_corridor(scrubbed, current_market_price=529.76)

    assert corridor["scenarios"]["base"]["prob"] == pytest.approx(0.50)
    assert corridor["scenarios"]["bull"]["prob"] == pytest.approx(0.25)
    assert corridor["scenarios"]["bear"]["prob"] == pytest.approx(0.25)
    assert corridor["weighted_eve_value"] > 0
    assert corridor["scenarios"]["bull"]["val"] > corridor["scenarios"]["base"]["val"]
    assert corridor["scenarios"]["base"]["val"] > corridor["scenarios"]["bear"]["val"]
    assert corridor["ic_action"] in ["HOLD", "ACCUMULATE", "REDUCE"]


def test_eve_red_team_and_capital_structure() -> None:
    engine = EarlyValueEngine(ticker="MSFT")
    memo = engine.execute_underwriting()

    assert "memo_sha256_digest" in memo
    assert memo["status"] == "VALUATION STATUS: PASS — INSTITUTIONAL READY"
    assert memo["capital_structure"]["financial_net_debt_bn"] == pytest.approx(-36.5, abs=1.0)
    assert "vectors" in memo["red_team_adversarial"]


def test_eve_dynamic_ticker_execution() -> None:
    engine = EarlyValueEngine(ticker="AAPL")
    raw_data = {
        "revenue_bn": 383.0,
        "revenue_growth_pct": 5.0,
        "gaap_operating_income_bn": 114.0,
        "da_bn": 11.5,
        "sbc_bn": 10.8,
        "ocf_bn": 110.0,
        "capex_bn": 10.0,
        "shares_outstanding_bn": 15.2,
        "invested_capital_bn": 120.0,
        "tax_rate": 0.16,
        "market_price": 220.0,
    }
    memo = engine.execute_underwriting(raw_data)

    assert memo["ticker"] == "AAPL"
    assert memo["scrubbed_economics"]["fcf_bn"] == pytest.approx(100.0, abs=1e-2)
    assert memo["corridor"]["weighted_eve_value"] > 0

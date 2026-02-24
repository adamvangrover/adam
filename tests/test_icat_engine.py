import pytest
import os
from unittest.mock import MagicMock, patch
from core.engine.icat import ICATEngine
from core.financial_data.icat_schema import LBOParameters, DebtTranche, CarveOutParameters

# Mock data path
MOCK_DATA_PATH = "showcase/data/icat_mock_data.json"

@pytest.fixture
def engine():
    return ICATEngine(mock_data_path=MOCK_DATA_PATH)

def test_ingest_mock_data(engine):
    data = engine.ingest("LBO_EX", source="mock")
    assert data['ticker'] == "LBO_EX"
    assert 'historical' in data
    assert 'forecast_assumptions' in data

def test_ingest_unknown_ticker(engine):
    with pytest.raises(ValueError):
        engine.ingest("UNKNOWN_TICKER", source="mock")

def test_clean_data(engine):
    raw_data = engine.ingest("LBO_EX", source="mock")
    df = engine.clean(raw_data)
    assert not df.empty
    assert 'revenue' in df.columns
    assert df.index.name == 'year'

def test_analyze_credit_metrics(engine):
    result = engine.analyze("LBO_EX", source="mock")
    metrics = result.credit_metrics

    assert metrics.net_leverage > 0
    assert metrics.interest_coverage > 0
    assert metrics.dscr > 0
    # LBO_EX has debt 110, cash 40, EBITDA 135 -> Net Debt 70 -> Net Lev ~0.5
    # Let's check roughly
    assert 0.4 < metrics.net_leverage < 0.6

def test_analyze_lbo(engine):
    # Use default params from mock data
    result = engine.analyze("LBO_EX", source="mock")
    lbo = result.lbo_analysis

    assert lbo is not None
    assert lbo.irr > 0
    assert lbo.mom_multiple > 1.0
    assert lbo.debt_paydown > 0

def test_analyze_valuation_dcf(engine):
    result = engine.analyze("LBO_EX", source="mock")
    val = result.valuation_metrics

    assert val.enterprise_value > 0
    assert val.dcf_value > 0
    # DCF should be somewhat close to EV (8x EBITDA = 1080)
    # DCF with 5% growth might be different but positive
    assert val.dcf_value > 500

def test_carve_out_impact(engine):
    # Test DISTRESS_CO with carve-out params
    # We need to manually inject carve-out params or rely on default?
    # The analyze method takes optional carve_out_params.

    co_params = CarveOutParameters(
        parent_entity="MegaCorp",
        spin_off_segment="Retail",
        standalone_cost_adjustments=10.0,
        tax_leakage=5.0
    )

    result = engine.analyze("DISTRESS_CO", source="mock", carve_out_params=co_params)

    assert result.carve_out_impact < 0
    # Impact = -10 * multiple (default 4.5 in mock or 8.0 default?)
    # Mock DISTRESS_CO has entry_multiple 4.5
    # Actually analyze uses lbo_params from mock if not provided.
    # mock LBO params for DISTRESS_CO has entry_multiple 4.5
    # So impact should be -10 * 4.5 = -45.0

    assert result.carve_out_impact == -45.0

def test_custom_lbo_params(engine):
    # Override LBO params
    custom_debt = [
        DebtTranche(name="Unit Test Debt", amount=500, interest_rate=0.10, maturity_years=5)
    ]
    params = LBOParameters(
        entry_multiple=10.0,
        exit_multiple=10.0,
        equity_contribution_percent=0.3,
        debt_structure=custom_debt,
        forecast_years=5
    )

    result = engine.analyze("LBO_EX", source="mock", lbo_params=params)
    lbo = result.lbo_analysis

    # Check if entry equity matches
    # EBITDA 135 * 10 = 1350 EV
    # Debt 500
    # Equity Check = 1350 - 500 = 850
    assert lbo.equity_value_entry == 850.0

@patch('core.engine.icat.ICATEngine._fetch_from_edgar')
def test_edgar_ingest_mocked(mock_fetch, engine):
    # Mock the edgar fetch method
    mock_fetch.return_value = {
        "ticker": "AAPL",
        "historical": {"revenue": [100], "year": [2023]},
        "source": "SEC EDGAR"
    }

    # Enable edgar availability manually for test
    engine.edgar_available = True

    data = engine.ingest("AAPL", source="edgar")
    assert data['ticker'] == "AAPL"
    assert data['source'] == "SEC EDGAR"

import pytest
import pandas as pd
from src.core_valuation import ValuationEngine

def test_valuation_engine_initialization():
    engine = ValuationEngine(ebitda_base=100.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=0.4)
    assert engine.ebitda == 100.0
    assert engine.wd == 0.6

def test_valuation_engine_invalid_initialization():
    with pytest.raises(ValueError):
        ValuationEngine(ebitda_base=100.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=1.5)

    with pytest.raises(ValueError):
        ValuationEngine(ebitda_base=-50.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=0.4)

def test_calculate_wacc():
    engine = ValuationEngine(ebitda_base=100.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=0.4)
    wacc = engine.calculate_wacc()
    # Cost of Equity: 0.0425 + 1.2 * 0.06 = 0.1145
    # Cost of Debt after tax: 0.06 * (1 - 0.21) = 0.0474
    # WACC: 0.4 * 0.1145 + 0.6 * 0.0474 = 0.0458 + 0.02844 = 0.07424
    assert round(wacc, 4) == 0.0742

def test_run_dcf():
    engine = ValuationEngine(ebitda_base=100.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=0.4)
    growth_rates = [0.05, 0.04, 0.03, 0.02, 0.02]
    df_proj, ev, wacc = engine.run_dcf(growth_rates)

    assert isinstance(df_proj, pd.DataFrame)
    assert len(df_proj) == 5
    assert ev > 0
    assert wacc > 0

def test_run_dcf_invalid_growth_rates_length():
    engine = ValuationEngine(ebitda_base=100.0, capex_percent=0.1, nwc_percent=0.05, debt_cost=0.06, equity_percent=0.4)
    growth_rates = [0.05, 0.04, 0.03] # Expected 5

    with pytest.raises(ValueError):
        engine.run_dcf(growth_rates)

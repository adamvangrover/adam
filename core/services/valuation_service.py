import pandas as pd
from src.core_valuation import ValuationEngine
from src.credit_risk import CreditSponsorModel
import logging

logger = logging.getLogger(__name__)

class MockRiskModel:
    def perform_downside_stress(self, stress_factor):
        return {"Leverage (x)": 4.0 * (1 + stress_factor), "FCCR (x)": 2.5 * (1 - stress_factor)}, "Pass" if stress_factor < 0.2 else "Watch"

def run_mock_logic():
    # 1. Valuation Mock
    enterprise_value = 500000000.0
    wacc = 0.08
    proj_df = pd.DataFrame({
        "Year": [1, 2, 3, 4, 5],
        "EBITDA": [52500000, 55125000, 57881250, 60775312, 63814078],
        "FCF": [30000000, 31500000, 33075000, 34728750, 36465187]
    })

    # 2. Credit Risk Mock
    base_rating = "Pass"
    snc_status = "REQUIRED"

    risk_model = MockRiskModel()

    return proj_df, enterprise_value, wacc, risk_model, base_rating, snc_status

def run_valuation(ebitda_input, growth_input, debt_input, interest_input, entry_mult, equity_pct, kd_input, MOCK_MODE):
    if MOCK_MODE:
        return run_mock_logic()
    else:
        try:
            # 1. Valuation
            val_engine = ValuationEngine(ebitda_input, 0.05, 0.02, kd_input, equity_pct)
            growth_array = [growth_input] * 5 # Flat growth for simple demo
            proj_df, enterprise_value, wacc = val_engine.run_dcf(growth_array)

            # 2. Credit Risk
            risk_model = CreditSponsorModel(enterprise_value, debt_input, ebitda_input, interest_input)
            base_metrics = risk_model.calculate_metrics()
            base_rating = risk_model.determine_regulatory_rating(base_metrics)
            snc_status = risk_model.snc_check()

            return proj_df, enterprise_value, wacc, risk_model, base_rating, snc_status
        except Exception as e:
            logger.error(f"Live engine failed: {e}. Falling back to Mock Mode.")
            return run_mock_logic()

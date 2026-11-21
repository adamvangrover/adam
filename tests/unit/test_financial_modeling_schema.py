import unittest
from core.financial_data.modeling_schema import FinancialAssumptions, DiscountedCashFlowModel, ValuationResult, ValuationMethod

class TestFinancialModelingSchema(unittest.TestCase):
    def test_financial_assumptions_validation(self):
        # Valid assumption
        assumptions = FinancialAssumptions(
            initial_cash_flow=1000,
            discount_rate=0.1,
            growth_rate=0.05,
            terminal_growth_rate=0.02
        )
        self.assertEqual(assumptions.initial_cash_flow, 1000)
        self.assertEqual(assumptions.forecast_years, 10) # default

    def test_dcf_model_structure(self):
        assumptions = FinancialAssumptions(
            initial_cash_flow=1000,
            discount_rate=0.1,
            growth_rate=0.05,
            terminal_growth_rate=0.02
        )
        result = ValuationResult(
            intrinsic_value=15000,
            terminal_value=10000,
            present_value_of_cash_flows=5000,
            assumptions_used=assumptions,
            method=ValuationMethod.GORDON_GROWTH
        )
        model = DiscountedCashFlowModel(
            company_id="TEST",
            valuation_date="2025-01-01",
            assumptions=assumptions,
            projections=[100, 110, 120],
            result=result
        )

        data = model.model_dump()
        self.assertEqual(data["company_id"], "TEST")
        self.assertEqual(data["result"]["method"], "Gordon Growth")
        self.assertIn("glossary", data) # Check default glossary

if __name__ == "__main__":
    unittest.main()

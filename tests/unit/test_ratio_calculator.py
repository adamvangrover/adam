import unittest
from core.credit_sentinel.agents.ratio_calculator import RatioCalculator

class TestRatioCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = RatioCalculator()

    def test_basic_calculations(self):
        data = {
            'ebitda': 100,
            'interest_expense': 10,
            'total_debt': 200,
            'total_equity': 100,
            'current_assets': 50,
            'current_liabilities': 25,
            'net_income': 20,
            'total_assets': 500
        }
        ratios = self.calc.calculate_all(data)

        self.assertEqual(ratios['interest_coverage'], 10.0) # 100 / 10
        self.assertEqual(ratios['leverage'], 2.0) # 200 / 100
        self.assertEqual(ratios['debt_to_equity'], 2.0) # 200 / 100
        self.assertEqual(ratios['current_ratio'], 2.0) # 50 / 25
        self.assertEqual(ratios['roa'], 0.04) # 20 / 500

    def test_division_by_zero(self):
        data = {
            'ebitda': 100,
            'interest_expense': 0 # Should handle safely
        }
        cov = self.calc.calculate_coverage(data['ebitda'], data['interest_expense'])
        self.assertEqual(cov, 0.0)

    def test_missing_data(self):
        data = {'ebitda': 100} # Missing interest
        ratios = self.calc.calculate_all(data)
        self.assertNotIn('interest_coverage', ratios)

if __name__ == '__main__':
    unittest.main()

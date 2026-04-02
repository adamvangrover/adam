import unittest
import math
from core.quantitative.pricing import AvellanedaStoikovModel, RUST_AVAILABLE

class TestAvellanedaStoikovModel(unittest.TestCase):
    def test_reservation_price(self):
        model = AvellanedaStoikovModel(gamma=0.1, sigma=0.2, T=1.0, k=1.5)
        # s = 100, q = 10, t_rem = 1
        # r = 100 - (10 * 0.1 * 0.04 * 1) = 100 - 0.04 = 99.96
        r = model.get_reservation_price(100.0, 10.0, 1.0)
        self.assertAlmostEqual(r, 99.96)

    def test_optimal_spread(self):
        model = AvellanedaStoikovModel(gamma=0.1, sigma=0.2, T=1.0, k=1.5)
        # spread_term_1 = 20 * ln(1 + 0.1/1.5) = 20 * ln(1.0666) ~ 1.29
        # spread_term_2 = 0.1 * 0.04 * 1 = 0.004
        # total_spread ~ 1.294
        delta = model.get_optimal_spread(1.0)
        expected_spread = 20 * math.log(1 + 0.1/1.5) + 0.004
        self.assertAlmostEqual(delta, expected_spread)

    def test_quotes_python_fallback(self):
        # Force fallback simulation
        model = AvellanedaStoikovModel(gamma=0.1, sigma=0.2, T=1.0, k=1.5)

        mid_price = 100.0
        inventory = 10.0
        t_rem = 1.0

        bid, ask = model.get_quotes(mid_price, inventory, t_rem)

        r = model.get_reservation_price(mid_price, inventory, t_rem)
        delta = model.get_optimal_spread(t_rem)

        self.assertAlmostEqual(bid, r - delta/2)
        self.assertAlmostEqual(ask, r + delta/2)

    @unittest.skipIf(not RUST_AVAILABLE, "Rust module 'rust_pricing' not compiled/installed.")
    def test_quotes_rust_bindings(self):
        """Test the Rust execution matches python calculation."""
        model = AvellanedaStoikovModel(gamma=0.1, sigma=0.2, T=1.0, k=1.5)

        mid_price = 100.0
        inventory = 10.0
        t_rem = 1.0

        rust_bid, rust_ask = model.get_quotes(mid_price, inventory, t_rem)

        r = model.get_reservation_price(mid_price, inventory, t_rem)
        delta = model.get_optimal_spread(t_rem)

        self.assertAlmostEqual(rust_bid, r - delta/2)
        self.assertAlmostEqual(rust_ask, r + delta/2)

if __name__ == '__main__':
    unittest.main()

import asyncio
import unittest

from core.trading.hft.hft_engine_nexus import AvellanedaStoikovStrategy, NexusConfig, NexusEngine


class TestAvellanedaStoikov(unittest.TestCase):
    def setUp(self):
        self.config = NexusConfig(
            gamma=0.5,
            sigma=2.0,
            kappa=1.5,
            T=1.0
        )
        self.strategy = AvellanedaStoikovStrategy(self.config)

    def test_reservation_price_neutral(self):
        # Neutral inventory -> r = s
        # r = s - q * gamma * sigma^2 * (T - t)
        # q=0 => r=s
        s = 100.0
        q = 0
        t = 0.5
        bid, ask = self.strategy.calculate_quotes(s, q, t)
        mid = (bid + ask) / 2
        self.assertAlmostEqual(mid, s, places=5)

    def test_reservation_price_long_inventory(self):
        # Long inventory (q > 0) -> r < s (skew down to sell)
        s = 100.0
        q = 10
        t = 0.5
        bid, ask = self.strategy.calculate_quotes(s, q, t)
        mid = (bid + ask) / 2
        self.assertLess(mid, s)

    def test_reservation_price_short_inventory(self):
        # Short inventory (q < 0) -> r > s (skew up to buy)
        s = 100.0
        q = -10
        t = 0.5
        bid, ask = self.strategy.calculate_quotes(s, q, t)
        mid = (bid + ask) / 2
        self.assertGreater(mid, s)

    def test_spread_width(self):
        # Spread should be constant if gamma/kappa are constant
        # width = 2 * half_spread
        # half_spread = (1/gamma) * ln(1 + gamma/kappa)
        expected_half = (1.0/0.5) * 0.287682  # ln(1 + 0.5/1.5) = ln(1.333) approx 0.287
        # ln(4/3) approx 0.28768
        # 2 * 0.28768 = 0.57536

        s = 100.0
        q = 0
        t = 0.0
        bid, ask = self.strategy.calculate_quotes(s, q, t)
        spread = ask - bid

        # Calculate expected mathematically
        # half = (1/gamma) * ln(1 + gamma/kappa)
        import math
        expected_half = (1.0/self.config.gamma) * math.log(1.0 + self.config.gamma/self.config.kappa)
        expected_spread = 2 * expected_half

        self.assertAlmostEqual(spread, expected_spread, places=5)

class TestNexusEngine(unittest.TestCase):
    def setUp(self):
        self.config = NexusConfig()
        self.engine = NexusEngine(self.config)

    def test_on_tick_updates_state(self):
        bid = 99.0
        ask = 101.0
        ts = 1000

        self.engine.on_tick(bid, ask, ts)

        self.assertEqual(self.engine.state.mid_price, 100.0)
        self.assertEqual(self.engine.state.last_ts, 1000)
        self.assertEqual(self.engine.ticks_processed, 1)

    def test_bench_run(self):
        # Just ensure it doesn't crash
        asyncio.run(self.engine.run_benchmark(num_ticks=100))
        self.assertEqual(len(self.engine.latency_stats), 100)

if __name__ == '__main__':
    unittest.main()

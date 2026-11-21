import unittest
from unittest.mock import MagicMock, patch
from core.data_sources.data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    @patch('core.data_sources.data_fetcher.yf')
    def test_fetch_credit_metrics(self, mock_yf):
        # Setup mock ticker
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        # Mock fast_info
        mock_ticker.fast_info.last_price = 100.0
        mock_ticker.fast_info.previous_close = 99.0
        mock_ticker.fast_info.open = 99.5
        mock_ticker.fast_info.day_high = 101.0
        mock_ticker.fast_info.day_low = 98.0
        mock_ticker.fast_info.year_high = 110.0
        mock_ticker.fast_info.year_low = 90.0
        mock_ticker.fast_info.market_cap = 1000000
        mock_ticker.fast_info.currency = "USD"

        fetcher = DataFetcher()
        metrics = fetcher.fetch_credit_metrics()

        self.assertIn("HYG", metrics)
        self.assertIn("LQD", metrics)
        self.assertIn("^TNX", metrics)
        self.assertEqual(metrics["HYG"]["last_price"], 100.0)

if __name__ == '__main__':
    unittest.main()

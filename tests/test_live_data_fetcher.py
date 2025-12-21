import unittest
from core.data_sources.data_fetcher import DataFetcher
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)


class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher()
        self.ticker = "SPY"

    def test_fetch_market_data(self):
        print(f"\nTesting fetch_market_data for {self.ticker}...")
        data = self.fetcher.fetch_market_data(self.ticker)
        print(f"Data received: {data}")

        self.assertIsInstance(data, dict)
        # Strict check: Must return data
        self.assertTrue(data, "Market data returned empty dict")
        self.assertEqual(data["symbol"], self.ticker)
        self.assertIsNotNone(data.get("current_price"), "current_price is None")
        self.assertIsNotNone(data.get("volume"), "volume is None")

    def test_fetch_historical_data_daily(self):
        print(f"\nTesting fetch_historical_data (daily) for {self.ticker}...")
        data = self.fetcher.fetch_historical_data(self.ticker, period="5d", interval="1d")
        print(f"Data received: {len(data)} records")

        self.assertIsInstance(data, list)
        self.assertTrue(data, "Historical data (daily) is empty")
        first_record = data[0]
        self.assertIn("close", first_record)
        self.assertIn("date", first_record)

    def test_fetch_historical_data_intraday(self):
        print(f"\nTesting fetch_historical_data (intraday) for {self.ticker}...")
        # 1m data for 1d
        data = self.fetcher.fetch_historical_data(self.ticker, period="1d", interval="1h")
        print(f"Data received: {len(data)} records")

        self.assertIsInstance(data, list)
        self.assertTrue(data, "Historical data (intraday) is empty")
        first_record = data[0]
        self.assertIn("close", first_record)
        self.assertIn("date", first_record)

    def test_fetch_news(self):
        print(f"\nTesting fetch_news for {self.ticker}...")
        news = self.fetcher.fetch_news(self.ticker)
        print(f"News received: {len(news)} items")

        self.assertIsInstance(news, list)
        # Strict check: news should exist for SPY
        self.assertTrue(news, "News returned empty list")
        self.assertIn("title", news[0])


if __name__ == "__main__":
    unittest.main()

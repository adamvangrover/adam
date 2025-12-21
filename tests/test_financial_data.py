import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from core.financial_data import DataLakehouse, MarketDiscoveryAgent, MarketTicker


class TestFinancialData(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lakehouse = DataLakehouse(root_path=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('core.financial_data.discovery.yf.Search')
    def test_discovery_agent(self, mock_search):
        # Mock yfinance response
        mock_response = MagicMock()
        mock_response.quotes = [
            {
                'symbol': 'AAPL',
                'shortname': 'Apple Inc.',
                'exchange': 'NMS',
                'quoteType': 'EQUITY',
                'sector': 'Technology',
                'score': 1000.0
            }
        ]
        mock_search.return_value = mock_response

        agent = MarketDiscoveryAgent()
        results = agent.search_universe("Technology")

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], MarketTicker)
        self.assertEqual(results[0].symbol, 'AAPL')
        self.assertEqual(results[0].short_name, 'Apple Inc.')

    @patch('core.financial_data.lakehouse.yf.Ticker')
    def test_lakehouse_ingest(self, mock_ticker_class):
        # Mock Ticker
        mock_ticker = MagicMock()
        # Create a sample DataFrame
        data = {
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000, 2000],
            'Dividends': [0.0, 0.0],
            'Stock Splits': [0.0, 0.0]
        }
        index = pd.to_datetime(['2023-01-01', '2023-01-02'])
        df = pd.DataFrame(data, index=index)
        mock_ticker.history.return_value = df

        mock_ticker_class.return_value = mock_ticker

        # Test Ingest
        self.lakehouse.ingest_tickers(['AAPL'], period='2d')

        # Verify file exists in new structure
        # data/daily/region=US/year=2023/{uuid}.parquet
        daily_path = Path(self.temp_dir) / "daily"
        self.assertTrue(daily_path.exists())

        # Check that we have region=US and year=2023
        region_path = daily_path / "region=US"
        self.assertTrue(region_path.exists())

        year_path = region_path / "year=2023"
        self.assertTrue(year_path.exists())

        parquet_files = list(year_path.glob("*.parquet"))
        self.assertTrue(len(parquet_files) >= 1)

        # Verify Load
        # Because UUIDs are random, loading logic is tested here too
        loaded_df = self.lakehouse.load_data('AAPL')
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(loaded_df.iloc[0]['Close'], 101.0)

    def test_metadata_storage(self):
        ticker = MarketTicker(symbol='AAPL', shortname='Apple', sector='Tech')
        self.lakehouse.store_metadata([ticker])

        metadata_path = Path(self.temp_dir) / "metadata" / "ticker_universe.parquet"
        self.assertTrue(metadata_path.exists())

        df = pd.read_parquet(metadata_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['symbol'], 'AAPL')

if __name__ == '__main__':
    unittest.main()

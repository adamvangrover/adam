# tests/test_data_sources.py

import unittest
from unittest.mock import patch
from core.data_sources.data_sources import DataSources

class TestDataSources(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of DataSources."""
        config = {
            'api_keys': {
                'twitter': {
                    'consumer_key': 'YOUR_CONSUMER_KEY',
                    'consumer_secret': 'YOUR_CONSUMER_SECRET',
                    'access_token': 'YOUR_ACCESS_TOKEN',
                    'access_token_secret': 'YOUR_ACCESS_TOKEN_SECRET'
                },
                #... (add API keys for other data sources)
            }
        }
        self.data_sources = DataSources(config)

    @patch('core.data_sources.data_sources.requests.get')
    def test_get_financial_news_headlines(self, mock_get):
        """Test fetching financial news headlines."""
        # Check if simulated data is being used
        if isinstance(self.data_sources.api_keys.get('financial_news_api'), dict) and \
           'simulated' in self.data_sources.api_keys['financial_news_api']:
            # Mock the API response with simulated data
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'articles': [
                    {'title': 'Tech Stock Soars on Earnings Report'},
                    {'title': 'Market Dips Amid Inflation Concerns'}
                ]
            }
            mock_get.return_value = mock_response
        else:
            # Use actual API keys and make real API calls
            pass  # No need to mock the response

        headlines = self.data_sources.get_financial_news_headlines(source="bloomberg", keywords=["technology"])
        self.assertIsNotNone(headlines)
        self.assertIsInstance(headlines, list)
        #... (add more assertions to validate the fetched headlines, e.g., check for keywords)

    @patch('core.data_sources.data_sources.requests.get')
    def test_get_historical_news(self, mock_get):
        """Test fetching historical news data."""
        #... (mock API response or use actual API based on configuration)

        historical_news = self.data_sources.get_historical_news(
            source="reuters", keywords=["inflation"], start_date="2023-01-01", end_date="2023-12-31"
        )
        self.assertIsNotNone(historical_news)
        self.assertIsInstance(historical_news, list)
        #... (add more assertions, e.g., check for date range)

    @patch('core.data_sources.data_sources.tweepy.API.search_tweets')
    def test_get_tweets(self, mock_search_tweets):
        """Test fetching tweets from Twitter."""
        #... (mock API response or use actual API based on configuration)

        tweets = self.data_sources.get_tweets(query="$AAPL", sentiment="positive")
        self.assertIsNotNone(tweets)
        self.assertIsInstance(tweets, list)
        #... (add more assertions, e.g., check for sentiment)

    #... (add tests for other data source methods with similar structure)

if __name__ == '__main__':
    unittest.main()

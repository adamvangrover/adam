# tests/test_data_sources.py

import unittest
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

    def test_get_financial_news_headlines(self):
        """Test fetching financial news headlines."""
        headlines = self.data_sources.get_financial_news_headlines(source="bloomberg", keywords=["technology"])
        self.assertIsNotNone(headlines)
        self.assertIsInstance(headlines, list)  # Check if it returns a list
        #... (add more assertions to validate the fetched headlines, e.g., check for keywords)

    def test_get_historical_news(self):
        """Test fetching historical news data."""
        historical_news = self.data_sources.get_historical_news(
            source="reuters", keywords=["inflation"], start_date="2023-01-01", end_date="2023-12-31"
        )
        self.assertIsNotNone(historical_news)
        self.assertIsInstance(historical_news, list)
        #... (add more assertions, e.g., check for date range)

    def test_get_tweets(self):
        """Test fetching tweets from Twitter."""
        tweets = self.data_sources.get_tweets(query="$AAPL", sentiment="positive")
        self.assertIsNotNone(tweets)
        self.assertIsInstance(tweets, list)
        #... (add more assertions, e.g., check for sentiment)

    def test_get_trending_topics(self):
        """Test fetching trending topics from Twitter."""
        trending_topics = self.data_sources.get_trending_topics(location="New York")
        self.assertIsNotNone(trending_topics)
        self.assertIsInstance(trending_topics, list)
        #... (add more assertions, e.g., check for location)

    def test_get_facebook_posts(self):
        """Test fetching Facebook posts."""
        posts = self.data_sources.get_facebook_posts(query="technology", sentiment="negative")
        self.assertIsNotNone(posts)
        self.assertIsInstance(posts, list)
        #... (add more assertions, e.g., check for sentiment)

    #... (add tests for other data source methods with similar structure)

if __name__ == '__main__':
    unittest.main()

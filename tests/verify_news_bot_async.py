import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure core is in path
sys.path.append(os.getcwd())

from core.agents.news_bot import NewsBot


class TestNewsBotAsync(unittest.IsolatedAsyncioTestCase):
    async def test_aggregate_news_async(self):
        config = {
            "news_api_key": "test_key",
            "user_preferences": {"topics": ["finance"]},
            "portfolio": {"stocks": ["Test"]}
        }
        # Mocking nltk data find/download to avoid errors
        with patch("nltk.data.find"), patch("nltk.download"):
             bot = NewsBot(config)

        # Mocking get_finance_news directly might be easier, but let's mock httpx
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": [{"title": "Test News", "description": "Analysis of Test stock."}]}
            mock_response.raise_for_status = MagicMock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            # Also mock feedparser
            with patch("feedparser.parse") as mock_feed:
                mock_feed.return_value.entries = []

                # Also mock coin gecko
                bot.cg = MagicMock()
                bot.cg.get_trending_searches = MagicMock(return_value={"coins": []})

                results = await bot.execute()

                self.assertIn("personalized_feed", results)
                # We expect at least one article from finance mock
                self.assertTrue(len(results["personalized_feed"]) > 0, "Personalized feed is empty")
                self.assertEqual(results["personalized_feed"][0]["title"], "Test News")

if __name__ == "__main__":
    unittest.main()

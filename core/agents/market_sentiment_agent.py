from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import logging
import asyncio
from core.agents.agent_base import AgentBase
from core.data_sources.financial_news_api import SimulatedFinancialNewsAPI
from core.data_sources.prediction_market_api import SimulatedPredictionMarketAPI
from core.data_sources.social_media_api import SimulatedSocialMediaAPI
from core.data_sources.web_traffic_api import SimulatedWebTrafficAPI


class MarketSentimentAgent(AgentBase):
    """
    Agent responsible for gauging market sentiment from a variety of sources,
    such as news articles, social media, and prediction markets.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the MarketSentimentAgent.
        """
        super().__init__(config, **kwargs)
        self.data_sources = config.get('data_sources', [])
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5)

        # Initialize sources
        # In a real system, these might be injected or configured
        self.news_api = SimulatedFinancialNewsAPI(self.config)
        self.prediction_market_api = SimulatedPredictionMarketAPI()
        self.social_media_api = SimulatedSocialMediaAPI(self.config)
        self.web_traffic_api = SimulatedWebTrafficAPI()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the sentiment analysis.

        Returns:
            Dict containing the sentiment score and analysis details.
        """
        logging.info("MarketSentimentAgent execution started.")

        overall_sentiment, details = await self.analyze_sentiment()

        result = {
            "sentiment_score": overall_sentiment,
            "details": details,
            "status": "success"
        }

        # Optionally send to message broker if configured
        if hasattr(self, 'message_broker') and self.message_broker:
            await self.send_message("system_monitor", result)  # Example target

        return result

    async def analyze_sentiment(self) -> Tuple[float, Dict[str, float]]:
        """
        Analyzes sentiment from configured sources.
        """
        # 1. News
        # Assuming get_headlines is synchronous/simulated.
        # If it were real I/O, we'd wrap it.
        try:
            headlines, news_sentiment = self.news_api.get_headlines()
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            news_sentiment = 0.0

        logging.info(f"News Sentiment: {news_sentiment}")

        # 2. Prediction Markets
        try:
            pred_sentiment = self.prediction_market_api.get_market_sentiment()
        except Exception as e:
            logging.error(f"Error fetching prediction market data: {e}")
            pred_sentiment = 0.0

        logging.info(f"Prediction Market Sentiment: {pred_sentiment}")

        # 3. Social Media
        try:
            social_sentiment = self.social_media_api.get_social_media_sentiment()
        except Exception as e:
            logging.error(f"Error fetching social media data: {e}")
            social_sentiment = 0.0

        logging.info(f"Social Media Sentiment: {social_sentiment}")

        # 4. Web Traffic
        try:
            web_sentiment = self.web_traffic_api.get_web_traffic_sentiment()
        except Exception as e:
            logging.error(f"Error fetching web traffic data: {e}")
            web_sentiment = 0.0

        logging.info(f"Web Traffic Sentiment: {web_sentiment}")

        # 5. Combine
        overall = self.combine_sentiment(news_sentiment, pred_sentiment, social_sentiment, web_sentiment)
        logging.info(f"Overall Market Sentiment: {overall}")

        details = {
            "news_sentiment": news_sentiment,
            "prediction_market_sentiment": pred_sentiment,
            "social_media_sentiment": social_sentiment,
            "web_traffic_sentiment": web_sentiment
        }

        return overall, details

    def combine_sentiment(self, news: float, pred: float, social: float, web: float) -> float:
        """
        Combines sentiment from different sources into an overall sentiment score.
        """
        # Simple weighted average
        weights = {
            'news': 0.4,
            'prediction': 0.3,
            'social': 0.2,
            'web': 0.1
        }

        # Ensure inputs are floats (mock APIs might return None or ints)
        def clean(val):
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            return 0.0

        score = (
            clean(news) * weights['news'] +
            clean(pred) * weights['prediction'] +
            clean(social) * weights['social'] +
            clean(web) * weights['web']
        )
        return score


if __name__ == "__main__":
    # Test harness
    logging.basicConfig(level=logging.INFO)

    async def main():
        config = {
            'data_sources': ['news', 'prediction_market', 'social_media', 'web_traffic'],
            'sentiment_threshold': 0.5
        }
        agent = MarketSentimentAgent(config)
        result = await agent.execute()
        print(f"Result: {result}")

    asyncio.run(main())

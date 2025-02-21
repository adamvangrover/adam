# core/agents/market_sentiment_agent.py

from core.data_sources.financial_news_api import SimulatedFinancialNewsAPI
from core.utils.data_utils import send_message

class MarketSentimentAgent:
    def __init__(self, config):
        self.data_sources = config['data_sources']
        self.sentiment_threshold = config['sentiment_threshold']
        self.news_api = SimulatedFinancialNewsAPI()

    def analyze_sentiment(self):
        headlines, sentiment = self.news_api.get_headlines()
        print("Analyzing market sentiment from news headlines...")
        # Placeholder for more sophisticated sentiment analysis
        # (For now, we'll just return the simulated sentiment from the API)

        # Send sentiment to message queue
        message = {'agent': 'market_sentiment_agent', 'sentiment': sentiment}
        send_message(message)

        return sentiment

# Example usage (eventually, this would be managed by the system)
if __name__ == "__main__":
    import yaml
    with open("../../config/agents.yaml", "r") as f:
        agent_config = yaml.safe_load(f)

    agent = MarketSentimentAgent(agent_config['market_sentiment_agent'])
    sentiment = agent.analyze_sentiment()
    print(f"Market Sentiment Score: {sentiment}")

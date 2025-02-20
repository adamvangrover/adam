# core/data_sources/financial_news_api.py

import random

class SimulatedFinancialNewsAPI:
    def get_headlines(self):
        # Simulate fetching news headlines
        headlines = [
            "Tech Stocks Surge on AI Breakthrough",
            "Inflation Concerns Weigh on Market",
            "Central Bank Holds Interest Rates Steady",
            "Geopolitical Tensions Rise in Eastern Europe",
            "New Healthcare Regulations Announced"
        ]
        # Simulate some variability in the news (positive/negative sentiment)
        simulated_sentiment = random.choice(["positive", "negative", "neutral"])
        return headlines, simulated_sentiment

# Example usage
if __name__ == "__main__":
    api = SimulatedFinancialNewsAPI()
    headlines, sentiment = api.get_headlines()
    print("Simulated News Headlines:")
    for headline in headlines:
        print(f"- {headline}")
    print(f"Overall Sentiment: {sentiment}")

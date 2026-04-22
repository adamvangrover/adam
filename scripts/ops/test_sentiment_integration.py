import asyncio
import os
import sys
import json
from pprint import pprint

# Ensure repo root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '../../'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.v30_architecture.python_intelligence.agents.market_sentiment_agent import MarketSentimentAgent

async def run_agent_test():
    print("Initializing MarketSentimentAgent...")
    agent = MarketSentimentAgent()

    asset = "NVDA"
    print(f"\n1. Executing Web Search for: {asset}...")

    # We call the internal method to test the tool spin up
    news_results = agent._search_news(asset)

    print("\n--- Raw Search Results ---")
    for idx, r in enumerate(news_results):
        print(f"[{idx+1}] {r.get('title')}\n    URL: {r.get('url')}\n    Snippet: {r.get('snippet')[:100]}...\n")

    print("\n2. Analyzing Sentiment, Filtering Noise, and Ranking Signals...")
    result = agent.analyze_sentiment(asset, news_results)

    print("\n--- AGENT OUTPUT (JSON) ---")
    print(json.dumps(result, indent=2))
    print("---------------------------\n")

if __name__ == "__main__":
    asyncio.run(run_agent_test())

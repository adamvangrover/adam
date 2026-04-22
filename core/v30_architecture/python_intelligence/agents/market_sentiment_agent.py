import asyncio
import random
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

# Try to import litellm, if not use mock
try:
    import litellm
except ImportError:
    litellm = None

# Try to import web search dependencies
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("MarketSentimentAgent")

class SentimentSignal(BaseModel):
    source_url: str = Field(default="mock_url")
    headline: str
    signal_type: str = Field(..., description="E.g., 'Earnings', 'Macro', 'Regulatory', 'Product'")
    is_noise: bool = Field(..., description="True if irrelevant or spam, False if actionable signal")
    impact_score: float = Field(..., description="Impact scale from -1.0 (highly negative) to 1.0 (highly positive)")

class SentimentOutput(BaseModel):
    asset_id: str
    sentiment_score: float = Field(..., description="Aggregated sentiment score from -1.0 (bearish) to 1.0 (bullish)")
    confidence: float = Field(..., description="Confidence level from 0.0 to 1.0 based on signal strength")
    signals: List[SentimentSignal] = Field(default_factory=list, description="Ranked list of signals, filtered for noise")
    key_drivers: List[str] = Field(default_factory=list, description="Key themes driving the sentiment")

class MarketSentimentAgent(BaseAgent):
    """
    V30 Market Sentiment Agent.
    Spins up search tools, fetches news, filters noise, ranks signals,
    and emits findings to the NeuralMesh.
    """
    def __init__(self):
        super().__init__(name="MarketSentimentAgent", role="SentimentAnalyst")
        self.target_assets = ["AAPL", "BTC", "TSLA", "NVDA", "SPY"]

    def _search_news(self, asset: str) -> List[Dict[str, str]]:
        """Uses DuckDuckGo to search for recent news."""
        results = []
        if DDGS_AVAILABLE:
            try:
                logger.info(f"{self.name} searching web for: {asset} financial news")
                with DDGS() as ddgs:
                    # Search for recent news
                    search_gen = ddgs.text(f"{asset} financial news stock", max_results=3)
                    for r in search_gen:
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", "")
                        })
            except Exception as e:
                logger.error(f"Search failed: {e}")

        if not results:
             # Fallback mock news if offline
             results = [
                 {"title": f"{asset} exceeds earnings estimates", "snippet": "Company reports strong growth in Q3.", "url": "http://mocknews.local/1"},
                 {"title": f"Regulatory concerns for {asset}", "snippet": "New probes launched into corporate practices.", "url": "http://mocknews.local/2"}
             ]
        return results

    def _mock_llm_sentiment(self, asset: str, news: List[Dict]) -> SentimentOutput:
        """Fallback mock sentiment logic when LLM is unavailable."""
        signals = []
        total_impact = 0

        for item in news:
            is_noise = random.choice([True, False])
            impact = random.uniform(-0.8, 0.8) if not is_noise else 0.0
            total_impact += impact
            signals.append(SentimentSignal(
                source_url=item.get("url", ""),
                headline=item.get("title", "Unknown"),
                signal_type="Macro",
                is_noise=is_noise,
                impact_score=impact
            ))

        score = total_impact / len(signals) if signals else 0.0
        # Sort by absolute impact
        signals.sort(key=lambda x: abs(x.impact_score), reverse=True)

        return SentimentOutput(
            asset_id=asset,
            sentiment_score=max(min(score, 1.0), -1.0),
            confidence=random.uniform(0.5, 0.9),
            signals=signals,
            key_drivers=["market volatility", "mock data"]
        )

    def analyze_sentiment(self, asset: str, news_data: List[Dict[str, str]]) -> dict:
        """Uses LLM (or mock) to analyze sentiment of a text blob."""
        if litellm is None or os.environ.get("MOCK_MODE") == "true":
            output = self._mock_llm_sentiment(asset, news_data)
            return output.model_dump()

        news_text = "\n".join([f"- {n['title']}: {n['snippet']} ({n['url']})" for n in news_data])
        prompt = f"""
        Analyze the following recent news for {asset}.
        Filter out noise. Extract actionable signals and rank them by impact.
        News:
        {news_text}

        Return JSON matching: {SentimentOutput.model_json_schema()}
        """
        try:
            # We use an environment check or try/except to handle missing api keys
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            # Ensure asset_id is correct
            parsed["asset_id"] = asset
            # Validate with pydantic
            validated = SentimentOutput(**parsed)

            # Ensure signals are sorted by impact (absolute value)
            validated.signals.sort(key=lambda x: abs(x.impact_score), reverse=True)

            return validated.model_dump()
        except Exception as e:
            logger.warning(f"LLM analysis failed for {asset}: {e}. Falling back to mock.")
            output = self._mock_llm_sentiment(asset, news_data)
            return output.model_dump()

    async def run(self):
        """Main loop: Gather inputs, analyze, filter, emit."""
        logger.info(f"{self.name} starting sentiment analysis loop.")
        while True:
            try:
                # 1. Spin up internal tools/search
                target_asset = random.choice(self.target_assets)
                news_results = self._search_news(target_asset)

                # 2. Analyze Sentiment (Signal filtering & ranking)
                analysis_result = self.analyze_sentiment(target_asset, news_results)

                # 3. Emit findings to the mesh
                await self.emit("THOUGHT", {
                    "task": "sentiment_analysis",
                    "status": "completed",
                    "output": analysis_result
                })

                logger.info(f"{self.name} emitted sentiment for {target_asset}: {analysis_result['sentiment_score']:.2f}")

            except Exception as e:
                logger.error(f"Error in {self.name} execution loop: {e}")

            # Sleep before next cycle
            await asyncio.sleep(random.uniform(8.0, 15.0))

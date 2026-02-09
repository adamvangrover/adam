from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Union, List
import logging
import asyncio
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.data_sources.financial_news_api import SimulatedFinancialNewsAPI
from core.data_sources.prediction_market_api import SimulatedPredictionMarketAPI
from core.data_sources.social_media_api import SimulatedSocialMediaAPI
from core.data_sources.web_traffic_api import SimulatedWebTrafficAPI
from core.data_sources.data_fetcher import DataFetcher


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
        self.data_fetcher = DataFetcher()

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the sentiment analysis.
        Supports both legacy string/dict input and new AgentInput schema.

        Returns:
            Dict containing the sentiment score and analysis details, or AgentOutput.
        """
        # 1. Input Normalization
        query = ""
        is_standard_mode = False

        # Handle explicit input_data argument
        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
                if input_data.context:
                    self.set_context(input_data.context)
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")
                if "context" in input_data:
                    self.set_context(input_data["context"])
                # Merge input_data into kwargs for backward compatibility if needed
                kwargs.update(input_data)

        # Fallback: Check kwargs for legacy usage where execute(param=val) was called
        # The agent base wraps execute, so args/kwargs are passed through.
        # But if input_data matches one of the expected types, we used it.

        logging.info(f"MarketSentimentAgent execution started. Mode: {'Standard' if is_standard_mode else 'Legacy'}")

        overall_sentiment, details = await self.analyze_sentiment()

        result = {
            "sentiment_score": overall_sentiment,
            "details": details,
            "status": "success"
        }

        # Optionally send to message broker if configured
        if hasattr(self, 'message_broker') and self.message_broker:
            await self.send_message("system_monitor", result)  # Example target

        if is_standard_mode:
            return self._format_output(result, query)

        return result

    def _format_output(self, result: Dict[str, Any], query: str) -> AgentOutput:
        """Helper to format output to AgentOutput."""
        sentiment_score = result.get("sentiment_score", 0.0)
        details = result.get("details", {})

        # Construct a textual answer
        answer = f"Market Sentiment Analysis for '{query}':\n"
        answer += f"Overall Sentiment Score: {sentiment_score:.2f} (0.0 Bearish - 1.0 Bullish)\n\n"

        if "credit_override" in details:
            answer += f"WARNING: Credit Dominance Rule Triggered: {details['credit_override']}\n"
            answer += "Sentiment has been overridden to reflect systemic risk.\n\n"

        answer += "Breakdown:\n"
        answer += f"- News: {details.get('news_sentiment', 'N/A')}\n"
        answer += f"- Prediction Markets: {details.get('prediction_market_sentiment', 'N/A')}\n"
        answer += f"- Social Media: {details.get('social_media_sentiment', 'N/A')}\n"
        answer += f"- Web Traffic: {details.get('web_traffic_sentiment', 'N/A')}\n"

        return AgentOutput(
            answer=answer,
            sources=["SimulatedFinancialNewsAPI", "SimulatedPredictionMarketAPI", "SimulatedSocialMediaAPI"],
            confidence=0.8, # Placeholder confidence
            metadata=result
        )

    async def analyze_sentiment(self) -> Tuple[float, Dict[str, float]]:
        """
        Analyzes sentiment from configured sources.
        """
        # 1. News
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

        # 6. Credit Dominance Logic Gate (Adam v24.1)
        credit_override, credit_details = await self.check_credit_dominance_rule()
        if credit_override:
            logging.warning(f"Credit Dominance Rule Triggered: {credit_override}")
            details["credit_override"] = credit_override

        # Merge credit details into main details for Persona use
        details.update(credit_details)

        if credit_override:
            # Force output to represent the risk (Liquidity Mirage / Systemic Tremor)
            # Overriding sentiment to Bearish/Cautionary (e.g., 0.2)
            overall = 0.2
            details["status_override"] = credit_override

        return overall, details

    async def check_credit_dominance_rule(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Checks for market plumbing warnings based on the 'Risk Architect' logic.
        Async execution to prevent blocking on DataFetcher I/O.
        """
        details = {}

        # Parallel Fetching
        loop = asyncio.get_running_loop()

        # Helper for threaded execution of synchronous DataFetcher methods
        def fetch(func, *args):
            return loop.run_in_executor(None, func, *args)

        # Launch tasks
        try:
            t_spy = fetch(self.data_fetcher.fetch_market_data, "SPY")
            t_qqq = fetch(self.data_fetcher.fetch_market_data, "QQQ")
            t_credit = fetch(self.data_fetcher.fetch_credit_metrics)
            t_vol = fetch(self.data_fetcher.fetch_volatility_metrics)
            t_crypto = fetch(self.data_fetcher.fetch_crypto_metrics)
            t_treasury = fetch(self.data_fetcher.fetch_treasury_metrics)
            t_liquidity = fetch(self.data_fetcher.fetch_macro_liquidity)

            # Await all
            spy_data, qqq_data, credit_data, vol_data, crypto_data, treasury_data, liquidity_data = await asyncio.gather(
                t_spy, t_qqq, t_credit, t_vol, t_crypto, t_treasury, t_liquidity
            )
        except Exception as e:
            logging.error(f"Error fetching data for Credit Dominance Rule: {e}")
            return None, {}

        # --- Populate Data Nexus Details ---

        # Volatility
        details["vix"] = vol_data.get("^VIX", {}).get("last_price")
        details["vix3m"] = vol_data.get("^VIX3M", {}).get("last_price")
        details["move_index"] = vol_data.get("^MOVE", {}).get("last_price")

        # Crypto
        btc = crypto_data.get("BTC-USD", {})
        eth = crypto_data.get("ETH-USD", {})
        details["btc_price"] = btc.get("last_price")
        details["eth_price"] = eth.get("last_price")
        if details.get("btc_price") and details.get("eth_price") and details["btc_price"] != 0:
             details["eth_btc_ratio"] = details["eth_price"] / details["btc_price"]

        # Credit / Treasury
        details["hyg_price"] = credit_data.get("HYG", {}).get("last_price")
        details["ief_price"] = treasury_data.get("IEF", {}).get("last_price")
        details["tnx_yield"] = treasury_data.get("^TNX", {}).get("last_price")
        details["irx_yield"] = treasury_data.get("^IRX", {}).get("last_price") # 13 Week

        # Liquidity
        details["usd_liquidity"] = liquidity_data.get("liquidity_index")

        # --- Calculate Changes ---

        def get_pct_change(data):
            if not data:
                return 0.0
            if data.get("current_price") and data.get("previous_close"):
                 return (data["current_price"] - data["previous_close"]) / data["previous_close"] * 100
            if data.get("last_price") and data.get("previous_close"):
                 return (data["last_price"] - data["previous_close"]) / data["previous_close"] * 100
            return 0.0

        spy_change = get_pct_change(spy_data)
        qqq_change = get_pct_change(qqq_data)
        hyg_change = get_pct_change(credit_data.get("HYG"))
        ief_change = get_pct_change(treasury_data.get("IEF"))
        btc_change = get_pct_change(btc)

        details["spy_change_pct"] = spy_change
        details["qqq_change_pct"] = qqq_change
        details["hyg_change_pct"] = hyg_change
        details["ief_change_pct"] = ief_change
        details["btc_change_pct"] = btc_change

        # --- Logic Gates ---

        trigger = None

        # Condition A: Liquidity Mirage
        # Equity Markets Green (> +0.5%) AND High Yield Spreads Widening.
        # Spread Widening proxy: HYG Change - IEF Change < -0.15%
        relative_credit_performance = hyg_change - ief_change
        details["relative_credit_performance"] = relative_credit_performance

        liquidity_mirage = (spy_change > 0.5 and relative_credit_performance < -0.15)

        # Condition B: Systemic Tremor
        # VIX Inversion (Spot > 3M) OR Curve Inversion (10Y < 3M).
        vix_inverted = (details.get("vix") is not None and details.get("vix3m") is not None and details["vix"] > details["vix3m"])

        curve_slope = None
        if details.get("tnx_yield") is not None and details.get("irx_yield") is not None:
             curve_slope = details["tnx_yield"] - details["irx_yield"]
             details["curve_slope_10y_3m"] = curve_slope

        curve_inverted = (curve_slope is not None and curve_slope < 0)

        systemic_tremor = (vix_inverted or curve_inverted)

        # Condition C: Risk-On Decoupling
        # BTC < -3% while Nasdaq > +0.5%
        risk_on_decoupling = (btc_change < -3.0 and qqq_change > 0.5)

        # Prioritization: Systemic Tremor > Risk-On Decoupling > Liquidity Mirage
        if systemic_tremor:
            trigger = "Systemic Tremor"
        elif risk_on_decoupling:
             trigger = "Risk-On Decoupling"
        elif liquidity_mirage:
             trigger = "Liquidity Mirage"

        return trigger, details

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

        print("--- Test Standard Mode ---")
        input_obj = AgentInput(query="Market Check")
        result = await agent.execute(input_obj)
        print(f"Result (Standard): {result}")

        print("\n--- Test Legacy Mode ---")
        result_legacy = await agent.execute() # Using default None input, relies on internal logic (or lack of kwargs)
        print(f"Result (Legacy): {result_legacy}")

    asyncio.run(main())

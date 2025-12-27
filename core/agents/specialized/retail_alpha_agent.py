import logging
import asyncio
import random
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.utils.logging_utils import get_logger

# Try to import specialized agents, but fail gracefully for the "runnable today" requirement
try:
    from core.agents.specialized.institutional_radar_agent import InstitutionalRadarAgent
    RADAR_AVAILABLE = True
except ImportError:
    RADAR_AVAILABLE = False

try:
    from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False

logger = get_logger("agents.retail_alpha")

class RetailAlphaAgent(AgentBase):
    """
    Retail Alpha Agent: 'The Retail Supplement'

    This agent bridges the gap between institutional data (13Fs, Risk Models) and
    retail trading needs (Signals, Hype, Simple Metrics).

    It generates 'Alpha Signals' by looking for divergences:
    - Smart Money Buying vs Retail Selling (Bullish Divergence)
    - Smart Money Selling vs Retail Euphoria (Bearish Trap)
    """

    def __init__(self, config: Dict[str, Any], constitution: Dict[str, Any] = None, kernel=None):
        super().__init__(config, constitution=constitution, kernel=kernel)
        self.mock_mode = config.get("mock_mode", True) # Default to mock for instant utility

        # Initialize sub-agents if available and not in mock mode
        self.radar_agent = InstitutionalRadarAgent(config) if RADAR_AVAILABLE and not self.mock_mode else None
        # Risk agent might need specific init, keeping it simple for now
        self.risk_agent = None

    async def execute(self, tickers: List[str] = None, **kwargs) -> Dict[str, Any]:
        if not tickers:
            tickers = ["AAPL", "NVDA", "TSLA", "GME", "AMC", "PLTR", "AMD", "MSFT"]

        logger.info(f"Generating Retail Alpha Signals for: {tickers}")

        results = []

        for ticker in tickers:
            signal = await self._analyze_ticker(ticker)
            results.append(signal)

        # Sort by conviction
        results.sort(key=lambda x: x['conviction'], reverse=True)

        return {
            "signals": results,
            "market_status": self._get_market_status(),
            "timestamp": "2025-10-24T14:30:00Z" # In a real app, use datetime.utcnow()
        }

    async def _analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a single ticker for Retail Alpha.
        """

        # 1. Institutional Flow (Smart Money)
        # In a real scenario, we'd call self.radar_agent.execute(ciks=[ticker_cik])
        # For this "Supplement" demo, we simulate the logic based on "The Great Divergence" theme
        inst_score = self._get_institutional_score(ticker)

        # 2. Retail Sentiment (Hype)
        # Simulating social media sentiment
        retail_score = self._get_retail_sentiment(ticker)

        # 3. Volatility/Risk
        risk_score = self._get_risk_profile(ticker)

        # 4. Alpha Logic (Divergence)
        signal_type = "HOLD"
        conviction = 0
        reason = "Neutral activity."

        # Logic: Smart Money > 0.7 and Retail < 0.4 => BUY (Contrarian)
        # Logic: Smart Money < 0.3 and Retail > 0.8 => SELL (Trap)
        # Logic: Both High => MOMENTUM BUY

        if inst_score > 70 and retail_score < 40:
            signal_type = "STRONG BUY"
            conviction = 92
            reason = "Whale accumulation despite retail fear. Classic accumulation setup."
        elif inst_score < 30 and retail_score > 80:
            signal_type = "STRONG SELL"
            conviction = 88
            reason = "Institutional distribution into retail liquidity. Bull trap warning."
        elif inst_score > 70 and retail_score > 70:
            signal_type = "MOMENTUM"
            conviction = 75
            reason = "Aligned Institutional and Retail flows. High velocity upward move."
        elif inst_score < 40 and retail_score < 40:
            signal_type = "AVOID"
            conviction = 60
            reason = "No interest from whales or retail. Dead money."
        else:
            signal_type = "WATCH"
            conviction = 50
            reason = "Mixed signals. Wait for clarity."

        return {
            "ticker": ticker,
            "signal": signal_type,
            "conviction": conviction,
            "institutional_score": inst_score, # 0-100
            "retail_score": retail_score, # 0-100
            "risk_score": risk_score, # 0-100 (Higher is riskier)
            "reason": reason,
            "catalyst": self._get_catalyst(ticker)
        }

    def _get_institutional_score(self, ticker: str) -> int:
        """Mock logic for demo - replace with InstitutionalRadarAgent"""
        # randomized but consistent for demo
        seed = sum(ord(c) for c in ticker)
        random.seed(seed)
        base = random.randint(20, 90)
        # Bias some popular ones for the narrative
        if ticker in ["NVDA", "MSFT"]: return 85 # Whales love these
        if ticker in ["GME", "AMC"]: return 25   # Whales hate these
        return base

    def _get_retail_sentiment(self, ticker: str) -> int:
        """Mock logic for demo - replace with MarketSentimentGraph"""
        if ticker in ["GME", "AMC", "TSLA"]: return 95 # Retail loves these
        if ticker in ["NVDA"]: return 80
        return 50

    def _get_risk_profile(self, ticker: str) -> int:
        """Mock logic"""
        if ticker in ["GME", "AMC"]: return 90
        if ticker in ["MSFT", "AAPL"]: return 30
        return 60

    def _get_catalyst(self, ticker: str) -> str:
        catalysts = [
            "Earnings Beat Expectation",
            "New Product Launch",
            "CEO Insider Buying",
            "Sector Rotation",
            "Short Squeeze Potential",
            "Analyst Upgrade Cycle",
            "Macro Headwinds"
        ]
        return random.choice(catalysts)

    def _get_market_status(self):
        return {
            "regime": "High Volatility / Distribution",
            "vix": 24.5,
            "sector_rotation": "Tech -> Energy"
        }

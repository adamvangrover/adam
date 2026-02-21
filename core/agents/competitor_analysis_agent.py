from typing import Any, Dict, List, Union, Optional
import logging
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class CompetitorAnalysisAgent(AgentBase):
    """
    Agent responsible for identifying competitors and comparing key metrics.
    """

    # Mock Data for Competitors
    COMPETITORS_MAP = {
        "AAPL": ["MSFT", "GOOGL", "Samsung"],
        "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "GOOGL": ["MSFT", "META", "AMZN"],
        "AMZN": ["WMT", "MSFT", "GOOGL"],
        "TSLA": ["BYD", "F", "GM"],
        "NVDA": ["AMD", "INTC", "QCOM"],
        "META": ["GOOGL", "SNAP", "TIKTOK"],
        "JPM": ["BAC", "C", "WFC"],
        "V": ["MA", "AXP", "PYPL"],
        "PG": ["UL", "CL", "KMB"]
    }

    # Mock Data for Metrics (P/E, Rev Growth)
    METRICS_DB = {
        "AAPL": {"pe": 28.5, "rev_growth": 0.05, "market_cap": "3.4T"},
        "MSFT": {"pe": 35.2, "rev_growth": 0.12, "market_cap": "3.1T"},
        "GOOGL": {"pe": 22.1, "rev_growth": 0.10, "market_cap": "2.0T"},
        "Samsung": {"pe": 15.0, "rev_growth": 0.02, "market_cap": "350B"},
        "AMZN": {"pe": 40.5, "rev_growth": 0.11, "market_cap": "1.9T"},
        "META": {"pe": 25.8, "rev_growth": 0.15, "market_cap": "1.2T"},
        "TSLA": {"pe": 55.0, "rev_growth": 0.08, "market_cap": "700B"},
        "BYD": {"pe": 20.0, "rev_growth": 0.25, "market_cap": "100B"},
        "NVDA": {"pe": 60.0, "rev_growth": 0.50, "market_cap": "2.5T"},
        "AMD": {"pe": 45.0, "rev_growth": 0.15, "market_cap": "250B"},
        "INTC": {"pe": 80.0, "rev_growth": -0.05, "market_cap": "150B"}
    }

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes competitor analysis.
        """
        query = ""
        is_standard_mode = False

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True
            elif isinstance(input_data, str):
                query = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")
                kwargs.update(input_data)

        # Extract ticker from query (simple heuristic)
        ticker = query.strip().upper() if query else kwargs.get("ticker", "").upper()

        # If query is "Analyze AAPL competitors", extract AAPL
        for t in self.COMPETITORS_MAP.keys():
            if t in ticker:
                ticker = t
                break

        logger.info(f"CompetitorAnalysisAgent analyzing for ticker: {ticker}")

        competitors = self.COMPETITORS_MAP.get(ticker, [])

        if not competitors:
            msg = f"No competitors found for {ticker} or ticker not supported."
            if is_standard_mode:
                return AgentOutput(
                    answer=msg,
                    sources=[],
                    confidence=0.0,
                    metadata={"error": "Not Found"}
                )
            return {"error": msg}

        # Compare
        target_metrics = self.METRICS_DB.get(ticker, {})
        comparison = []

        for comp in competitors:
            m = self.METRICS_DB.get(comp, {})
            if m:
                comparison.append({
                    "ticker": comp,
                    "pe": m.get("pe"),
                    "rev_growth": m.get("rev_growth"),
                    "market_cap": m.get("market_cap")
                })

        # Generate Report
        report = f"Competitor Analysis for {ticker}:\n\n"
        report += f"Target ({ticker}): P/E={target_metrics.get('pe', 'N/A')}, Growth={target_metrics.get('rev_growth', 'N/A')}\n"
        report += "-" * 40 + "\n"

        for c in comparison:
            pe_diff = c['pe'] - target_metrics.get('pe', 0) if c['pe'] and target_metrics.get('pe') else 0
            report += f"{c['ticker']}:\n"
            report += f"  P/E: {c['pe']} ({pe_diff:+.1f} vs Target)\n"
            report += f"  Growth: {c['rev_growth']*100:.1f}%\n"
            report += f"  Cap: {c['market_cap']}\n"

        result = {
            "target": ticker,
            "competitors": comparison,
            "report": report
        }

        if is_standard_mode:
            return AgentOutput(
                answer=report,
                sources=["Internal Knowledge Base"],
                confidence=0.9,
                metadata=result
            )

        return result

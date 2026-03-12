# core/agents/macroeconomic_analysis_agent.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import logging
import asyncio
from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.utils.data_utils import send_message

logger = logging.getLogger(__name__)

class MacroeconomicAnalysisAgent(AgentBase):
    """
    Agent responsible for analyzing macroeconomic indicators (GDP, Inflation, etc.)
    to provide a broad market context.

    Refactored for v23 Architecture (Path A).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.data_sources = config.get('data_sources', {})
        self.indicators = config.get('indicators', ['GDP', 'inflation', 'unemployment'])

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the macroeconomic analysis workflow.
        Supports standard AgentInput schema.
        """
        logger.info("Executing MacroeconomicAnalysisAgent...")

        is_standard_mode = False
        query = "Macroeconomic Analysis"
        country = "US"
        period = "current"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                country = input_data.context.get("country", country)
                period = input_data.context.get("period", period)
                is_standard_mode = True
            elif isinstance(input_data, dict):
                country = input_data.get("country", country)
                period = input_data.get("period", period)
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                query = input_data

        country = kwargs.get("country", country)
        period = kwargs.get("period", period)

        insights = self.analyze_macroeconomic_data(country, period)

        # Check if the user is asking for the specific 2026 Reflation thesis
        if "reflation" in query.lower() or "2026" in query.lower():
            insights["strategic_report"] = self.generate_reflation_report()

        if is_standard_mode:
            answer = f"Macroeconomic Analysis for {country} ({period}):\n\n"
            answer += f"- GDP Trend: {insights.get('GDP_growth_trend', 'Unknown').title()}\n"
            answer += f"- Inflation Outlook: {insights.get('inflation_outlook', 'Unknown').title()}\n"

            if "strategic_report" in insights:
                report = insights["strategic_report"]
                answer += f"\n--- Strategic Outlook: {report['regime']} ---\n"
                answer += f"Thesis: {report['core_thesis']}\n"
                answer += "Drivers:\n"
                for d in report['key_drivers']:
                    answer += f"  - {d['factor']} ({d['impact']}): {d['mechanism']}\n"
                answer += f"Implication: {report['strategic_implication']}\n"

            return AgentOutput(
                answer=answer,
                sources=["Government Stats API", "Internal Economic Models"],
                confidence=0.8,
                metadata=insights
            )

        return insights

    def analyze_macroeconomic_data(self, country: str = "US", period: str = "current") -> Dict[str, Any]:
        """
        Performs the analysis of macroeconomic data.
        """
        logger.info(f"Analyzing macroeconomic data for {country}...")

        stats_api = self.data_sources.get('government_stats_api')

        gdp_trend = "neutral"
        inflation_outlook = "stable"

        if stats_api:
            try:
                gdp_growth = stats_api.get_gdp(country=country, period="quarterly")
                inflation_rate = stats_api.get_inflation(country=country, period="monthly")

                gdp_trend = self.analyze_gdp_trend(gdp_growth)
                inflation_outlook = self.analyze_inflation_outlook(inflation_rate)
            except Exception as e:
                logger.error(f"Error fetching/analyzing stats: {e}")
        else:
            logger.warning("government_stats_api not configured. Using default/mock heuristic values.")
            # Heuristic/Mock values for standard testing
            if country == "US":
                gdp_trend = "positive"
                inflation_outlook = "sticky"
            elif country == "EU":
                gdp_trend = "neutral"
                inflation_outlook = "cooling"
            elif country == "CN":
                gdp_trend = "slowing"
                inflation_outlook = "deflationary"

        insights = {
            'country': country,
            'GDP_growth_trend': gdp_trend,
            'inflation_outlook': inflation_outlook,
            'timestamp': "2025-Q1"
        }

        try:
            message = {'agent': 'macroeconomic_analysis_agent', 'insights': insights}
            send_message(message)
        except Exception as e:
            logger.debug(f"Legacy send_message failed (no broker): {e}")

        return insights

    def analyze_gdp_trend(self, gdp_growth: Any) -> str:
        if not gdp_growth:
            return "neutral"
        if isinstance(gdp_growth, dict) and 'value' in gdp_growth:
            val = gdp_growth['value']
            if val > 2.0:
                return "positive"
            if val < 0.0:
                return "negative"
        return "stable"

    def analyze_inflation_outlook(self, inflation_rate: Any) -> str:
        if not inflation_rate:
             return "stable"
        if isinstance(inflation_rate, dict) and 'value' in inflation_rate:
             val = inflation_rate['value']
             if val > 3.0: return "sticky/high"
             if val < 1.0: return "cooling/deflationary"
        return "stable"

    def generate_reflation_report(self) -> Dict[str, Any]:
        """
        Generates the 'Reflationary Agentic Boom' narrative report.
        """
        return {
            "regime": "Reflationary Agentic Boom",
            "year": 2026,
            "core_thesis": "The 'Apex Paradox': Simultaneous supply-side deflation (AI) and demand-side inflation (Fiscal).",
            "key_drivers": [
                {
                    "factor": "Agentic Productivity",
                    "impact": "Deflationary",
                    "mechanism": "Zero-marginal cost labor for cognitive tasks."
                },
                {
                    "factor": "Sovereign Fiscal Dominance",
                    "impact": "Inflationary",
                    "mechanism": "Monetization of debt to fund AI/Energy infrastructure."
                },
                {
                    "factor": "Geopolitical Fragmentation",
                    "impact": "Inflationary",
                    "mechanism": "Supply chain duplication and tariff wars."
                }
            ],
            "strategic_implication": "Avoid the 'Muddled Middle'. Barbell allocation required."
        }

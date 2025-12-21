import logging
import asyncio
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.institutional_radar.ingestion import SECEdgarScraper
from core.institutional_radar.analytics import InstitutionalRadarAnalytics
from core.institutional_radar.reporting import InstitutionalRadarReporter
from core.utils.logging_utils import get_logger

logger = get_logger("agents.institutional_radar")


class InstitutionalRadarAgent(AgentBase):
    """
    Agent responsible for executing the Institutional Radar blueprint:
    Ingesting 13F data, analyzing trends, and generating narrative reports.
    """

    def __init__(self, config: Dict[str, Any], constitution: Dict[str, Any] = None, kernel=None):
        super().__init__(config, constitution=constitution, kernel=kernel)
        self.mock_mode = config.get("mock_mode", False)
        self.scraper = SECEdgarScraper(mock_mode=self.mock_mode)
        self.analytics = InstitutionalRadarAnalytics()  # Manages its own session
        self.reporter = InstitutionalRadarReporter()  # Manages its own LLM

    async def execute(self, year: int = None, quarter: int = None, ciks: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Main execution flow.
        """
        if not year or not quarter:
            # Default to current if not provided (mocking logic)
            # In production, derive from date
            year = 2025
            quarter = 3

        logger.info(f"Executing Institutional Radar for Q{quarter} {year}")

        # 1. Ingestion
        # If CIKs are not provided, we might default to a watchlist or skip ingestion if data exists
        if ciks:
            logger.info("Starting Ingestion Phase...")
            # Run in thread pool to avoid blocking async loop if it takes long
            try:
                # Synchronous call wrapped in executor
                await asyncio.to_thread(self.scraper.run_pipeline, ciks, year, quarter)
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                return {"error": f"Ingestion failed: {str(e)}"}
        else:
            logger.info("No CIKs provided, skipping ingestion or assuming data exists.")

        # 2. Analytics
        logger.info("Starting Analytics Phase...")
        try:
            crowding = await asyncio.to_thread(self.analytics.calculate_crowding_score, year, quarter)
            flows = await asyncio.to_thread(self.analytics.calculate_sector_flows, year, quarter)
            clusters = await asyncio.to_thread(self.analytics.detect_cluster_buys, year, quarter)

            analytics_results = {
                "crowding_score": crowding.to_dict(orient='records') if not crowding.empty else [],
                "sector_flows": flows.to_dict(orient='records') if not flows.empty else [],
                "cluster_buys": clusters.reset_index().to_dict(orient='records') if not clusters.empty else []
            }
        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            return {"error": f"Analytics failed: {str(e)}"}

        # 3. Reporting
        logger.info("Starting Reporting Phase...")
        try:
            quarter_label = f"Q{quarter} {year}"
            narrative = await asyncio.to_thread(
                self.reporter.generate_report,
                crowding,
                flows,
                clusters,
                quarter_label
            )

            html_output = self.reporter.format_html_report(narrative, {"quarter": quarter_label})

            final_output = {
                "narrative": narrative,
                "html": html_output,
                "data": analytics_results,
                "status": "success"
            }

            # Close session
            self.analytics.close()

            return final_output

        except Exception as e:
            logger.error(f"Reporting failed: {e}")
            self.analytics.close()
            return {"error": f"Reporting failed: {str(e)}"}

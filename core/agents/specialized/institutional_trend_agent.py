from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.vertical_risk_agent.ingestion.sec_13f_handler import Sec13FHandler
import pandas as pd
import json

class InstitutionalTrendAgent(AgentBase):
    """
    Agent responsible for monitoring institutional capital flows via 13F filings.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.handler = Sec13FHandler()
        self.watchlist = {
            "Berkshire Hathaway": "0001067983",
            "Renaissance Technologies": "0001037389"
        }

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the agent.
        Can be triggered to run the analysis or generate a report.
        """
        task = kwargs.get("task", "analyze")
        period = kwargs.get("period", "2025-Q3")

        if task == "analyze":
            return self.run_analysis(period)
        elif task == "synthesis":
            context = kwargs.get("context", "")
            return self.generate_synthesis_report(context)
        else:
            return {"error": f"Unknown task: {task}"}

    def run_analysis(self, period: str = "2025-Q3") -> Dict[str, Any]:
        """
        Executes the quarterly trend analysis.
        """
        report_data = {
            "period": period,
            "funds_analyzed": [],
            "synthesis": "Analysis pending generation..."
        }

        previous_period = self._get_previous_period(period)

        for fund_name, cik in self.watchlist.items():
            curr_holdings = self.handler.fetch_holdings(cik, period)
            prev_holdings = self.handler.fetch_holdings(cik, previous_period)

            delta = self.handler.calculate_delta(curr_holdings, prev_holdings)

            # Identify top moves
            new_positions = delta[delta['action_calculated'] == 'NEW']
            exits = delta[delta['action_calculated'] == 'EXIT']

            fund_summary = {
                "name": fund_name,
                "cik": cik,
                "top_buys": new_positions.nlargest(3, 'shares_curr')[['ticker', 'shares_curr']].to_dict('records') if not new_positions.empty else [],
                "top_sells": exits.nlargest(3, 'shares_prev')[['ticker', 'shares_prev']].to_dict('records') if not exits.empty else []
            }
            report_data["funds_analyzed"].append(fund_summary)

        return report_data

    def _get_previous_period(self, period: str) -> str:
        # Simple logic for Q3 -> Q2
        if "Q3" in period:
            return period.replace("Q3", "Q2")
        elif "Q2" in period:
            return period.replace("Q2", "Q1")
        elif "Q1" in period:
            year = int(period.split('-')[0]) - 1
            return f"{year}-Q4"
        elif "Q4" in period:
            return period.replace("Q4", "Q3")
        return period

    def generate_synthesis_report(self, context_text: str) -> Dict[str, Any]:
        """
        Generates a synthesis report based on provided context text.
        """
        return {
            "title": "Institutional Flow Report",
            "date": "2025-10-15",
            "content": context_text,
            "source": "13F Regulatory Filings"
        }

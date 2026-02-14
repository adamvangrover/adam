from typing import Dict, List, Any, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class FraudDetectionAgent(AgentBase):
    """
    A specialized agent for detecting financial anomalies and simulating restatements.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Any = None):
        super().__init__(config, constitution, kernel)
        self.anomalies_detected = []

    async def execute(self, command: str = "audit", data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes the fraud detection or restatement logic.

        Args:
            command (str): The command to execute ("audit", "restate", "investigate").
            data (Dict): The financial data to process.

        Returns:
            Dict: Result of the operation.
        """
        data = data or kwargs.get("data", {})

        if command == "audit" or command == "investigate":
            anomalies = self.detect_anomalies(data)
            return {
                "status": "Audit Complete",
                "anomalies": anomalies,
                "risk_score": len(anomalies) * 10
            }

        elif command == "restate":
            restated_data = self.restate_financials(data)
            return {
                "status": "Restatement Complete",
                "original_data": data,
                "restated_data": restated_data,
                "impact": "Significant downward revision"
            }

        else:
            return {"error": f"Unknown command: {command}"}

    def detect_anomalies(self, data: Dict[str, Any]) -> List[str]:
        """
        Analyzes financial data for inconsistencies.
        """
        anomalies = []

        # Mock Logic for Detection
        revenue = float(data.get("revenue", 0))
        cash_flow = float(data.get("cash_flow", 0))
        expenses = float(data.get("expenses", 0))

        if revenue > 0 and cash_flow < (revenue * 0.1):
            anomalies.append("Revenue high but operating cash flow dangerously low (<10%).")

        if expenses == 0 and revenue > 0:
            anomalies.append("Zero expenses reported with positive revenue (Highly Suspicious).")

        if data.get("growth_rate", 0) > 0.50:
            anomalies.append("Unusual growth rate (>50%) flagged for review.")

        self.anomalies_detected = anomalies
        return anomalies

    def restate_financials(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates a financial restatement by adjusting metrics downwards.
        """
        restated = data.copy()

        # Simulate 'cooking the books' correction
        if "revenue" in restated:
            restated["revenue"] = float(restated["revenue"]) * 0.85  # 15% reduction

        if "expenses" in restated:
            restated["expenses"] = float(restated["expenses"]) * 1.20 # 20% increase (underreported costs)

        if "net_income" in restated:
            # Re-calculate net income if possible, or just slash it
            restated["net_income"] = float(restated.get("revenue", 0)) - float(restated.get("expenses", 0))

        restated["restatement_date"] = "2026-04-01"
        restated["notes"] = "Restated due to accounting irregularities discovered by FraudDetectionAgent."

        return restated

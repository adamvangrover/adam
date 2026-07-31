import json
import asyncio
from datetime import datetime, timezone
from json_logic import jsonLogic
from typing import Dict, Any, List

class MacroSyntheticEngine:
    def __init__(self, rules_path: str = "config/covenants.json"):
        self.rules = self._load_rules(rules_path)
        self.state_matrix = {}
        self.prov_o_telemetry = []

    def _load_rules(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _log_telemetry(self, agent_id: str, action: str, data_node: Any):
        """W3C PROV-O compliant telemetry logging."""
        entry = {
            "prov:Activity": action,
            "prov:wasAssociatedWith": f"urn:adam:agent:{agent_id}",
            "prov:used": data_node,
            "prov:generatedAtTime": datetime.now(timezone.utc).isoformat()
        }
        self.prov_o_telemetry.append(entry)
        # In production, dispatch to Qdrant vector memory layer

    async def ingest_market_ledger(self, jsonl_data: List[str]):
        """Parses the live web search JSONL output into the state matrix."""
        for line in jsonl_data:
            if not line.strip(): continue
            record = json.loads(line)

            # Map variable nodes to the internal state matrix
            node_key = record.get("variable_node")
            if node_key:
                self.state_matrix[node_key] = float(record.get("market_level_value", 0))

        self._log_telemetry("IngestionAgent", "ParsedMarketLedger", self.state_matrix)

    async def evaluate_covenants(self) -> Dict[str, Any]:
        """Runs the state matrix through the jsonLogic risk parameters."""
        results = {}

        # 10Y Yield & HY Spread Check
        results['duration_mirage'] = jsonLogic(
            self.rules['systemic_regime_covenants']['century_duration_mirage_trigger'],
            {"live_ingestion": self.state_matrix}
        )

        # ARM Collateral Check
        results['arm_collateral_status'] = jsonLogic(
            self.rules['systemic_regime_covenants']['arm_ltv_breach'],
            {"live_ingestion": self.state_matrix}
        )

        self._log_telemetry("RiskEvaluationAgent", "EvaluatedCovenants", results)
        return results

    async def generate_dashboard_payload(self) -> Dict:
        """Packages data for the Streamlit/HTML frontend."""
        await self.ingest_market_ledger(self.state_matrix)
        risk_flags = await self.evaluate_covenants()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": "Restrictive" if risk_flags.get('duration_mirage') else "Transitional",
            "market_coordinates": self.state_matrix,
            "risk_flags": risk_flags
        }

# Usage execution handled by core Adam OS orchestration
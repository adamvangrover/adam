# core/agents/legal_agent.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalAgent(AgentBase):
    """
    Agent responsible for legal reasoning, covenant checking, and document review.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the Legal Agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.knowledge_base_path = self.config.get("knowledge_base_path", "knowledge_base/Knowledge_Graph.json")
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    async def execute(self, *args, **kwargs):
        """
        Executes legal analysis tasks.

        Tasks:
        - "review_agreement": Reviews credit agreements or legal docs.
        - "check_covenants": Checks financial covenants.
        - "detect_fraud": Scans for fraud signals.
        - "suggest_strategy": Suggests restructuring strategy.
        """
        task = kwargs.get('task')
        logger.info(f"LegalAgent executing task: {task}")

        if task == "review_agreement":
            doc_text = kwargs.get("document_text", "")
            return self.review_credit_agreement(doc_text)

        elif task == "check_covenants":
            financials = kwargs.get("financials", {})
            return self.check_covenants(financials)

        elif task == "detect_fraud":
            doc_text = kwargs.get("document_text", "")
            financials = kwargs.get("financials", {})
            return self.detect_fraud_signals(doc_text, financials)

        elif task == "suggest_strategy":
            distress_level = kwargs.get("distress_level", "Low")
            return {"strategy": self.suggest_restructuring_strategy(distress_level)}

        else:
            logger.warning(f"Unknown task: {task}")
            return {"error": f"Unknown task: {task}"}

    def review_credit_agreement(self, doc_text: str) -> Dict[str, Any]:
        """
        Simulates reviewing a credit agreement for key clauses.
        """
        logger.info("Reviewing credit agreement...")
        findings = []
        clauses_found = []
        risk_level = "Low"

        doc_lower = doc_text.lower()

        # Keyword based detection
        if "negative pledge" in doc_lower:
            clauses_found.append("Negative Pledge")
            findings.append("Negative Pledge clause limits further secured debt.")

        if "cross-default" in doc_lower:
            clauses_found.append("Cross-Default")
            findings.append("Cross-Default provision present (standard LMA/LSTA).")

        if "change of control" in doc_lower:
            clauses_found.append("Change of Control")
            findings.append("Change of Control put option identified (101%).")

        if "asset sale sweep" in doc_lower:
            clauses_found.append("Asset Sale Sweep")
            findings.append("Mandatory prepayment from asset sale proceeds detected.")

        if "financial covenant" in doc_lower:
            clauses_found.append("Financial Covenants")
            findings.append("Maintenance covenants present (Leverage/Interest Coverage).")

        if not clauses_found:
            findings.append("Standard documentation assumed (no special clauses detected in snippet).")

        # Risk Assessment based on clauses
        if "cross-default" in doc_lower and "change of control" in doc_lower:
             risk_level = "Medium" # Standard but adds complexity

        return {
            "status": "Review Complete",
            "clauses_identified": clauses_found,
            "key_findings": findings,
            "risk_assessment": risk_level
        }

    def check_covenants(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks financial covenants against standard thresholds.
        """
        logger.info("Checking covenants...")
        violations = []
        status = "Pass"

        # Standard Covenants
        max_lev = 4.0
        min_dscr = 1.25

        leverage = financials.get("leverage_ratio", 0)
        dscr = financials.get("dscr", 0)

        if leverage > max_lev:
            violations.append(f"Leverage Ratio {leverage:.2f}x exceeds limit of {max_lev}x")
            status = "Breach"

        if dscr > 0 and dscr < min_dscr: # Check dscr > 0 to avoid div/0 or bad data noise
            violations.append(f"DSCR {dscr:.2f}x is below minimum of {min_dscr}x")
            status = "Breach"

        return {
            "covenant_status": status,
            "violations": violations,
            "tested_metrics": {
                "Leverage": f"{leverage:.2f}x (Limit: {max_lev}x)",
                "DSCR": f"{dscr:.2f}x (Limit: >{min_dscr}x)"
            }
        }

    def detect_fraud_signals(self, doc_text: str, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans for red flags indicating potential fraud or misrepresentation.
        """
        logger.info("Scanning for fraud signals...")
        signals = []
        risk_level = "Low"

        # Textual Signals
        red_flag_keywords = ["material weakness", "restatement", "investigation", "accounting irregularities"]
        for kw in red_flag_keywords:
            if kw in doc_text.lower():
                signals.append(f"Keyword alert: '{kw}' found in documentation.")
                risk_level = "High"

        # Financial Signals (Benford's Law is too complex for this, simple heuristic)
        # Check for perfect integers where unlikely
        rev = financials.get("revenue", 0)
        if rev > 1000 and rev % 100 == 0:
             signals.append("Revenue figure is suspiciously round.")
             if risk_level == "Low": risk_level = "Medium"

        return {
            "fraud_risk_level": risk_level,
            "signals_detected": signals
        }

    def suggest_restructuring_strategy(self, distress_level: str) -> str:
        """
        Suggests a strategy based on distress level.
        """
        if distress_level == "High":
            return "Immediate forbearance agreement recommended. Engage restructuring counsel. Prepare for Chapter 11 filing or distressed exchange."
        elif distress_level == "Medium":
            return "Negotiate covenant waiver or amendment. Consider asset sales to deleverage. Review cash sweep provisions."
        else:
            return "Monitor compliance. No immediate restructuring action required."

    # --- Legacy Stubs (Preserved) ---

    def analyze_legal_aspects(self, acquirer_name, target_name):
        return {"status": "Not implemented"}

    def analyze_legal_standing(self, company_name, company_data):
        return {"status": "Not implemented"}

    def analyze_legal_document(self, document_text):
        # Forward to new method
        return self.review_credit_agreement(document_text)

    def assess_geopolitical_legal_impact(self, geopolitical_event):
        return {}

    def assess_regulatory_legal_impact(self, regulation_change):
        return {}

    def provide_legal_advice(self, query):
        return "Consult outside counsel."

if __name__ == "__main__":
    agent = LegalAgent({})
    async def main():
        res = await agent.execute(task="review_agreement", document_text="This agreement contains a negative pledge and cross-default clause.")
        print(res)
    asyncio.run(main())

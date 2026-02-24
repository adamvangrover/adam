# core/agents/legal_agent.py

import json
import logging

logger = logging.getLogger(__name__)

class LegalAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the Legal Agent with access to legal knowledge
        and reasoning capabilities.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            # logger.error(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            # logger.error(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def review_credit_agreement(self, doc_text: str) -> dict:
        """
        Simulates reviewing a credit agreement for key clauses.
        """
        logger.info("Reviewing credit agreement...")
        findings = []
        clauses_found = []

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
            findings.append("Change of Control put option identified.")

        if not clauses_found:
            findings.append("Standard documentation assumed (no special clauses detected in snippet).")

        return {
            "status": "Review Complete",
            "clauses_identified": clauses_found,
            "key_findings": findings
        }

    def check_covenants(self, financials: dict) -> dict:
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

    def detect_fraud_signals(self, doc_text: str, financials: dict) -> dict:
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

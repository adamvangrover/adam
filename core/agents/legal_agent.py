# core/agents/legal_agent.py

import json
import logging
from typing import List, Dict, Any, Union

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class LegalAgent(AgentBase):
    def __init__(self, config: Dict[str, Any] = None, knowledge_base_path="knowledge_base/Knowledge_Graph.json", **kwargs):
        """
        Initializes the Legal Agent with access to legal knowledge
        and reasoning capabilities.
        """
        super().__init__(config or {}, **kwargs)
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes legal analysis. Supports checking credit agreements, covenants, and fraud signals.
        """
        logger.info("Executing Legal Analysis...")

        is_standard_mode = False
        query = "Legal Analysis"
        doc_text = ""
        financials = {}
        task = "review_agreement"

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                is_standard_mode = True

                context = input_data.context
                doc_text = context.get("document_text", query)
                financials = context.get("financials", {})
                task = context.get("task", "review_agreement")
            elif isinstance(input_data, dict):
                doc_text = input_data.get("document_text", "")
                financials = input_data.get("financials", {})
                task = input_data.get("task", "review_agreement")
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                doc_text = input_data
                query = input_data

        # Fallback to kwargs
        if not doc_text: doc_text = kwargs.get("document_text", "")
        if not financials: financials = kwargs.get("financials", {})
        task = kwargs.get("task", task)

        result = {}

        if task == "review_agreement":
            result = self.review_credit_agreement(doc_text)
        elif task == "check_covenants":
            result = self.check_covenants(financials)
        elif task == "detect_fraud":
            result = self.detect_fraud_signals(doc_text, financials)
        else:
            result = {"error": f"Unknown legal task: {task}"}

        if is_standard_mode:
            answer = f"Legal Analysis Report ({task}):\n\n"
            if "error" in result:
                answer += f"Error: {result['error']}"
            elif task == "review_agreement":
                answer += f"Risk Level: {result.get('risk_assessment')}\n"
                answer += "Key Findings:\n" + "\n".join([f"- {f}" for f in result.get('key_findings', [])])
            elif task == "check_covenants":
                answer += f"Status: {result.get('covenant_status')}\n"
                if result.get("violations"):
                    answer += "Violations:\n" + "\n".join([f"- {v}" for v in result['violations']])
            elif task == "detect_fraud":
                answer += f"Fraud Risk Level: {result.get('fraud_risk_level')}\n"
                if result.get("signals_detected"):
                    answer += "Signals:\n" + "\n".join([f"- {s}" for s in result['signals_detected']])

            return AgentOutput(
                answer=answer,
                sources=["Legal Knowledge Base", "Contract Logic"],
                confidence=0.85 if "error" not in result else 0.0,
                metadata=result
            )

        return result

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
        return self.review_credit_agreement(document_text)

    def assess_geopolitical_legal_impact(self, geopolitical_event):
        return {}

    def assess_regulatory_legal_impact(self, regulation_change):
        return {}

    def provide_legal_advice(self, query):
        return "Consult outside counsel."

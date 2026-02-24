import json
import re
import os
import random
import logging
import sys
from datetime import datetime

# Ensure we can import core modules
sys.path.append(os.getcwd())

# Mock imports if core is not fully available in this context, or use real ones
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent
    from core.agents.legal_agent import LegalAgent
except ImportError:
    # Minimal mock classes for standalone execution
    class RiskAssessmentAgent:
        def __init__(self, config): pass
        def assess_loan_risk(self, details, borrower):
            return {"risk_quant_metrics": {"PD": 0.05, "LGD": 0.45, "RWA": 1200, "RAROC": 0.15}, "overall_risk_score": 0.4}

    class LegalAgent:
        def __init__(self): pass
        def review_credit_agreement(self, text):
            return {"clauses_identified": ["Negative Pledge", "Cross-Default"]}
        def detect_fraud_signals(self, text, fin):
            return {"fraud_risk_level": "Low", "signals_detected": []}
        def suggest_restructuring_strategy(self, level):
            return "Monitor compliance."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimulateInteraction")

def load_mock_data():
    path = "showcase/js/mock_data.js"
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return {}

    with open(path, "r") as f:
        content = f.read()

    # Strip JS assignment: window.MOCK_DATA = { ... };
    match = re.search(r"window\.MOCK_DATA\s*=\s*(\{.*)", content, re.DOTALL)
    if not match:
        return {}

    json_str = match.group(1).strip()
    if json_str.endswith(";"):
        json_str = json_str[:-1]

    try:
        data = json.loads(json_str)
        return data
    except:
        return {}

def extract_companies(data):
    companies = {}
    if 'credit_memos' in data:
        return data['credit_memos']
    for k, v in data.items():
        if isinstance(v, dict) and ('borrower_name' in v):
            name = v.get('borrower_name', k).replace(" ", "_")
            companies[name] = v
    return companies

def run_simulation():
    data = load_mock_data()
    companies = extract_companies(data)

    # Fallback data if load fails
    if not companies:
        companies = {
            "Apple_Inc": { "borrower_name": "Apple Inc.", "historical_financials": [{"revenue": 383000, "ebitda": 125000}] },
            "Tesla_Inc": { "borrower_name": "Tesla Inc.", "historical_financials": [{"revenue": 96000, "ebitda": 15000}] }
        }

    risk_agent = RiskAssessmentAgent({"knowledge_base_path": "core/data/risk_rating_mapping.json"})
    legal_agent = LegalAgent()

    interactions = {}

    for company_id, company_data in companies.items():
        name = company_data.get('borrower_name', company_id)
        # Use underscores for keys to match JS logic
        key_name = name.replace(" ", "_")

        log = []
        highlights = []
        ui_events = [] # NEW: Visual choreography

        financials = {}
        if 'historical_financials' in company_data and company_data['historical_financials']:
            financials = company_data['historical_financials'][0]

        # --- 1. Risk Assessment Phase ---

        # Event 1: RiskBot goes to Financials
        ui_events.append({
            "order": 1,
            "actor": "RiskBot",
            "tab": "annex-a",
            "target": "#financials-table",
            "action": "highlight",
            "duration": 2000,
            "message": "Scanning historical revenue trends..."
        })

        log.append({
            "actor": "RiskBot",
            "message": f"Initiating credit assessment for {name}. Analyzing historical financials.",
            "timestamp": datetime.now().isoformat()
        })

        # Event 2: RiskBot checks EBITDA Adjustments
        ui_events.append({
            "order": 2,
            "actor": "RiskBot",
            "tab": "annex-a",
            "target": "#fin-adjustments-panel",
            "action": "move",
            "duration": 1500,
            "message": "Verifying EBITDA add-backs consistency..."
        })

        loan_details = {"loan_amount": 1000, "interest_rate": "5.5%", "seniority": "Senior Secured"}
        risk_result = risk_agent.assess_loan_risk(loan_details, {"financial_data": financials})
        qm = risk_result.get("risk_quant_metrics", {})

        log.append({
            "actor": "RiskBot",
            "message": f"Calculated preliminary risk metrics: PD={qm.get('PD', 0.05):.2%}, LGD={qm.get('LGD', 0.4):.2%}.",
            "timestamp": datetime.now().isoformat()
        })

        # Event 3: RiskBot Updates Quant Tab
        ui_events.append({
            "order": 3,
            "actor": "RiskBot",
            "tab": "risk-quant",
            "target": "#risk-quant-container",
            "action": "highlight",
            "duration": 2000,
            "message": "Updating PD/LGD models based on new inputs."
        })

        # --- 2. Legal Review Phase ---

        # Event 4: LegalAI checks Document
        ui_events.append({
            "order": 4,
            "actor": "LegalAI",
            "tab": "memo", # Assuming doc viewer is visible or linked
            "target": "#pdf-viewer",
            "action": "highlight",
            "duration": 3000,
            "message": "Scanning 10-K and Credit Agreement..."
        })

        log.append({
            "actor": "LegalAI",
            "message": f"Reviewing credit documentation and 10-K filings for {name}.",
            "timestamp": datetime.now().isoformat()
        })

        doc_text = f"Credit Agreement for {name}. Standard LMA terms. Negative Pledge included."
        legal_review = legal_agent.review_credit_agreement(doc_text)

        if legal_review['clauses_identified']:
            # Event 5: LegalAI flags clauses
            ui_events.append({
                 "order": 5,
                 "actor": "LegalAI",
                 "tab": "annex-c", # Capital Structure / Covenants
                 "target": "#cap-structure-container",
                 "action": "highlight",
                 "duration": 2000,
                 "message": "Mapping restrictive covenants to capital structure."
            })

            clauses = ", ".join(legal_review['clauses_identified'])
            log.append({
                "actor": "LegalAI",
                "message": f"Identified key protective clauses: {clauses}.",
                "timestamp": datetime.now().isoformat()
            })
            for c in legal_review['clauses_identified']:
                highlights.append({"type": "clause", "label": c, "status": "Protected"})

        # --- 3. Convergence Phase ---

        # Event 6: Both Agents Meet in Interlock Tab
        ui_events.append({
            "order": 6,
            "actor": "System", # Special event to switch tab for user
            "tab": "interlock",
            "target": "#interlock-log",
            "action": "focus",
            "duration": 1000,
            "message": "Initiating Interlock Sequence..."
        })

        fraud_check = legal_agent.detect_fraud_signals(doc_text, financials)
        if fraud_check['fraud_risk_level'] != "Low":
             log.append({ "actor": "LegalAI", "message": f"FRAUD SIGNAL: {fraud_check['signals_detected'][0]}", "timestamp": datetime.now().isoformat() })
             highlights.append({"type": "risk", "label": "Fraud Alert", "status": "Critical"})
        else:
             log.append({ "actor": "LegalAI", "message": "No significant fraud indicators detected.", "timestamp": datetime.now().isoformat() })

        strategy = legal_agent.suggest_restructuring_strategy("Low")
        log.append({ "actor": "LegalAI", "message": f"Strategy: {strategy}", "timestamp": datetime.now().isoformat() })

        log.append({ "actor": "System", "message": "Consensus Reached.", "timestamp": datetime.now().isoformat() })

        interactions[key_name] = {
            "borrower_name": name,
            "logs": log,
            "highlights": highlights,
            "ui_events": ui_events # New Field
        }

    output_path = "showcase/data/risk_legal_interaction.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(interactions, f, indent=2)

    logger.info(f"Simulation complete. Data saved to {output_path}")

if __name__ == "__main__":
    run_simulation()

from __future__ import annotations
import logging
import datetime
import uuid
import random
from typing import Dict, Any, List, Optional

from core.engine.icat import ICATEngine
from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.legal_agent import LegalAgent
from core.financial_data.icat_schema import LBOParameters, DebtTranche, ICATOutput

logger = logging.getLogger("CreditMemoOrchestrator")

class CreditMemoOrchestrator:
    """
    Orchestrates the end-to-end credit memo generation process.
    Integrates ICAT (Financials), Risk Agent (Quant), and Legal Agent (Qual/Docs).
    Simulates System 2 Interlock and RAG.
    """

    def __init__(self, mock_library: Dict[str, Any], output_dir: str = "showcase/data"):
        self.mock_library = mock_library
        self.output_dir = output_dir

        # Initialize Agents
        self.risk_agent = RiskAssessmentAgent(config={})
        self.legal_agent = LegalAgent()

        # Initialize ICAT with mock DB injection
        self.icat_engine = ICATEngine(mock_data_path="showcase/data/icat_mock_data.json")
        self.icat_engine.mock_db = self.mock_library

    def process_entity(self, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full processing pipeline for a single entity.
        """
        logger.info(f"Orchestrator processing {data['name']}...")

        # 1. Financial Analysis
        icat_output = self._run_financials(key, data)
        if not icat_output:
            logger.error(f"Financial analysis failed for {key}")
            return {}

        # 2. Risk Analysis
        risk_output = self._run_risk(data, icat_output)

        # 3. Legal Review
        legal_output, fraud_check = self._run_legal(data)

        # 4. Interlock Simulation
        logs, ui_events = self.run_interlock(data['name'], icat_output, risk_output, legal_output, fraud_check)

        # 5. RAG Simulation (Citation Generation)
        rag_citations = self.simulate_rag(data['docs']['10-K'], "risk factors")

        # 6. Construct Final Memo
        memo_data = self.construct_memo(data, icat_output, risk_output, legal_output, logs, rag_citations)

        return {
            "memo": memo_data,
            "interaction_log": {
                "borrower_name": data['name'],
                "logs": logs,
                "highlights": self._extract_highlights(legal_output, fraud_check),
                "ui_events": ui_events
            }
        }

    def _run_financials(self, key: str, data: Dict[str, Any]) -> Optional[ICATOutput]:
        # LBO Logic
        lbo_params = None
        if "Distressed" in data['sector'] or "Consumer" in data['sector']:
             latest_ebitda = data['historical']['ebitda'][-1]
             senior_amt = latest_ebitda * 3.0
             mezz_amt = latest_ebitda * 1.5

             lbo_params = LBOParameters(
                 entry_multiple=8.0,
                 exit_multiple=8.0,
                 equity_contribution_percent=0.3,
                 tax_rate=0.25,
                 debt_structure=[
                     DebtTranche(name="Senior", amount=senior_amt, interest_rate=0.07, amortization_rate=0.01),
                     DebtTranche(name="Mezz", amount=mezz_amt, interest_rate=0.12, amortization_rate=0.0)
                 ]
             )

        try:
            return self.icat_engine.analyze(ticker=key, source="mock", lbo_params=lbo_params)
        except Exception as e:
            logger.error(f"ICAT Error: {e}")
            return None

    def _run_risk(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        fin_data_risk = {
            "credit_rating": "BBB" if icat.credit_metrics.z_score > 1.8 else "CCC",
            "z_score": icat.credit_metrics.z_score,
            "total_assets": data['historical']['total_assets'][-1],
            "total_debt": data['historical']['total_debt'][-1],
            "cash": data['historical']['cash'][-1],
            "monthly_burn_rate": 0,
            "liquidity_ratio": icat.credit_metrics.interest_coverage
        }

        loan_details = {
            "seniority": "Senior Secured" if "Distressed" in data['sector'] else "Senior Unsecured",
            "collateral_value": data['historical']['total_assets'][-1] * 0.5,
            "loan_amount": data['historical']['total_debt'][-1] * 0.2,
            "interest_rate": 0.07
        }

        return self.risk_agent.assess_loan_risk(loan_details, fin_data_risk)

    def _run_legal(self, data: Dict[str, Any]):
        doc_text = data['docs'].get('Credit_Agreement', '')
        ten_k = data['docs'].get('10-K', '')

        legal_output = self.legal_agent.review_credit_agreement(doc_text)
        fraud_check = self.legal_agent.detect_fraud_signals(ten_k, {"revenue": data['historical']['revenue'][-1]})
        return legal_output, fraud_check

    def run_interlock(self, name, icat, risk, legal, fraud):
        """Simulates the back-and-forth between agents."""
        logs = []
        ui_events = []
        timestamp = datetime.datetime.now()

        # 1. Risk Phase
        logs.append({
            "actor": "RiskBot",
            "message": f"Initiating credit assessment for {name}.",
            "timestamp": timestamp.isoformat()
        })
        ui_events.append({
            "order": 1, "actor": "RiskBot", "tab": "annex-a", "target": "#financials-table",
            "action": "highlight", "duration": 2000, "message": "Analyzing trends..."
        })

        pd = risk['risk_quant_metrics']['PD']
        lgd = risk['risk_quant_metrics']['LGD']
        logs.append({
            "actor": "RiskBot",
            "message": f"Metrics: PD={pd*100:.2f}%, LGD={lgd*100:.2f}%.",
            "timestamp": (timestamp + datetime.timedelta(seconds=2)).isoformat()
        })

        # 2. Legal Phase
        logs.append({
            "actor": "LegalAI",
            "message": f"Reviewing docs for {name}.",
            "timestamp": (timestamp + datetime.timedelta(seconds=4)).isoformat()
        })
        ui_events.append({
            "order": 3, "actor": "LegalAI", "tab": "memo", "target": "#pdf-viewer",
            "action": "highlight", "duration": 3000, "message": "Scanning Docs..."
        })

        # 3. Interlock Logic
        clauses = legal['clauses_identified']
        if clauses:
             logs.append({
                "actor": "LegalAI",
                "message": f"Clauses found: {', '.join(clauses)}.",
                "timestamp": (timestamp + datetime.timedelta(seconds=6)).isoformat()
            })
             if "Negative Pledge" in clauses:
                  ui_events.append({
                    "order": 4, "actor": "LegalAI", "tab": "annex-c", "target": "#cap-structure-container",
                    "action": "highlight", "duration": 2000, "message": "Verifying security..."
                  })

        # 4. Fraud
        if fraud['fraud_risk_level'] != "Low":
             logs.append({
                "actor": "LegalAI",
                "message": f"FRAUD SIGNAL: {fraud['signals_detected'][0]}",
                "timestamp": (timestamp + datetime.timedelta(seconds=7)).isoformat()
            })

        logs.append({
            "actor": "System",
            "message": "Consensus Reached.",
            "timestamp": (timestamp + datetime.timedelta(seconds=8)).isoformat()
        })

        return logs, ui_events

    def simulate_rag(self, doc_text: str, query: str) -> List[Dict[str, Any]]:
        """
        Simulates Retrieval Augmented Generation by extracting chunks.
        In a real system, this would query a Vector DB.
        """
        sentences = doc_text.split('. ')
        citations = []

        # Simple extraction strategy: random sentence or keyword match
        # Logic: find sentences with keywords relevant to credit
        keywords = ["risk", "debt", "revenue", "competition", "regulation", "growth"]

        candidates = [s for s in sentences if any(k in s.lower() for k in keywords)]

        # Fallback if no keywords
        if not candidates:
            candidates = sentences[:2]

        # Select 1-2 chunks
        selected = candidates[:2]

        for i, text in enumerate(selected):
            citations.append({
                "doc_id": "10-K_FY2025.pdf",
                "chunk_id": str(uuid.uuid4())[:8],
                "page_number": random.randint(10, 50),
                "text": text + "."
            })

        return citations

    def construct_memo(self, data, icat, risk, legal, logs, citations):
        # Calculate Risk Score (inverted)
        raw_risk = min(risk['overall_risk_score'], 1.0)
        risk_score = int((1.0 - raw_risk) * 100)

        # Construct Hist Data
        hist = []
        years = data['historical']['year']
        for i, year in enumerate(years):
            record = {"period": str(year)}
            for k, v in data['historical'].items():
                if k != 'year': record[k] = v[i]
            hist.append(record)
        hist.sort(key=lambda x: x['period'], reverse=True)

        # Sections with Citations
        risk_content = f"Primary Risk Factors:\n1. {risk['risk_factors'].get('geopolitical_risk', ['N/A'])[0]}\n2. Market Volatility (Beta: {data['market_data']['beta']})\n\nQuantitative Model:\nProbability of Default: {risk['risk_quant_metrics']['PD']*100:.2f}%\nLoss Given Default: {risk['risk_quant_metrics']['LGD']*100:.2f}%"

        # Inject simulated citations into Executive Summary
        exec_summary = f"{data['description']}\n\nKey Credit Stats:\n- Net Leverage: {icat.credit_metrics.net_leverage:.2f}x\n- Interest Coverage: {icat.credit_metrics.interest_coverage:.2f}x\n- Z-Score: {icat.credit_metrics.z_score:.2f}"
        if citations:
            exec_summary += f"  [Ref: {citations[0]['doc_id']}]"

        sections = [
            {
                "title": "Executive Summary",
                "content": exec_summary,
                "citations": citations,
                "author_agent": "Writer"
            },
            {
                "title": "Risk Analysis",
                "content": risk_content,
                "citations": [],
                "author_agent": "Risk Assessment Agent"
            },
            {
                "title": "Legal & Covenants",
                "content": f"Document Review Summary:\n{legal['key_findings'][0]}\n\nClauses Identified: {', '.join(legal['clauses_identified'])}",
                "citations": [],
                "author_agent": "Legal Agent"
            }
        ]

        return {
            "borrower_name": data['name'],
            "borrower_details": {"name": data['name'], "sector": data['sector']},
            "report_date": datetime.datetime.now().isoformat(),
            "risk_score": risk_score,
            "historical_financials": hist,
            "sections": sections,
            "key_strengths": ["Market Leader", "Strong Cash Flow"],
            "key_weaknesses": ["Cyclical Industry", "Regulatory Risk"],
            "dcf_analysis": {
                "enterprise_value": icat.valuation_metrics.enterprise_value,
                "share_price": icat.valuation_metrics.dcf_value / (data['market_data']['market_cap'] / data['market_data']['share_price']),
                "wacc": data['forecast_assumptions']['discount_rate'],
                "growth_rate": data['forecast_assumptions']['terminal_growth_rate'],
                "terminal_value": icat.valuation_metrics.dcf_value * 0.4,
                "free_cash_flow": [1000, 1100, 1200, 1300, 1400]
            },
            "pd_model": {
                "model_score": int((1 - risk['risk_quant_metrics']['PD']) * 100),
                "implied_rating": "BBB",
                "one_year_pd": risk['risk_quant_metrics']['PD'],
                "five_year_pd": risk['risk_quant_metrics']['PD'] * 3,
                "input_factors": {"Leverage": icat.credit_metrics.net_leverage, "Z-Score": icat.credit_metrics.z_score}
            },
            "system_two_critique": {
                "critique_points": ["Valuation aligns with sector.", "Legal review confirms standard protections."],
                "conviction_score": 0.85,
                "verification_status": "PASS",
                "author_agent": "System 2",
                "quantitative_analysis": {
                    "ratios_checked": ["Leverage", "DSCR", "Z-Score"],
                    "variance_analysis": "Consistent",
                    "dcf_validation": "WACC within range"
                }
            },
            "equity_data": data['market_data']
        }

    def _extract_highlights(self, legal, fraud):
        highlights = []
        for c in legal['clauses_identified']:
            highlights.append({"type": "clause", "label": c, "status": "Protected"})
        if fraud['fraud_risk_level'] != "Low":
            highlights.append({"type": "risk", "label": "Fraud Alert", "status": "Critical"})
        return highlights

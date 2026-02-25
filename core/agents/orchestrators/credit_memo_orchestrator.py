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

    def process_entity(self, key: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full processing pipeline for a single entity.
        If 'data' is not provided, it attempts to load from library or generate synthetic data.
        """
        # If data is missing, try to find in library
        if not data:
            if key in self.mock_library:
                data = self.mock_library[key]
            else:
                # Runtime Generation: Create synthetic profile
                logger.info(f"Ticker {key} not found in library. Generating synthetic profile...")
                data = self.generate_synthetic_profile(key)

        logger.info(f"Orchestrator processing {data['name']}...")

        # Update ICAT mock db for this run if needed
        if key not in self.icat_engine.mock_db:
             self.icat_engine.mock_db[key] = data

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
        rag_citations = self.simulate_rag(data['docs']['10-K'], data['sector'])

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

    def generate_synthetic_profile(self, ticker: str, sector: str = None) -> Dict[str, Any]:
        """
        Generates a plausible synthetic financial profile for an unknown ticker.
        """
        sectors = ["Technology", "Healthcare", "Energy", "Consumer Cyclical", "Financial Services", "Industrials"]
        if not sector:
            sector = random.choice(sectors)

        # Base multiplier logic (randomize slightly)
        scale_factor = random.uniform(0.5, 5.0) # Scale of company size (billions)

        # Financial Templates
        if sector == "Technology":
            margin = 0.35
            growth = 0.15
            leverage = 1.5
            pe = 35.0
        elif sector == "Energy":
            margin = 0.18
            growth = 0.02
            leverage = 2.5
            pe = 12.0
        else: # General
            margin = 0.20
            growth = 0.05
            leverage = 3.0
            pe = 18.0

        # Generate History (3 years)
        base_rev = 10000 * scale_factor
        revenue = [base_rev * (1 - growth), base_rev, base_rev * (1 + growth)]
        ebitda = [r * margin for r in revenue]
        net_income = [e * 0.6 for e in ebitda] # Simple tax/interest proxy

        total_debt = [e * leverage for e in ebitda]
        cash = [d * 0.2 for d in total_debt]
        total_assets = [d * 2.5 for d in total_debt] # 40% debt to assets
        total_liabilities = [a * 0.6 for a in total_assets]
        interest_expense = [d * 0.06 for d in total_debt]
        capex = [r * 0.05 for r in revenue]

        share_price = random.uniform(50, 200)
        shares_out = (net_income[-1] * pe) / share_price
        market_cap = shares_out * share_price

        return {
            "ticker": ticker,
            "name": f"{ticker} Inc.",
            "sector": sector,
            "description": f"Generated synthetic profile for {ticker}, a leading player in the {sector} sector.",
            "historical": {
                "revenue": revenue,
                "ebitda": ebitda,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_debt": total_debt,
                "cash": cash,
                "interest_expense": interest_expense,
                "capex": capex,
                "year": [2023, 2024, 2025]
            },
            "forecast_assumptions": {
                "revenue_growth": [growth] * 5,
                "ebitda_margin": [margin] * 5,
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.03
            },
            "market_data": {
                "share_price": share_price,
                "market_cap": market_cap,
                "beta": random.uniform(0.8, 1.5),
                "pe_ratio": pe,
                "price_data": [share_price * (1 + random.uniform(-0.1, 0.1)) for _ in range(6)]
            },
            "docs": {
                "10-K": self._generate_synthetic_docs(sector, "10-K"),
                "Credit_Agreement": self._generate_synthetic_docs(sector, "Credit_Agreement")
            }
        }

    def _generate_synthetic_docs(self, sector: str, doc_type: str) -> str:
        """Generates mock text for RAG simulation."""
        if doc_type == "10-K":
            common = "The company faces significant competition. Global economic conditions may affect results. "
            if sector == "Technology":
                return common + "Rapid technological changes could render products obsolete. Cybersecurity risks are elevated. Intellectual property protection is critical."
            elif sector == "Energy":
                return common + "Commodity price volatility significantly impacts margins. Environmental regulations are becoming stricter. Geopolitical instability in production regions is a risk."
            elif sector == "Healthcare":
                return common + "Regulatory approval for new products is uncertain. Patent expirations may lead to generic competition. Healthcare policy reform could impact pricing."
            else:
                return common + "Supply chain disruptions could increase costs. Labor shortages may impact operations."
        else: # Credit Agreement
            return "Standard LSTA terms. Negative Pledge on significant assets. Cross-Default threshold $50M. Change of Control trigger. Financial covenants: Net Leverage < 4.5x, Interest Coverage > 2.5x."

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

    def simulate_rag(self, doc_text: str, sector: str = "General") -> List[Dict[str, Any]]:
        """
        Simulates Retrieval Augmented Generation with context awareness.
        """
        sentences = doc_text.split('. ')
        citations = []

        # Keywords logic enhanced
        keywords = ["risk", "debt", "revenue", "competition", "regulation", "growth"]
        if sector == "Technology":
            keywords.extend(["cybersecurity", "intellectual property", "obsolete"])
        elif sector == "Energy":
             keywords.extend(["commodity", "environmental", "geopolitical"])
        elif sector == "Healthcare":
             keywords.extend(["patent", "approval", "reform"])

        candidates = [s for s in sentences if any(k in s.lower() for k in keywords)]

        # Fallback
        if not candidates:
            candidates = sentences[:3]

        # Select up to 3 chunks
        selected = candidates[:3]

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
        risk_content = f"Primary Risk Factors:\n1. {risk['risk_factors'].get('geopolitical_risk', ['N/A'])[0]}\n2. Market Volatility (Beta: {data['market_data']['beta']:.2f})\n\nQuantitative Model:\nProbability of Default: {risk['risk_quant_metrics']['PD']*100:.2f}%\nLoss Given Default: {risk['risk_quant_metrics']['LGD']*100:.2f}%"

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

        # Peer Comps Generation (if not present)
        peer_comps = []
        if 'peer_comps' not in data:
            # Generate synthetic peers
            for i in range(3):
                peer_name = f"{data['sector']} Peer {i+1}"
                peer_comps.append({
                    "ticker": f"PEER{i+1}",
                    "name": peer_name,
                    "ev_ebitda": random.uniform(8, 20),
                    "pe_ratio": random.uniform(15, 40),
                    "leverage_ratio": random.uniform(1, 4),
                    "market_cap": data['market_data']['market_cap'] * random.uniform(0.5, 1.5)
                })
        else:
            peer_comps = data['peer_comps']

        # Debt Facilities Generation (if not present)
        debt_facilities = []
        if 'debt_facilities' not in data:
             debt_facilities = [
                 {"facility_type": "Revolver", "amount_committed": 2000, "amount_drawn": 500, "interest_rate": "S+200", "snc_rating": "Pass", "ltv": 0.2},
                 {"facility_type": "Term Loan B", "amount_committed": 5000, "amount_drawn": 5000, "interest_rate": "S+350", "snc_rating": "Pass", "ltv": 0.5}
             ]
        else:
             debt_facilities = data['debt_facilities']

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
                "share_price": icat.valuation_metrics.dcf_value / (data['market_data']['market_cap'] / data['market_data']['share_price']) if data['market_data']['market_cap'] else 0,
                "wacc": data['forecast_assumptions']['discount_rate'],
                "growth_rate": data['forecast_assumptions']['terminal_growth_rate'],
                "terminal_value": icat.valuation_metrics.dcf_value * 0.4,
                "free_cash_flow": [1000, 1100, 1200, 1300, 1400] # Should ideally be calculated
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
            "equity_data": data['market_data'],
            "peer_comps": peer_comps,
            "debt_facilities": debt_facilities,
            "repayment_schedule": [
                {"year": "2025", "amount": 500, "source": "Amortization"},
                {"year": "2026", "amount": 500, "source": "Amortization"},
                {"year": "2027", "amount": 4000, "source": "Maturity Wall"}
            ]
        }

    def _extract_highlights(self, legal, fraud):
        highlights = []
        for c in legal['clauses_identified']:
            highlights.append({"type": "clause", "label": c, "status": "Protected"})
        if fraud['fraud_risk_level'] != "Low":
            highlights.append({"type": "risk", "label": "Fraud Alert", "status": "Critical"})
        return highlights

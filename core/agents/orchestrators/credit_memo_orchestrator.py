from __future__ import annotations
import logging
import datetime
import uuid
import random
import asyncio
from typing import Dict, Any, List, Optional

from core.engine.icat import ICATEngine
from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.legal_agent import LegalAgent
from core.agents.regulatory_compliance_agent import RegulatoryComplianceAgent
from core.financial_data.icat_schema import LBOParameters, DebtTranche, ICATOutput
from core.data.realtime_fetcher import RealtimeFetcher
from core.tools.web_search_tool import WebSearchTool
from core.engine.valuation_utils import calculate_dcf, get_price_targets

logger = logging.getLogger("CreditMemoOrchestrator")

class CreditMemoOrchestrator:
    """
    Orchestrates the end-to-end credit memo generation process.
    Integrates ICAT (Financials), Risk Agent (Quant), and Legal Agent (Qual/Docs).
    Simulates System 2 Interlock and RAG.
    """

    def __init__(self, mock_library: Dict[str, Any] = None, output_dir: str = "showcase/data", mode: str = "mock"):
        self.mock_library = mock_library or {}
        self.output_dir = output_dir
        self.mode = mode

        # Initialize Agents
        self.risk_agent = RiskAssessmentAgent(config={})
        self.legal_agent = LegalAgent()

        # Initialize Regulatory Compliance Agent (using mock config for now)
        self.regulatory_agent = RegulatoryComplianceAgent(config={
            "agent_id": "RegulatoryBot",
            "llm_config": {"model": "gpt-3.5-turbo"} # Placeholder if needed
        })

        # Initialize Tools for Live Mode
        if self.mode == "live":
            self.realtime_fetcher = RealtimeFetcher()
            self.web_search = WebSearchTool()
        else:
            self.realtime_fetcher = None
            self.web_search = None

        # Initialize ICAT with mock DB injection
        # In live mode, we will inject fetched data into the mock_db dynamically
        self.icat_engine = ICATEngine(mock_data_path="showcase/data/icat_mock_data.json")
        self.icat_engine.mock_db = self.mock_library

    def process_entity(self, key: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full processing pipeline for a single entity.
        """
        logger.info(f"Orchestrator processing {key} in {self.mode} mode...")

        # 0. Fetch Data if Live
        if self.mode == "live":
            fetched_data = self.realtime_fetcher.fetch_data(key)
            if not fetched_data:
                logger.error(f"Could not fetch live data for {key}")
                return {}

            # Enrich with Docs via Web Search
            fetched_data = self._enrich_with_docs(key, fetched_data)

            data = fetched_data
            # Inject into ICAT's mock_db so it can 'find' it
            self.icat_engine.mock_db[key] = data

        if not data:
            logger.error(f"No data provided for {key}")
            return {}

        # 1. Financial Analysis
        icat_output = self._run_financials(key, data)
        if not icat_output:
            logger.error(f"Financial analysis failed for {key}")
            return {}

        # 1.1. Enhanced Valuation (DCF Scenarios)
        valuation_report = self._run_valuation_scenarios(data, icat_output)

        # 2. Risk Analysis
        risk_output = self._run_risk(data, icat_output)

        # 3. Legal Review
        legal_output, fraud_check = self._run_legal(data)

        # 3.1. Regulatory Compliance Check
        regulatory_report = self._run_regulatory_check(data, icat_output)

        # 4. Interlock Simulation
        logs, ui_events = self.run_interlock(data['name'], icat_output, risk_output, legal_output, fraud_check, regulatory_report)

        # 5. RAG Simulation (Citation Generation)
        # Use real doc text if available
        doc_text = data['docs'].get('10-K', '')
        rag_citations = self.simulate_rag(doc_text, "risk factors")

        # 6. System 2 Critique (Logic Consistency)
        system_2_critique = self._run_system_2_critique(risk_output, valuation_report, regulatory_report)

        # 7. Construct Final Memo
        memo_data = self.construct_memo(
            data, icat_output, risk_output, legal_output, logs, rag_citations,
            valuation_report, regulatory_report, system_2_critique
        )

        return {
            "memo": memo_data,
            "interaction_log": {
                "borrower_name": data['name'],
                "logs": logs,
                "highlights": self._extract_highlights(legal_output, fraud_check, regulatory_report),
                "ui_events": ui_events
            }
        }

    def _enrich_with_docs(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses WebSearchTool to find 10-K text or summary.
        """
        try:
            # Synchronous wrapper for async tool
            query = f"{ticker} 10-K risk factors summary"
            logger.info(f"Searching for docs: {query}")

            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            content = loop.run_until_complete(self.web_search.execute(query=query))
            loop.close()

            if content and not content.startswith("Error"):
                data['docs']['10-K'] = content[:5000] # Limit size
            else:
                data['docs']['10-K'] = "Could not retrieve 10-K content via search."

        except Exception as e:
            logger.error(f"Error fetching docs for {ticker}: {e}")

        return data

    def _run_financials(self, key: str, data: Dict[str, Any]) -> Optional[ICATOutput]:
        # LBO Logic
        lbo_params = None
        if "Distressed" in data['sector'] or "Consumer" in data['sector']:
             # Use safe get
             ebitda_list = data['historical'].get('ebitda', [0])
             latest_ebitda = ebitda_list[-1] if ebitda_list else 0
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
            # We use source="mock" because we injected the data into mock_db
            return self.icat_engine.analyze(ticker=key, source="mock", lbo_params=lbo_params)
        except Exception as e:
            logger.error(f"ICAT Error: {e}")
            return None

    def _run_valuation_scenarios(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        """
        Runs Base, Bull, and Bear case valuations using DCF Engine.
        """
        try:
            # Extract inputs from ICAT or Data
            fcf = icat.valuation_metrics.dcf_value * 0.05 # Approximate if not directly exposed
            if not fcf:
                # Fallback
                fcf = data['historical']['net_income'][-1] + data['historical']['capex'][-1]

            base_inputs = {
                "fcf": fcf,
                "growth_rate": data['forecast_assumptions']['terminal_growth_rate'],
                "beta": data['market_data'].get('beta', 1.0),
                "shares_outstanding": data['market_data']['market_cap'] / data['market_data']['share_price'] if data['market_data']['share_price'] else 1,
                "net_debt": data['historical']['total_debt'][-1] - data['historical']['cash'][-1]
            }

            # 1. Base Case
            base_dcf = calculate_dcf(base_inputs)

            # 2. Bull Case (Higher Growth, Lower Risk)
            bull_scenario = {
                "growth_rate": base_inputs['growth_rate'] * 1.5,
                "market_risk_premium": 0.04
            }
            bull_dcf = calculate_dcf(base_inputs, scenario=bull_scenario)

            # 3. Bear Case (Lower Growth, Higher WACC)
            bear_scenario = {
                "growth_rate": base_inputs['growth_rate'] * 0.5,
                "risk_free_rate": 0.06 # Higher rates
            }
            bear_dcf = calculate_dcf(base_inputs, scenario=bear_scenario)

            return {
                "base_case": base_dcf,
                "bull_case": bull_dcf,
                "bear_case": bear_dcf,
                "price_targets": get_price_targets(base_dcf['intrinsic_share_price'], 0.25) # 25% vol assumption
            }
        except Exception as e:
            logger.error(f"Valuation failed: {e}")
            return {}

    def _run_risk(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        # Handle potential missing keys safely
        hist = data.get('historical', {})
        total_assets = hist.get('total_assets', [0])[-1]
        total_debt = hist.get('total_debt', [0])[-1]
        cash = hist.get('cash', [0])[-1]

        fin_data_risk = {
            "credit_rating": "BBB" if icat.credit_metrics.z_score > 1.8 else "CCC",
            "z_score": icat.credit_metrics.z_score,
            "total_assets": total_assets,
            "total_debt": total_debt,
            "cash": cash,
            "monthly_burn_rate": 0,
            "liquidity_ratio": icat.credit_metrics.interest_coverage
        }

        loan_details = {
            "seniority": "Senior Secured" if "Distressed" in data.get('sector','') else "Senior Unsecured",
            "collateral_value": total_assets * 0.5,
            "loan_amount": total_debt * 0.2,
            "interest_rate": 0.07
        }

        borrower_data = fin_data_risk # Using fin data as borrower profile

        # Pass "borrower_data" for correct PD calculation in RiskAgent
        combined_data = {
            "loan_details": loan_details,
            "borrower_data": borrower_data
        }

        # Use loop to run async execute method of agent if needed,
        # but RiskAssessmentAgent.assess_loan_risk is synchronous logic inside execute.
        # We can call assess_loan_risk directly as it's a public method on the class instance.
        return self.risk_agent.assess_loan_risk(loan_details, borrower_data)

    def _run_legal(self, data: Dict[str, Any]):
        docs = data.get('docs', {})
        doc_text = docs.get('Credit_Agreement', '')
        ten_k = docs.get('10-K', '')

        # Safe get for revenue
        revenue = data['historical'].get('revenue', [0])[-1]

        legal_output = self.legal_agent.review_credit_agreement(doc_text)
        fraud_check = self.legal_agent.detect_fraud_signals(ten_k, {"revenue": revenue})
        return legal_output, fraud_check

    def _run_regulatory_check(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        """
        Runs regulatory compliance checks on the proposed facility.
        """
        try:
            # Construct a "Transaction" object for the agent
            transaction = {
                "id": f"TX-{uuid.uuid4().hex[:6]}",
                "customer": data['name'],
                "ticker": data.get('ticker', ''),
                "amount": data['historical']['total_debt'][-1] * 0.2, # Mock loan amount
                "country": "US", # Default
                "sector": data.get('sector', 'General')
            }

            # Directly call the analysis method (synchronous part of the agent)
            analysis = self.regulatory_agent._analyze_transaction(transaction)
            return analysis
        except Exception as e:
            logger.error(f"Regulatory check failed: {e}")
            return {"compliance_status": "Unknown", "risk_score": 0.0}

    def _run_system_2_critique(self, risk: Dict, valuation: Dict, regulatory: Dict) -> Dict:
        """
        Simulates System 2 (Slow Thinking) critique by checking logical consistency across modules.
        """
        critique_points = []
        score = 0.85 # Start high

        # Check 1: Valuation vs Risk
        # If Bear Case value < Share Price, Risk should be high
        if valuation and 'bear_case' in valuation:
            intrinsic = valuation['base_case']['intrinsic_share_price']
            bear_val = valuation['bear_case']['intrinsic_share_price']
            if bear_val < intrinsic * 0.7:
                 critique_points.append("Significant downside valuation risk detected in Bear Case.")
                 score -= 0.1

        # Check 2: Regulatory vs Risk
        if regulatory.get('compliance_status') == 'non-compliant':
            critique_points.append("CRITICAL: Regulatory non-compliance flagged. Credit approval unlikely.")
            score -= 0.5

        # Check 3: PD Calibration
        pd = risk['risk_quant_metrics']['PD']
        if pd > 0.1 and score > 0.7:
            critique_points.append("High Probability of Default contradicts initial high conviction.")
            score -= 0.2

        if not critique_points:
            critique_points.append("Cross-module analysis confirms consistent risk-return profile.")

        return {
            "author_agent": "System 2 Consensus Engine",
            "critique_points": critique_points,
            "conviction_score": max(0.0, score),
            "verification_status": "PASS" if score > 0.5 else "REVIEW_REQUIRED"
        }

    def run_interlock(self, name, icat, risk, legal, fraud, regulatory):
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

        # 3. Regulatory Phase
        logs.append({
            "actor": "RegulatoryBot",
            "message": f"Compliance Check: {regulatory.get('compliance_status', 'Unknown')}. Score: {regulatory.get('risk_score', 0):.2f}",
            "timestamp": (timestamp + datetime.timedelta(seconds=5)).isoformat()
        })

        # 4. Interlock Logic
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

        # 5. Fraud
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
        if not doc_text:
            return []

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

    def construct_memo(self, data, icat, risk, legal, logs, citations, valuation, regulatory, system2):
        # Calculate Risk Score (inverted)
        raw_risk = min(risk['overall_risk_score'], 1.0)
        risk_score = int((1.0 - raw_risk) * 100)

        # Construct Hist Data
        hist = []
        years = data['historical']['year']
        for i, year in enumerate(years):
            record = {"period": str(year)}
            for k, v in data['historical'].items():
                if k != 'year':
                    # Handle if list is shorter than years
                    if i < len(v):
                        record[k] = v[i]
            hist.append(record)
        hist.sort(key=lambda x: x['period'], reverse=True)

        # Sections with Citations
        risk_content = f"Primary Risk Factors:\n1. {risk['risk_factors'].get('geopolitical_risk', ['N/A'])[0]}\n2. Market Volatility (Beta: {data['market_data'].get('beta', 1.0)})\n\nQuantitative Model:\nProbability of Default: {risk['risk_quant_metrics']['PD']*100:.2f}%\nLoss Given Default: {risk['risk_quant_metrics']['LGD']*100:.2f}%"

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
                "title": "Valuation & Scenarios",
                "content": f"Base Case Equity Value: ${valuation.get('base_case', {}).get('intrinsic_value', 0):,.2f}M\n"
                           f"Bull Case: ${valuation.get('bull_case', {}).get('intrinsic_value', 0):,.2f}M\n"
                           f"Bear Case: ${valuation.get('bear_case', {}).get('intrinsic_value', 0):,.2f}M\n"
                           f"WACC: {valuation.get('base_case', {}).get('wacc', 0.1)*100:.2f}%",
                "citations": [],
                "author_agent": "Valuation Engine"
            },
            {
                "title": "Risk Analysis",
                "content": risk_content,
                "citations": [],
                "author_agent": "Risk Assessment Agent"
            },
             {
                "title": "Regulatory Compliance",
                "content": f"Status: {regulatory.get('compliance_status', 'Unknown')}\nViolated Rules: {', '.join(regulatory.get('violated_rules', [])) or 'None'}",
                "citations": [],
                "author_agent": "Regulatory Compliance Agent"
            },
            {
                "title": "Legal & Covenants",
                "content": f"Document Review Summary:\n{legal['key_findings'][0]}\n\nClauses Identified: {', '.join(legal['clauses_identified'])}",
                "citations": [],
                "author_agent": "Legal Agent"
            }
        ]

        # Safe get for forecast assumptions
        forecast = data.get('forecast_assumptions', {})
        discount_rate = forecast.get('discount_rate', 0.10)
        terminal_growth = forecast.get('terminal_growth_rate', 0.03)

        return {
            "borrower_name": data['name'],
            "borrower_details": {"name": data['name'], "sector": data.get('sector', 'Unknown')},
            "report_date": datetime.datetime.now().isoformat(),
            "risk_score": risk_score,
            "historical_financials": hist,
            "sections": sections,
            "key_strengths": ["Market Leader", "Strong Cash Flow"],
            "key_weaknesses": ["Cyclical Industry", "Regulatory Risk"],
            "dcf_analysis": {
                "enterprise_value": icat.valuation_metrics.enterprise_value,
                "share_price": valuation.get('base_case', {}).get('intrinsic_share_price', 0),
                "wacc": valuation.get('base_case', {}).get('wacc', discount_rate),
                "growth_rate": terminal_growth,
                "terminal_value": icat.valuation_metrics.dcf_value * 0.4,
                "scenarios": {
                    "bull": valuation.get('bull_case', {}).get('intrinsic_share_price', 0),
                    "bear": valuation.get('bear_case', {}).get('intrinsic_share_price', 0)
                }
            },
            "pd_model": {
                "model_score": int((1 - risk['risk_quant_metrics']['PD']) * 100),
                "implied_rating": "BBB",
                "one_year_pd": risk['risk_quant_metrics']['PD'],
                "five_year_pd": risk['risk_quant_metrics']['PD'] * 3,
                "input_factors": {"Leverage": icat.credit_metrics.net_leverage, "Z-Score": icat.credit_metrics.z_score},
                "lgd": risk['risk_quant_metrics']['LGD'],
                "rwa": risk['risk_quant_metrics']['RWA'],
                "raroc": risk['risk_quant_metrics']['RAROC']
            },
            "system_two_critique": system2,
            "equity_data": data['market_data']
        }

    def _extract_highlights(self, legal, fraud, regulatory):
        highlights = []
        for c in legal['clauses_identified']:
            highlights.append({"type": "clause", "label": c, "status": "Protected"})
        if fraud['fraud_risk_level'] != "Low":
            highlights.append({"type": "risk", "label": "Fraud Alert", "status": "Critical"})
        if regulatory.get('compliance_status') == 'non-compliant':
             highlights.append({"type": "risk", "label": "Compliance Breach", "status": "Critical"})
        return highlights

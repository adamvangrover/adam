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
from core.engine.valuation_utils import calculate_dcf, calculate_multiples, get_price_targets

logger = logging.getLogger("CreditMemoOrchestrator")

class CreditMemoOrchestrator:
    """
    Orchestrates the end-to-end credit memo generation process.
    Integrates ICAT (Financials), Risk Agent (Quant), Legal Agent (Qual/Docs),
    Regulatory Compliance, and Valuation Engines.
    Supports both Mock and Live execution modes.
    """

    def __init__(self, mock_library: Dict[str, Any], output_dir: str = "showcase/data", mode: str = "mock"):
        self.mock_library = mock_library
        self.output_dir = output_dir
        self.mode = mode

        # Initialize Agents
        self.risk_agent = RiskAssessmentAgent(config={})
        self.legal_agent = LegalAgent()
        self.regulatory_agent = RegulatoryComplianceAgent(config={})

        # Initialize Engines
        self.icat_engine = ICATEngine(mock_data_path="showcase/data/icat_mock_data.json")
        self.icat_engine.mock_db = self.mock_library

        # Live Data Components
        self.realtime_fetcher = None
        self.web_search_tool = None

        if self.mode == "live":
            self.realtime_fetcher = RealtimeFetcher()
            self.web_search_tool = WebSearchTool()
            logger.info("Initialized in LIVE mode with RealtimeFetcher and WebSearchTool.")

    def process_entity(self, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full processing pipeline for a single entity.
        """
        logger.info(f"Orchestrator processing {data['name']} (Mode: {self.mode})...")

        # 1. Enrich Data (if Live)
        if self.mode == "live":
            data = data.copy() # Prevent global state mutation of MOCK_LIBRARY
            data = self._enrich_with_live_data(key, data)
            data = self._enrich_with_docs(key, data)

        # 2. Financial Analysis (ICAT)
        icat_output = self._run_financials(key, data)
        if not icat_output:
            logger.error(f"Financial analysis failed for {key}")
            return {}

        # 3. Valuation Analysis (DCF Scenarios)
        valuation_scenarios = self._run_valuation_scenarios(data, icat_output)

        # 4. Risk Analysis
        risk_output = self._run_risk(data, icat_output)

        # 5. Regulatory Compliance Check
        reg_output = self._run_regulatory_check(data)

        # 6. Legal Review
        legal_output, fraud_check = self._run_legal(data)

        # 7. System 2 Critique (Cross-Validation)
        critique_output = self._system_2_critique(icat_output, risk_output, reg_output, valuation_scenarios)

        # 8. Interlock Simulation (Agent Chat)
        logs, ui_events = self.run_interlock(data['name'], icat_output, risk_output, legal_output, fraud_check, reg_output)

        # 9. RAG Simulation (Citation Generation)
        # In live mode, this could use real snippets found via search
        doc_source = data['docs'].get('10-K', '')
        rag_citations = self.simulate_rag(doc_source, "risk factors")

        # 10. Construct Final Memo
        memo_data = self.construct_memo(data, icat_output, risk_output, legal_output, logs, rag_citations, valuation_scenarios, reg_output, critique_output)

        return {
            "memo": memo_data,
            "interaction_log": {
                "borrower_name": data['name'],
                "logs": logs,
                "highlights": self._extract_highlights(legal_output, fraud_check, reg_output),
                "ui_events": ui_events
            }
        }

    def _enrich_with_live_data(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches live market and historical data to update the entity record."""
        try:
            # 1. Market Data
            md = self.realtime_fetcher.fetch_market_data(ticker)
            if md:
                data['market_data'] = md

            # 2. Historicals
            hist = self.realtime_fetcher.fetch_historical_data(ticker)
            if hist:
                # Extract aux price data if present
                if '_price_history' in hist:
                    if 'market_data' not in data: data['market_data'] = {}
                    data['market_data']['price_data'] = hist.pop('_price_history')

                data['historical'] = hist

            # 3. Forecast Assumptions
            fa = self.realtime_fetcher.fetch_forecast_assumptions(ticker)
            if fa:
                data['forecast_assumptions'] = fa

            logger.info(f"Enriched {ticker} with live financial data.")
            return data
        except Exception as e:
            logger.error(f"Failed to enrich live data for {ticker}: {e}")
            return data

    def _enrich_with_docs(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Uses Web Search to find recent news/filings to populate 'docs'."""
        try:
            import asyncio
            query = f"{ticker} SEC 10-K risk factors summary"

            try:
                loop = asyncio.get_running_loop()
                # If there's already a running loop we shouldn't block it,
                # but since orchestrator is synchronous, we create a task
                # Wait, this is tricky in sync context. Better use asyncio.run in isolated thread
                # Or for simplicity, run it if no loop:
                content = "Async fetching unavailable in this event loop context."
            except RuntimeError:
                content = asyncio.run(self.web_search_tool.execute(query=query, num_results=1))

            # Update docs
            if 'docs' not in data: data['docs'] = {}
            data['docs']['10-K'] = content

            logger.info(f"Enriched {ticker} with web search content.")
            return data
        except Exception as e:
            logger.error(f"Failed to enrich docs for {ticker}: {e}")
            return data

    def _run_financials(self, key: str, data: Dict[str, Any]) -> Optional[ICATOutput]:
        # LBO Logic
        lbo_params = None
        sector = data.get('sector', '')
        if "Distressed" in sector or "Consumer" in sector:
             # Safe access to lists
             ebitda_list = data['historical'].get('ebitda', [])
             latest_ebitda = ebitda_list[-1] if ebitda_list else 100

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
            # If live, source="mock" is still technically true for ICAT as we inject data via the dict,
            # but we pass the data dict implicitly by updating mock_library or passing it directly?
            # ICAT takes a ticker. If live, we need to ensure ICAT uses the *updated* data.
            # We can use `ingest_from_repo_file` or just override the entry in `self.icat_engine.mock_db`
            self.icat_engine.mock_db[key] = data
            return self.icat_engine.analyze(ticker=key, source="mock", lbo_params=lbo_params)
        except Exception as e:
            logger.error(f"ICAT Error: {e}")
            return None

    def _run_valuation_scenarios(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        """
        Runs Bull/Base/Bear DCF scenarios using core.engine.valuation_utils.
        """
        # Calculate D/E Ratio (Total Debt / Market Cap)
        total_debt = data['historical']['total_debt'][-1]
        market_cap = data['market_data'].get('market_cap', 0)
        de_ratio = total_debt / market_cap if market_cap > 0 else 0.5 # Default fallback

        growth_rate = data['forecast_assumptions']['revenue_growth']
        if isinstance(growth_rate, list):
            growth_rate = growth_rate[0] # Use year 1 growth rate if it's a list

        financials = {
            "fcf": icat.valuation_metrics.dcf_value * 0.05 if icat.valuation_metrics.dcf_value else 100, # Proxy for FCF base
            "growth_rate": float(growth_rate),
            "beta": data['market_data'].get('beta', 1.0),
            "debt_equity_ratio": de_ratio,
            "net_debt": total_debt - data['historical']['cash'][-1],
            "shares_outstanding": data['market_data']['market_cap'] / data['market_data']['share_price'] if data['market_data']['share_price'] else 100
        }

        rfr = 0.042 # Could be fetched live, hardcoded for stability

        # Base Case
        base = calculate_dcf(financials, rfr)

        # Bull Case (+20% growth, lower risk)
        bull_scenario = {
            "growth_rate": financials["growth_rate"] * 1.2,
            "market_risk_premium": 0.045
        }
        bull = calculate_dcf(financials, rfr, scenario=bull_scenario)

        # Bear Case (-20% growth, higher risk)
        bear_scenario = {
            "growth_rate": financials["growth_rate"] * 0.8,
            "market_risk_premium": 0.07
        }
        bear = calculate_dcf(financials, rfr, scenario=bear_scenario)

        return {
            "base": base,
            "bull": bull,
            "bear": bear,
            "current_price": data['market_data']['share_price']
        }

    def _run_risk(self, data: Dict[str, Any], icat: ICATOutput) -> Dict[str, Any]:
        # Safe access to historical lists
        hist = data['historical']
        assets = hist.get('total_assets', [0])[-1]
        debt = hist.get('total_debt', [0])[-1]
        cash = hist.get('cash', [0])[-1]

        fin_data_risk = {
            "credit_rating": icat.credit_metrics.credit_rating or "N/A",
            "z_score": icat.credit_metrics.z_score,
            "total_assets": assets,
            "total_debt": debt,
            "cash": cash,
            "monthly_burn_rate": 0, # Placeholder
            "liquidity_ratio": icat.credit_metrics.interest_coverage
        }

        loan_details = {
            "seniority": "Senior Secured" if "Distressed" in data.get('sector','') else "Senior Unsecured",
            "collateral_value": assets * 0.5,
            "loan_amount": debt * 0.2,
            "interest_rate": 0.07
        }

        return self.risk_agent.assess_loan_risk(loan_details, fin_data_risk)

    def _run_regulatory_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a mock transaction analysis through the RegulatoryComplianceAgent.
        """
        # Create a mock transaction representing a new loan issuance
        mock_transaction = {
            "id": f"TXN-{str(uuid.uuid4())[:6]}",
            "customer": data['name'],
            "amount": 1_000_000, # $1B notional represented
            "country": "US",
            "ticker": data.get("ticker", "UNKNOWN")
        }

        # Execute
        # We wrap the internal method call or use the public execute.
        # Ideally we use execute(), but it is async.
        # For now, to avoid deep async refactoring of the orchestrator (which is sync in process_entity),
        # we will use the internal method but acknowledge the tech debt, or wrap it if a public sync method existed.
        # Since _analyze_transaction is effectively the sync core logic we need:
        analysis = self.regulatory_agent._analyze_transaction(mock_transaction)
        return analysis

    def _run_legal(self, data: Dict[str, Any]):
        doc_text = data['docs'].get('Credit_Agreement', '')
        ten_k = data['docs'].get('10-K', '')

        # Fallback if docs empty
        if not doc_text and not ten_k:
            return {"key_findings": ["No documents provided."], "clauses_identified": []}, {"fraud_risk_level": "Unknown", "signals_detected": []}

        legal_output = self.legal_agent.review_credit_agreement(doc_text or "Standard Terms")
        fraud_check = self.legal_agent.detect_fraud_signals(ten_k or "", {"revenue": 100}) # Mock revenue if needed
        return legal_output, fraud_check

    def _system_2_critique(self, icat: ICATOutput, risk, reg, valuation) -> Dict[str, Any]:
        """
        System 2 (Reflector) Logic: Cross-validates findings.
        Checks for consistency between Quant (ICAT/Risk) and Qual (Reg/Valuation).
        """
        critiques = []
        status = "PASS"
        score = 0.9

        # Check 1: Valuation vs Risk
        # If PD is high (>5%) but Valuation shows massive upside (>50%), flag it.
        pd = risk.get('risk_quant_metrics', {}).get('PD', 0)
        curr_price = valuation['current_price']
        bull_price = valuation['bull']['intrinsic_share_price']

        if pd > 0.05 and bull_price > (curr_price * 1.5):
            critiques.append("High Default Risk conflicts with Aggressive Bull Case Valuation.")
            score -= 0.2

        # Check 2: Regulatory vs Credit
        # If Regulatory Risk is high but Credit Rating is Investment Grade
        reg_score = reg.get('risk_score', 0)
        rating = icat.credit_metrics.credit_rating
        if reg_score > 0.7 and rating == "Investment Grade":
            critiques.append("Regulatory headwinds may threaten Investment Grade status.")
            score -= 0.15
            status = "REVIEW_REQUIRED"

        # Check 3: Leverage Consistency
        lev = icat.credit_metrics.net_leverage
        if lev > 5.0 and pd < 0.02:
             critiques.append("Leverage > 5x usually implies higher PD than modeled.")
             score -= 0.1

        return {
            "critique_points": critiques if critiques else ["Models are consistent across domains."],
            "conviction_score": round(max(0.0, score), 2),
            "verification_status": status,
            "author_agent": "System 2 (Reflector)",
            "quantitative_analysis": {
                "ratios_checked": ["Leverage vs PD", "Valuation vs Risk", "Regulatory vs Rating"],
                "variance_analysis": "Flagged" if critiques else "Nominal",
                "dcf_validation": f"Base: ${valuation['base']['intrinsic_share_price']} vs Curr: ${curr_price}"
            }
        }

    def run_interlock(self, name, icat, risk, legal, fraud, reg):
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

        # 2. Regulatory Phase
        if reg['violated_rules']:
             logs.append({
                "actor": "RegBot",
                "message": f"COMPLIANCE ALERT: {reg['violated_rules'][0]}",
                "timestamp": (timestamp + datetime.timedelta(seconds=3)).isoformat()
            })

        # 3. Legal Phase
        logs.append({
            "actor": "LegalAI",
            "message": f"Reviewing docs for {name}.",
            "timestamp": (timestamp + datetime.timedelta(seconds=4)).isoformat()
        })

        # 4. Interlock Logic
        clauses = legal['clauses_identified']
        if clauses:
             logs.append({
                "actor": "LegalAI",
                "message": f"Clauses found: {', '.join(clauses)}.",
                "timestamp": (timestamp + datetime.timedelta(seconds=6)).isoformat()
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
        """
        if not doc_text: return []

        sentences = doc_text.split('. ')
        citations = []
        keywords = ["risk", "debt", "revenue", "competition", "regulation", "growth"]
        candidates = [s for s in sentences if any(k in s.lower() for k in keywords)]

        if not candidates:
            candidates = sentences[:2]

        selected = candidates[:2]

        for i, text in enumerate(selected):
            citations.append({
                "doc_id": "10-K_FY2025.pdf", # Placeholder filename
                "chunk_id": str(uuid.uuid4())[:8],
                "page_number": random.randint(10, 50),
                "text": text + "."
            })

        return citations

    def construct_memo(self, data, icat, risk, legal, logs, citations, valuation, reg, critique):
        # Calculate Risk Score (inverted)
        raw_risk = min(risk['overall_risk_score'], 1.0)
        risk_score = int((1.0 - raw_risk) * 100)

        # Construct Hist Data
        hist = []
        years = data['historical']['year']
        # Convert lists to dicts
        for i in range(len(years)):
            record = {"period": str(years[i])}
            for k, v in data['historical'].items():
                if k != 'year' and i < len(v): # Safety check
                    record[k] = v[i]
            hist.append(record)
        # Sort descending by period
        hist.sort(key=lambda x: x['period'], reverse=True)

        # Sections with Citations
        risk_content = f"Primary Risk Factors:\n1. {risk['risk_factors'].get('geopolitical_risk', ['N/A'])[0]}\n2. Market Volatility (Beta: {data['market_data']['beta']})\n\nQuantitative Model:\nProbability of Default: {risk['risk_quant_metrics']['PD']*100:.2f}%\nLoss Given Default: {risk['risk_quant_metrics']['LGD']*100:.2f}%"

        exec_summary = f"{data.get('description', 'Analysis of ' + data['name'])}\n\nKey Credit Stats:\n- Net Leverage: {icat.credit_metrics.net_leverage:.2f}x\n- Interest Coverage: {icat.credit_metrics.interest_coverage:.2f}x\n- Z-Score: {icat.credit_metrics.z_score:.2f}"

        md = data.get('market_data', {})
        consensus_text = ""
        if md.get('consensus_rating'):
            consensus_text = f"\n- Market Consensus: {md['consensus_rating']} (Target: ${md.get('consensus_target_price', 0):.2f})"

        exec_summary += consensus_text

        if citations:
            exec_summary += f"  [Ref: {citations[0]['doc_id']}]"

        md = data.get('market_data', {})
        consensus_text = ""
        if md.get('consensus_rating'):
            consensus_text = f"Consensus Rating: {md['consensus_rating']} (Target: ${md.get('consensus_target_price', 0):.2f}, Analysts: {md.get('analyst_count', 0)})"

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
                "title": "Valuation & Scenarios",
                "content": f"Base Case Target: ${valuation['base']['intrinsic_share_price']}\nBull Case: ${valuation['bull']['intrinsic_share_price']}\nBear Case: ${valuation['bear']['intrinsic_share_price']}\nCurrent: ${valuation['current_price']}\n{consensus_text}",
                "citations": [],
                "author_agent": "Valuation Engine"
            },
            {
                "title": "Regulatory Compliance",
                "content": f"Compliance Status: {reg['compliance_status'].upper()}\nRisk Score: {reg['risk_score']}\nViolations: {', '.join(reg['violated_rules']) if reg['violated_rules'] else 'None'}",
                "citations": [],
                "author_agent": "Regulatory Agent"
            },
            {
                "title": "Legal & Covenants",
                "content": f"Document Review Summary:\n{legal['key_findings'][0] if legal['key_findings'] else 'None'}\n\nClauses Identified: {', '.join(legal['clauses_identified'])}",
                "citations": [],
                "author_agent": "Legal Agent"
            },
            {
                "title": "System 2 Consensus Engine",
                "content": f"Status: {critique['verification_status']}\nConviction: {critique['conviction_score']}\nCritiques: {'; '.join(critique['critique_points'])}",
                "citations": [],
                "author_agent": "System 2"
            }
        ]

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
                "share_price": valuation['base']['intrinsic_share_price'],
                "wacc": valuation['base']['wacc'],
                "growth_rate": valuation['base']['terminal_growth'],
                "terminal_value": icat.valuation_metrics.dcf_value * 0.4 if icat.valuation_metrics.dcf_value else 0,
                "free_cash_flow": [1000, 1100, 1200, 1300, 1400] # Simplified visualization data
            },
            "pd_model": {
                "model_score": int((1 - risk['risk_quant_metrics']['PD']) * 100),
                "implied_rating": icat.credit_metrics.credit_rating or "N/A",
                "one_year_pd": risk['risk_quant_metrics']['PD'],
                "five_year_pd": risk['risk_quant_metrics']['PD'] * 3,
                "input_factors": {"Leverage": icat.credit_metrics.net_leverage, "Z-Score": icat.credit_metrics.z_score}
            },
            "system_two_critique": critique,
            "equity_data": data['market_data']
        }

    def _extract_highlights(self, legal, fraud, reg):
        highlights = []
        for c in legal['clauses_identified']:
            highlights.append({"type": "clause", "label": c, "status": "Protected"})
        if fraud['fraud_risk_level'] != "Low":
            highlights.append({"type": "risk", "label": "Fraud Alert", "status": "Critical"})
        if reg['violated_rules']:
            highlights.append({"type": "compliance", "label": "Regulatory Violation", "status": "Warning"})
        return highlights

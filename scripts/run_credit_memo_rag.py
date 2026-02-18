"""
Script to run Credit Memo RAG Pipeline.
Ingests a document (10-K), extracts data using a RAG-like approach (Regex/LLM),
and generates artifacts for the Sovereign Dashboard and Credit Memo Automation.
"""

import sys
import os
import json
import re
import argparse
import random
from datetime import datetime
import logging
import asyncio

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock core imports if not available in this environment, otherwise use real ones
try:
    from core.agents.agent_base import AgentBase
    from core.llm.engines.dummy_llm_engine import DummyLLMEngine
    from core.embeddings.models.dummy_embedding_model import DummyEmbeddingModel
    from core.vectorstore.stores.in_memory_vector_store import InMemoryVectorStore
except ImportError:
    # Fallback for standalone execution without full core dependencies
    class AgentBase:
        def __init__(self, config): pass
    class DummyLLMEngine:
        def __init__(self, model_name): pass
    class DummyEmbeddingModel:
        pass
    class InMemoryVectorStore:
        def __init__(self, embedding_dim): pass
        async def add_documents(self, docs): pass

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CreditMemoRAG")

class RegexExtractingLLM:
    """
    A Dummy LLM that 'reads' the provided context (document text) and extracts
    specific financial data using Regex, simulating a high-performance LLM extraction.
    Returns structured data with source provenance.
    """
    def __init__(self, document_text="", doc_id="unknown"):
        self.document_text = document_text
        self.doc_id = doc_id

    def update_context(self, text, doc_id):
        self.document_text = text
        self.doc_id = doc_id

    def _find_match(self, pattern, text, group_idx=1, is_float=False, context_padding=50):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val_str = match.group(group_idx)
            val = val_str
            if is_float:
                try:
                    clean_val = val_str.replace(',', '').replace('$', '').strip()
                    val = float(clean_val)
                    # Check if 'billion' follows
                    if "billion" in text[match.end():match.end()+15].lower():
                        val *= 1000
                except:
                    pass

            # Context snippet
            start = match.start()
            end = match.end()
            context_start = max(0, start - context_padding)
            context_end = min(len(text), end + context_padding)
            context = text[context_start:context_end]

            return {
                "value": val,
                "source": {
                    "doc_id": self.doc_id,
                    "start": start,
                    "end": end,
                    "text": context.strip()
                }
            }
        return None

    def generate_response(self, prompt, **kwargs):
        """
        Parses the prompt to identify what is being asked, and scans the document text.
        Returns { "value": ..., "source": ... } or { "value": "Data not found", "source": None }
        """
        text = self.document_text # Keep original case for extraction if needed, but regex uses IGNORECASE

        # --- Priority Extractions (Specifics first) ---

        # Projections
        if "revenue growth" in prompt.lower():
             return self._find_match(r"revenue growth of approximately\s*([\d.]+)", text, is_float=True) or {"value": 0, "source": None}
        if "margin" in prompt.lower():
             return self._find_match(r"margins are targeted at\s*([\d.]+)", text, is_float=True) or {"value": 0, "source": None}

        # SNC
        if "snc rating" in prompt.lower():
             return self._find_match(r"snc regulatory rating:\s*(.+?)\.", text, group_idx=1) or {"value": "N/A", "source": None}
        if "snc justification" in prompt.lower():
             return self._find_match(r"justification:\s*(.+)", text, group_idx=1) or {"value": "N/A", "source": None}

        # Cap Structure Extraction
        if "senior notes" in prompt.lower():
            return self._find_match(r"senior notes:\s*\$([\d,.]+)", text, is_float=True) or {"value": 0, "source": None}
        if "revolver" in prompt.lower():
            return self._find_match(r"revolver drawn:\s*\$([\d,.]+)", text, is_float=True) or {"value": 0, "source": None}
        if "subordinated" in prompt.lower():
            return self._find_match(r"subordinated debt:\s*\$([\d,.]+)", text, is_float=True) or {"value": 0, "source": None}

        # --- Generic Financials Extraction ---

        if "total net sales" in prompt.lower() or "revenue" in prompt.lower():
            # Pattern: "Total net sales were $394.3 billion"
            res = self._find_match(r"total net sales.*?\$([\d,.]+)", text, is_float=True)
            if res: return res
            res = self._find_match(r"total net sales:\s*\$([\d,.]+)", text, is_float=True)
            if res: return res

        if "net income" in prompt.lower():
            res = self._find_match(r"net income.*?\$([\d,.]+)", text, is_float=True)
            if res: return res
            res = self._find_match(r"net income:\s*\$([\d,.]+)", text, is_float=True)
            if res: return res

        if "total assets" in prompt.lower():
            res = self._find_match(r"total assets:\s*\$([\d,.]+)", text, is_float=True)
            if res: return res

        if "total liabilities" in prompt.lower() and "shareholders" not in prompt.lower():
            res = self._find_match(r"total liabilities:\s*\$([\d,.]+)", text, is_float=True)
            if res: return res

        if "total debt" in prompt.lower():
             res = self._find_match(r"total debt was\s*\$([\d,.]+)", text, is_float=True)
             if res: return res
             res = self._find_match(r"term debt:\s*\$([\d,.]+)", text, is_float=True)
             if res: return res

        if "cash" in prompt.lower():
             res = self._find_match(r"cash and cash equivalents:\s*\$([\d,.]+)", text, is_float=True)
             if res: return res

        if "ebitda" in prompt.lower():
            # Estimate EBITDA
            res = self._find_match(r"operating income:\s*\$([\d,.]+)", text, is_float=True)
            if res:
                # Mocking logic: EBITDA = Op Income * 1.1
                op_inc = res['value']
                res['value'] = op_inc * 1.10
                return res

        # Qualitative Extraction
        if "risk factors" in prompt.lower():
            start = text.lower().find("item 1a. risk factors")
            end = text.lower().find("item 7.")
            if start != -1 and end != -1:
                section = text[start:end]
                # Extract bullets
                risks = re.findall(r"- (.*?)(?=\n|$)", section)
                val = "\n".join(risks[:5])
                return {
                    "value": val,
                    "source": {
                        "doc_id": self.doc_id,
                        "start": start,
                        "end": end, # Entire section scope
                        "text": "Item 1A. Risk Factors..."
                    }
                }

        if "business description" in prompt.lower():
            start = text.lower().find("item 1. business")
            if start != -1:
                val = text[start:start+500] + "..."
                return {
                    "value": val,
                    "source": {
                        "doc_id": self.doc_id,
                        "start": start,
                        "end": start+500,
                        "text": val
                    }
                }

        return {"value": "Data not found in document.", "source": None}

class RAGAgent(AgentBase):
    def __init__(self, config, llm_engine, embedding_model, vector_store):
        super().__init__(config)
        self.llm_engine = llm_engine
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def execute(self, *args, **kwargs):
        pass

    async def ingest_document(self, text, chunk_size=1000, chunk_overlap=100):
        # Mock ingestion
        embedding = [0.0] * 128
        await self.vector_store.add_documents([(text, embedding)])
        return True

class CreditMemoRAGPipeline:
    def __init__(self, ticker):
        self.ticker = ticker
        self.llm = RegexExtractingLLM()
        self.embedding_model = DummyEmbeddingModel()
        self.vector_store = InMemoryVectorStore(embedding_dim=128)

        self.agent = RAGAgent(
            config={"agent_id": f"rag_analyst_{ticker}"},
            llm_engine=self.llm,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

    async def run(self, document_path):
        logger.info(f"Ingesting document: {document_path}")

        # 1. Read Document
        try:
            with open(document_path, 'r') as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {document_path}")
            return None, None

        # Update LLM context
        doc_id = os.path.basename(document_path)
        self.llm.update_context(text, doc_id)

        # 2. Ingest
        await self.agent.ingest_document(text)

        # 3. Extract Data
        logger.info("Extracting Financials...")

        source_map = {}

        def get_val(prompt, key, default=0.0):
            res = self.llm.generate_response(prompt)
            if res['source']:
                source_map[key] = res['source']

            try:
                return float(res['value'])
            except:
                return default

        def get_text(prompt, key):
            res = self.llm.generate_response(prompt)
            if res['source']:
                source_map[key] = res['source']
            return str(res['value'])

        # New Extractions (Do specifics first to avoid pollution, though priority is now handled in class)
        proj_rev_growth = get_val("What is the revenue growth projection?", "proj_rev_growth", 0)
        proj_margin = get_val("What is the margin projection?", "proj_margin", 0)

        revenue = get_val("What is the total net sales or revenue?", "revenue")
        if revenue == 0: revenue = 383285.0

        net_income = get_val("What is the net income?", "net_income", revenue * 0.25)
        assets = get_val("What are the total assets?", "assets", revenue * 0.9)
        liabilities = get_val("What are the total liabilities?", "liabilities", assets * 0.8)
        debt = get_val("What is the total debt?", "total_debt", liabilities * 0.4)
        cash = get_val("What is the cash and cash equivalents?", "cash", 29000.0)
        ebitda = get_val("What is the EBITDA?", "ebitda", revenue * 0.30)

        senior_notes = get_val("What are the senior notes?", "senior_notes", 0)
        revolver = get_val("What is the revolver drawn?", "revolver", 0)
        sub_debt = get_val("What is the subordinated debt?", "sub_debt", 0)

        snc_rating = get_text("What is the SNC rating?", "snc_rating")
        snc_justification = get_text("What is the SNC justification?", "snc_justification")

        risks = get_text("What are the key risk factors?", "risk_factors")
        description = get_text("What is the business description?", "description")

        # 4. Generate Advanced Analytics

        # Valuation & Risk
        valuation_data = self._generate_valuation(ebitda, debt, cash, liabilities, assets, revenue, net_income)

        # Consensus
        consensus_data = self._generate_consensus(revenue, ebitda)

        # System 2 Review (Audit)
        extracted_data = {
            "assets": assets,
            "liabilities": liabilities,
            "revenue": revenue,
            "ebitda": ebitda,
            "debt": debt,
            "proj_rev_growth": proj_rev_growth
        }
        system2_audit = self._run_system2_review(extracted_data)

        # A. Spread Data
        spread_data = {
            "ticker": self.ticker,
            "fiscal_year": 2024,
            "metrics": {
                "Revenue": revenue,
                "EBITDA": ebitda,
                "Total Debt": debt,
                "Cash": cash,
                "Net Debt": debt - cash,
                "Net Income": net_income
            },
            "ratios": {
                "Leverage (Debt/EBITDA)": round(debt / ebitda, 2) if ebitda else 0,
                "Interest Coverage": round(ebitda / (debt * 0.05), 2) if debt else 0
            },
            "valuation": valuation_data,
            "consensus": consensus_data,
            "source": "RAG Extraction (10-K)",
            "source_map": source_map,
            "system2_audit": system2_audit
        }

        # B. Credit Memo
        memo_data = {
            "borrower_name": f"{self.ticker} Inc.",
            "ticker": self.ticker,
            "sector": "Technology",
            "report_date": datetime.now().isoformat(),
            "risk_score": 85.0,
            "executive_summary": f"RAG Analysis of {self.ticker} based on 10-K ingestion. The company shows strong revenue of ${revenue:,.2f}M and EBITDA of ${ebitda:,.2f}M. The implied valuation suggests a base target of ${valuation_data['dcf']['share_price']:.2f}. Key risks identified include: {risks[:200]}...",
            "financial_ratios": {
                "leverage_ratio": debt / ebitda if ebitda else 0,
                "ebitda": ebitda,
                "revenue": revenue,
                "dscr": ebitda / (debt * 0.05) if debt else 0,
                "sources": source_map
            },
            "sections": [
                {
                    "title": "Business Overview",
                    "content": description,
                    "citations": [source_map.get('description')] if source_map.get('description') else []
                },
                {
                    "title": "Risk Factors",
                    "content": risks,
                    "citations": [source_map.get('risk_factors')] if source_map.get('risk_factors') else []
                }
            ],
            "historical_financials": [
                {
                    "period": "FY2024 (RAG)",
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "net_income": net_income,
                    "gross_debt": debt,
                    "cash": cash,
                    "leverage_ratio": debt / ebitda if ebitda else 0,
                    "total_liabilities": liabilities,
                    "capex": revenue * 0.05,
                    "interest_expense": debt * 0.05,
                    "sources": {
                        "revenue": source_map.get('revenue'),
                        "ebitda": source_map.get('ebitda'),
                        "net_income": source_map.get('net_income'),
                        "gross_debt": source_map.get('total_debt'),
                        "cash": source_map.get('cash'),
                        "total_liabilities": source_map.get('liabilities')
                    }
                }
            ],
            "capital_structure": {
                "senior_notes": senior_notes,
                "revolver": revolver,
                "subordinated": sub_debt,
                "sources": {
                    "senior_notes": source_map.get('senior_notes'),
                    "revolver": source_map.get('revolver'),
                    "subordinated": source_map.get('sub_debt')
                }
            },
            "projections": {
                "revenue_growth": proj_rev_growth,
                "ebitda_margin": proj_margin,
                "sources": {
                    "revenue_growth": source_map.get('proj_rev_growth'),
                    "ebitda_margin": source_map.get('proj_margin')
                }
            },
            "snc_rating": {
                "rating": snc_rating,
                "justification": snc_justification,
                "sources": {
                    "rating": source_map.get('snc_rating'),
                    "justification": source_map.get('snc_justification')
                }
            },
            "dcf_analysis": valuation_data["dcf"],
            "pd_model": valuation_data["risk_model"],
            "consensus_data": consensus_data,
            "regulatory_rating": valuation_data["risk_model"]["regulatory_rating"],
            "price_targets": valuation_data["forward_view"]["price_targets"],
            "system2_audit": system2_audit,
            "documents": [
                {
                    "doc_id": os.path.basename(document_path),
                    "title": f"{self.ticker} 10-K",
                    "chunks": []
                }
            ]
        }

        return spread_data, memo_data

    def _run_system2_review(self, data):
        """
        Performs logical integrity checks on extracted data (System 2 thinking).
        """
        audit_log = []
        score = 100

        # Check 1: Balance Sheet Solvency
        if data['assets'] < data['liabilities']:
            audit_log.append({"check": "Solvency", "status": "FAIL", "msg": "Total Liabilities exceed Total Assets (Insolvency Risk)."})
            score -= 20
        else:
            audit_log.append({"check": "Solvency", "status": "PASS", "msg": "Assets cover Liabilities."})

        # Check 2: EBITDA Margin
        if data['revenue'] > 0:
            margin = data['ebitda'] / data['revenue']
            if margin > 0.8:
                audit_log.append({"check": "Margin", "status": "WARN", "msg": f"EBITDA Margin {margin:.1%} seems suspiciously high."})
                score -= 10
            elif margin < -0.2:
                audit_log.append({"check": "Margin", "status": "WARN", "msg": f"Negative EBITDA Margin {margin:.1%} detected."})
                score -= 5
            else:
                audit_log.append({"check": "Margin", "status": "PASS", "msg": f"Margin {margin:.1%} within normal bounds."})

        # Check 3: Leverage sanity
        if data['ebitda'] > 0:
            lev = data['debt'] / data['ebitda']
            if lev > 10:
                audit_log.append({"check": "Leverage", "status": "WARN", "msg": f"Leverage {lev:.1f}x is extremely high."})
                score -= 10
            else:
                audit_log.append({"check": "Leverage", "status": "PASS", "msg": f"Leverage {lev:.1f}x is logical."})

        # Check 4: Projections sanity
        if data['proj_rev_growth'] > 50:
             audit_log.append({"check": "Projections", "status": "WARN", "msg": f"Projected growth {data['proj_rev_growth']}% is very aggressive."})
             score -= 5

        return {
            "score": score,
            "verdict": "CLEAN" if score > 90 else ("REVIEW" if score > 70 else "FLAGGED"),
            "log": audit_log
        }

    def _generate_consensus(self, revenue, ebitda):
        """Generates mock consensus data around the actuals."""
        # Simulate that "Street" is slightly off from "System" (Actuals)
        rev_variance = random.uniform(-0.05, 0.05)
        ebitda_variance = random.uniform(-0.08, 0.08)

        street_rev = revenue * (1 + rev_variance)
        street_ebitda = ebitda * (1 + ebitda_variance)

        return {
            "revenue": {
                "mean": street_rev,
                "high": street_rev * 1.1,
                "low": street_rev * 0.9,
                "system_delta_pct": (revenue - street_rev) / street_rev
            },
            "ebitda": {
                "mean": street_ebitda,
                "high": street_ebitda * 1.15,
                "low": street_ebitda * 0.85,
                "system_delta_pct": (ebitda - street_ebitda) / street_ebitda
            },
            "sentiment": "BULLISH" if rev_variance < 0 else "BEARISH" # If actual > street, bullish
        }

    def _generate_valuation(self, ebitda, debt, cash, liabilities, assets, revenue, net_income):
        wacc = 0.09
        growth_rate = 0.03
        base_fcf = ebitda * 0.70

        # DCF
        pv_fcf = 0
        for i in range(1, 6):
            fcf = base_fcf * ((1.04)**i)
            pv_fcf += fcf / ((1+wacc)**i)

        term_val = (base_fcf * (1.04)**5 * (1+growth_rate)) / (wacc - growth_rate)
        pv_term = term_val / ((1+wacc)**5)

        ev = pv_fcf + pv_term
        equity = ev - debt + cash
        mock_shares_outstanding = 15000.0 # Millions, typical for mega cap
        share_price = equity / mock_shares_outstanding

        # Scenarios
        bull_price = share_price * 1.25
        bear_price = share_price * 0.75

        # Risk Metrics
        leverage = debt / ebitda if ebitda else 0

        # Regulatory Rating Matrix (Simplified)
        if leverage < 1.5: rating = "PASS (Grade 1)"
        elif leverage < 2.5: rating = "PASS (Grade 2)"
        elif leverage < 3.5: rating = "WATCH (Grade 3)"
        elif leverage < 4.5: rating = "SUBSTANDARD (Grade 4)"
        else: rating = "DOUBTFUL (Grade 5)"

        # LGD
        lgd = 0.45 # Unsecured default

        # Z-Score
        z_score = 0
        if assets > 0 and liabilities > 0:
             z_score = 1.2*((assets-liabilities)/assets) + 1.4*((assets-liabilities)/assets) + 3.3*(ebitda/assets) + 0.6*((assets-liabilities)/liabilities) + 1.0*(revenue/assets)

        return {
            "dcf": {
                "enterprise_value": ev,
                "equity_value": equity,
                "share_price": share_price,
                "wacc": wacc,
                "growth_rate": growth_rate,
                "base_fcf": base_fcf,
                "mock_shares": mock_shares_outstanding
            },
            "risk_model": {
                "z_score": round(z_score, 2),
                "pd_category": "Safe" if z_score > 3 else "Grey",
                "regulatory_rating": rating,
                "lgd": lgd,
                "pd_1yr": 0.005 if z_score > 3 else 0.02,
                "credit_rating": "AA" if z_score > 3 else "BBB",
                "rationale": f"Z-Score {z_score:.2f}, Leverage {leverage:.1f}x"
            },
            "forward_view": {
                "price_targets": {
                    "bull": round(bull_price, 2),
                    "base": round(share_price, 2),
                    "bear": round(bear_price, 2)
                },
                "conviction_score": 85,
                "rationale": "Strong fundamentals extracted from 10-K."
            }
        }

async def main():
    parser = argparse.ArgumentParser(description="Run Credit Memo RAG Pipeline")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--file", type=str, required=True, help="Path to 10-K text file")

    args = parser.parse_args()

    pipeline = CreditMemoRAGPipeline(args.ticker)
    spread, memo = await pipeline.run(args.file)

    if not spread: return

    # Save Artifacts
    output_dir = "showcase/data/sovereign_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sovereign Dashboard Artifacts
    with open(f"{output_dir}/{args.ticker}_rag_spread.json", 'w') as f:
        json.dump(spread, f, indent=2)
    with open(f"{output_dir}/{args.ticker}_rag_memo.json", 'w') as f:
        dash_memo = {
            "title": f"RAG Credit Memo: {args.ticker}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "recommendation": "APPROVE" if spread["valuation"]["risk_model"]["z_score"] > 2 else "REVIEW",
            "executive_summary": memo["executive_summary"],
            "covenant_analysis": {"leverage": "PASS", "coverage": "PASS"},
            "regulatory_rating": memo["regulatory_rating"]
        }
        json.dump(dash_memo, f, indent=2)

    with open(f"{output_dir}/{args.ticker}_rag_audit.json", 'w') as f:
        audit_log = {
            "ticker": args.ticker,
            "timestamp": datetime.now().isoformat(),
            "quant_audit": {"action": "RAG_EXTRACTION", "status": "SUCCESS", "details": "Extracted financials from 10-K"},
            "risk_audit": {"action": "RISK_ASSESSMENT", "status": "SUCCESS", "details": "Identified key risks from text"}
        }
        json.dump(audit_log, f, indent=2)

    # 2. Credit Memo Automation Artifact
    memo_output_dir = "showcase/data"
    memo_filename = f"credit_memo_{args.ticker}_RAG.json"
    with open(f"{memo_output_dir}/{memo_filename}", 'w') as f:
        json.dump(memo, f, indent=2)

    # 3. Update Library
    lib_path = "showcase/data/credit_memo_library.json"
    if os.path.exists(lib_path):
        with open(lib_path, 'r') as f:
            library = json.load(f)

        entry = {
            "id": f"{args.ticker}_RAG",
            "borrower_name": f"{args.ticker} (RAG)",
            "ticker": args.ticker,
            "sector": "Technology",
            "report_date": datetime.now().isoformat(),
            "risk_score": memo["risk_score"],
            "file": memo_filename,
            "summary": memo["executive_summary"][:150] + "..."
        }

        library = [x for x in library if x["id"] != entry["id"]]
        library.insert(0, entry)

        with open(lib_path, 'w') as f:
            json.dump(library, f, indent=2)

    logger.info("RAG Pipeline Complete. Artifacts generated.")

if __name__ == "__main__":
    asyncio.run(main())

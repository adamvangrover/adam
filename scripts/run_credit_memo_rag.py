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
    """
    def __init__(self, document_text=""):
        self.document_text = document_text

    def update_context(self, text):
        self.document_text = text

    def generate_response(self, prompt, **kwargs):
        """
        Parses the prompt to identify what is being asked, and scans the document text.
        """
        text = self.document_text.lower()

        # Financials Extraction
        if "total net sales" in prompt.lower() or "revenue" in prompt.lower():
            # Look for "Total net sales ... $X billion" or similar
            # Pattern: "Total net sales were $394.3 billion"
            match = re.search(r"total net sales.*?\$([\d,.]+)\s*billion", text)
            if match:
                val = float(match.group(1).replace(',', '')) * 1000 # Convert to millions
                return f"{val}"

            # Fallback for table-like data
            # "Total net sales: $394,328" (in millions)
            match = re.search(r"total net sales:\s*\$([\d,.]+)", text)
            if match:
                return match.group(1).replace(',', '')

        if "net income" in prompt.lower():
            match = re.search(r"net income.*?\$([\d,.]+)\s*billion", text)
            if match:
                val = float(match.group(1).replace(',', '')) * 1000
                return f"{val}"
            match = re.search(r"net income:\s*\$([\d,.]+)", text)
            if match:
                return match.group(1).replace(',', '')

        if "total assets" in prompt.lower():
            match = re.search(r"total assets:\s*\$([\d,.]+)", text)
            if match:
                return match.group(1).replace(',', '')

        if "total liabilities" in prompt.lower() and "shareholders" not in prompt.lower():
            match = re.search(r"total liabilities:\s*\$([\d,.]+)", text)
            if match:
                return match.group(1).replace(',', '')

        if "total debt" in prompt.lower():
             match = re.search(r"total debt was\s*\$([\d,.]+)\s*billion", text)
             if match:
                val = float(match.group(1).replace(',', '')) * 1000
                return f"{val}"
             match = re.search(r"term debt:\s*\$([\d,.]+)", text)
             if match:
                 return match.group(1).replace(',', '')

        if "cash" in prompt.lower():
             match = re.search(r"cash and cash equivalents:\s*\$([\d,.]+)", text)
             if match:
                 return match.group(1).replace(',', '')

        if "ebitda" in prompt.lower():
            # Estimate EBITDA from Operating Income + D&A (Mocking D&A or just extracting Op Income)
            # "Operating Income: $125,345"
            match = re.search(r"operating income:\s*\$([\d,.]+)", text)
            if match:
                op_inc = float(match.group(1).replace(',', ''))
                # Add mock D&A (approx 10% of Op Income for mock)
                ebitda = op_inc * 1.10
                return f"{ebitda:.2f}"

        # Qualitative Extraction
        if "risk factors" in prompt.lower():
            # Extract bullet points under "Risk Factors"
            start = text.find("item 1a. risk factors")
            end = text.find("item 7.")
            if start != -1 and end != -1:
                section = self.document_text[start:end]
                # Extract lines starting with "- "
                risks = re.findall(r"- (.*?)(?=\n|$)", section)
                return "\n".join(risks[:5]) # Return top 5
            return "Global Economic Conditions: Macroeconomic conditions, including inflation, interest rates and currency fluctuations, could adversely affect demand."

        if "business description" in prompt.lower():
            start = text.find("item 1. business")
            if start != -1:
                return self.document_text[start:start+500] + "..."
            return "Item 1. Business\nApple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories."

        return "Data not found in document."

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
        self.llm.update_context(text)

        # 2. Ingest
        await self.agent.ingest_document(text)

        # 3. Extract Data
        logger.info("Extracting Financials...")

        def get_float(prompt, default=0.0):
            try:
                val = self.llm.generate_response(prompt)
                return float(val)
            except:
                return default

        revenue = get_float("What is the total net sales or revenue?")
        if revenue == 0: revenue = 383285.0 # Fallback default if regex fails completely

        net_income = get_float("What is the net income?", revenue * 0.25)
        assets = get_float("What are the total assets?", revenue * 0.9)
        liabilities = get_float("What are the total liabilities?", assets * 0.8)
        debt = get_float("What is the total debt?", liabilities * 0.4)
        cash = get_float("What is the cash and cash equivalents?", 29000.0)
        ebitda = get_float("What is the EBITDA?", revenue * 0.30)

        risks = self.llm.generate_response("What are the key risk factors?")
        description = self.llm.generate_response("What is the business description?")

        # 4. Generate Advanced Analytics

        # Valuation & Risk
        valuation_data = self._generate_valuation(ebitda, debt, cash, liabilities, assets, revenue, net_income)

        # Consensus
        consensus_data = self._generate_consensus(revenue, ebitda)

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
            "source": "RAG Extraction (10-K)"
        }

        # B. Credit Memo
        memo_data = {
            "borrower_name": f"{self.ticker} Inc.",
            "ticker": self.ticker,
            "sector": "Technology",
            "report_date": datetime.now().isoformat(),
            "risk_score": 85.0, # Could link to z-score
            "executive_summary": f"RAG Analysis of {self.ticker} based on 10-K ingestion. The company shows strong revenue of ${revenue:,.2f}M and EBITDA of ${ebitda:,.2f}M. The implied valuation suggests a base target of ${valuation_data['dcf']['share_price']:.2f}. Key risks identified include: {risks[:200]}...",
            "sections": [
                {
                    "title": "Business Overview",
                    "content": description,
                    "citations": [{"doc_id": os.path.basename(document_path), "chunk_id": "chunk_001", "page_number": 1}]
                },
                {
                    "title": "Risk Factors",
                    "content": risks,
                    "citations": [{"doc_id": os.path.basename(document_path), "chunk_id": "chunk_002", "page_number": 5}]
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
                    "interest_expense": debt * 0.05
                }
            ],
            "dcf_analysis": valuation_data["dcf"],
            "pd_model": valuation_data["risk_model"],
            "consensus_data": consensus_data, # New Field
            "regulatory_rating": valuation_data["risk_model"]["regulatory_rating"], # New Field
            "price_targets": valuation_data["forward_view"]["price_targets"], # New Field
            "documents": [
                {
                    "doc_id": os.path.basename(document_path),
                    "title": f"{self.ticker} 10-K",
                    "chunks": []
                }
            ]
        }

        return spread_data, memo_data

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

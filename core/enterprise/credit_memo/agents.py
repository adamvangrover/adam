from typing import Dict, Any, List
from .model import CreditMemoSection, Citation, EvidenceChunk, CreditMemo
from .citation_engine import citation_engine
from .spreading_engine import spreading_engine
from .graph_engine import graph_engine

class CreditAgent:
    """Base class for Credit Memo Agents."""
    def __init__(self, name: str):
        self.name = name

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class ArchivistAgent(CreditAgent):
    """Retrieves unstructured and structured context."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        borrower = context.get("borrower_name")
        query = context.get("query", "")

        # 1. Vector Search (Spatial RAG)
        chunks = citation_engine.retrieve_context(borrower, query)

        # 2. Graph Search (Entity Resolution)
        graph_data = graph_engine.query_relationships(borrower)

        return {
            "evidence_chunks": chunks,
            "graph_context": graph_data
        }

class QuantAgent(CreditAgent):
    """Handles financial spreading, PD/LGD modeling, and Scenario Analysis."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        borrower = context.get("borrower_name")
        raw_financials = context.get("raw_financial_text", "")

        # 1. Spreading
        spread = spreading_engine.spread_financials(borrower, raw_financials)

        # 2. Advanced Quant Models
        pd_model = spreading_engine.calculate_pd_model(spread)
        scenarios = spreading_engine.generate_scenarios(spread)

        # Note: LGD requires debt facilities which are fetched separately in Orchestrator for now,
        # but we can do a partial LGD here or let Orchestrator handle it.
        # For simplicity, we'll let Orchestrator call get_debt_facilities and then LGD.
        # Or better, we return what we can here.

        return {
            "financial_spread": spread,
            "pd_model": pd_model,
            "scenario_analysis": scenarios
        }

class RiskOfficerAgent(CreditAgent):
    """Identifies Red Flags and compliance issues."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        spread = context.get("financial_spread")
        graph = context.get("graph_context", [])

        risks = []

        # Financial Risks
        if spread.dscr < 1.25:
            risks.append("CRITICAL: DSCR below 1.25x covenant threshold.")

        if spread.leverage_ratio > 4.0:
            risks.append("HIGH: Leverage ratio exceeds 4.0x policy limit.")

        # Graph Risks
        for rel in graph:
            if rel['risk_level'] in ['High', 'Speculative']:
                risks.append(f"CONNECTED PARTY RISK: {rel['entity']} via {rel['path']} is {rel['risk_level']} risk.")

        return {"identified_risks": risks}

class WriterAgent(CreditAgent):
    """Synthesizes the final memo sections with citations."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Mock LLM Synthesis
        borrower = context.get("borrower_name")
        risks = context.get("identified_risks", [])
        spread = context.get("financial_spread")
        chunks = context.get("evidence_chunks", [])

        # Generate 'Executive Summary'
        exec_summary = f"The borrower {borrower} presents a mixed credit profile. "
        exec_summary += f"Financial performance shows strong EBITDA (${spread.ebitda}M) "
        exec_summary += f"but elevated leverage ({spread.leverage_ratio:.1f}x). "

        # Add citations (Mock logic: associate chunks)
        citations = []
        if chunks:
            # Cite first chunk
            c = chunks[0]
            citations.append(Citation(doc_id=c.doc_id, chunk_id=c.chunk_id, page_number=c.page_number))
            exec_summary += f" [Ref: {c.doc_id}]"

        # Generate 'Key Risks'
        risk_text = "Key Risks:\n"
        for r in risks:
            risk_text += f"- {r}\n"

        # Generate SWOT (Mock)
        strengths = [
            "Market leading position in core segment.",
            f"Strong EBITDA generation (${spread.ebitda}M).",
            "Diversified customer base."
        ]
        weaknesses = [
            f"Elevated leverage at {spread.leverage_ratio:.1f}x.",
            "Exposure to cyclical end-markets.",
            "Recent management turnover."
        ]
        mitigants = [
            "Strong free cash flow conversion.",
            "Demonstrated ability to deleverage.",
            "Sponsor support."
        ]

        return {
            "executive_summary": exec_summary,
            "risk_section": risk_text,
            "citations": citations,
            "key_strengths": strengths,
            "key_weaknesses": weaknesses,
            "mitigants": mitigants
        }

class SystemTwoAgent(CreditAgent):
    """
    Performs a second-pass critique and validation of the generated memo.
    Protocol: System 2 Thinking
    """
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        memo = context.get("credit_memo")
        if not memo:
            return {"error": "No memo provided for critique"}

        critique = spreading_engine.generate_critique(memo)

        return {
            "system_two_critique": critique
        }

from typing import Dict, Any, List
from .model import CreditMemoSection, Citation, EvidenceChunk
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
    """Handles financial spreading and ratio analysis."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        borrower = context.get("borrower_name")
        raw_financials = context.get("raw_financial_text", "")

        spread = spreading_engine.spread_financials(borrower, raw_financials)

        return {
            "financial_spread": spread
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

        return {
            "executive_summary": exec_summary,
            "risk_section": risk_text,
            "citations": citations
        }

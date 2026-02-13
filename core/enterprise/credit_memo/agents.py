from typing import Dict, Any, List
from .model import CreditMemoSection, Citation, EvidenceChunk, CreditMemo, Attribution, DebtFacilityRating
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
    """Handles financial spreading, ratio analysis, and valuation."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        borrower = context.get("borrower_name")
        raw_financials = context.get("raw_financial_text", "")

        spread = spreading_engine.spread_financials(borrower, raw_financials)

        # Valuation Logic (Mock DCF/Multiple)
        price_target = 0.0
        price_level = "Fair Value"
        market_cap = 0.0 # Millions
        enterprise_value = 0.0 # Millions
        reasoning = []

        if "apple" in borrower.lower():
            price_target = 245.0
            price_level = "Undervalued"
            market_cap = 3400000.0 # ~$3.4T
            enterprise_value = 3450000.0 # EV > MC due to net cash adjustment if mock logic was complex, but let's assume simplified
            reasoning = ["Strong Free Cash Flow yield", "Services revenue mix shift", "DCF Terminal Growth: 3.5%"]
        elif "tesla" in borrower.lower():
             price_target = 310.0
             price_level = "Overvalued"
             market_cap = 950000.0
             enterprise_value = 970000.0
             reasoning = ["Margin compression risk", "High valuation multiple vs peers", "WACC: 11.5%"]
        elif "jpmorgan" in borrower.lower():
             price_target = 195.0
             price_level = "Fair Value"
             market_cap = 580000.0
             enterprise_value = 600000.0 # Rough proxy
             reasoning = ["NII tailwinds", "Defensive balance sheet", "P/TBV: 2.1x"]
        elif "techcorp" in borrower.lower():
             price_target = 42.0
             price_level = "Undervalued"
             market_cap = 4200.0 # Small/Mid cap
             enterprise_value = 7200.0 # High Debt (3000)
             reasoning = ["Sum-of-parts valuation discount", "High Leverage Penalty applied to WACC"]

        attribution = Attribution(
            agent_id=self.name,
            model_version="Quant-v2.2 (DCF Enhanced)",
            justification=f"Valuation derived from DCF (WACC/Terminal Value) and Comparable Multiples.",
            key_factors=reasoning
        )

        return {
            "financial_spread": spread,
            "price_target": price_target,
            "price_level": price_level,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "attribution": attribution
        }

class RiskOfficerAgent(CreditAgent):
    """Identifies Red Flags, compliance issues, and assigns Credit Rating."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        spread = context.get("financial_spread")
        graph = context.get("graph_context", [])

        risks = []
        rating = "BBB"
        score = 50.0
        factors = []

        # Debt Ratings by Facility (Notching)
        debt_ratings = []

        # Financial Risks
        if spread.dscr < 1.25:
            risks.append("CRITICAL: DSCR below 1.25x covenant threshold.")
            rating = "B-"
            score = 30.0
            factors.append("Low DSCR (<1.25x)")

            debt_ratings = [
                DebtFacilityRating(facility_type="Senior Secured Term Loan", rating="B+", recovery_rating="RR2 (85%)", amount_outstanding=spread.total_liabilities * 0.6),
                DebtFacilityRating(facility_type="Senior Unsecured Notes", rating="CCC+", recovery_rating="RR5 (20%)", amount_outstanding=spread.total_liabilities * 0.4)
            ]

        elif spread.leverage_ratio > 4.0:
            risks.append("HIGH: Leverage ratio exceeds 4.0x policy limit.")
            rating = "BB"
            score = 45.0
            factors.append("High Leverage (>4.0x)")

            debt_ratings = [
                DebtFacilityRating(facility_type="Revolving Credit Facility", rating="BBB-", recovery_rating="RR1 (95%)", amount_outstanding=spread.total_liabilities * 0.1),
                DebtFacilityRating(facility_type="Senior Secured Notes", rating="BB", recovery_rating="RR3 (60%)", amount_outstanding=spread.total_liabilities * 0.9)
            ]

        elif spread.leverage_ratio < 2.0 and spread.dscr > 3.0:
            rating = "AA"
            score = 90.0
            factors.append("Strong Leverage (<2.0x)")

            debt_ratings = [
                DebtFacilityRating(facility_type="Senior Unsecured Debentures", rating="AA", recovery_rating="RR1", amount_outstanding=spread.total_liabilities * 1.0)
            ]

        elif spread.leverage_ratio < 3.0:
            rating = "A-"
            score = 75.0
            factors.append("Moderate Leverage (<3.0x)")

            debt_ratings = [
                DebtFacilityRating(facility_type="Commercial Paper", rating="A-1", recovery_rating="N/A", amount_outstanding=spread.total_liabilities * 0.2),
                DebtFacilityRating(facility_type="Senior Unsecured Notes", rating="A-", recovery_rating="RR3", amount_outstanding=spread.total_liabilities * 0.8)
            ]

        # Graph Risks
        for rel in graph:
            if rel['risk_level'] in ['High', 'Speculative']:
                risks.append(f"CONNECTED PARTY RISK: {rel['entity']} via {rel['path']} is {rel['risk_level']} risk.")
                score -= 10.0 # Penalty
                factors.append(f"Connected Party Risk: {rel['entity']}")

        attribution = Attribution(
            agent_id=self.name,
            model_version="RiskOfficer-v1.5 (Facility Notching)",
            justification=f"Assigned Corporate Rating {rating} and facility ratings based on LGD/RR models.",
            key_factors=factors
        )

        return {
            "identified_risks": risks,
            "credit_rating": rating,
            "debt_ratings": debt_ratings,
            "risk_score": max(0.0, min(100.0, score)),
            "attribution": attribution
        }

class MarketAnalystAgent(CreditAgent):
    """
    Analyzes market sentiment and conviction.
    Protocol: Sentiment Analysis
    """
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        borrower = context.get("borrower_name", "").lower()

        sentiment = 50.0
        conviction = 50.0
        factors = []

        if "apple" in borrower:
            sentiment = 65.0 # Positive product cycle
            conviction = 80.0 # High analyst consensus
            factors = ["Product Supercycle", "Analyst Consensus: Buy"]
        elif "tesla" in borrower:
            sentiment = 45.0 # Mixed news
            conviction = 40.0 # Divergent views
            factors = ["Regulatory Scrutiny", "Competition in China"]
        elif "jpmorgan" in borrower:
            sentiment = 70.0 # Strong earnings
            conviction = 85.0 # Blue chip
            factors = ["Sector Rotation into Value", "Rate Hike Beneficiary"]
        elif "techcorp" in borrower:
            sentiment = 30.0 # Sector headwinds
            conviction = 60.0 # Bearish view
            factors = ["Inventory Correction", "Cyclical Downturn"]

        attribution = Attribution(
            agent_id=self.name,
            model_version="MarketAnalyst-v3.0",
            justification=f"Sentiment derived from news flow and analyst dispersion.",
            key_factors=factors
        )

        return {
            "sentiment_score": sentiment,
            "conviction_score": conviction,
            "attribution": attribution
        }

class SystemTwoAgent(CreditAgent):
    """
    The 'System 2' Reviewer.
    Verifies coherence between Financials, Risks, and Sentiment.
    Protocol: Metacognition
    """
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Context contains the draft memo components
        memo_data = context.get("draft_memo", {})

        risk_score = memo_data.get("risk_score", 50.0)
        sentiment = memo_data.get("sentiment_score", 50.0)
        leverage = memo_data.get("financial_ratios", {}).get("leverage_ratio", 0.0)

        notes = []
        adjustments = []

        # Logic 1: High Leverage + High Sentiment -> Warning
        if leverage > 4.0 and sentiment > 60.0:
            notes.append("System 2 Correction: Market sentiment ignores high leverage risk. Lowering conviction.")
            memo_data["conviction_score"] = max(0, memo_data.get("conviction_score", 50) - 20)
            adjustments.append("lowered_conviction_score")

        # Logic 2: Low Risk Score but Investment Grade Rating -> Inconsistency
        rating = memo_data.get("credit_rating", "NR")
        if risk_score < 40 and rating.startswith("A"):
             notes.append(f"System 2 Alert: Risk Score ({risk_score}) inconsistent with Rating ({rating}).")
             # Adjust score up
             memo_data["risk_score"] = 60.0
             adjustments.append("raised_risk_score")

        # Logic 3: Verify Citations
        citations = context.get("citations", [])
        if not citations:
            notes.append("System 2 Warning: No citations found. Reducing validation confidence.")
            adjustments.append("citation_missing_penalty")

        if not notes:
             notes.append("System 2 Verified: No inconsistencies found.")

        attribution = Attribution(
            agent_id=self.name,
            model_version="SystemTwo-v1.0 (Metacognition)",
            justification="Review of coherence between financial metrics and qualitative signals.",
            key_factors=adjustments if adjustments else ["Consistent Profile"]
        )

        return {
            "system_two_notes": " | ".join(notes),
            "adjusted_memo_data": memo_data,
            "attribution": attribution
        }

class WriterAgent(CreditAgent):
    """Synthesizes the final memo sections with citations."""
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Mock LLM Synthesis
        borrower = context.get("borrower_name")
        risks = context.get("identified_risks", [])
        spread = context.get("financial_spread")
        chunks = context.get("evidence_chunks", [])

        sentiment = context.get("sentiment_score")
        rating = context.get("credit_rating")

        # Generate 'Executive Summary'
        exec_summary = f"The borrower {borrower} presents a {rating} credit profile. "
        exec_summary += f"Financial performance shows strong EBITDA (${spread.ebitda}M) "
        exec_summary += f"with leverage at {spread.leverage_ratio:.1f}x. "

        if sentiment > 60:
            exec_summary += "Market sentiment is bullish. "
        elif sentiment < 40:
            exec_summary += "Market sentiment is bearish. "

        # Add citations (Mock logic: associate chunks)
        citations = []
        if chunks:
            # Cite first chunk
            c = chunks[0]
            # Embed bbox and text for frontend
            citations.append(Citation(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                page_number=c.page_number,
                bbox=c.bbox,
                text_snippet=c.text[:50] + "..."
            ))
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

from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime

from .model import CreditMemo, CreditMemoSection, Citation, AuditLogEntry, Attribution
from .agents import ArchivistAgent, QuantAgent, RiskOfficerAgent, WriterAgent, MarketAnalystAgent, SystemTwoAgent
from .audit_logger import audit_logger
from .prompt_registry import registry as prompt_registry
from .auditor import AuditAgent

class CreditMemoOrchestrator:
    """
    Orchestrates the Credit Memo generation pipeline.
    Protocol: ADAM-V-NEXT (Agent Swarm)
    """
    def __init__(self):
        self.archivist = ArchivistAgent("Archivist")
        self.quant = QuantAgent("Quant")
        self.risk = RiskOfficerAgent("RiskOfficer")
        self.market = MarketAnalystAgent("MarketAnalyst")
        self.writer = WriterAgent("Writer")
        self.system_two = SystemTwoAgent("SystemTwo")
        self.auditor = AuditAgent()

    def generate_credit_memo(self, borrower_name: str, query: str = "", user_id: str = "system") -> CreditMemo:
        """
        Runs the full pipeline.
        """
        transaction_id = str(uuid.uuid4())
        logging.info(f"Starting Credit Memo Generation for {borrower_name} (Tx: {transaction_id})")

        # 1. Load Prompt (for Audit)
        prompt_id = "commercial_credit_risk_analysis"
        prompt_def = prompt_registry.get_prompt(prompt_id)
        prompt_ver = prompt_def.version if prompt_def else "unknown"

        # 2. Archivist (Retrieval)
        archivist_out = self.archivist.execute({"borrower_name": borrower_name, "query": query})
        chunks = archivist_out.get("evidence_chunks", [])
        graph_data = archivist_out.get("graph_context", [])

        # 3. Quant (Spreading & Valuation)
        raw_text = "ASSETS: 5000\nLIABILITIES: 3000\nEQUITY: 2000" if "TechCorp" in borrower_name else ""
        quant_out = self.quant.execute({"borrower_name": borrower_name, "raw_financial_text": raw_text})
        spread = quant_out.get("financial_spread")
        price_target = quant_out.get("price_target")
        price_level = quant_out.get("price_level")
        market_cap = quant_out.get("market_cap")
        enterprise_value = quant_out.get("enterprise_value")
        quant_attribution = quant_out.get("attribution")

        # 4. Risk (Analysis & Rating)
        risk_out = self.risk.execute({"financial_spread": spread, "graph_context": graph_data})
        risks = risk_out.get("identified_risks", [])
        rating = risk_out.get("credit_rating", "NR")
        debt_ratings = risk_out.get("debt_ratings", [])
        risk_score = risk_out.get("risk_score", 50.0)
        risk_attribution = risk_out.get("attribution")

        # 5. Market Analyst (Sentiment & Conviction)
        market_out = self.market.execute({"borrower_name": borrower_name})
        sentiment = market_out.get("sentiment_score", 50.0)
        conviction = market_out.get("conviction_score", 50.0)
        market_attribution = market_out.get("attribution")

        # 6. Writer (Synthesis)
        writer_ctx = {
            "borrower_name": borrower_name,
            "evidence_chunks": chunks,
            "financial_spread": spread,
            "identified_risks": risks,
            "sentiment_score": sentiment,
            "credit_rating": rating
        }
        writer_out = self.writer.execute(writer_ctx)

        exec_summary = writer_out.get("executive_summary", "")
        risk_text = writer_out.get("risk_section", "")
        citations = writer_out.get("citations", [])

        # 7. System Two (Review & Correction)
        draft_memo_data = {
            "borrower_name": borrower_name,
            "risk_score": risk_score,
            "sentiment_score": sentiment,
            "conviction_score": conviction,
            "credit_rating": rating,
            "financial_ratios": {
                "leverage_ratio": spread.leverage_ratio,
                "dscr": spread.dscr,
                "current_ratio": spread.current_ratio
            }
        }

        sys2_out = self.system_two.execute({"draft_memo": draft_memo_data, "citations": citations})
        adjusted_data = sys2_out.get("adjusted_memo_data", draft_memo_data)
        sys2_notes = sys2_out.get("system_two_notes", "System 2 Verified.")
        sys2_attribution = sys2_out.get("attribution")

        # Collect Attributions
        score_attributions = {
            "Valuation": quant_attribution,
            "Risk Score": risk_attribution,
            "Sentiment": market_attribution,
            "System Two": sys2_attribution
        }

        # Remove None values
        score_attributions = {k: v for k, v in score_attributions.items() if v is not None}

        # 8. Construct Final Memo
        memo = CreditMemo(
            borrower_name=borrower_name,
            executive_summary=exec_summary,
            sections=[
                CreditMemoSection(title="Executive Summary", content=exec_summary, citations=citations),
                CreditMemoSection(title="Key Risks & Mitigants", content=risk_text, citations=[]),
                CreditMemoSection(title="Financial Analysis", content=f"EBITDA: ${spread.ebitda}M | Leverage: {spread.leverage_ratio:.1f}x", citations=[])
            ],
            financial_ratios={
                "leverage_ratio": spread.leverage_ratio,
                "dscr": spread.dscr,
                "current_ratio": spread.current_ratio
            },
            risk_score=adjusted_data.get("risk_score", risk_score),
            sentiment_score=adjusted_data.get("sentiment_score", sentiment),
            conviction_score=adjusted_data.get("conviction_score", conviction),
            credit_rating=adjusted_data.get("credit_rating", rating),
            price_target=price_target,
            price_level=price_level,
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            debt_ratings=debt_ratings,
            system_two_notes=sys2_notes,
            score_attributions=score_attributions
        )

        # 9. Audit Logging (Pass/Fail Check)
        audit_result = self.auditor.audit_generation(memo, chunks)

        # Construct Full Trace for Audit
        trace = {k: v.model_dump() for k, v in score_attributions.items()}

        audit_entry = AuditLogEntry(
            transaction_id=transaction_id,
            user_id=user_id,
            action="GENERATE_CREDIT_MEMO",
            model_version="gpt-4-32k",
            prompt_version=prompt_ver,
            inputs={"borrower_name": borrower_name, "query": query},
            outputs={
                "risk_score": memo.risk_score,
                "rating": memo.credit_rating,
                "sys2_notes": sys2_notes
            },
            citations_count=len(citations),
            validation_status=audit_result["status"],
            validation_errors=audit_result["errors"],
            attribution_trace=trace
        )

        audit_logger.log_event(audit_entry)

        return memo

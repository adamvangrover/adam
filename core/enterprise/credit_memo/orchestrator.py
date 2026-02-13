from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime

from .model import CreditMemo, CreditMemoSection, Citation, AuditLogEntry
from .agents import ArchivistAgent, QuantAgent, RiskOfficerAgent, WriterAgent
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
        self.writer = WriterAgent("Writer")
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
        if not prompt_def:
            logging.warning(f"Prompt {prompt_id} not found. Using default context.")
            prompt_ver = "unknown"
        else:
            prompt_ver = prompt_def.version

        # 2. Archivist (Retrieval)
        archivist_out = self.archivist.execute({"borrower_name": borrower_name, "query": query})
        chunks = archivist_out.get("evidence_chunks", [])
        graph_data = archivist_out.get("graph_context", [])

        # 3. Quant (Spreading)
        # Simulate raw text extraction
        raw_text = "ASSETS: 5000\nLIABILITIES: 3000\nEQUITY: 2000" if "TechCorp" in borrower_name else ""
        quant_out = self.quant.execute({"borrower_name": borrower_name, "raw_financial_text": raw_text})
        spread = quant_out.get("financial_spread")

        # 4. Risk (Analysis)
        risk_out = self.risk.execute({"financial_spread": spread, "graph_context": graph_data})
        risks = risk_out.get("identified_risks", [])

        # 5. Writer (Synthesis)
        writer_ctx = {
            "borrower_name": borrower_name,
            "evidence_chunks": chunks,
            "financial_spread": spread,
            "identified_risks": risks
        }
        writer_out = self.writer.execute(writer_ctx)

        exec_summary = writer_out.get("executive_summary", "")
        risk_text = writer_out.get("risk_section", "")
        citations = writer_out.get("citations", [])

        # 6. Construct Memo
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
            risk_score=75.0 if spread.leverage_ratio < 4.0 else 45.0
        )

        # 7. Audit Logging (Pass/Fail Check)
        # Run System 2 Audit
        audit_result = self.auditor.audit_generation(memo, chunks)

        audit_entry = AuditLogEntry(
            transaction_id=transaction_id,
            user_id=user_id,
            action="GENERATE_CREDIT_MEMO",
            model_version="gpt-4-32k", # From prompt config
            prompt_version=prompt_ver,
            inputs={"borrower_name": borrower_name, "query": query},
            outputs={"memo_summary_len": len(exec_summary), "risk_score": memo.risk_score},
            citations_count=len(citations),
            validation_status=audit_result["status"],
            validation_errors=audit_result["errors"]
        )

        audit_logger.log_event(audit_entry)

        return memo

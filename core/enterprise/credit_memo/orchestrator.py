from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime

from .model import CreditMemo, CreditMemoSection, Citation, AuditLogEntry, DCFAnalysis
from .agents import ArchivistAgent, QuantAgent, RiskOfficerAgent, WriterAgent, SystemTwoAgent
from .audit_logger import audit_logger
from .prompt_registry import registry as prompt_registry
from .auditor import AuditAgent
from .spreading_engine import spreading_engine # Import global instance

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
        if not prompt_def:
            logging.warning(f"Prompt {prompt_id} not found. Using default context.")
            prompt_ver = "unknown"
        else:
            prompt_ver = prompt_def.version

        # 2. Archivist (Retrieval)
        archivist_out = self.archivist.execute({"borrower_name": borrower_name, "query": query})
        chunks = archivist_out.get("evidence_chunks", [])
        graph_data = archivist_out.get("graph_context", [])

        # 3. Quant (Spreading & Financials)
        # Simulate raw text extraction
        raw_text = "ASSETS: 5000\nLIABILITIES: 3000\nEQUITY: 2000" if "TechCorp" in borrower_name else ""
        quant_out = self.quant.execute({"borrower_name": borrower_name, "raw_financial_text": raw_text})
        spread = quant_out.get("financial_spread")
        pd_model = quant_out.get("pd_model")
        scenario_analysis = quant_out.get("scenario_analysis")

        # 3.1 Advanced Quant: Historicals & DCF
        # Note: QuantAgent normally calls spreading_engine internally, but for this demo
        # we access the engine logic directly or extend the agent.
        # Here we extend the workflow explicitly.
        historicals = spreading_engine.get_historicals(spread)
        dcf = spreading_engine.calculate_dcf(spread)

        # 3.2 Advanced Quant: Capital Structure (New)
        ratings = spreading_engine.get_credit_ratings(borrower_name)
        debt = spreading_engine.get_debt_facilities(borrower_name)
        equity = spreading_engine.get_equity_data(borrower_name)
        repayment_schedule = spreading_engine.get_debt_repayment_schedule(debt)

        # 3.3 Advanced Quant: LGD Analysis
        lgd = spreading_engine.calculate_lgd_analysis(debt, spread.total_assets)

        # 3.4 Advanced Quant: Peer Comps (New)
        peer_comps = spreading_engine.get_peer_comps(borrower_name)

        # 3.5 Agent Workflow Log (New)
        agent_log = spreading_engine.get_agent_log(borrower_name)

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
        strengths = writer_out.get("key_strengths", [])
        weaknesses = writer_out.get("key_weaknesses", [])
        mitigants = writer_out.get("mitigants", [])

        # 6. Construct Memo (Draft)
        memo = CreditMemo(
            borrower_name=borrower_name,
            executive_summary=exec_summary,
            sections=[
                CreditMemoSection(
                    title="Executive Summary",
                    content=exec_summary,
                    citations=citations,
                    author_agent="Writer"
                ),
                CreditMemoSection(
                    title="Key Risks & Mitigants",
                    content=risk_text,
                    citations=[],
                    author_agent="Risk Officer"
                ),
                CreditMemoSection(
                    title="Financial Analysis",
                    content=f"EBITDA: ${spread.ebitda}M | Leverage: {spread.leverage_ratio:.1f}x",
                    citations=[],
                    author_agent="Quant"
                )
            ],
            financial_ratios={
                "leverage_ratio": spread.leverage_ratio,
                "dscr": spread.dscr,
                "current_ratio": spread.current_ratio,
                "revenue": spread.revenue,
                "ebitda": spread.ebitda,
                "net_income": spread.net_income
            },
            historical_financials=[h.model_dump() for h in historicals],
            dcf_analysis=dcf,
            pd_model=pd_model,
            lgd_analysis=lgd,
            scenario_analysis=scenario_analysis,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            mitigants=mitigants,
            risk_score=pd_model.model_score if pd_model else 75.0, # Use PD model score if available
            credit_ratings=ratings,
            debt_facilities=debt,
            repayment_schedule=repayment_schedule,
            equity_data=equity,
            peer_comps=peer_comps,
            agent_log=agent_log
        )

        # 7. System 2 Critique (Validation)
        # We manually invoke spreading_engine.generate_critique to ensure we get the new rich fields
        # In a full system, the SystemTwoAgent would call this.
        # s2_out = self.system_two.execute({"credit_memo": memo})
        # critique = s2_out.get("system_two_critique")
        critique = spreading_engine.generate_critique(memo)
        memo.system_two_critique = critique

        # 8. Audit Logging (Pass/Fail Check)
        # Run System 2 Audit (using old auditor for compliance checks)
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

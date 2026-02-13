from typing import Any, Dict, List, Optional
import logging
import asyncio
from abc import abstractmethod

from core.agents.agent_base import AgentBase
from core.audit.audit_logger import AuditLogger, AuditLog

class CreditAgentBase(AgentBase):
    """
    Base class for all Credit Memo agents.
    Enforces audit logging and strict output schemas.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.audit_logger = AuditLogger()
        self.agent_type = config.get("agent_type", "generic_credit_agent")

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the agent's task.
        Must return a dictionary containing at least 'output', 'confidence', and 'citations'.
        """
        pass

    def log_execution(self,
                      inputs: Dict[str, Any],
                      output: str,
                      citations: List[str],
                      metadata: Dict[str, Any] = None):
        """
        Helper to write to the centralized Audit Log.
        """
        try:
            log_entry = AuditLog(
                user_id=self.config.get("user_id", "system"),
                prompt_version_id=metadata.get("prompt_version_id", "unknown"),
                model_id=self.config.get("model_id", "gpt-4"),
                hyperparameters=self.config.get("hyperparameters", {}),
                retrieved_chunks=metadata.get("retrieved_chunks", []),
                graph_context=metadata.get("graph_context", []),
                raw_output=output,
                citations=citations,
                guardrail_status="PASS" # Mocked for now
            )
            self.audit_logger.log_event(log_entry)
        except Exception as e:
            logging.error(f"Failed to log execution for {self.agent_type}: {e}")

    def format_citation(self, doc_id: str, chunk_id: str) -> str:
        """
        Returns the standardized citation format: [doc_id:chunk_id]
        """
        return f"[{doc_id}:{chunk_id}]"

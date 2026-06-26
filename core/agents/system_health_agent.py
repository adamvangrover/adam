from src.pdil.models import ProvenanceHeader
import time
from typing import Dict, Any

from pydantic import BaseModel

from core.schemas.agent_schema import AgentInput, AgentOutput
from core.agents.agent_base import AgentBase


class HealthMetrics(BaseModel):
    agent_id: str
    uptime_seconds: float
    error_count: int


class SystemHealthAgent(AgentBase):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.start_time = time.time()
        self.error_count = 0

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        metrics = HealthMetrics(
            agent_id=self.config.get("agent_id", "unknown"),
            uptime_seconds=time.time() - self.start_time,
            error_count=self.error_count
        )
        return AgentOutput(provenance_trace=ProvenanceHeader(git_commit_hash="legacy", timestamp="1970-01-01T00:00:00Z", content_hash="legacy", jsonLogic_version="legacy", confidence_score=1.0, derivation_path="legacy", source_data_object="legacy"),
            answer="System is healthy",
            confidence=1.0,
            metadata={"status": "healthy", "metrics": metrics.model_dump()}
        )

    # Dummy additive method for daily expansion
    def ping(self) -> str:
        return "pong"

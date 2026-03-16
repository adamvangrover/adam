import time
from typing import Any, Dict

from pydantic import BaseModel

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

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        metrics = HealthMetrics(
            agent_id=self.config.get("agent_id", "unknown"),
            uptime_seconds=time.time() - self.start_time,
            error_count=self.error_count
        )
        return {"status": "healthy", "metrics": metrics.model_dump()}

    # Dummy additive method for daily expansion
    def ping(self) -> str:
        return "pong"

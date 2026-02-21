from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging
import psutil
import platform
import os
import datetime

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    query: str = Field(..., description="The user's objective or question.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state.")

class AgentOutput(BaseModel):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0-1).")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemHealthAgent:
    """
    Agent responsible for monitoring system health metrics using psutil.
    """
    def __init__(self, agent_name: str = "SystemHealthAgent"):
        self.agent_name = agent_name
        self.logger = logger.getChild(agent_name)

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        self.logger.info(f"Checking system health for: {input_data.query}")

        try:
            # Gather metrics
            # Note: interval=0.1 blocks for 100ms. In async context, this might block event loop slightly.
            # Ideally use run_in_executor, but for this simple agent it's acceptable.
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            system_info = {
                "os": platform.system(),
                "release": platform.release(),
                "python_version": platform.python_version(),
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": disk.percent,
                "timestamp": datetime.datetime.now().isoformat()
            }

            summary = (
                f"System Health Check:\n"
                f"- OS: {system_info['os']} {system_info['release']}\n"
                f"- Python: {system_info['python_version']}\n"
                f"- CPU Usage: {system_info['cpu_percent']}%\n"
                f"- Memory: {system_info['memory_available_gb']}GB available / {system_info['memory_total_gb']}GB total ({system_info['memory_percent']}%)\n"
                f"- Disk: {system_info['disk_free_gb']}GB free / {system_info['disk_total_gb']}GB total ({system_info['disk_percent']}%)"
            )

            return AgentOutput(
                answer=summary,
                sources=["psutil", "platform"],
                confidence=1.0,
                metadata=system_info
            )

        except Exception as e:
            self.logger.error(f"Error checking system health: {e}", exc_info=True)
            return AgentOutput(
                answer=f"Error checking system health: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )

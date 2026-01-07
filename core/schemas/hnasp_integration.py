from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class IntegratedAgentState(BaseModel):
    """
    Placeholder state for HNASP integration.
    """
    agent_id: str
    state_data: Dict[str, Any] = Field(default_factory=dict)

class AgentPacket(BaseModel):
    """
    Placeholder packet for agent communication.
    """
    source: str
    destination: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

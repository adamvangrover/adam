from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class AgentPacket(BaseModel):
    """
    Standardized packet for inter-agent communication.
    """
    id: str = Field(..., description="Unique ID of the packet")
    sender: str = Field(..., description="Name of the sending agent")
    recipient: str = Field(..., description="Name of the receiving agent")
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: Dict[str, Any] = Field(default_factory=dict)
    message_type: str = Field("default", description="Type of message (e.g., 'command', 'response')")

class IntegratedAgentState(BaseModel):
    """
    Represents the integrated state of an agent, combining HNASP and other state models.
    """
    agent_id: str
    current_task: Optional[str] = None
    memory_context: Dict[str, Any] = Field(default_factory=dict)
    active_tools: List[str] = Field(default_factory=list)
    status: str = "idle"

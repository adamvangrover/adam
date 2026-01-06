from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class AgentPacket(BaseModel):
    """
    Standardized packet for inter-agent communication.
    Merged: Uses 'packet_id' (main) for explicit naming, but retains 
    routing fields (sender, recipient) from v24.
    """
    packet_id: str = Field(..., description="Unique ID of the packet")
    sender: str = Field(..., description="Name of the sending agent")
    recipient: str = Field(..., description="Name of the receiving agent")
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: str = Field("default", description="Type of message (e.g., 'command', 'response')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured content payload")

class IntegratedAgentState(BaseModel):
    """
    Represents the integrated state of an agent.
    Merged: Retains specific operational fields (task, memory, tools) from v24 
    instead of the generic 'data' dict from main.
    """
    agent_id: str = Field(..., description="ID of the agent this state belongs to")
    current_task: Optional[str] = None
    memory_context: Dict[str, Any] = Field(default_factory=dict, description="Active working memory")
    active_tools: List[str] = Field(default_factory=list, description="Currently enabled tools")
    status: str = Field("idle", description="Current operational status")
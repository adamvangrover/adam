from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class IntegratedAgentState(BaseModel):
    """
    Minimal placeholder for IntegratedAgentState.
    """
    state_id: str = Field(..., description="Unique ID of the state.")
    data: Dict[str, Any] = Field(default_factory=dict, description="State data.")

class AgentPacket(BaseModel):
    """
    Minimal placeholder for AgentPacket.
    """
    packet_id: str = Field(..., description="Unique ID of the packet.")
    payload: Any = Field(..., description="Packet payload.")

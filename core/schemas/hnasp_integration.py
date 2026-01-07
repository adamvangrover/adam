from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class AgentPacket(BaseModel):
    """
    Standardized packet for inter-agent communication.
    Merged: Adopts 'sender'/'recipient' (main) for clarity over 'source'/'destination'.
    Retains 'metadata' (guide) for extensibility and adds automatic UUID generation.
    """
    packet_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique ID of the packet"
    )
    sender: str = Field(..., description="Name/ID of the sending agent")
    recipient: str = Field(..., description="Name/ID of the receiving agent")
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: str = Field("default", description="Type of message (e.g., 'command', 'response')")
    
    # Merged payload/content into 'payload' but kept 'metadata' from guide branch
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured content payload")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Routing or tracing metadata")

class IntegratedAgentState(BaseModel):
    """
    Represents the integrated state of an agent.
    Merged: Uses specific fields (memory, tools) from 'main' for structure,
    but adds 'custom_state' to capture the generic 'state_data' from the 'guide' branch.
    """
    agent_id: str = Field(..., description="ID of the agent this state belongs to")
    status: str = Field("idle", description="Current operational status")
    current_task: Optional[str] = None
    
    # Structured context from 'main'
    memory_context: Dict[str, Any] = Field(default_factory=dict, description="Active working memory")
    active_tools: List[str] = Field(default_factory=list, description="Currently enabled tools")
    
    # Flexible fallback from 'guide' (originally state_data)
    custom_state: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific or legacy state data")
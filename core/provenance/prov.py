from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime

class ProvEntity(BaseModel):
    id: str
    type: str = "prov:Entity"
    attributes: Dict[str, Any] = Field(default_factory=dict)

class ProvActivity(BaseModel):
    id: str
    type: str = "prov:Activity"
    startTime: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    endTime: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class ProvAgent(BaseModel):
    id: str
    type: str = "prov:Agent"
    agent_type: str # 'person', 'software', 'model'
    attributes: Dict[str, Any] = Field(default_factory=dict)

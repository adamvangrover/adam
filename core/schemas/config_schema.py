from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    """
    Configuration for a single agent.
    """
    persona: Optional[str] = None
    description: Optional[str] = None
    expertise: Optional[List[str]] = None
    data_sources: Optional[List[str]] = None
    communication_style: Optional[str] = None
    # Allow arbitrary other fields for specific agents (e.g., 'alerting_thresholds')
    model_config = ConfigDict(extra='allow', populate_by_name=True)


class AgentsYamlConfig(BaseModel):
    """
    Schema for config/agents.yaml
    """
    agents: Dict[str, AgentConfig]
    default: Optional[Dict[str, Any]] = None

    # Allow other top-level keys like 'skills', 'alerting', etc.
    model_config = ConfigDict(extra='allow')


class WorkflowConfig(BaseModel):
    """
    Schema for config/workflows.yaml
    """
    agents: List[str]
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    model_config = ConfigDict(extra='allow')


class WorkflowsYamlConfig(BaseModel):
    """
    Top level schema for config/workflows.yaml
    """
    # The file structure seems to be { "workflow_name": { "agents": [], ... } }
    # So it's a Dict[str, WorkflowConfig] effectively.
    # However, since pydantic expects a model, we can map it or just use RootModel.
    pass

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

# --- Evolutionary Architect Schemas ---

class EvolutionGoal(str, Enum):
    REFACTOR = "refactor"
    FEATURE_ADDITION = "feature_addition"
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"

class EvolutionaryArchitectInput(BaseModel):
    goal_type: EvolutionGoal = Field(..., description="The type of evolution to perform.")
    description: str = Field(..., description="Natural language description of the desired change.")
    target_files: Optional[List[str]] = Field(None, description="Specific files to target, if known.")
    constraints: List[str] = Field(default_factory=list, description="Constraints to adhere to (e.g., 'no external deps').")

class ProposedChange(BaseModel):
    file_path: str
    diff: str
    rationale: str

class EvolutionaryArchitectOutput(BaseModel):
    proposal_id: str
    changes: List[ProposedChange]
    impact_analysis: str
    safety_score: float = Field(..., ge=0.0, le=1.0)
    status: str = "proposed"

# --- Didactic Architect Schemas ---

class AudienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class DidacticArchitectInput(BaseModel):
    topic: str = Field(..., description="The topic to explain or teach.")
    target_audience: AudienceLevel = Field(default=AudienceLevel.INTERMEDIATE)
    context_files: List[str] = Field(default_factory=list, description="Files relevant to the topic.")
    output_format: str = Field("markdown", description="Format of the output (markdown, html, notebook).")

class TutorialSection(BaseModel):
    title: str
    content: str
    code_snippets: List[str]

class PortableConfig(BaseModel):
    filename: str
    content: str
    description: str

class DidacticArchitectOutput(BaseModel):
    title: str
    sections: List[TutorialSection]
    setup_guide: str
    portable_configs: List[PortableConfig] = Field(default_factory=list, description="Generated config files like Dockerfiles.")

# --- Chronos Agent Schemas ---

class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"   # Current session / immediate context
    MEDIUM_TERM = "medium_term" # Recent interactions / active project
    LONG_TERM = "long_term"     # Historical data / deep archival

class ChronosInput(BaseModel):
    query: str = Field(..., description="The query or context to analyze temporally.")
    horizons: List[TimeHorizon] = Field(default_factory=lambda: [TimeHorizon.SHORT_TERM], description="Time horizons to check.")
    reference_date: Optional[datetime] = Field(None, description="The 'current' date for relative comparison.")

class MemoryFragment(BaseModel):
    source: str
    content: str
    timestamp: Optional[datetime]
    relevance_score: float

class HistoricalComparison(BaseModel):
    period_name: str
    similarity_score: float
    key_similarities: List[str]
    key_differences: List[str]

class ChronosOutput(BaseModel):
    query_context: str
    memories: Dict[TimeHorizon, List[MemoryFragment]]
    historical_analogs: List[HistoricalComparison]
    temporal_synthesis: str = Field(..., description="Synthesis of the temporal analysis.")

from typing import TypedDict, List, Optional, Annotated
import operator

class ResearchArtifact(TypedDict):
    title: str
    content: str
    source: str
    credibility_score: float

class RiskAssessmentState(TypedDict):
    """
    The 'Memory' of a v23 Reasoning Loop. 
    Tracks the evolution of the analysis from draft to final.
    """
    # Input
    ticker: str
    user_intent: str
    
    # Internal Reasoning State
    research_data: Annotated[List[ResearchArtifact], operator.add] # Append-only log
    draft_analysis: Optional[str]
    critique_notes: List[str]
    iteration_count: int
    
    # Control Flow
    quality_score: float
    needs_correction: bool
    
    # Explainability
    human_readable_status: str # "I am currently verifying the debt ratio..."

def init_risk_state(ticker: str, intent: str) -> RiskAssessmentState:
    return {
        "ticker": ticker,
        "user_intent": intent,
        "research_data": [],
        "draft_analysis": None,
        "critique_notes": [],
        "iteration_count": 0,
        "quality_score": 0.0,
        "needs_correction": False,
        "human_readable_status": "Initializing analysis..."
    }

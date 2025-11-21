from typing import TypedDict, List, Optional, Annotated, Dict, Any
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

class SNCAnalysisState(TypedDict):
    """
    The State for the Shared National Credit (SNC) Analysis Graph.
    """
    # Input
    obligor_id: str
    syndicate_data: Dict[str, Any] # Banks, vote shares, etc.
    financials: Dict[str, Any]

    # Internal Reasoning State
    structure_analysis: Optional[str] # Analysis of the syndicate structure
    regulatory_rating: Optional[str] # Pass, SM, SS, D, L
    rationale: Optional[str]
    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    is_compliant: bool
    needs_revision: bool

    # Explainability
    human_readable_status: str

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

def init_snc_state(obligor_id: str, syndicate_data: Dict, financials: Dict) -> SNCAnalysisState:
    return {
        "obligor_id": obligor_id,
        "syndicate_data": syndicate_data,
        "financials": financials,
        "structure_analysis": None,
        "regulatory_rating": None,
        "rationale": None,
        "critique_notes": [],
        "iteration_count": 0,
        "is_compliant": True,
        "needs_revision": False,
        "human_readable_status": "Initializing SNC analysis..."
    }

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

class MarketSentimentState(TypedDict):
    """
    State for the Market Sentiment & News Monitoring Graph.
    """
    # Input
    ticker: str
    target_sector: str

    # Internal Reasoning State
    news_feed: List[Dict[str, Any]]  # List of articles/tweets
    sentiment_score: float           # -1.0 to 1.0
    sentiment_trend: Literal["bullish", "bearish", "neutral"]
    key_drivers: List[str]           # "Inflation", "Earnings", etc.

    # KG Integration
    related_entities: List[str]      # Entities found in KG related to the news

    # Output
    alert_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    final_report: Optional[str]

    # Control Flow
    iteration_count: int

    # Explainability
    human_readable_status: str

class RedTeamState(TypedDict):
    """
    State for the Adversarial Red Team Loop.
    """
    target_entity: str
    scenario_type: str # e.g. "Cyber", "Macro", "Regulatory"
    current_scenario_description: str
    simulated_impact_score: float # 0.0 to 10.0
    severity_threshold: float

    # Internal
    critique_notes: List[str]
    iteration_count: int
    is_sufficiently_severe: bool

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

def init_sentiment_state(ticker: str, sector: str) -> MarketSentimentState:
    return {
        "ticker": ticker,
        "target_sector": sector,
        "news_feed": [],
        "sentiment_score": 0.0,
        "sentiment_trend": "neutral",
        "key_drivers": [],
        "related_entities": [],
        "alert_level": "LOW",
        "final_report": None,
        "iteration_count": 0,
        "human_readable_status": "Initializing Sentiment Monitor..."
    }

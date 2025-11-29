from typing import TypedDict, List, Optional, Annotated, Dict, Any, Literal
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
    data_context: Dict[str, Any] # Raw data from MCP
    
    # Internal Reasoning State
    analysis_draft: Optional[str] # Current version of the output
    critique_feedback: List[str] # List of strings from the Reviewer node
    revision_count: int # Integer
    
    # Legacy/Helper fields
    research_data: Annotated[List[ResearchArtifact], operator.add] # Append-only log
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

class ESGAnalysisState(TypedDict):
    """
    State for the ESG (Environmental, Social, Governance) Analysis Graph.
    """
    # Input
    company_name: str
    sector: str

    # Internal Reasoning State
    env_score: float
    social_score: float
    gov_score: float
    total_esg_score: float
    controversies: List[str]

    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    needs_revision: bool

    # Explainability
    human_readable_status: str
    final_report: Optional[str]

class ComplianceState(TypedDict):
    """
    State for the Regulatory Compliance Graph.
    """
    # Input
    entity_id: str
    jurisdiction: str

    # Internal Reasoning
    applicable_regulations: List[str]
    potential_violations: List[str]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    needs_revision: bool

    # Explainability
    human_readable_status: str
    final_report: Optional[str]

def init_risk_state(ticker: str, intent: str) -> RiskAssessmentState:
    return {
        "ticker": ticker,
        "user_intent": intent,
        "data_context": {},
        "analysis_draft": None,
        "critique_feedback": [],
        "revision_count": 0,
        "research_data": [],
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

def init_esg_state(company: str, sector: str) -> ESGAnalysisState:
    return {
        "company_name": company,
        "sector": sector,
        "env_score": 0.0,
        "social_score": 0.0,
        "gov_score": 0.0,
        "total_esg_score": 0.0,
        "controversies": [],
        "critique_notes": [],
        "iteration_count": 0,
        "needs_revision": False,
        "human_readable_status": "Initializing ESG Analysis...",
        "final_report": None
    }

def init_compliance_state(entity: str, jurisdiction: str) -> ComplianceState:
    return {
        "entity_id": entity,
        "jurisdiction": jurisdiction,
        "applicable_regulations": [],
        "potential_violations": [],
        "risk_level": "LOW",
        "critique_notes": [],
        "iteration_count": 0,
        "needs_revision": False,
        "human_readable_status": "Initializing Compliance Check...",
        "final_report": None
    }

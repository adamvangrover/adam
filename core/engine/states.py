from typing import TypedDict, List, Optional, Annotated, Dict, Any, Literal
import operator
from datetime import datetime


class ResearchArtifact(TypedDict):
    title: str
    content: str
    source: str
    credibility_score: float


class PlanOnGraph(TypedDict):
    """A symbolic scaffold representing the causal links and logical steps."""
    id: str
    steps: List[Dict[str, Any]]
    is_complete: bool
    cypher_query: Optional[str]


class GraphState(TypedDict):
    """
    Represents the state of the general purpose Adaptive System Graph.
    Used by the NeuroSymbolicPlanner.
    """
    request: str
    plan: Optional[PlanOnGraph]
    current_task_index: int
    assessment: Optional[Dict[str, Any]]
    critique: Optional[Dict[str, Any]]
    human_feedback: Optional[str]
    iteration: int
    max_iterations: int


class RiskAssessmentState(TypedDict):
    """
    The 'Memory' of a v23 Reasoning Loop. 
    Tracks the evolution of the analysis from draft to final.
    """
    # Input
    ticker: str
    user_intent: str

    # Internal Reasoning State
    research_data: Annotated[List[ResearchArtifact], operator.add]  # Append-only log
    draft_analysis: Optional[str]
    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    quality_score: float
    needs_correction: bool

    # Explainability
    human_readable_status: str  # "I am currently verifying the debt ratio..."


class SNCAnalysisState(TypedDict):
    """
    The State for the Shared National Credit (SNC) Analysis Graph.
    """
    # Input
    obligor_id: str
    syndicate_data: Dict[str, Any]  # Banks, vote shares, etc.
    financials: Dict[str, Any]

    # Internal Reasoning State
    structure_analysis: Optional[str]  # Analysis of the syndicate structure
    regulatory_rating: Optional[str]  # Pass, SM, SS, D, L
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
    scenario_type: str  # e.g. "Cyber", "Macro", "Regulatory"
    current_scenario_description: str
    simulated_impact_score: float  # 0.0 to 10.0
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


class QuantumRiskState(TypedDict):
    """
    State for the Quantum-Enhanced Risk Graph.
    Tracks the execution of QMC simulations and Hybrid QNN inference.
    """
    # Input
    portfolio_id: str
    risk_factors: Dict[str, Any]

    # Internal Reasoning
    simulation_type: Literal["QMC_MERTON", "HYBRID_QNN", "GENERATIVE_STRESS"]
    simulation_results: Dict[str, Any]  # e.g. PD, VaR, Asset Value
    quantum_execution_time: float
    classical_fallback_triggered: bool

    # Explainability
    icaa_score: float  # Inter-Class Attribution Alignment
    human_readable_status: str
    final_report: Optional[str]

    # Control Flow
    iteration_count: int


class CrisisSimulationState(TypedDict):
    """
    State for the Macro-Economic Crisis Simulation Graph.
    """
    # Input
    scenario_description: str  # "Interest rates +5%"
    portfolio_data: Dict[str, Any]

    # Internal Reasoning
    macro_variables: Dict[str, float]  # { "rates": 0.08, "gdp": -0.02 }
    first_order_impacts: List[str]
    second_order_impacts: List[str]
    estimated_loss: float

    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    is_realistic: bool
    needs_refinement: bool

    # Explainability
    human_readable_status: str
    final_report: Optional[str]


class ReflectorState(TypedDict):
    """
    State for the Reflector (Meta-Cognition) Graph.
    """
    input_content: str
    context: Dict[str, Any]

    critique_notes: List[str]
    score: float
    is_valid: bool

    refined_content: Optional[str]
    iteration_count: int
    human_readable_status: str

# --- Omniscient State (v23.5) ---


class EntityEcosystem(TypedDict):
    legal_entity: Dict[str, str]
    management_assessment: Dict[str, Any]
    competitive_positioning: Dict[str, str]


class EquityAnalysis(TypedDict):
    fundamentals: Dict[str, str]
    valuation_engine: Dict[str, Any]


class CreditAnalysis(TypedDict):
    snc_rating_model: Dict[str, Any]
    cds_market_implied_rating: str
    covenant_risk_analysis: Dict[str, Any]


class SimulationEngine(TypedDict):
    monte_carlo_default_prob: float
    quantum_scenarios: List[Dict[str, Any]]
    trading_dynamics: Dict[str, str]


class StrategicSynthesis(TypedDict):
    m_and_a_posture: str
    final_verdict: Dict[str, Any]


class OmniscientNodes(TypedDict):
    entity_ecosystem: EntityEcosystem
    equity_analysis: EquityAnalysis
    credit_analysis: CreditAnalysis
    simulation_engine: SimulationEngine
    strategic_synthesis: StrategicSynthesis


class OmniscientMeta(TypedDict):
    target: str
    generated_at: str
    model_version: str


class OmniscientKnowledgeGraph(TypedDict):
    meta: OmniscientMeta
    nodes: OmniscientNodes


class OmniscientState(TypedDict):
    """
    State for the v23.5 'AI Partner' Omniscient Workflow.
    """
    ticker: str  # Input
    v23_knowledge_graph: OmniscientKnowledgeGraph  # Output
    human_readable_status: str

# --- Initializers ---


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


def init_quantum_state(portfolio_id: str, risk_factors: Dict) -> QuantumRiskState:
    return {
        "portfolio_id": portfolio_id,
        "risk_factors": risk_factors,
        "simulation_type": "QMC_MERTON",
        "simulation_results": {},
        "quantum_execution_time": 0.0,
        "classical_fallback_triggered": False,
        "icaa_score": 0.0,
        "human_readable_status": "Initializing Quantum Risk Engine...",
        "final_report": None,
        "iteration_count": 0
    }


def init_crisis_state(scenario: str, portfolio: Dict) -> CrisisSimulationState:
    return {
        "scenario_description": scenario,
        "portfolio_data": portfolio,
        "macro_variables": {},
        "first_order_impacts": [],
        "second_order_impacts": [],
        "estimated_loss": 0.0,
        "critique_notes": [],
        "iteration_count": 0,
        "is_realistic": False,
        "needs_refinement": True,
        "human_readable_status": "Initializing Crisis Simulation...",
        "final_report": None
    }


def init_reflector_state(content: str, context: Dict = None) -> ReflectorState:
    return {
        "input_content": content,
        "context": context or {},
        "critique_notes": [],
        "score": 0.0,
        "is_valid": False,
        "refined_content": None,
        "iteration_count": 0,
        "human_readable_status": "Initializing Reflection..."
    }


def init_omniscient_state(ticker: str) -> OmniscientState:
    empty_nodes: OmniscientNodes = {
        "entity_ecosystem": {},
        "equity_analysis": {},
        "credit_analysis": {},
        "simulation_engine": {},
        "strategic_synthesis": {}
    }

    empty_meta: OmniscientMeta = {
        "target": ticker,
        "generated_at": datetime.now().isoformat(),
        "model_version": "Adam-v23.5"
    }

    return {
        "ticker": ticker,
        "v23_knowledge_graph": {
            "meta": empty_meta,
            "nodes": empty_nodes
        },
        "human_readable_status": "Initializing AI Partner Deep Dive..."
    }

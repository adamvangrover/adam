from typing import TypedDict, List, Optional, Annotated, Dict, Any, Literal
import operator
from datetime import datetime


# NOTE: TypedDicts use total=False to allow for partial updates during graph execution (fixing strictness issues).
class ResearchArtifact(TypedDict, total=False):
    """
    A specific piece of research data collected by an agent.

    Attributes:
        title (str): Title of the research artifact.
        content (str): The actual text content.
        source (str): Source URL or citation.
        credibility_score (float): Assessment of source credibility (0.0-1.0).
    """
    title: str
    content: str
    source: str
    credibility_score: float


class PlanOnGraph(TypedDict, total=False):
    """
    A symbolic scaffold representing the causal links and logical steps.

    Attributes:
        id (str): Unique identifier for the plan.
        steps (List[Dict[str, Any]]): Sequence of planned steps.
        is_complete (bool): Whether the plan is fully executed.
        cypher_query (Optional[str]): Associated Cypher query for KG retrieval.
    """
    id: str
    steps: List[Dict[str, Any]]
    is_complete: bool
    cypher_query: Optional[str]


class GraphState(TypedDict, total=False):
    """
    Represents the state of the general purpose Adaptive System Graph.
    Used by the NeuroSymbolicPlanner.

    Attributes:
        request (str): The original user request.
        plan (Optional[PlanOnGraph]): The current execution plan.
        current_task_index (int): Index of the current step in the plan.
        assessment (Optional[Dict[str, Any]]): Assessment data.
        critique (Optional[Dict[str, Any]]): Critique data.
        human_feedback (Optional[str]): Feedback from human user.
        iteration (int): Current iteration count.
        max_iterations (int): Maximum allowed iterations.
    """
    request: str
    plan: Optional[PlanOnGraph]
    current_task_index: int
    assessment: Optional[Dict[str, Any]]
    critique: Optional[Dict[str, Any]]
    human_feedback: Optional[str]
    iteration: int
    max_iterations: int


class RiskAssessmentState(TypedDict, total=False):
    """
    The 'Memory' of a v23 Reasoning Loop. 
    Tracks the evolution of the analysis from draft to final.

    Attributes:
        ticker (str): The stock ticker symbol.
        user_intent (str): The user's goal (e.g., "assess credit risk").
        research_data (List[ResearchArtifact]): Collected research.
        draft_analysis (Optional[str]): The draft report.
        critique_notes (List[str]): Notes from the reviewer.
        iteration_count (int): Number of refinement loops.
        quality_score (float): Quality score of the draft.
        needs_correction (bool): Flag for needing revision.
        human_readable_status (str): Status message for UI.
    """
    # Input
    ticker: str
    user_intent: str

    # Internal Reasoning State
    # Append-only log
    research_data: Annotated[List[ResearchArtifact], operator.add]
    draft_analysis: Optional[str]
    critique_notes: List[str]
    iteration_count: int

    # Control Flow
    quality_score: float
    needs_correction: bool

    # Explainability
    human_readable_status: str  # "I am currently verifying the debt ratio..."


class SNCAnalysisState(TypedDict, total=False):
    """
    The State for the Shared National Credit (SNC) Analysis Graph.

    Attributes:
        obligor_id (str): The unique ID of the borrower.
        syndicate_data (Dict[str, Any]): Data about the bank syndicate.
        financials (Dict[str, Any]): Financial statements.
        structure_analysis (Optional[str]): Analysis of deal structure.
        regulatory_rating (Optional[str]): Rating (Pass, SM, SS, D, L).
        rationale (Optional[str]): Reasoning for the rating.
        critique_notes (List[str]): Reviewer notes.
        iteration_count (int): Current iteration.
        is_compliant (bool): Whether it meets regulatory standards.
        needs_revision (bool): Flag for revision.
        human_readable_status (str): Status message for UI.
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


class MarketSentimentState(TypedDict, total=False):
    """
    State for the Market Sentiment & News Monitoring Graph.

    Attributes:
        ticker (str): Stock ticker.
        target_sector (str): Sector to monitor.
        news_feed (List[Dict[str, Any]]): Collected news items.
        sentiment_score (float): Aggregate sentiment (-1.0 to 1.0).
        sentiment_trend (Literal): "bullish", "bearish", "neutral".
        key_drivers (List[str]): Main drivers of sentiment.
        related_entities (List[str]): Entities found in KG.
        alert_level (Literal): "LOW", "MEDIUM", "HIGH", "CRITICAL".
        final_report (Optional[str]): The final sentiment report.
        iteration_count (int): Iteration counter.
        human_readable_status (str): Status message.
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


class RedTeamState(TypedDict, total=False):
    """
    State for the Adversarial Red Team Loop.

    Attributes:
        target_entity (str): The entity being attacked.
        scenario_type (str): Type of attack (e.g. "Cyber").
        current_scenario_description (str): Description of the scenario.
        simulated_impact_score (float): Impact score (0-10).
        severity_threshold (float): Threshold for concern.
        critique_notes (List[str]): Reviewer notes.
        iteration_count (int): Iteration counter.
        is_sufficiently_severe (bool): Whether scenario is severe enough.
        human_readable_status (str): Status message.
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


class ESGAnalysisState(TypedDict, total=False):
    """
    State for the ESG (Environmental, Social, Governance) Analysis Graph.

    Attributes:
        company_name (str): Name of the company.
        sector (str): Industry sector.
        env_score (float): Environmental score.
        social_score (float): Social score.
        gov_score (float): Governance score.
        total_esg_score (float): Aggregated score.
        controversies (List[str]): List of known controversies.
        critique_notes (List[str]): Reviewer notes.
        iteration_count (int): Iteration counter.
        needs_revision (bool): Flag for revision.
        human_readable_status (str): Status message.
        final_report (Optional[str]): Final ESG report.
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


class ComplianceState(TypedDict, total=False):
    """
    State for the Regulatory Compliance Graph.

    Attributes:
        entity_id (str): ID of the entity.
        jurisdiction (str): Legal jurisdiction.
        applicable_regulations (List[str]): List of regulations.
        potential_violations (List[str]): Identified risks.
        risk_level (Literal): "LOW", "MEDIUM", "HIGH", "CRITICAL".
        critique_notes (List[str]): Reviewer notes.
        iteration_count (int): Iteration counter.
        needs_revision (bool): Flag for revision.
        human_readable_status (str): Status message.
        final_report (Optional[str]): Final report.
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


class QuantumRiskState(TypedDict, total=False):
    """
    State for the Quantum-Enhanced Risk Graph.
    Tracks the execution of QMC simulations and Hybrid QNN inference.

    Attributes:
        portfolio_id (str): Portfolio identifier.
        risk_factors (Dict[str, Any]): Input risk factors.
        simulation_type (Literal): Type of simulation.
        simulation_results (Dict[str, Any]): Results (VaR, etc.).
        quantum_execution_time (float): Execution time.
        classical_fallback_triggered (bool): If quantum failed.
        icaa_score (float): Inter-Class Attribution Alignment.
        human_readable_status (str): Status message.
        final_report (Optional[str]): Final report.
        iteration_count (int): Iteration counter.
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


class CrisisSimulationState(TypedDict, total=False):
    """
    State for the Macro-Economic Crisis Simulation Graph.

    Attributes:
        scenario_description (str): Description of the crisis.
        portfolio_data (Dict[str, Any]): Portfolio details.
        macro_variables (Dict[str, float]): Economic indicators.
        first_order_impacts (List[str]): Direct impacts.
        second_order_impacts (List[str]): Knock-on effects.
        estimated_loss (float): Estimated financial loss.
        critique_notes (List[str]): Reviewer notes.
        iteration_count (int): Iteration counter.
        is_realistic (bool): Reality check flag.
        needs_refinement (bool): Refinement flag.
        human_readable_status (str): Status message.
        final_report (Optional[str]): Final report.
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


class ReflectorState(TypedDict, total=False):
    """
    State for the Reflector (Meta-Cognition) Graph.

    Attributes:
        input_content (str): Content to critique.
        context (Dict[str, Any]): Additional context.
        critique_notes (List[str]): Generated critique.
        score (float): Quality score.
        is_valid (bool): Validity flag.
        refined_content (Optional[str]): Improved content.
        iteration_count (int): Iteration counter.
        human_readable_status (str): Status message.
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


class EntityEcosystem(TypedDict, total=False):
    """Data about the entity, management, and competitors."""
    legal_entity: Dict[str, str]
    management_assessment: Dict[str, Any]
    competitive_positioning: Dict[str, str]


class EquityAnalysis(TypedDict, total=False):
    """Data about fundamentals and valuation."""
    fundamentals: Dict[str, str]
    valuation_engine: Dict[str, Any]


class CreditAnalysis(TypedDict, total=False):
    """Data about creditworthiness, SNC ratings, and covenants."""
    snc_rating_model: Dict[str, Any]
    cds_market_implied_rating: str
    covenant_risk_analysis: Dict[str, Any]


class SimulationEngine(TypedDict, total=False):
    """Data from Monte Carlo and Quantum simulations."""
    monte_carlo_default_prob: float
    quantum_scenarios: List[Dict[str, Any]]
    trading_dynamics: Dict[str, str]


class StrategicSynthesis(TypedDict, total=False):
    """Final strategic analysis and conviction."""
    m_and_a_posture: str
    final_verdict: Dict[str, Any]


class OmniscientNodes(TypedDict, total=False):
    """The collection of analysis nodes in the Knowledge Graph."""
    entity_ecosystem: EntityEcosystem
    equity_analysis: EquityAnalysis
    credit_analysis: CreditAnalysis
    simulation_engine: SimulationEngine
    strategic_synthesis: StrategicSynthesis


class OmniscientMeta(TypedDict, total=False):
    """Metadata for the Knowledge Graph."""
    target: str
    generated_at: str
    model_version: str


class OmniscientKnowledgeGraph(TypedDict, total=False):
    """The v23.5 Hyper-Dimensional Knowledge Graph structure."""
    meta: OmniscientMeta
    nodes: OmniscientNodes


class OmniscientState(TypedDict, total=False):
    """
    State for the v23.5 'AI Partner' Omniscient Workflow.

    Attributes:
        ticker (str): The target ticker symbol.
        v23_knowledge_graph (OmniscientKnowledgeGraph): The output graph.
        human_readable_status (str): Status message.
    """
    ticker: str  # Input
    v23_knowledge_graph: OmniscientKnowledgeGraph  # Output
    human_readable_status: str

# --- Initializers ---


def init_risk_state(ticker: str, intent: str) -> RiskAssessmentState:
    """Initializes the Risk Assessment State."""
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
    """Initializes the SNC Analysis State."""
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
    """Initializes the Market Sentiment State."""
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
    """Initializes the ESG Analysis State."""
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
    """Initializes the Compliance State."""
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
    """Initializes the Quantum Risk State."""
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
    """Initializes the Crisis Simulation State."""
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


def init_reflector_state(content: str, context: Optional[Dict[str, Any]] = None) -> ReflectorState:
    """Initializes the Reflector State."""
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
    """Initializes the Omniscient State (v23.5)."""
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

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- Core Definitions ---


class Meta(BaseModel):
    target: str
    generated_at: str
    model_version: Literal["Adam-v23.5-Apex-Architect"]


class LegalEntity(BaseModel):
    name: str
    lei: str
    jurisdiction: str
    sector: str

# --- Phase 1: Entity & Management ---


class ManagementAssessment(BaseModel):
    capital_allocation_score: float = Field(description="Score 0-10 on capital allocation efficiency")
    alignment_analysis: str = Field(description="Analysis of insider buying/selling and compensation alignment")
    key_person_risk: Literal["High", "Med", "Low"]


class CompetitivePositioning(BaseModel):
    moat_status: Literal["Wide", "Narrow", "None"]
    technology_risk_vector: str


class EntityEcosystem(BaseModel):
    legal_entity: LegalEntity
    management_assessment: ManagementAssessment
    competitive_positioning: CompetitivePositioning

# --- Phase 2: Deep Fundamental & Valuation ---


class Fundamentals(BaseModel):
    revenue_cagr_3yr: str
    ebitda_margin_trend: Literal["Expanding", "Contracting", "Stable"]


class DCFModel(BaseModel):
    wacc_assumption: str
    terminal_growth: str
    intrinsic_value_estimate: float


class MultiplesAnalysis(BaseModel):
    current_ev_ebitda: float
    peer_median_ev_ebitda: float
    verdict: Literal["Undervalued", "Overvalued", "Fair"]


class PriceTargets(BaseModel):
    bear_case: float
    base_case: float
    bull_case: float


class ValuationEngine(BaseModel):
    dcf_model: DCFModel
    multiples_analysis: MultiplesAnalysis
    price_targets: PriceTargets


class EquityAnalysis(BaseModel):
    fundamentals: Fundamentals
    valuation_engine: ValuationEngine

# --- Phase 3: Credit, Covenants & SNC Ratings ---


class PrimaryFacilityAssessment(BaseModel):
    facility_type: str
    collateral_coverage: Literal["Strong", "Adequate", "Weak"]
    repayment_capacity: str


class SNCRatingModel(BaseModel):
    overall_borrower_rating: Literal["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]
    rationale: str
    primary_facility_assessment: PrimaryFacilityAssessment


class CovenantRiskAnalysis(BaseModel):
    primary_constraint: str = Field(description="The primary restrictive covenant or 'Choke Point'")
    current_level: float
    breach_threshold: float
    headroom_assessment: str = Field(description="Analysis of headroom (e.g., '15% Cushion')")


class CreditAnalysis(BaseModel):
    snc_rating_model: SNCRatingModel
    cds_market_implied_rating: str
    covenant_risk_analysis: CovenantRiskAnalysis

# --- Phase 4: Risk, Simulation & Quantum Modeling ---


class QuantumScenario(BaseModel):
    scenario_name: str
    probability: Literal["Low", "Med", "High"]
    impact_severity: Literal["Critical", "High", "Moderate"]
    estimated_impact_ev: str


class TradingDynamics(BaseModel):
    short_interest: str
    liquidity_risk: Literal["Low", "Med", "High"]


class SimulationEngine(BaseModel):
    monte_carlo_default_prob: str
    quantum_scenarios: List[QuantumScenario]
    trading_dynamics: TradingDynamics

# --- Phase 5: Synthesis & Conviction ---


class FinalVerdict(BaseModel):
    recommendation: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    conviction_level: int = Field(ge=1, le=10)
    time_horizon: str
    rationale_summary: str
    justification_trace: List[str]


class StrategicSynthesis(BaseModel):
    m_and_a_posture: Literal["Buyer", "Seller", "Neutral"]
    final_verdict: FinalVerdict

# --- Unified Graph Structure ---


class Nodes(BaseModel):
    entity_ecosystem: EntityEcosystem
    equity_analysis: EquityAnalysis
    credit_analysis: CreditAnalysis
    simulation_engine: SimulationEngine
    strategic_synthesis: StrategicSynthesis


class V23KnowledgeGraph(BaseModel):
    meta: Meta
    nodes: Nodes
    model_config = ConfigDict(populate_by_name=True)


class HyperDimensionalKnowledgeGraph(BaseModel):
    v23_knowledge_graph: V23KnowledgeGraph
    model_config = ConfigDict(populate_by_name=True)

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class Meta(BaseModel):
    target: str
    generated_at: str
    model_version: Literal["Adam-v23.5"]

class LegalEntity(BaseModel):
    name: str
    lei: str
    jurisdiction: str

class ManagementAssessment(BaseModel):
    capital_allocation_score: float
    alignment_analysis: str
    key_person_risk: Literal["High", "Med", "Low"]

class CompetitivePositioning(BaseModel):
    moat_status: Literal["Wide", "Narrow", "None"]
    technology_risk_vector: str

class EntityEcosystem(BaseModel):
    legal_entity: LegalEntity
    management_assessment: ManagementAssessment
    competitive_positioning: CompetitivePositioning

class Fundamentals(BaseModel):
    revenue_cagr_3yr: str
    ebitda_margin_trend: Literal["Expanding", "Contracting"]

class DCFModel(BaseModel):
    wacc: float
    terminal_growth: float
    intrinsic_value: float

class MultiplesAnalysis(BaseModel):
    current_ev_ebitda: float
    peer_median_ev_ebitda: float

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

class Facility(BaseModel):
    id: str
    amount: str
    regulatory_rating: str
    collateral_coverage: str
    covenant_headroom: str

class SNCRatingModel(BaseModel):
    overall_borrower_rating: Literal["Pass", "Special Mention", "Substandard", "Doubtful"]
    facilities: List[Facility]

class CovenantRiskAnalysis(BaseModel):
    primary_constraint: str
    current_level: float
    breach_threshold: float
    risk_assessment: str

class CreditAnalysis(BaseModel):
    snc_rating_model: SNCRatingModel
    cds_market_implied_rating: str
    covenant_risk_analysis: CovenantRiskAnalysis

class QuantumScenario(BaseModel):
    name: str
    probability: float
    estimated_impact_ev: str

class TradingDynamics(BaseModel):
    short_interest: str
    liquidity_risk: str

class SimulationEngine(BaseModel):
    monte_carlo_default_prob: float
    quantum_scenarios: List[QuantumScenario]
    trading_dynamics: TradingDynamics

class FinalVerdict(BaseModel):
    recommendation: Literal["Long", "Short", "Hold"]
    conviction_level: int = Field(ge=0, le=10)
    time_horizon: str
    rationale_summary: str
    justification_trace: List[str]

class StrategicSynthesis(BaseModel):
    m_and_a_posture: Literal["Buyer", "Seller", "Neutral"]
    final_verdict: FinalVerdict

class Nodes(BaseModel):
    entity_ecosystem: EntityEcosystem
    equity_analysis: EquityAnalysis
    credit_analysis: CreditAnalysis
    simulation_engine: SimulationEngine
    strategic_synthesis: StrategicSynthesis

class V23KnowledgeGraph(BaseModel):
    meta: Meta
    nodes: Nodes

class HyperDimensionalKnowledgeGraph(BaseModel):
    v23_knowledge_graph: V23KnowledgeGraph
    model_config = ConfigDict(populate_by_name=True)

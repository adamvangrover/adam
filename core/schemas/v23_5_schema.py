from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- Phase 1: Entity, Ecosystem & Management ---

class LegalEntity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    lei: str
    jurisdiction: str

class ManagementAssessment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    capital_allocation_score: float
    alignment_analysis: str
    key_person_risk: str = Field(description="High/Med/Low")

class CompetitivePositioning(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    moat_status: str = Field(description="Wide/Narrow/None")
    technology_risk_vector: str

class EntityEcosystem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    legal_entity: LegalEntity
    management_assessment: ManagementAssessment
    competitive_positioning: CompetitivePositioning

# --- Phase 2: Deep Fundamental & Valuation ---

class Fundamentals(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    revenue_cagr_3yr: str
    ebitda_margin_trend: str = Field(description="Expanding/Contracting")

class DCFModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    wacc: float
    terminal_growth: float
    intrinsic_value: float

class MultiplesAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    current_ev_ebitda: float
    peer_median_ev_ebitda: float

class PriceTargets(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    bear_case: float
    base_case: float
    bull_case: float

class ValuationEngine(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    dcf_model: DCFModel
    multiples_analysis: MultiplesAnalysis
    price_targets: PriceTargets

class EquityAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    fundamentals: Fundamentals
    valuation_engine: ValuationEngine

# --- Phase 3: Credit, Covenants & SNC Ratings ---

class Facility(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    amount: str
    regulatory_rating: str
    collateral_coverage: str
    covenant_headroom: str

class SNCRatingModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    overall_borrower_rating: str = Field(description="Pass/SpecialMention/Substandard")
    facilities: List[Facility]

class CovenantRiskAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    primary_constraint: str
    current_level: float
    breach_threshold: float
    risk_assessment: str

class CreditAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    snc_rating_model: SNCRatingModel
    cds_market_implied_rating: str
    covenant_risk_analysis: CovenantRiskAnalysis

# --- Phase 4: Risk, Simulation & Quantum Modeling ---

class QuantumScenario(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    probability: float
    estimated_impact_ev: str

class TradingDynamics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    short_interest: str
    liquidity_risk: str

class SimulationEngine(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    monte_carlo_default_prob: float
    quantum_scenarios: List[QuantumScenario]
    trading_dynamics: TradingDynamics

# --- Phase 5: Synthesis, Conviction & Strategy ---

class FinalVerdict(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    recommendation: str = Field(description="Long/Short/Hold")
    conviction_level: int = Field(ge=0, le=10)
    time_horizon: str
    rationale_summary: str
    justification_trace: List[str]

class StrategicSynthesis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    m_and_a_posture: str = Field(description="Buyer/Seller/Neutral")
    final_verdict: FinalVerdict

# --- Root Objects ---

class KnowledgeGraphNodes(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    entity_ecosystem: EntityEcosystem
    equity_analysis: EquityAnalysis
    credit_analysis: CreditAnalysis
    simulation_engine: SimulationEngine
    strategic_synthesis: StrategicSynthesis

class KnowledgeGraphMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    target: str
    generated_at: str
    model_version: str

class V23OmniscientKnowledgeGraph(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    meta: KnowledgeGraphMeta
    nodes: KnowledgeGraphNodes

class V23Output(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    v23_omniscient_knowledge_graph: V23OmniscientKnowledgeGraph

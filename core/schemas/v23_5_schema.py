from __future__ import annotations
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

# --- Core Definitions ---

class Meta(BaseModel):
    target: str
    generated_at: str
    model_version: Literal["Adam-v23.5"]

class LegalEntity(BaseModel):
    name: str
    lei: str
    jurisdiction: str
    sector: Optional[str] = None
    industry: Optional[str] = None

# --- Phase 1: Entity & Management ---

class ManagementAssessment(BaseModel):
    capital_allocation_score: float = Field(description="Score 0-10 on capital allocation efficiency")
    alignment_analysis: str = Field(description="Analysis of insider buying/selling and compensation alignment")
    key_person_risk: Literal["High", "Med", "Low"]
    ceo_tone_score: Optional[float] = Field(None, description="NLP sentiment score of CEO earnings calls")

class CompetitivePositioning(BaseModel):
    moat_status: Literal["Wide", "Narrow", "None"]
    technology_risk_vector: str
    market_share_trend: Optional[str] = None

class EntityEcosystem(BaseModel):
    legal_entity: LegalEntity
    management_assessment: ManagementAssessment
    competitive_positioning: CompetitivePositioning

# --- Phase 2: Deep Fundamental & Valuation ---

class Fundamentals(BaseModel):
    revenue_cagr_3yr: str
    ebitda_margin_trend: Literal["Expanding", "Contracting", "Stable"]
    net_debt_ebitda: Optional[float] = None

class DCFModel(BaseModel):
    wacc: float
    terminal_growth: float
    intrinsic_value: float
    implied_upside: Optional[float] = None

class MultiplesAnalysis(BaseModel):
    current_ev_ebitda: float
    peer_median_ev_ebitda: float
    premium_discount: Optional[str] = None

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

class Facility(BaseModel):
    id: str
    amount: str
    regulatory_rating: Literal["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]
    collateral_coverage: str
    covenant_headroom: str
    pricing: Optional[str] = None
    maturity: Optional[str] = None

class SNCRatingModel(BaseModel):
    overall_borrower_rating: Literal["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]
    facilities: List[Facility]
    rationale: Optional[str] = None

class CovenantRiskAnalysis(BaseModel):
    primary_constraint: str
    current_level: float
    breach_threshold: float
    risk_assessment: str

class CreditAnalysis(BaseModel):
    snc_rating_model: SNCRatingModel
    cds_market_implied_rating: str
    covenant_risk_analysis: CovenantRiskAnalysis

# --- Phase 4: Risk, Simulation & Quantum Modeling ---

class QuantumScenario(BaseModel):
    name: str
    probability: float
    estimated_impact_ev: str
    description: Optional[str] = None

class TradingDynamics(BaseModel):
    short_interest: str
    liquidity_risk: str
    put_call_ratio: Optional[float] = None

class SimulationEngine(BaseModel):
    monte_carlo_default_prob: float
    quantum_scenarios: List[QuantumScenario]
    trading_dynamics: TradingDynamics

# --- Phase 5: Synthesis & Conviction ---

class FinalVerdict(BaseModel):
    recommendation: Literal["Long", "Short", "Hold"]
    conviction_level: int = Field(ge=1, le=10)
    time_horizon: str
    rationale_summary: str
    justification_trace: List[str]

class StrategicSynthesis(BaseModel):
    m_and_a_posture: Literal["Buyer", "Seller", "Neutral"]
    final_verdict: FinalVerdict
    activist_risk: Optional[Literal["High", "Medium", "Low"]] = None

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

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Organization(BaseModel):
    id: str = Field(..., description="Unique identifier for the organization")
    name: str = Field(..., description="Legal name of the organization")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction of incorporation")
    industry_code: Optional[str] = Field(None, description="Industry classification code (e.g., NAICS)")

class Borrower(Organization):
    credit_score: Optional[float] = Field(None, description="Internal or external credit score")

class Sponsor(Organization):
    fund_size: Optional[float] = Field(None, description="Total size of the sponsoring fund")

class Parent(Organization):
    consolidated_revenue: Optional[float] = Field(None, description="Consolidated revenue of the parent entity")

class Subsidiary(Organization):
    parent_id: str = Field(..., description="ID of the parent organization")

class FinancialInstrument(BaseModel):
    id: str = Field(..., description="Unique identifier for the financial instrument")
    type: str = Field(..., description="Type of the instrument")
    currency: str = Field(default="USD", description="Currency of the instrument")
    notional_amount: float = Field(..., description="Principal or notional amount")

class Facility(FinancialInstrument):
    maturity_date: datetime = Field(..., description="Maturity date of the facility")
    interest_rate: float = Field(..., description="Current interest rate")

class Bond(FinancialInstrument):
    coupon_rate: float = Field(..., description="Coupon rate of the bond")
    isin: Optional[str] = Field(None, description="International Securities Identification Number")

class Revolver(FinancialInstrument):
    drawn_amount: float = Field(default=0.0, description="Amount currently drawn")
    available_amount: float = Field(..., description="Amount available to draw")

class Swap(FinancialInstrument):
    fixed_rate: float = Field(..., description="Fixed rate leg")
    floating_spread: float = Field(..., description="Floating rate spread")

class LegalArtifact(BaseModel):
    id: str = Field(..., description="Unique identifier for the legal artifact")
    content: str = Field(..., description="Text content or reference to the document")
    effective_date: datetime = Field(..., description="Date the artifact becomes effective")

class Agreement(LegalArtifact):
    parties: List[str] = Field(default_factory=list, description="List of party IDs involved")

class Covenant(LegalArtifact):
    metric: str = Field(..., description="Financial metric being monitored")
    threshold: float = Field(..., description="Threshold value for the covenant")
    is_breached: bool = Field(default=False, description="Whether the covenant is currently breached")

class Amendment(LegalArtifact):
    original_agreement_id: str = Field(..., description="ID of the agreement being amended")

class RiskConcept(BaseModel):
    id: str = Field(..., description="Unique identifier for the risk concept")
    value: float = Field(..., description="Calculated value of the risk metric")
    confidence_interval: Optional[float] = Field(None, description="Confidence interval of the calculation")

class Rating(RiskConcept):
    scale: str = Field(..., description="Rating scale used (e.g., S&P, Moody's, Internal)")

class Default(RiskConcept):
    probability: float = Field(..., description="Probability of default (PD)")

class Recovery(RiskConcept):
    rate: float = Field(..., description="Loss given default (LGD) or recovery rate")

class Watchlist(RiskConcept):
    status: str = Field(..., description="Current watchlist status (e.g., Normal, Monitored, Critical)")

class Portfolio(BaseModel):
    id: str = Field(..., description="Unique identifier for the portfolio")
    name: str = Field(..., description="Name of the portfolio")
    instruments: List[FinancialInstrument] = Field(default_factory=list, description="List of instruments in the portfolio")

class Exposure(BaseModel):
    id: str = Field(..., description="Unique identifier for the exposure record")
    amount: float = Field(..., description="Calculated total exposure amount")
    counterparty_id: str = Field(..., description="ID of the counterparty")

class Event(BaseModel):
    id: str = Field(..., description="Unique identifier for the event")
    type: str = Field(..., description="Type of event (e.g., Payment, Default, RatingChange)")
    timestamp: datetime = Field(..., description="When the event occurred")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data payload")

class Decision(BaseModel):
    id: str = Field(..., description="Unique identifier for the decision")
    outcome: str = Field(..., description="Final outcome (e.g., Approved, Rejected)")
    rationale: str = Field(..., description="Textual explanation of the decision")
    evidence_ids: List[str] = Field(default_factory=list, description="List of evidence IDs used to make the decision")

class Policy(BaseModel):
    id: str = Field(..., description="Unique identifier for the policy")
    version: str = Field(..., description="Version of the policy")
    rules: str = Field(..., description="Policy DSL or ruleset content")

class Evidence(BaseModel):
    id: str = Field(..., description="Unique identifier for the evidence")
    source: str = Field(..., description="Source system or process that generated the evidence")
    data: Dict[str, Any] = Field(default_factory=dict, description="The raw evidence data")

class Scenario(BaseModel):
    id: str = Field(..., description="Unique identifier for the scenario")
    description: str = Field(..., description="Description of the macro shock or scenario parameters")
    shock_factors: Dict[str, float] = Field(default_factory=dict, description="Specific variable shock factors")

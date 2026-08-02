from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

# Base class for the AFOS Canonical Risk Ontology
class CanonicalEntity(BaseModel):
    id: str = Field(..., description="Unique identifier for the entity.")

# ------------------------------------------------------------------
# Organization Hierarchy
# ------------------------------------------------------------------
class Organization(CanonicalEntity):
    name: str = Field(..., description="Name of the organization.")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction.")

class Sponsor(Organization):
    sponsor_type: str = Field(default="Private Equity", description="Type of sponsor.")

class Parent(Organization):
    pass

class Subsidiary(Organization):
    parent_id: str = Field(..., description="ID of the parent organization.")

class Borrower(Organization):
    sponsor_id: Optional[str] = Field(None, description="ID of the financial sponsor, if any.")
    parent_id: Optional[str] = Field(None, description="ID of the parent company, if any.")

# ------------------------------------------------------------------
# Financial Instrument Hierarchy
# ------------------------------------------------------------------
class FinancialInstrument(CanonicalEntity):
    borrower_id: str = Field(..., description="The borrower this instrument is issued to.")
    currency: str = Field(default="USD", description="Base currency.")

class Facility(FinancialInstrument):
    facility_size: float = Field(..., description="Total committed amount of the facility.")

class Bond(FinancialInstrument):
    coupon: float = Field(..., description="Coupon rate of the bond.")
    maturity_date: datetime = Field(..., description="Maturity date.")

class Revolver(Facility):
    drawn_amount: float = Field(default=0.0, description="Amount currently drawn.")

class Swap(FinancialInstrument):
    notional: float = Field(..., description="Notional value of the swap.")

# ------------------------------------------------------------------
# Legal Artifact Hierarchy
# ------------------------------------------------------------------
class LegalArtifact(CanonicalEntity):
    effective_date: datetime = Field(..., description="Date the artifact becomes effective.")

class Agreement(LegalArtifact):
    parties: List[str] = Field(..., description="List of Organization IDs involved.")

class Covenant(LegalArtifact):
    instrument_id: str = Field(..., description="ID of the associated Financial Instrument.")
    metric: str = Field(..., description="The financial metric to track (e.g., Leverage).")
    threshold: float = Field(..., description="The limit for the covenant.")
    is_max: bool = Field(True, description="True if the metric must stay below the threshold, False if above.")

class Amendment(LegalArtifact):
    original_agreement_id: str = Field(..., description="ID of the agreement being amended.")

# ------------------------------------------------------------------
# Risk Concept Hierarchy
# ------------------------------------------------------------------
class RiskConcept(CanonicalEntity):
    description: str = Field(..., description="Description of the risk concept.")

class Rating(RiskConcept):
    value: str = Field(..., description="The rating value (e.g., AAA, BB-).")
    agency: str = Field(default="Internal", description="Agency providing the rating.")

class Default(RiskConcept):
    default_date: datetime = Field(..., description="Date of default.")
    reason: str = Field(..., description="Reason for the default.")

class Recovery(RiskConcept):
    recovery_rate: float = Field(..., description="Estimated or actual recovery rate.")

class Watchlist(RiskConcept):
    status: str = Field(..., description="Current status on the watchlist.")

# ------------------------------------------------------------------
# Operational Primitives
# ------------------------------------------------------------------
class Portfolio(CanonicalEntity):
    instruments: List[str] = Field(default_factory=list, description="List of Financial Instrument IDs.")

class Exposure(CanonicalEntity):
    instrument_id: str = Field(..., description="ID of the Financial Instrument.")
    amount: float = Field(..., description="Amount exposed.")

class Event(CanonicalEntity):
    timestamp: datetime = Field(..., description="When the event occurred.")
    event_type: str = Field(..., description="Classification of the event.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data.")

class Policy(CanonicalEntity):
    version: str = Field(..., description="Version of the policy.")
    ruleset: str = Field(..., description="Identifier or reference to the executable rules.")

class Evidence(CanonicalEntity):
    source_uri: str = Field(..., description="URI or path to the source material.")
    hash: str = Field(..., description="Cryptographic hash of the evidence for provenance.")

class Scenario(CanonicalEntity):
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Shock parameters and assumptions.")

class Decision(CanonicalEntity):
    timestamp: datetime = Field(..., description="When the decision was made.")
    policy_id: str = Field(..., description="ID of the Policy applied.")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of Evidence used.")
    outcome: Any = Field(..., description="The result of the decision.")

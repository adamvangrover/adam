import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator

# ==================================================================
# TYPE DEFINITIONS
# ==================================================================
T = TypeVar('T', bound=BaseModel)

# ==================================================================
# 1. THE ASSURANCE & TRUST LAYER (DAF)
# ==================================================================
class AssuredDependency(BaseModel):
    """Zero-Trust Dependency Manifest."""
    dependency_id: str = Field(..., description="Unique ID for the loaded module/model.")
    version: str = Field(..., description="Strict semantic versioning.")
    source_uri: str = Field(..., description="Where the dependency is fetched from.")
    expected_sha384: str = Field(..., description="Cryptographic hash for integrity.")
    execution_environment: str = Field(default="isolated_container")

class AdversarialContext(BaseModel):
    """Guardrails against AI hallucinations and adversarial inputs."""
    model_id: str = Field(..., description="ID of the inference model used.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model's internal confidence.")
    perturbation_bound: float = Field(..., description="Calculated epsilon (robustness radius).")
    entropy_score: float = Field(..., description="Information entropy of the input.")
    flagged_anomalies: List[str] = Field(default_factory=list, description="Any detected edge-case triggers.")
    
    @field_validator('confidence_score')
    @classmethod
    def minimum_confidence(cls, v):
        if v < 0.85:
            raise ValueError("Decision rejected: Confidence score below safety threshold (0.85).")
        return v

class ImmutableDecisionBlock(BaseModel, Generic[T]):
    """Cryptographically sealed state machine block."""
    block_id: str = Field(..., description="UUID of this block.")
    previous_block_hash: str = Field(..., description="Hash of the preceding block (Chain of truth).")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies: List[AssuredDependency] = Field(..., description="State of the Base Loader.")
    adversarial_defense: AdversarialContext = Field(..., description="Proof of resilience.")
    payload: T = Field(..., description="The canonical ontology payload.")
    block_hash: Optional[str] = Field(default=None, description="The final sealed hash of this block.")

    def seal(self) -> str:
        """Calculates a deterministic SHA-384 hash of the block contents."""
        # Use Pydantic's model_dump_json to handle datetimes correctly, then sort keys deterministically
        state_json = self.model_dump_json(exclude={'block_hash'})
        state_dict = json.loads(state_json)
        deterministic_str = json.dumps(state_dict, sort_keys=True)
        
        calculated_hash = hashlib.sha384(deterministic_str.encode('utf-8')).hexdigest()
        self.block_hash = calculated_hash
        return calculated_hash

# ==================================================================
# 2. BASE ENTITY LAYER
# ==================================================================
class CanonicalEntity(BaseModel):
    id: str = Field(..., description="Unique identifier for the entity.")

# ==================================================================
# 3. ORGANIZATION HIERARCHY
# ==================================================================
class Organization(CanonicalEntity):
    name: str = Field(..., description="Legal name of the organization")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction of incorporation")
    industry_code: Optional[str] = Field(None, description="Industry classification code (e.g., NAICS)")

class Sponsor(Organization):
    sponsor_type: str = Field(default="Private Equity", description="Type of sponsor.")
    fund_size: Optional[float] = Field(None, description="Total size of the sponsoring fund")

class Parent(Organization):
    consolidated_revenue: Optional[float] = Field(None, description="Consolidated revenue of the parent entity")

class Subsidiary(Organization):
    parent_id: str = Field(..., description="ID of the parent organization")

class Borrower(Organization):
    credit_score: Optional[float] = Field(None, description="Internal or external credit score")
    sponsor_id: Optional[str] = Field(None, description="ID of the financial sponsor, if any.")
    parent_id: Optional[str] = Field(None, description="ID of the parent company, if any.")

# ==================================================================
# 4. FINANCIAL INSTRUMENT HIERARCHY
# ==================================================================
class FinancialInstrument(CanonicalEntity):
    type: str = Field(..., description="Type of the instrument")
    borrower_id: str = Field(..., description="The borrower this instrument is issued to.")
    currency: str = Field(default="USD", description="Currency of the instrument")
    notional_amount: float = Field(..., description="Principal or notional amount")

class Facility(FinancialInstrument):
    maturity_date: datetime = Field(..., description="Maturity date of the facility")
    interest_rate: float = Field(..., description="Current interest rate")
    facility_size: float = Field(..., description="Total committed amount of the facility.")

class Bond(FinancialInstrument):
    coupon_rate: float = Field(..., description="Coupon rate of the bond")
    maturity_date: datetime = Field(..., description="Maturity date.")
    isin: Optional[str] = Field(None, description="International Securities Identification Number")

class Revolver(Facility):
    drawn_amount: float = Field(default=0.0, description="Amount currently drawn")
    available_amount: float = Field(..., description="Amount available to draw")

class Swap(FinancialInstrument):
    fixed_rate: float = Field(..., description="Fixed rate leg")
    floating_spread: float = Field(..., description="Floating rate spread")
    notional: float = Field(..., description="Notional value of the swap.")

# ==================================================================
# 5. LEGAL ARTIFACT HIERARCHY
# ==================================================================
class LegalArtifact(CanonicalEntity):
    content: Optional[str] = Field(None, description="Text content or reference to the document")
    effective_date: datetime = Field(..., description="Date the artifact becomes effective")

class Agreement(LegalArtifact):
    parties: List[str] = Field(default_factory=list, description="List of Organization IDs involved")

class Covenant(LegalArtifact):
    instrument_id: str = Field(..., description="ID of the associated Financial Instrument.")
    metric: str = Field(..., description="Financial metric being monitored")
    threshold: float = Field(..., description="Threshold value for the covenant")
    is_max: bool = Field(True, description="True if the metric must stay below the threshold, False if above.")
    is_breached: bool = Field(default=False, description="Whether the covenant is currently breached")

class Amendment(LegalArtifact):
    original_agreement_id: str = Field(..., description="ID of the agreement being amended")

# ==================================================================
# 6. RISK CONCEPT HIERARCHY
# ==================================================================
class RiskConcept(CanonicalEntity):
    description: Optional[str] = Field(None, description="Description of the risk concept.")
    value: float = Field(..., description="Calculated value of the risk metric")
    confidence_interval: Optional[float] = Field(None, description="Confidence interval of the calculation")

class Rating(RiskConcept):
    scale: str = Field(..., description="Rating scale used (e.g., S&P, Moody's, Internal)")
    agency: str = Field(default="Internal", description="Agency providing the rating.")
    rating_value: str = Field(..., description="The rating value (e.g., AAA, BB-).")

class Default(RiskConcept):
    probability: float = Field(..., description="Probability of default (PD)")
    default_date: Optional[datetime] = Field(None, description="Date of default.")
    reason: Optional[str] = Field(None, description="Reason for the default.")

class Recovery(RiskConcept):
    rate: float = Field(..., description="Loss given default (LGD) or recovery rate")

class Watchlist(RiskConcept):
    status: str = Field(..., description="Current watchlist status (e.g., Normal, Monitored, Critical)")

# ==================================================================
# 7. OPERATIONAL PRIMITIVES
# ==================================================================
class Portfolio(CanonicalEntity):
    name: str = Field(..., description="Name of the portfolio")
    instruments: List[str] = Field(default_factory=list, description="List of Financial Instrument IDs.")

class Exposure(CanonicalEntity):
    instrument_id: str = Field(..., description="ID of the Financial Instrument.")
    counterparty_id: str = Field(..., description="ID of the counterparty")
    amount: float = Field(..., description="Calculated total exposure amount")

class Event(CanonicalEntity):
    type: str = Field(..., description="Classification of the event.")
    timestamp: datetime = Field(..., description="When the event occurred.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data payload.")

class Decision(CanonicalEntity):
    timestamp: datetime = Field(..., description="When the decision was made.")
    policy_id: str = Field(..., description="ID of the Policy applied.")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of Evidence used.")
    outcome: str = Field(..., description="Final outcome (e.g., Approved, Rejected).")
    rationale: str = Field(..., description="Textual explanation of the decision.")

class Policy(CanonicalEntity):
    version: str = Field(..., description="Version of the policy.")
    ruleset: str = Field(..., description="Identifier or reference to the executable rules.")
    rules: Optional[str] = Field(None, description="Policy DSL or ruleset content.")

class Evidence(CanonicalEntity):
    source_uri: str = Field(..., description="URI or path to the source material.")
    source: Optional[str] = Field(None, description="Source system or process that generated the evidence.")
    hash: str = Field(..., description="Cryptographic hash of the evidence for provenance.")
    data: Dict[str, Any] = Field(default_factory=dict, description="The raw evidence data.")

class Scenario(CanonicalEntity):
    description: str = Field(..., description="Description of the macro shock or scenario parameters.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Shock parameters and assumptions.")

# ==================================================================
# EXECUTION / FUNCTIONAL TEST
# ==================================================================
if __name__ == "__main__":
    # 1. Create a core operational decision
    credit_decision = Decision(
        id="dec_001",
        timestamp=datetime.now(timezone.utc),
        policy_id="pol_credit_v2",
        evidence_ids=["evd_994", "evd_995"],
        outcome="Approved",
        rationale="Borrower leverage ratio within covenant threshold of 4.5x."
    )

    # 2. Record the dependencies that executed this decision
    model_dependency = AssuredDependency(
        dependency_id="credit_risk_llm",
        version="1.4.2",
        source_uri="s3://afos-models/credit_risk_v1.bin",
        expected_sha384="a4b3c2d1..."
    )

    # 3. Capture the AI adversarial context
    ai_context = AdversarialContext(
        model_id="credit_risk_llm",
        confidence_score=0.96,
        perturbation_bound=0.015,
        entropy_score=2.34
    )

    # 4. Wrap it in the Immutable Decision Block
    block = ImmutableDecisionBlock[Decision](
        block_id="blk_00000001",
        previous_block_hash="0000000000000000000000000000000000000000000000000000000000000000",
        dependencies=[model_dependency],
        adversarial_defense=ai_context,
        payload=credit_decision
    )

    # 5. Cryptographically seal the block
    final_hash = block.seal()
    
    print("\n✅ Immutable Block Successfully Created & Sealed")
    print("-" * 50)
    print(f"Block Hash: {final_hash}")
    print("-" * 50)
    print(block.model_dump_json(indent=2))
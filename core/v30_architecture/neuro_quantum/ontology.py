from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class SemanticLabel(Enum):
    """
    Standardized semantic labels for system states,
    aligning with the "Semantic Labelset" requirement.
    """
    RISK = "RISK"
    GROWTH = "GROWTH"
    STAGNATION = "STAGNATION"
    CHAOS = "CHAOS"
    STABILITY = "STABILITY"
    # Expanded Semantic Labels
    RECOVERY = "RECOVERY"
    TRANSITION = "TRANSITION"
    CRISIS = "CRISIS"

class FIBOConcept(Enum):
    """
    Mappings to Financial Industry Business Ontology (FIBO) URIs.
    Aligned with 'Market Mayhem' and Crisis Response prompts.
    """
    # Core Instruments
    LOAN = "fibo-loan-ln-ln:Loan"
    SYNDICATED_LOAN = "fibo-loan-ln-ln:SyndicatedLoan"
    DERIVATIVE = "fibo-der-drc-ff:DerivativeInstrument"
    SWAP = "fibo-der-drc-swp:Swap"
    BOND = "fibo-sec-dbt-bnd:Bond"
    EQUITY = "fibo-sec-eq-eq:ListedShare"
    CORPORATE_DEBT = "fibo-sec-dbt-dbti:CorporateDebtInstrument"
    
    # Expanded Asset Classes (Aligned with new Scenarios)
    COMMODITY = "fibo-der-drc-comm:Commodity"
    REAL_ESTATE = "fibo-fbc-fct-rle:RealEstateAsset"
    CRYPTO_ASSET = "fibo-curr-curr:CryptoAsset" 
    PRECIOUS_METAL = "fibo-der-drc-comm:PreciousMetal"
    INFRASTRUCTURE = "fibo-fbc-fct-fse:InfrastructureAsset"

class MarketRegime(Enum):
    """
    Market Regimes defined in Market Mayhem configurations.
    """
    # Core Regimes
    STAGFLATIONARY_DIVERGENCE = "Stagflationary_Divergence"
    CREDIT_EVENT = "Credit_Event_Shadow_Bank"
    GEOPOLITICAL_ESCALATION = "Geopolitical_Escalation"
    GOLDILOCKS = "Goldilocks_Soft_Landing"
    DISINFLATION = "Disinflationary_Boom"
    
    # Merged Regimes
    LIQUIDITY_SHOCK = "Liquidity_Shock_Event"
    BIFURCATED_NORMALIZATION = "Bifurcated_Normalization"
    
    # Expanded Regimes (Black Swans & Future States)
    HYPER_AGENTIC_BOOM = "Hyper_Agentic_Boom"
    AI_BUBBLE_BURST = "AI_Bubble_Burst"
    QUANTUM_EVENT = "Quantum_Decryption_Event"
    SUPPLY_CHAIN_COLLAPSE = "Global_Supply_Chain_Collapse"
    CYBER_BLACKOUT = "Global_Cloud_Outage"
    SOVEREIGN_DEFAULT = "Sovereign_Default_Contagion"
    ENERGY_SHOCK = "Energy_Crisis_Strait_Closure"
    CRE_CRISIS = "Commercial_Real_Estate_Collapse"
    
    DEFAULT = "Neutral"

@dataclass
class EnvironmentDefinition:
    """
    Defines the parameters for a specific simulation environment (context).
    Acts as the 'Environment Definitions' component.
    """
    name: str
    tau_mean: float = 1.0
    bias_shift: float = 0.0
    noise_level: float = 0.1
    synapse_density: float = 0.5
    description: str = ""

@dataclass
class OntologyTuple:
    """
    Represents a formal tuple linking a concept (Prompt/Context)
    to a Semantic Label and its expected Neural State signature.
    """
    concept: str
    label: SemanticLabel
    # Expected values for key neurons (mock signature)
    expected_signature: Dict[str, float] = field(default_factory=dict)
    # Optional mapping to a FIBO concept
    fibo_mapping: Optional[FIBOConcept] = None
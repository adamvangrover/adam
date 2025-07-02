# backend/src/main/python/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class DriverType(Enum):
    MACROECONOMIC = "Macroeconomic"
    FUNDAMENTAL = "Fundamental"
    TECHNICAL = "Technical"
    GEOPOLITICAL = "Geopolitical"
    COMPANY_SPECIFIC = "CompanySpecific"
    INDUSTRY_SPECIFIC = "IndustrySpecific"

class ImpactPotential(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class TimeHorizon(Enum):
    SHORT_TERM = "Short-term"
    MEDIUM_TERM = "Medium-term"
    LONG_TERM = "Long-term"

class Trend(Enum):
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    STABLE = "Stable"

@dataclass
class Industry:
    id: str
    name: str
    description: Optional[str] = None
    macro_driver_ids: List[str] = field(default_factory=list)
    industry_specific_driver_ids: List[str] = field(default_factory=list)

@dataclass
class TradingLevelData:
    price: Optional[float] = None
    volume: Optional[int] = None
    volatility: Optional[float] = None
    timestamp: Optional[str] = None # ISO format date string

@dataclass
class Company:
    id: str  # e.g., ticker symbol
    name: str
    industry_id: str
    ownership_structure: Optional[str] = None
    corporate_structure: Optional[str] = None
    financials: Dict[str, Any] = field(default_factory=dict)
    company_specific_driver_ids: List[str] = field(default_factory=list)
    trading_levels: Optional[TradingLevelData] = None

@dataclass
class Driver:
    id: str
    name: str
    description: str
    type: DriverType
    impact_potential: Optional[ImpactPotential] = None
    time_horizon: Optional[TimeHorizon] = None
    metrics: Dict[str, str] = field(default_factory=dict)
    related_macro_factor_ids: List[str] = field(default_factory=list)

@dataclass
class MacroEnvironmentFactor:
    id: str
    name: str
    current_value: Optional[Any] = None # Could be str or number
    trend: Optional[Trend] = None
    impact_narrative: Optional[str] = None
    source: Optional[str] = None

@dataclass
class NarrativeExplanation:
    id: str
    generated_for_entity_id: str
    entity_type: str # 'Company' or 'Industry'
    driver_ids: List[str] = field(default_factory=list)
    explanation_text: str = ""
    linked_trading_level_context: Optional[str] = None
    timestamp: str = "" # ISO format date string

# For knowledge graph representation (simplified)
@dataclass
class Node:
    id: str
    label: str # e.g., 'Company', 'Industry', 'Driver'
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source_id: str
    target_id: str
    relationship_type: str # e.g., 'BELONGS_TO', 'AFFECTED_BY'
    properties: Dict[str, Any] = field(default_factory=dict)

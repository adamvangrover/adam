from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import random

class Resource(BaseModel):
    name: str
    amount: float = 0.0 # Normalized 0.0 to 1.0 (local scarcity/abundance)
    strategic_importance: float = 0.5  # 0.0 to 1.0
    volatility: float = 0.1
    market_price: float = 1.0

class PopulationSegment(BaseModel):
    label: str  # e.g., "Urban Youth", "Rural Agrarian"
    percentage: float
    sentiment: float = 0.5  # 0.0 (Revolutionary) to 1.0 (Loyalist)
    tech_adoption: float = 0.0
    economic_strain: float = 0.0 # 0.0 (Secure) to 1.0 (Desperate)

class Military(BaseModel):
    readiness: float = 0.5
    personnel: int = 100000
    kinetic_capability: float = 0.5 # Tanks, Jets, etc.
    cyber_capability: float = 0.5 # CNE, CNO capabilities
    nuclear_capable: bool = False
    doctrine: str = "Defensive" # Defensive, Expansionist, Asymmetric

class EconomicIndicators(BaseModel):
    gdp_growth: float = 0.02
    inflation_rate: float = 0.02
    unemployment_rate: float = 0.05
    trade_balance: float = 0.0 # Positive = Surplus
    debt_to_gdp: float = 0.5
    currency_strength: float = 1.0

class Sovereign(BaseModel):
    id: str
    name: str
    region: str
    ideology: str  # e.g., "Digital Autocracy", "Market Democracy"
    stability_index: float = 1.0

    # Components
    economy: EconomicIndicators = Field(default_factory=EconomicIndicators)
    military: Military = Field(default_factory=Military)

    resources: List[Resource] = []
    demographics: List[PopulationSegment] = []

    # Strategy
    strategic_goals: List[str] = []

    # Nexus of Exposure
    allies: List[str] = []
    adversaries: List[str] = []
    sanctions_active: List[str] = [] # List of entity IDs sanctioning this sovereign

    def calculate_stability(self) -> float:
        """
        Calculates internal stability based on weighted factors:
        - Population Sentiment (40%)
        - Economic Strain (Inflation + Unemployment) (30%)
        - Resource Stress (Strategic resource shortages) (20%)
        - Military Readiness (Confidence in security) (10%)
        """
        # 1. Population Sentiment
        avg_sentiment = sum(s.sentiment * s.percentage for s in self.demographics) if self.demographics else 0.5

        # 2. Economic Strain
        econ_strain = min(1.0, (self.economy.inflation_rate * 2) + self.economy.unemployment_rate)

        # 3. Resource Stress
        # High importance resources with low amount contribute to stress
        strategic_resources = [r for r in self.resources if r.strategic_importance > 0.7]
        if strategic_resources:
            resource_stress = sum((1.0 - r.amount) for r in strategic_resources) / len(strategic_resources)
        else:
            resource_stress = 0.0

        # 4. Military Confidence
        mil_confidence = self.military.readiness

        # Composite Score
        # Sentiment contributes positively, others negatively
        stability = (
            (avg_sentiment * 0.4) +
            ((1.0 - econ_strain) * 0.3) +
            ((1.0 - resource_stress) * 0.2) +
            (mil_confidence * 0.1)
        )

        self.stability_index = max(0.0, min(1.0, stability))
        return self.stability_index

    def generate_policy(self) -> str:
        # Placeholder for narrative generation
        return f"Policy focusing on {self.ideology} to boost {self.resources[0].name if self.resources else 'economy'}."

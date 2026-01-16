from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import random

class Resource(BaseModel):
    name: str
    amount: float = 0.0
    strategic_importance: float = 0.5  # 0.0 to 1.0

class PopulationSegment(BaseModel):
    label: str  # e.g., "Urban Youth", "Rural Agrarian"
    percentage: float
    sentiment: float = 0.5  # 0.0 (Revolutionary) to 1.0 (Loyalist)
    tech_adoption: float = 0.0

class Sovereign(BaseModel):
    id: str
    name: str
    region: str
    ideology: str  # e.g., "Digital Autocracy", "Market Democracy"
    stability_index: float = 1.0
    gdp_growth: float = 0.02

    resources: List[Resource] = []
    demographics: List[PopulationSegment] = []

    # Nexus of Exposure
    allies: List[str] = []
    adversaries: List[str] = []

    def calculate_stability(self) -> float:
        # Simple simulation logic
        avg_sentiment = sum(s.sentiment * s.percentage for s in self.demographics)
        resource_stress = sum(1.0 - r.amount for r in self.resources if r.strategic_importance > 0.8) / max(1, len(self.resources))

        self.stability_index = (avg_sentiment * 0.7) - (resource_stress * 0.3)
        return self.stability_index

    def generate_policy(self) -> str:
        # Placeholder for narrative generation
        return f"Policy focusing on {self.ideology} to boost {self.resources[0].name if self.resources else 'economy'}."

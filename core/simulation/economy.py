from typing import List, Dict
from pydantic import BaseModel

class Industry(BaseModel):
    name: str
    dependence_map: Dict[str, float]  # Resource -> Importance (0-1)
    output: float = 1.0

class EconomyEngine:
    INDUSTRIES = [
        "Semiconductors", "Energy", "Agriculture", "Defense", "Biotech", "Fintech"
    ]

    @staticmethod
    def generate_industries(ideology: str) -> List[Industry]:
        industries = []
        if "Digital" in ideology or "Tech" in ideology:
            industries.append(Industry(name="Semiconductors", dependence_map={"Rare Earths": 0.9, "Water": 0.4}))
            industries.append(Industry(name="Fintech", dependence_map={"Compute": 0.8}))
        elif "Agrarian" in ideology:
            industries.append(Industry(name="Agriculture", dependence_map={"Water": 0.9, "Fertilizer": 0.7}))
        else:
            industries.append(Industry(name="Energy", dependence_map={"Oil": 0.9}))
            industries.append(Industry(name="Manufacturing", dependence_map={"Steel": 0.8, "Energy": 0.7}))

        return industries

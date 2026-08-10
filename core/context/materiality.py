from enum import Enum
from pydantic import BaseModel

class MaterialityLevel(str, Enum):
    IMMATERIAL = "immaterial"
    MINOR = "minor"
    MATERIAL = "material"
    CRITICAL = "critical"

class MaterialityThreshold(BaseModel):
    metric_name: str
    threshold_value: float
    level: MaterialityLevel

class MaterialityEvaluator:
    def __init__(self, thresholds: list[MaterialityThreshold]):
        self.thresholds = thresholds

    def evaluate(self, metric: str, value: float) -> MaterialityLevel:
        for t in self.thresholds:
            if t.metric_name == metric and value >= t.threshold_value:
                return t.level
        return MaterialityLevel.IMMATERIAL

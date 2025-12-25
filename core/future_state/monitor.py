from typing import List, Optional, Callable
from pydantic import BaseModel, Field

class SignPost(BaseModel):
    """
    An identifiable event or metric that signals a shift in the trajectory.
    """
    id: str
    description: str
    metric: str
    threshold: float
    current_value: float = 0.0
    is_triggered: bool = False
    probability_impact: float = Field(0.0, description="Shift in singularity probability (-1.0 to 1.0)")
    era_impact: Optional[str] = None # e.g., "POST_SCARCITY"

    def check(self, value: float) -> bool:
        self.current_value = value
        if value >= self.threshold and not self.is_triggered:
            self.is_triggered = True
            return True
        return False

class SignalMonitor(BaseModel):
    """
    Monitors global signals to update the simulation state.
    """
    active_sign_posts: List[SignPost] = []
    triggered_history: List[str] = []

    def register_default_signals(self):
        defaults = [
            SignPost(
                id="SP_001",
                description="Commercial AGI Valuation Parity",
                metric="SSI_VALUATION_B",
                threshold=100.0, # $100 Billion
                probability_impact=0.1
            ),
            SignPost(
                id="SP_002",
                description="Universal Basic Compute Legislation",
                metric="COUNTRIES_WITH_UBC",
                threshold=1.0,
                era_impact="POST_SCARCITY",
                probability_impact=0.2
            ),
            SignPost(
                id="SP_003",
                description="Break of the Turing-Lovelace Barrier (Recursive Coding)",
                metric="AUTO_CODE_ACCURACY",
                threshold=0.99,
                probability_impact=0.3
            ),
            SignPost(
                id="SP_004",
                description="Energy Cost Collapse (Fusion/Solar)",
                metric="ENERGY_COST_CENT_KWH",
                threshold=-0.01, # Effective negative cost or near zero (logic inversion)
                # We'll treat threshold as "cross below" for cost in logic, but for simplicity let's say inverted metric "Energy Abundance"
                probability_impact=0.1
            )
        ]
        self.active_sign_posts.extend(defaults)

    def update_signal(self, metric: str, value: float) -> Optional[SignPost]:
        for sp in self.active_sign_posts:
            if sp.metric == metric:
                if sp.check(value):
                    self.triggered_history.append(f"TRIGGERED: {sp.description} (Value: {value})")
                    return sp
        return None

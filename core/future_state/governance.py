from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import math
from enum import Enum

class PoliticalAxis(str, Enum):
    AUTHORITARIAN = "AUTHORITARIAN"
    LIBERTARIAN = "LIBERTARIAN"
    HUMAN_CENTRIC = "HUMAN_CENTRIC"
    AI_CENTRIC = "AI_CENTRIC"

class PoliticalCompass(BaseModel):
    """
    Tracks the ideological drift of the society.
    """
    authority_score: float = Field(0.0, ge=-1.0, le=1.0, description="-1 (Anarchy) to +1 (Totalitarian)")
    substrate_score: float = Field(0.0, ge=-1.0, le=1.0, description="-1 (Bio-Purist) to +1 (Transhumanist)")

    def drift(self, event_impact_auth: float, event_impact_sub: float):
        self.authority_score = max(-1.0, min(1.0, self.authority_score + event_impact_auth))
        self.substrate_score = max(-1.0, min(1.0, self.substrate_score + event_impact_sub))

class DataDividend(BaseModel):
    """
    Tracks payments made to users for their data contribution (Data Dignity).
    """
    user_did: str
    data_points_contributed: int
    rate_per_point_compute_credits: float = 0.001

    def calculate_payout(self) -> float:
        return self.data_points_contributed * self.rate_per_point_compute_credits

class Proposal(BaseModel):
    id: str
    description: str
    votes_for: int = 0
    votes_against: int = 0
    category: str = "GENERAL"

class QuadraticVoting(BaseModel):
    """
    Implements Quadratic Voting logic.
    Cost = (Votes)^2
    """
    voice_credits_balance: int = 100

    def calculate_cost(self, votes: int) -> int:
        return votes * votes

    def vote(self, proposal: Proposal, votes: int, direction: str) -> bool:
        cost = self.calculate_cost(votes)
        if self.voice_credits_balance >= cost:
            self.voice_credits_balance -= cost
            if direction == "for":
                proposal.votes_for += votes
            else:
                proposal.votes_against += votes
            return True
        return False

class Asset(BaseModel):
    id: str
    owner: str
    self_assessed_value: float
    tax_rate: float = 0.02 # 2% per year

    def calculate_daily_tax(self) -> float:
        return (self.self_assessed_value * self.tax_rate) / 365.0

    def buy(self, new_owner: str, offer_amount: float) -> bool:
        """Harberger Tax rule: Must sell if offer >= self_assessed_value"""
        if offer_amount >= self.self_assessed_value:
            self.owner = new_owner
            self.self_assessed_value = offer_amount # Reset valuation
            return True
        return False

class RadicalExchange(BaseModel):
    """
    Container for Radical Exchange mechanisms.
    """
    assets: Dict[str, Asset] = {}
    dividend_pool: float = 0.0 # Total collected taxes to be redistributed

    def collect_taxes(self):
        daily_tax = sum(a.calculate_daily_tax() for a in self.assets.values())
        self.dividend_pool += daily_tax

class AlgocraticCouncil(BaseModel):
    """
    Automated governance system using AGI optimization.
    """
    active_proposals: List[Proposal] = []
    political_compass: PoliticalCompass = Field(default_factory=PoliticalCompass)
    radical_exchange_system: RadicalExchange = Field(default_factory=RadicalExchange)

    def optimize_allocation(self, proposals: List[Proposal]) -> Optional[Proposal]:
        """Selects the proposal with highest aggregate quadratic utility."""
        if not proposals:
            return None
        # Simplified: just max net votes
        return max(proposals, key=lambda p: p.votes_for - p.votes_against)

    def enact_policy(self, proposal: Proposal):
        # Drift compass based on policy
        if "SURVEILLANCE" in proposal.description:
            self.political_compass.drift(0.1, 0.0)
        elif "PRIVACY" in proposal.description:
            self.political_compass.drift(-0.1, 0.0)

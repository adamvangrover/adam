from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from core.schemas.v23_5_schema import RiskDialogue, DialogueTurn

class ConsensusMetric(BaseModel):
    final_rating: str
    conviction_score: float
    divergence_penalty: float
    narrative: str
    risk_dialogue: Optional[RiskDialogue] = None

class RiskConsensusEngine:
    """
    Mathematical Core for Agentic Conviction.
    Implements the formula: C(x) = alpha * I(agree) + beta * conf(strat) - gamma * div(reg, strat)
    Generates a 'Risk Dialogue' simulating the debate between agents.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Severity Map for penalty calculation
        self.severity_map = {
            "Pass": 1,
            "Special Mention": 2,
            "Substandard": 3,
            "Doubtful": 4,
            "Loss": 5
        }

    def calculate_consensus(self,
                          reg_rating: str,
                          strat_rating: str,
                          strat_confidence: float,
                          reg_rationale: str = "",
                          strat_rationale: str = "") -> ConsensusMetric:
        """
        Calculates the final consensus rating and conviction score,
        and generates a dialogue trace.
        """

        # 1. Indicator Function I(agree)
        agreement = 1.0 if reg_rating == strat_rating else 0.0

        # 2. Divergence Penalty
        reg_sev = self.severity_map.get(reg_rating, 3)
        strat_sev = self.severity_map.get(strat_rating, 3)
        diff = abs(reg_sev - strat_sev)

        # Normalize diff (max diff is 4: Pass vs Loss)
        normalized_div = diff / 4.0
        penalty = normalized_div * self.gamma

        # 3. Calculate Base Conviction
        score = (self.alpha * agreement) + (self.beta * strat_confidence) - penalty
        score = max(0.0, min(1.0, score))

        # 4. Generate Dialogue
        dialogue = self._generate_dialogue(reg_rating, strat_rating, reg_rationale, strat_rationale, diff)

        # 5. Determine Final Rating Strategy
        final = strat_rating # Default to strategic if override logic passes
        override = False

        if reg_sev >= strat_sev:
            # Regulator is stricter. Default to Regulator.
            final = reg_rating
            narrative = f"Adopted Regulatory Rating ({reg_rating}) as constraint."

            # OVERRIDE LOGIC:
            # If Strategist has HIGH confidence (>0.9) AND divergence is small (1 step),
            # AND Strategist explicitly cites "Strong Liquidity" or "Deleveraging", allow override.
            if strat_confidence > 0.9 and diff <= 1 and ("Liquidity" in strat_rationale or "Deleveraging" in strat_rationale):
                final = strat_rating
                narrative = f"STRATEGIC OVERRIDE: Adopted ({strat_rating}) despite Regulatory flag ({reg_rating}) due to high conviction."
                override = True
                dialogue.override_applied = True
                dialogue.conclusion = narrative

        else:
            # Strategist is stricter. Always safe to take stricter.
            final = strat_rating
            narrative = f"Adopted Strategic Rating ({strat_rating}) identifying hidden risk."

        if agreement:
            narrative = f"Consensus Reached ({final}). High Conviction."

        return ConsensusMetric(
            final_rating=final,
            conviction_score=score,
            divergence_penalty=penalty,
            narrative=narrative,
            risk_dialogue=dialogue
        )

    def _generate_dialogue(self, reg_rating, strat_rating, reg_txt, strat_txt, diff) -> RiskDialogue:
        turns = []

        # Turn 1: Regulator Speaks
        turns.append(DialogueTurn(
            speaker="Regulator",
            argument=f"I assess this borrower as '{reg_rating}'. {reg_txt}"
        ))

        # Turn 2: Strategist Responds
        if diff == 0:
            resp = f"I agree with the classification of '{strat_rating}'. {strat_txt}"
        elif self.severity_map.get(reg_rating) > self.severity_map.get(strat_rating):
             resp = f"I disagree. I see it as '{strat_rating}'. {strat_txt} The regulatory model is too rigid on leverage."
        else:
             resp = f"I advise caution. While you see '{reg_rating}', my models indicate '{strat_rating}'. {strat_txt}"

        turns.append(DialogueTurn(
            speaker="Strategist",
            argument=resp,
            counter_point="Regulatory Leverage Cap" if diff > 0 else None
        ))

        return RiskDialogue(
            turns=turns,
            conclusion="Pending Consensus..."
        )

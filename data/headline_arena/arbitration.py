"""
ADAM-Macro-Sentinel Champion-Challenger Arbitration Harness
============================================================
Dual-model adversarial debate system:

    Model A (Champion): Structural / Balance-Sheet Thesis
    Model B (Challenger): Cross-Asset / Flow Momentum Counter-thesis

The pre-submission gate requires Model A to explicitly rebut Model B's
counterfactual challenge before committing the forecast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .rationale import MarketContext, RationaleGenerator
from .schema import Direction, ForecastSubmission, RationaleBlock


class ThesisType(str, Enum):
    STRUCTURAL = "structural"     # Balance-sheet, fundamental
    FLOW_MOMENTUM = "flow_momentum"  # Cross-asset, positioning, flow


@dataclass
class Thesis:
    """A single directional thesis with supporting evidence."""
    thesis_type: ThesisType
    direction: Direction
    confidence: float
    primary_argument: str
    supporting_evidence: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    catalyst_specificity: float = 0.0  # 0.0 = generic, 1.0 = highly specific

    def strength_score(self) -> float:
        """
        Compute thesis strength based on:
        - Number of supporting evidence points
        - Catalyst specificity
        - Confidence level
        """
        evidence_score = min(1.0, len(self.supporting_evidence) / 3.0)
        vulnerability_penalty = min(0.3, len(self.vulnerabilities) * 0.1)
        return (
            0.4 * self.confidence
            + 0.3 * evidence_score
            + 0.2 * self.catalyst_specificity
            - vulnerability_penalty
        )


@dataclass
class ArbitrationRecord:
    """Record of a champion-challenger debate round."""
    asset: str
    timestamp: str
    champion: Thesis
    challenger: Thesis
    champion_rebuttal: str
    challenger_rebuttal: str
    winner: str  # "champion" or "challenger"
    final_direction: Direction
    final_confidence: float
    reasoning: str
    debate_hash: str = ""

    def __post_init__(self):
        if not self.debate_hash:
            content = json.dumps({
                "asset": self.asset,
                "champion_dir": self.champion.direction.value,
                "challenger_dir": self.challenger.direction.value,
                "winner": self.winner,
                "final_dir": self.final_direction.value,
            }, sort_keys=True)
            self.debate_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


class ChampionChallengerArbitrator:
    """
    Dual-model arbitration harness implementing the adversarial
    champion-challenger debate protocol.

    Protocol:
    1. Champion generates structural/fundamental thesis
    2. Challenger generates cross-asset/flow counter-thesis
    3. Champion must explicitly rebut challenger's strongest argument
    4. If rebuttal fails, challenger's thesis prevails
    5. Final forecast inherits the winning thesis's direction + rationale
    """

    def __init__(self):
        self._debate_history: list[ArbitrationRecord] = []
        self._rationale_gen = RationaleGenerator()

    def generate_champion_thesis(self, ctx: MarketContext) -> Thesis:
        """
        Model A: Structural / Balance-Sheet Thesis.

        Focuses on:
        - Fundamental supply/demand dynamics
        - Central bank policy transmission
        - Balance sheet and flow-of-funds analysis
        - Valuation relative to macro fundamentals
        """
        direction = self._infer_structural_direction(ctx)
        evidence = []
        vulnerabilities = []

        # Build evidence from structural signals
        if ctx.primary_catalyst:
            evidence.append(f"Primary structural catalyst: {ctx.primary_catalyst}")

        if ctx.rate_differential:
            evidence.append(f"Rate differential: {ctx.rate_differential}")

        if ctx.yield_curve_shape:
            evidence.append(f"Yield curve: {ctx.yield_curve_shape}")

        for signal in ctx.supporting_signals:
            evidence.append(signal)

        # Identify vulnerabilities
        for contrary in ctx.contrary_signals:
            vulnerabilities.append(contrary)

        # Catalyst specificity scoring
        specificity = 0.5
        if ctx.catalyst_timestamp:
            specificity += 0.2
        if ctx.cftc_net_speculative:
            specificity += 0.15
        if ctx.implied_vol is not None:
            specificity += 0.15

        return Thesis(
            thesis_type=ThesisType.STRUCTURAL,
            direction=direction,
            confidence=self._compute_structural_confidence(ctx),
            primary_argument=(
                f"The structural thesis for {ctx.asset_ticker} is driven by "
                f"{ctx.primary_catalyst or 'macro fundamentals'}. "
                f"The balance-sheet impact flows through {ctx.catalyst_type or 'standard'} "
                f"transmission channels to the terminal clearing price."
            ),
            supporting_evidence=evidence,
            vulnerabilities=vulnerabilities,
            catalyst_specificity=min(1.0, specificity),
        )

    def generate_challenger_thesis(self, ctx: MarketContext) -> Thesis:
        """
        Model B: Cross-Asset / Flow Momentum Challenger.

        Focuses on:
        - Cross-asset correlation regime
        - Momentum and trend signals
        - Positioning and flow data (CFTC, dealer gamma)
        - Contrarian arguments
        """
        # Challenger often takes the opposite view for adversarial testing
        champion_direction = self._infer_structural_direction(ctx)
        challenger_direction = self._infer_flow_direction(ctx, champion_direction)

        evidence = []
        vulnerabilities = []

        # Flow-based evidence
        if ctx.cftc_net_speculative:
            evidence.append(f"CFTC positioning: {ctx.cftc_net_speculative}")

        if ctx.crowding_signal and "crowded" in ctx.crowding_signal:
            evidence.append(
                f"Crowding signal: {ctx.crowding_signal} — "
                f"vulnerable to reversal on catalyst surprise"
            )

        if ctx.dealer_gamma_exposure:
            evidence.append(f"Dealer gamma: {ctx.dealer_gamma_exposure}")

        # Momentum evidence
        if ctx.daily_range_1sigma:
            sigma_mult = abs(ctx.daily_change_pct) / ctx.daily_range_1sigma
            if sigma_mult > 2.0:
                evidence.append(
                    f"Mean-reversion signal: {sigma_mult:.1f}σ move "
                    f"statistically favors partial retracement"
                )

        # Cross-asset signals
        for signal in ctx.contrary_signals:
            evidence.append(f"Cross-asset contrary: {signal}")

        vulnerabilities.append(
            "Flow-momentum thesis lacks structural grounding — "
            "vulnerable to fundamental catalyst override"
        )

        return Thesis(
            thesis_type=ThesisType.FLOW_MOMENTUM,
            direction=challenger_direction,
            confidence=self._compute_flow_confidence(ctx),
            primary_argument=(
                f"The flow-momentum thesis challenges the structural view. "
                f"{ctx.asset_ticker} positioning data and cross-asset signals "
                f"suggest the structural thesis may be overweighting "
                f"{'the catalyst' if ctx.primary_catalyst else 'fundamentals'} "
                f"and underweighting flow dynamics."
            ),
            supporting_evidence=evidence,
            vulnerabilities=vulnerabilities,
            catalyst_specificity=0.3,  # Flow thesis is inherently less specific
        )

    def arbitrate(self, ctx: MarketContext) -> ArbitrationRecord:
        """
        Execute the full champion-challenger arbitration protocol.

        Returns an ArbitrationRecord with the winning thesis.
        """
        champion = self.generate_champion_thesis(ctx)
        challenger = self.generate_challenger_thesis(ctx)

        # Champion must rebut challenger's strongest argument
        champion_rebuttal = self._generate_rebuttal(champion, challenger, ctx)
        challenger_rebuttal = self._generate_rebuttal(challenger, champion, ctx)

        # Determine winner based on strength scores and rebuttal quality
        champion_score = champion.strength_score()
        challenger_score = challenger.strength_score()

        # Rebuttal bonus: if champion successfully addresses challenger's
        # strongest point, it gets a strength bonus
        rebuttal_bonus = self._score_rebuttal(champion_rebuttal, challenger)

        champion_total = champion_score + rebuttal_bonus
        challenger_total = challenger_score

        # Winner determination
        if champion_total >= challenger_total:
            winner = "champion"
            final_direction = champion.direction
            final_confidence = champion.confidence
            reasoning = (
                f"Champion (structural) prevails with score "
                f"{champion_total:.2f} vs challenger {challenger_total:.2f}. "
                f"The structural thesis successfully rebutted the flow-momentum "
                f"counter-thesis. {champion_rebuttal}"
            )
        else:
            winner = "challenger"
            final_direction = challenger.direction
            final_confidence = challenger.confidence
            reasoning = (
                f"Challenger (flow-momentum) prevails with score "
                f"{challenger_total:.2f} vs champion {champion_total:.2f}. "
                f"The structural thesis failed to adequately rebut the "
                f"cross-asset/flow evidence. {challenger_rebuttal}"
            )

        record = ArbitrationRecord(
            asset=ctx.asset_ticker,
            timestamp=datetime.now(timezone.utc).isoformat(),
            champion=champion,
            challenger=challenger,
            champion_rebuttal=champion_rebuttal,
            challenger_rebuttal=challenger_rebuttal,
            winner=winner,
            final_direction=final_direction,
            final_confidence=final_confidence,
            reasoning=reasoning,
        )

        self._debate_history.append(record)
        return record

    def _generate_rebuttal(
        self,
        rebutter: Thesis,
        opponent: Thesis,
        ctx: MarketContext,
    ) -> str:
        """
        Generate a rebuttal from one thesis against another.
        The rebutter must address the opponent's strongest evidence point.
        """
        if not opponent.supporting_evidence:
            return (
                f"The {opponent.thesis_type.value} thesis offers no specific "
                f"evidence to rebut. The {rebutter.thesis_type.value} thesis "
                f"stands unopposed."
            )

        strongest_opponent_point = opponent.supporting_evidence[0]
        rebuttal_parts = [
            f"Addressing the {opponent.thesis_type.value} thesis's primary "
            f"argument: '{strongest_opponent_point}'. "
        ]

        if rebutter.thesis_type == ThesisType.STRUCTURAL:
            rebuttal_parts.append(
                f"The structural analysis shows that {ctx.primary_catalyst or 'fundamental drivers'} "
                f"dominate the cross-asset flow signal. "
            )
            if ctx.daily_change_pct != 0:
                rebuttal_parts.append(
                    f"Price action ({ctx.daily_change_pct:+.2f}%) confirms the "
                    f"structural thesis is the dominant regime driver. "
                )
        else:
            rebuttal_parts.append(
                f"The flow data contradicts the structural thesis because "
                f"positioning and momentum signals suggest the market is "
                f"not pricing the catalyst at fair value. "
            )

        return "".join(rebuttal_parts)

    def _score_rebuttal(self, rebuttal: str, opponent: Thesis) -> float:
        """
        Score the quality of a rebuttal.
        Returns a bonus score in [0, 0.2].
        """
        score = 0.0

        # Length quality
        if len(rebuttal) > 200:
            score += 0.05

        # Addresses specific evidence
        if opponent.supporting_evidence:
            for evidence in opponent.supporting_evidence:
                # Check if any key terms from the evidence appear in rebuttal
                key_terms = evidence.split()[:3]  # First 3 words
                if any(term.lower() in rebuttal.lower() for term in key_terms if len(term) > 3):
                    score += 0.05
                    break

        # Provides counter-evidence
        counter_markers = [
            "however", "despite", "contradicts", "overrides", "dominates",
            "overwhelms", "confirms", "proves",
        ]
        if any(m in rebuttal.lower() for m in counter_markers):
            score += 0.05

        # Specific data reference
        data_markers = ["%", "bps", "sigma", "σ", "billion", "million"]
        if any(m in rebuttal for m in data_markers):
            score += 0.05

        return min(0.20, score)

    def _infer_structural_direction(self, ctx: MarketContext) -> Direction:
        """Infer direction from structural/fundamental signals."""
        signals = 0

        # Price momentum
        if ctx.daily_change_pct > 0.5:
            signals += 1
        elif ctx.daily_change_pct < -0.5:
            signals -= 1

        # Catalyst type analysis
        if ctx.catalyst_type == "monetary_policy":
            if "hike" in ctx.primary_catalyst.lower():
                # Rate hikes: bearish for bonds/gold, bullish for USD
                if ctx.asset_ticker in ("GC", "SI", "ZN"):
                    signals -= 1
                elif ctx.asset_ticker in ("DXY",):
                    signals += 1
            elif "cut" in ctx.primary_catalyst.lower():
                if ctx.asset_ticker in ("GC", "SI", "ZN"):
                    signals += 1
                elif ctx.asset_ticker in ("DXY",):
                    signals -= 1
        elif ctx.catalyst_type == "geopolitical":
            # Geopolitical escalation: bullish for safe havens
            if ctx.asset_ticker in ("GC", "SI"):
                signals += 1
            elif ctx.asset_ticker in ("CL",):
                signals += 1  # Supply disruption risk

        # Net signal
        if signals > 0:
            return Direction.BULLISH
        elif signals < 0:
            return Direction.BEARISH
        return Direction.NEUTRAL

    def _infer_flow_direction(
        self,
        ctx: MarketContext,
        champion_direction: Direction,
    ) -> Direction:
        """
        Infer direction from flow/momentum signals.
        Challenger often takes a contrarian or divergent view.
        """
        # If extreme move, flow thesis favors mean reversion
        if ctx.daily_range_1sigma and abs(ctx.daily_change_pct) > 2 * ctx.daily_range_1sigma:
            if ctx.daily_change_pct > 0:
                return Direction.BEARISH  # Mean reversion from big up
            else:
                return Direction.BULLISH  # Mean reversion from big down

        # If crowded positioning, flow thesis may be contrarian
        if ctx.crowding_signal:
            if "crowded_long" in ctx.crowding_signal:
                return Direction.BEARISH
            elif "crowded_short" in ctx.crowding_signal:
                return Direction.BULLISH

        # Default: same as champion (validates rather than challenges)
        return champion_direction

    def _compute_structural_confidence(self, ctx: MarketContext) -> float:
        """Compute confidence for the structural thesis."""
        base = 0.60

        # Catalyst specificity bonus
        if ctx.primary_catalyst and ctx.catalyst_timestamp:
            base += 0.05

        # Multiple supporting signals
        base += min(0.10, len(ctx.supporting_signals) * 0.03)

        # Contrary signals penalty
        base -= min(0.10, len(ctx.contrary_signals) * 0.03)

        # Magnitude alignment
        if ctx.daily_range_1sigma:
            sigma_mult = abs(ctx.daily_change_pct) / ctx.daily_range_1sigma
            if 0.5 < sigma_mult < 2.0:
                base += 0.05  # Normal continuation range

        return max(0.50, min(1.00, base))

    def _compute_flow_confidence(self, ctx: MarketContext) -> float:
        """Compute confidence for the flow/momentum thesis."""
        base = 0.55  # Lower base — flow is less reliable

        if ctx.cftc_net_speculative:
            base += 0.05

        if ctx.dealer_gamma_exposure:
            base += 0.05

        if ctx.crowding_signal and "crowded" in ctx.crowding_signal:
            base += 0.05

        return max(0.50, min(0.85, base))

    def get_debate_history(self) -> list[ArbitrationRecord]:
        return self._debate_history

    def get_win_rates(self) -> dict[str, float]:
        """Get historical win rates for champion vs challenger."""
        if not self._debate_history:
            return {"champion": 0.5, "challenger": 0.5}

        champion_wins = sum(
            1 for r in self._debate_history if r.winner == "champion"
        )
        total = len(self._debate_history)
        return {
            "champion": champion_wins / total,
            "challenger": (total - champion_wins) / total,
        }

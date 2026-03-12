from typing import List, Dict, Any
import random

class NarrativeWeaver:
    """
    Protocol: ADAM-V-NEXT
    Synthesizes disparate agent outputs into a cohesive 'Mission Brief' story.
    Acts as the 'Editor-in-Chief' for the system's internal monologue.
    """

    def __init__(self):
        self.templates = {
            "RISK": [
                "CRITICAL ALERT: System has detected {risk_factor} in {driver}. Immediate defensive positioning recommended.",
                "Red lights flashing. {risk_factor} poses a systemic threat. Agents advise maximum caution.",
                "Structural integrity compromised by {risk_factor}. Sentiment override: BEARISH."
            ],
            "CONFLICT": [
                "Consensus fracture detected. Agents are split on {driver}. Volatility spike imminent.",
                "Battleground sector: {sector}. Bulls and Bears are clashing over {driver} valuation.",
                "Divergence in signal processing. High uncertainty regarding {driver}. Awaiting resolution."
            ],
            "BULLISH": [
                "Market sentiment is surging, driven by {driver}. {risk_agent} advises caution, but the momentum is undeniable.",
                "Green shoots detected in {sector}, fueled by {driver}. Despite {risk_factor}, the consensus leans towards accumulation.",
                "Alpha signal generated: {driver} is leading the charge. System advises risk-on positioning."
            ],
            "BEARISH": [
                "Defensive posture initiated. {driver} is weighing heavily on sentiment. {risk_agent} flags critical levels.",
                "Capital preservation mode. {sector} weakness is spreading. {risk_factor} remains the primary concern.",
                "Systemic tremors detected. {driver} suggests a decoupling event. Recommend reducing exposure."
            ],
            "NEUTRAL": [
                "Markets are in a holding pattern. {driver} is balanced by {risk_factor}. Awaiting clearer signals.",
                "Volatility compression in effect. {sector} is flat. System stands ready for the breakout.",
                "Consensus is split. {risk_agent} and {growth_agent} are at odds regarding {driver}."
            ]
        }

    def weave(self, context: Dict[str, Any]) -> str:
        """
        Weaves a narrative from the provided context.

        Args:
            context: Dict containing 'sentiment', 'driver', 'risk_factor', 'sector',
                     plus optional 'conflicts' and 'risks' lists.
        """

        # Determine dominance hierarchy
        mode = "NEUTRAL"

        risks = context.get('risks', [])
        conflicts = context.get('conflicts', [])
        sentiment = context.get('sentiment', 'NEUTRAL').upper()

        if risks:
            mode = "RISK"
        elif conflicts:
            mode = "CONFLICT"
        else:
            mode = sentiment

        if mode not in self.templates:
            mode = "NEUTRAL"

        template = random.choice(self.templates[mode])

        # Safe formatting (Graceful Degradation)
        try:
            return template.format(
                driver=context.get('driver', 'market structure'),
                risk_agent=context.get('risk_agent', 'Risk Officer'),
                growth_agent=context.get('growth_agent', 'Growth Strategist'),
                risk_factor=context.get('risk_factor', 'uncertainty'),
                sector=context.get('sector', 'broad market')
            )
        except KeyError:
            return "Systems operational. Awaiting data coherence."

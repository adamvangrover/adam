from typing import Dict, Any, List
import random

class CritiqueSwarm:
    """
    A multi-agent swarm that provides independent critiques of strategic briefings.
    """

    def __init__(self):
        self.agents = {
            "RedTeam": self._red_team_critique,
            "ModelRisk": self._model_risk_critique,
            "Contrarian": self._contrarian_critique,
            "ConsensusReview": self._consensus_critique
        }

    def critique(self, briefing: Dict[str, Any], sim_data: Dict[str, Any]) -> List[Dict[str, str]]:
        critiques = []
        for name, func in self.agents.items():
            # ConsensusReview handles a different signature or context
            if name == "ConsensusReview":
                continue

            result = func(briefing, sim_data)
            if result:
                critiques.append({
                    "agent": name,
                    "perspective": result["perspective"],
                    "critique": result["text"]
                })
        return critiques

    def _red_team_critique(self, briefing, sim_data):
        # Technical/Security perspective
        scenario = sim_data.get("scenario", "")
        if "Cyber" in scenario or "Quantum" in scenario:
            return {
                "perspective": "Adversarial Simulation",
                "text": "The proposed mitigation assumes rational actor models. Adversary may employ 'scorched earth' logic not captured in current intensity models. Recommend creating air-gapped backups immediately."
            }
        elif "Blockade" in scenario:
            return {
                "perspective": "Logistics Denial",
                "text": "Blockade efficacy likely overstated. Smuggling routes via non-aligned third parties (Grey Zone tactics) are not accounted for in impact score."
            }
        return None

    def _model_risk_critique(self, briefing, sim_data):
        # Math/Stats perspective
        stats = sim_data.get("statistics", {})
        stdev = stats.get("stdev_impact", 0)
        mean = stats.get("mean_impact", 1)

        cv = stdev / mean if mean > 0 else 0

        if cv > 0.2:
            return {
                "perspective": "Statistical Reliability",
                "text": f"High volatility (CV: {cv:.2f}) detected in Monte Carlo runs. Tail risk events are significantly fatter than normal distribution suggests. Confidence intervals may be too narrow."
            }
        return {
            "perspective": "Statistical Reliability",
            "text": "Model convergence is high. Distribution suggests a stable, predictable outcome path."
        }

    def _contrarian_critique(self, briefing, sim_data):
        # Macro/Political perspective
        if "Critical" in briefing.get("executive_summary", ""):
            return {
                "perspective": "Market Contrarian",
                "text": "The 'Critical' designation may induce panic selling. Historical precedent suggests markets price in conflict premiums early. Recommendation: Fade the fear trade in non-affected sectors."
            }
        return {
            "perspective": "Geopolitical Realist",
            "text": "The briefing is too optimistic about diplomatic channels. Current alliance fracture points suggest coordination costs will be 30-40% higher than baseline."
        }

    def critique_consensus(self, consensus_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Specialized entry point for critiquing a ConsensusEngine result.
        """
        result = self._consensus_critique(consensus_result, {})
        return [result] if result else []

    def _consensus_critique(self, consensus_result: Dict[str, Any], _):
        """
        Analyzes the consensus rationale for overconfidence or weak signal support.
        """
        score = consensus_result.get("score", 0)
        rationale = consensus_result.get("rationale", "")

        # Check for Overconfidence in Neutral markets
        if abs(score) > 0.8 and "No signals provided" in rationale:
             return {
                "agent": "ConsensusReview",
                "perspective": "Logical Consistency",
                "critique": "High confidence score derived despite lack of underlying signals. Suspected hallucination or default bias."
            }

        # Check for 'Groupthink' (e.g., all agents agree too perfectly)
        if abs(score) > 0.95:
             return {
                "agent": "ConsensusReview",
                "perspective": "Groupthink Monitor",
                "critique": "Near-perfect consensus detected (>0.95). In complex adaptive systems, this often precedes a phase shift. Recommend re-sampling with higher volatility parameters."
            }

        return None

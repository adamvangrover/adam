import datetime
import random

class ConvictionManager:
    """
    Tracks and manages the 'Conviction Score' of various agents over time.
    Simulates a 'Reputation System' where agents gain/lose conviction based on consensus alignment.
    """

    def __init__(self):
        # Default starting conviction
        self.agent_scores = {
            "RiskOfficer": 0.85,
            "TechAnalyst": 0.70,
            "MacroSentinel": 0.75,
            "BlindspotScanner": 0.60, # Starts lower, unproven
            "Fundamentalist": 0.80
        }
        self.history = {} # Store history for heatmaps

    def get_conviction_map(self):
        """
        Returns the current conviction scores formatted for the UI.
        """
        # Simulate slight dynamic shifts
        for agent in self.agent_scores:
            change = (random.random() - 0.5) * 0.05
            self.agent_scores[agent] = max(0.1, min(1.0, self.agent_scores[agent] + change))

        return {
            "scores": {k: round(v, 2) for k, v in self.agent_scores.items()},
            "timestamp": datetime.datetime.now().isoformat()
        }

    def record_outcome(self, agent, success: bool):
        """
        Adjusts conviction based on a prediction outcome.
        """
        if agent not in self.agent_scores:
            return

        if success:
            self.agent_scores[agent] = min(1.0, self.agent_scores[agent] + 0.05)
        else:
            self.agent_scores[agent] = max(0.1, self.agent_scores[agent] - 0.10)

conviction_manager = ConvictionManager()

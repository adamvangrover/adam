from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

# Import the Critique Swarm
from core.agents.critique_swarm import CritiqueSwarm

# Try importing AgentBase, but allow fallback if running in a restricted script environment
try:
    from core.agents.agent_base import AgentBase
except ImportError:
    class AgentBase:
        def __init__(self, config, kernel=None):
            self.config = config
        async def execute(self, **kwargs): pass

class StrategicForesightAgent(AgentBase):
    """
    Strategic Foresight Agent

    Analyzes geopolitical simulation data to produce high-level National Security Council (NSC)
    style briefings. Focuses on "Global Macro 2026" themes.
    Includes Monte Carlo analysis and Independent Critique.
    """

    def __init__(self, config: Dict[str, Any], kernel=None):
        super().__init__(config, kernel)
        self.role = "National Security Advisor"
        self.clearance = "TOP SECRET // NOFORN"
        self.critique_swarm = CritiqueSwarm()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the foresight analysis.

        Args:
            simulation_data (Dict): Output from SovereignConflictSimulation.

        Returns:
            Dict: Structured briefing object.
        """
        simulation_data = kwargs.get('simulation_data')
        if not simulation_data:
            return {"error": "No simulation data provided"}

        briefing = self.generate_briefing(simulation_data)

        # Add Swarm Critique
        critiques = self.critique_swarm.critique(briefing, simulation_data)
        briefing["independent_critiques"] = critiques

        return briefing

    def generate_briefing(self, sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a structured briefing based on simulation data.
        """
        scenario = sim_data.get("scenario", "Unknown Scenario")

        # Check if Monte Carlo data exists, otherwise fallback to single run
        stats = sim_data.get("statistics", {})
        if stats:
             total_impact = stats.get("mean_impact", 0)
             confidence_95 = stats.get("confidence_95", [0,0])
             stats_text = f" (MC Mean: {total_impact}, 95% CI: [{confidence_95[0]}-{confidence_95[1]}])"
        else:
             total_impact = sim_data.get("total_impact", 0)
             stats_text = ""

        sector_impact = sim_data.get("representative_sector_impact", sim_data.get("sector_impact", {}))

        # Determine Threat Level
        if total_impact > 600:
            threat_level = "CRITICAL"
            tone = "Urgent action required. Sovereign integrity at risk."
        elif total_impact > 300:
            threat_level = "ELEVATED"
            tone = "Monitor closely. Prepare contingency protocols."
        else:
            threat_level = "MODERATE"
            tone = "Standard diplomatic channels sufficient."

        # Generate Strategic Recommendations based on scenario
        recommendations = self._get_recommendations(scenario, threat_level)

        # Generate Economic Fallout text
        economic_fallout = self._analyze_economic_fallout(sector_impact)

        briefing = {
            "header": {
                "to": "The President",
                "from": self.role,
                "date": datetime.now().strftime("%Y-%m-%d %H00Z"),
                "subject": f"SITREP: {scenario.upper()} - {threat_level} THREAT"
            },
            "executive_summary": f"Simulation indicates a rapidly evolving {scenario} scenario. Total economic and geopolitical impact is estimated at index {total_impact}{stats_text}. {tone}",
            "key_judgments": [
                f"Projected Impact Range: {stats.get('min_impact',0)} - {stats.get('max_impact',0)}.",
                f"Direct threat to {self._get_affected_sector(scenario)} infrastructure.",
                "Adversary intent assessed as strategic denial of access."
            ],
            "economic_fallout": economic_fallout,
            "recommendations": recommendations,
            "classification": self.clearance
        }

        return briefing

    def _get_affected_sector(self, scenario):
        if "Semiconductor" in scenario:
            return "Advanced Technology & Defense"
        elif "Energy" in scenario:
            return "Critical Energy & Logistics"
        elif "Cyber" in scenario or "Quantum" in scenario:
            return "Financial & Information Systems"
        return "National Economic"

    def _analyze_economic_fallout(self, sector_impact: Dict[str, int]) -> str:
        """Analyzes sector impact scores to generate a fallout summary."""
        if not sector_impact:
            return "Insufficient data to project economic fallout."

        sorted_sectors = sorted(sector_impact.items(), key=lambda item: item[1], reverse=True)
        top_3 = sorted_sectors[:3]

        analysis = "Contagion Risk Analysis: "
        for sector, val in top_3:
             if val > 50:
                 analysis += f"{sector} ({val}/100) severe risk. "
             elif val > 20:
                 analysis += f"{sector} ({val}/100) moderate strain. "

        if sector_impact.get("Finance", 0) > 70 or sector_impact.get("Shadow Banking", 0) > 70:
            analysis += "Capital flight and liquidity crunch imminent."

        return analysis

    def _get_recommendations(self, scenario, threat_level):
        recs = []
        if scenario == "Semiconductor Blockade":
            recs = [
                "Invoke Defense Production Act for domestic fabs.",
                "Initiate 'Silicon Shield' diplomatic protocols with allies.",
                "Authorize strategic reserve release of rare earth elements."
            ]
        elif scenario == "Energy Shock":
            recs = [
                "Deploy naval assets to secure key transit straits.",
                "Activate strategic petroleum reserve drawdown.",
                "Implement emergency industrial rationing guidelines."
            ]
        elif scenario == "Cyber Infrastructure Attack":
            recs = [
                "Sever international links to critical banking nodes.",
                "Activate National Cyber Mission Force (NCMF) counter-strike options.",
                "Mandate paper-trail backups for all clearing houses."
            ]
        elif scenario == "Quantum Decryption Event":
            recs = [
                "Initiate Post-Quantum Cryptography (PQC) transition immediately.",
                "Isolate sovereign ledger nodes from public internet.",
                "Freeze all blockchain-based asset transfers pending validation."
            ]

        if threat_level == "CRITICAL":
            recs.append("Raise DEFCON level. Mobilize cyber-command response.")

        return recs

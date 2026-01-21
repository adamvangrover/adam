from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import numpy as np

# --- Imports from Feature Branch ---
from core.research.oswm.inference import OSWMInference

# --- Imports from Main Branch ---
from core.agents.critique_swarm import CritiqueSwarm

# Try importing AgentBase, but allow fallback if running in a restricted script environment
try:
    from core.agents.agent_base import AgentBase
except ImportError:
    class AgentBase:
        def __init__(self, config, kernel=None):
            self.config = config
        async def execute(self, **kwargs): pass

# Configure logging
logger = logging.getLogger(__name__)

class StrategicForesightAgent(AgentBase):
    """
    Strategic Foresight Agent

    A unified intelligence unit acting as the system's "Pre-Crime" and National Security division.
    It combines:
    1. OSWM (One-Shot World Model) for financial regime shift detection (Market Pre-Crime).
    2. Geopolitical Simulation analysis for National Security Council (NSC) style briefings.
    """

    def __init__(self, config: Dict[str, Any], kernel=None):
        super().__init__(config, kernel)
        
        # Identity & Security (from Main)
        self.role = "National Security Advisor"
        self.clearance = "TOP SECRET // NOFORN"
        
        # Subsystems
        # 1. Initialize OSWM (from Feature Branch)
        try:
            self.oswm = OSWMInference()
            self.oswm.pretrain_on_synthetic_prior(steps=10) # Quick init for demo
            self.oswm_enabled = True
        except Exception as e:
            logger.error(f"Failed to initialize OSWM: {e}")
            self.oswm_enabled = False

        # 2. Initialize Critique Swarm (from Main)
        self.critique_swarm = CritiqueSwarm()
        
        logger.info("Strategic Foresight Agent initialized with Hybrid Capabilities (OSWM + GeoSim).")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the foresight analysis. Dispatches logic based on input type.
        """
        # Dispatch 1: Financial Market Analysis (Feature Branch Logic)
        if "ticker" in kwargs:
            ticker = kwargs.get("ticker")
            horizon = kwargs.get("horizon", 10)
            return self._generate_market_briefing(ticker, horizon)

        # Dispatch 2: Geopolitical Simulation (Main Branch Logic)
        elif "simulation_data" in kwargs:
            simulation_data = kwargs.get('simulation_data')
            briefing = self._generate_nsc_briefing(simulation_data)
            
            # Add Swarm Critique (Main Branch Feature)
            critiques = self.critique_swarm.critique(briefing, simulation_data)
            briefing["independent_critiques"] = critiques
            return briefing

        else:
            return {"error": "Invalid input. Provide 'ticker' for market analysis or 'simulation_data' for geopolitical sitrep."}

    # =========================================================================
    # Capability A: Financial Foresight (OSWM) - From Feature Branch
    # =========================================================================

    def _generate_market_briefing(self, ticker: str, horizon: int) -> Dict[str, Any]:
        """Generates a strategic briefing based on OSWM predictions."""
        if not self.oswm_enabled:
            return {"status": "OSWM_OFFLINE", "error": "Model not loaded"}

        logger.info(f"Generating strategic market briefing for {ticker} (Horizon: {horizon})...")

        # 1. Load Context
        context, stats = self.oswm.load_market_context(ticker, period="1mo")

        if len(context) < 10:
            logger.warning("Insufficient data for robust prediction.")
            return {"status": "INSUFFICIENT_DATA"}

        # 2. Run Simulations (Monte Carlo approximation)
        num_sims = 5
        predictions = []
        for _ in range(num_sims):
            # Perturb input slightly to simulate uncertainty
            perturbed_context = [c + np.random.normal(0, 0.01) for c in context[-10:]]
            pred = self.oswm.generate_scenario(perturbed_context, steps=horizon)
            predictions.append(pred)

        # 3. Analyze Results
        predictions = np.array(predictions)
        avg_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Denormalize
        mean, std = stats["mean"], stats["std"]
        avg_price = avg_pred * std + mean
        current_price = context[-1] * std + mean
        final_price = avg_price[-1]

        # Detect Regime Shift
        drift = (final_price - current_price) / current_price
        volatility_forecast = np.mean(std_pred)

        status = "STABLE"
        if abs(drift) > 0.05:
            status = "REGIME_SHIFT_IMMINENT"
        elif volatility_forecast > 0.5:
            status = "HIGH_UNCERTAINTY"

        briefing = {
            "type": "MARKET_INTELLIGENCE",
            "target": ticker,
            "status": status,
            "forecast_drift_pct": float(drift * 100),
            "volatility_index": float(volatility_forecast),
            "projected_path": avg_price.tolist(),
            "narrative": self._construct_market_narrative(ticker, status, drift)
        }

        logger.info(f"Market Briefing complete: {status}")
        return briefing

    def _construct_market_narrative(self, ticker, status, drift):
        if status == "REGIME_SHIFT_IMMINENT":
            direction = "CRASH" if drift < 0 else "melt-up"
            return f"CRITICAL ALERT: OSWM detects a {abs(drift*100):.1f}% {direction} trajectory for {ticker}. Hedging recommended."
        elif status == "HIGH_UNCERTAINTY":
            return f"WARNING: Market physics are breaking down for {ticker}. Predictive confidence is low."
        else:
            return f"Conditions for {ticker} appear nominal. Standard risk parameters apply."

    # =========================================================================
    # Capability B: Geopolitical Foresight (NSC) - From Main Branch
    # =========================================================================

    def _generate_nsc_briefing(self, sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured NSC briefing based on simulation data."""
        scenario = sim_data.get("scenario", "Unknown Scenario")

        # Check if Monte Carlo data exists
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

        # Generate Recommendations & Fallout
        recommendations = self._get_recommendations(scenario, threat_level)
        economic_fallout = self._analyze_economic_fallout(sector_impact)

        briefing = {
            "header": {
                "to": "The President",
                "from": self.role,
                "date": datetime.now().strftime("%Y-%m-%d %H00Z"),
                "subject": f"SITREP: {scenario.upper()} - {threat_level} THREAT"
            },
            "type": "NATIONAL_SECURITY_BRIEF",
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

    # --- Helpers for Geopolitical Analysis ---

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
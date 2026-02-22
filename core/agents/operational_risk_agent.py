from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import numpy as np
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class OperationalRiskAgent(AgentBase):
    """
    Agent responsible for assessing Operational Risk.
    Evaluates risks using Scorecard (Heuristic) and Loss Distribution Approach (LDA - Monte Carlo).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the operational risk assessment.

        Args:
            company_profile: Dictionary containing:
                - years_in_business (int)
                - employee_turnover_rate (float)
                - compliance_incidents (int)
                - management_tenure (float)
                - it_downtime_hours (float)
                - revenue (float, optional, for scaling severity)

        Returns:
            Dict containing operational risk assessment including OpVaR.
        """
        logger.info("Starting Operational Risk Assessment...")

        # 1. Heuristic Scorecard
        years = company_profile.get("years_in_business", 5)
        turnover = company_profile.get("employee_turnover_rate", 0.10)
        incidents = company_profile.get("compliance_incidents", 0)
        mgmt_tenure = company_profile.get("management_tenure", 5)
        revenue = company_profile.get("revenue", 1_000_000)

        risk_score = 0.0

        if years < 2: risk_score += 20
        elif years < 5: risk_score += 10

        if turnover > 0.20: risk_score += 20
        elif turnover > 0.15: risk_score += 10

        risk_score += (incidents * 10)

        if mgmt_tenure < 2: risk_score += 15

        risk_score = min(100.0, risk_score)

        # 2. Loss Distribution Approach (LDA) Simulation
        # Estimate Parameters
        # Frequency (Lambda): Base 0.5 + incidents + turnover adjustment
        lambda_freq = 0.5 + float(incidents)
        if turnover > 0.20: lambda_freq += 0.5

        # Severity (LogNormal): Scale with revenue
        # Assume Expected Severity per event is 5% of Revenue (conservative)
        expected_severity = revenue * 0.05
        if expected_severity < 1000: expected_severity = 1000

        # LogNormal Parameters calculation
        sigma_severity = 1.5 # High variance for operational losses (fat tail)
        mu_severity = np.log(expected_severity) - (0.5 * sigma_severity**2)

        lda_results = self._run_lda_simulation(lambda_freq, mu_severity, sigma_severity)

        level = "Low"
        if risk_score > 60: level = "High"
        elif risk_score > 30: level = "Medium"

        result = {
            "operational_risk_score": float(risk_score),
            "risk_level": level,
            "lda_simulation": {
                "annual_frequency_lambda": lambda_freq,
                "severity_mu": mu_severity,
                "op_var_99_9": lda_results["op_var_99_9"],
                "expected_annual_loss": lda_results["expected_loss"],
                "max_simulated_loss": lda_results["max_loss_simulation"]
            },
            "factors": {
                "stability": "Low" if years < 5 else "High",
                "compliance_history": "Clean" if incidents == 0 else "Issues",
                "workforce_stability": "Low" if turnover > 0.20 else "High"
            }
        }

        logger.info(f"Operational Risk Assessment Complete: {result}")
        return result

    def _run_lda_simulation(self, lambda_freq: float, mu_severity: float, sigma_severity: float, num_simulations: int = 10000) -> Dict[str, float]:
        """
        Runs Loss Distribution Approach (LDA) simulation using Monte Carlo.
        Frequency: Poisson(lambda)
        Severity: LogNormal(mu, sigma)
        """
        if lambda_freq <= 0:
            return {"op_var_99_9": 0.0, "expected_loss": 0.0, "max_loss_simulation": 0.0}

        # 1. Simulate number of events for all scenarios (Frequency)
        num_events_per_sim = np.random.poisson(lambda_freq, num_simulations)

        total_event_count = np.sum(num_events_per_sim)

        if total_event_count == 0:
            return {"op_var_99_9": 0.0, "expected_loss": 0.0, "max_loss_simulation": 0.0}

        # 2. Simulate severities for all events (Severity)
        severities = np.random.lognormal(mu_severity, sigma_severity, total_event_count)

        # 3. Aggregate severities by simulation ID (Convolution)
        # Create an array of simulation IDs corresponding to each event
        # e.g., if num_events_per_sim = [2, 0, 1], then simulation_ids = [0, 0, 2]
        simulation_ids = np.repeat(np.arange(num_simulations), num_events_per_sim)

        # Sum severities for each simulation ID
        total_losses = np.bincount(simulation_ids, weights=severities, minlength=num_simulations)

        # 4. Calculate Risk Metrics
        op_var_999 = np.percentile(total_losses, 99.9)
        expected_loss = np.mean(total_losses)

        return {
            "op_var_99_9": float(op_var_999),
            "expected_loss": float(expected_loss),
            "max_loss_simulation": float(np.max(total_losses))
        }

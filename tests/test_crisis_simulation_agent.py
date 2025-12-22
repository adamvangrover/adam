from __future__ import annotations
import unittest
import asyncio
from core.agents.meta_agents.crisis_simulation_agent import CrisisSimulationMetaAgent
from core.schemas.crisis_simulation import CrisisSimulationInput, RiskEntity


class TestCrisisSimulationMetaAgent(unittest.TestCase):

    def test_crisis_simulation_execution(self):
        """
        Tests the full execution of the CrisisSimulationMetaAgent with mock data.
        """
        # 1. Arrange
        agent_config = {"name": "CrisisSimulationTestAgent"}
        agent = CrisisSimulationMetaAgent(config=agent_config)

        # Create a mock risk portfolio
        mock_portfolio = [
            RiskEntity(
                risk_id="R-CYB-01",
                description="ERP Security Compromise.",
                velocity="Instant",
                persistence="Persistent",
                interconnectivity=["R-FIN-01", "R-OPS-03"],
                strategic_objective="Q4 Revenue Growth",
                quantitative_exposure=5000000.0,
                control_effectiveness=0.4,
                control_strength="Weak"
            ),
            RiskEntity(
                risk_id="R-OPS-03",
                description="Supply Chain Halt.",
                velocity="Instant",
                persistence="Transient",
                interconnectivity=[],
                strategic_objective="Operational Efficiency",
                quantitative_exposure=500000.0,
                control_effectiveness=0.8,
                control_strength="Strong"
            ),
            RiskEntity(
                risk_id="R-FIN-01",
                description="Quarterly Revenue Miss.",
                velocity="Gradual",
                persistence="Persistent",
                interconnectivity=[],
                strategic_objective="Q4 Revenue Growth",
                quantitative_exposure=200000.0,
                control_effectiveness=0.6,
                control_strength="Moderate"
            )
        ]

        simulation_input = CrisisSimulationInput(
            risk_portfolio=mock_portfolio,
            current_date="2024-11-25",
            user_scenario="A sophisticated ransomware gang encrypts the main ERP database on the last day of the fiscal quarter. Backups are found to be corrupted."
        )

        # 2. Act
        # The agent's execute method is async, so we run it in an event loop.
        result = asyncio.run(agent.execute(simulation_input))

        # 3. Assert
        self.assertIsNotNone(result)
        self.assertEqual(result.executive_summary, "The ransomware attack simulation indicates a high risk of cascading failures, primarily due to weak backup controls. The estimated financial impact is $5.7M, jeopardizing the 'Q4 Revenue Growth' strategic objective.")
        self.assertEqual(len(result.crisis_simulation_log), 4)
        self.assertEqual(result.crisis_simulation_log[0].risk_id_cited, "R-CYB-01")
        self.assertEqual(result.crisis_simulation_log[3].risk_id_cited, "R-FIN-01")
        self.assertIn("Cyber Incident Response Team", result.recommendations)


if __name__ == '__main__':
    unittest.main()

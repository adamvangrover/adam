from core.simulations.adam_van_grover_search import AdamVanGroverSearch
from core.engine.quantum_recommendation_engine import QuantumRecommendationEngine
import logging

# Configure logging
logger = logging.getLogger("QuantumStrategyAgent")

class QuantumStrategyAgent:
    """
    Specialized Agent that orchestrates the AdamVanGrover simulation and
    Quantum Recommendation Engine to generate high-level strategic advice.
    """

    def __init__(self):
        self.simulation = AdamVanGroverSearch()
        self.engine = QuantumRecommendationEngine()
        logger.info("QuantumStrategyAgent initialized.")

    def generate_quantum_thesis(self, market_volatility=0.2, entanglement=0.5):
        """
        Runs a simulation iteration and generates a strategic thesis based on the results.

        Args:
            market_volatility (float): External input for market noise (0.0 to 1.0).
            entanglement (float): External input for asset correlation (0.0 to 1.0).

        Returns:
            dict: Comprehensive report including search odds and strategic action.
        """
        logger.info("Generating Quantum Thesis...")

        # 1. Run Search Simulation (The "Needle" Search)
        sim_results = self.simulation.run_simulation()

        # Extract the optimized probability (our "Alpha Signal Strength")
        success_prob = sim_results.get('p_adam', 0.0)

        # 2. Analyze Regime via Engine
        # We pass the simulation result + external market factors
        analysis = self.engine.analyze_regime(
            success_prob=success_prob,
            coherence_time=sim_results.get('run_time_us', 50.0), # Proxy for trend duration
            volatility=market_volatility,
            correlation=entanglement
        )

        # 3. Synthesize Thesis
        thesis = {
            "agent_id": "QuantumStrategyAgent",
            "simulation_metrics": sim_results,
            "quantum_analysis": analysis,
            "narrative": f"The system has detected a {analysis['market_state']} regime with {analysis['confidence']*100:.1f}% confidence. Recommended action is {analysis['strategy']['action']}."
        }

        logger.info(f"Thesis Generated: {thesis['narrative']}")
        return thesis

if __name__ == "__main__":
    # Test Run
    agent = QuantumStrategyAgent()
    report = agent.generate_quantum_thesis(market_volatility=0.1, entanglement=0.2)
    print(report)

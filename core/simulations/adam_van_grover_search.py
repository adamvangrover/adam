import numpy as np
import logging
import time
from core.engine.quantum_recommendation_engine import QuantumRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdamVanGrover")

class AdamVanGroverSearch:
    """
    Implements the AdamVanGrover hybrid computational framework simulation.

    This simulation models the "Needle in a Haystack" problem at Enterprise Scale (N=10^15),
    calculating the probabilities of success using:
    1. Classical Search (Baseline)
    2. Linear Quantum Annealing (No Advantage)
    3. Roland-Cerf Schedule (Theoretical Optimal)
    4. Adam-Optimized Diabatic Schedule (Hybrid/Practical)
    """

    def __init__(self, n_records=10**15, coherence_time_us=100.0):
        self.N = n_records
        self.coherence_time = coherence_time_us * 1e-6 # Convert to seconds
        self.grover_time_theoretical = (np.pi / 2) * np.sqrt(self.N) # Unitless time steps, mapping needed

        # Mapping assumption from report: T_G approx 50ms for N=10^15
        self.t_grover_seconds = 50e-3

        logger.info(f"Initialized AdamVanGrover Search for N={self.N:.0e}")
        logger.info(f"Theoretical Grover Time: {self.t_grover_seconds*1000:.2f} ms")
        logger.info(f"Hardware Coherence Time: {self.coherence_time*1000:.2f} ms")

    def classical_probability(self, n_queries=1):
        """Probability of finding needle classically in n queries."""
        return n_queries / self.N

    def linear_anneal_probability(self):
        """
        Linear schedule fails because minimal gap g_min ~ N^(-1/2).
        Adiabatic condition T >> 1/g_min^2 ~ N.
        So success prob behaves like classical for T << N.
        """
        return 1.0 / self.N

    def roland_cerf_probability(self, run_time_seconds):
        """
        Theoretical probability if we could maintain adiabaticity.
        But strictly limited by run_time vs optimal time.
        If run_time < t_grover, probability is approx (run_time / t_grover)^2.
        """
        if run_time_seconds >= self.t_grover_seconds:
            return 1.0

        # Sinusoidal evolution of amplitude
        theta = (run_time_seconds / self.t_grover_seconds) * (np.pi / 2)
        return np.sin(theta)**2

    def adam_optimized_probability(self, run_time_seconds, optimization_steps=1000):
        """
        Simulates the AdamVanGrover framework where Adam optimizes the schedule
        to maximize probability within the limited coherence window (Diabatic).

        The report states: P ~ (T_run / T_grover)^2
        Resulting in ~ 2.5e-6 for T_run=50us, T_grover=50ms.
        """
        # Base diabatic probability (Landau-Zener limit approximation for search)
        ratio = run_time_seconds / self.t_grover_seconds
        base_prob = ratio**2

        # Adam Optimization Simulation
        # In a real system, Adam would tune betas/gammas.
        # Here we simulate the convergence of the cost function (overlap with target).

        # We start with a random schedule (poor overlap)
        current_prob = base_prob * 0.01

        logger.info(f"Starting Adam Optimization Loop ({optimization_steps} steps)...")

        # Simulate learning curve
        history = []
        for i in range(optimization_steps):
            # Adam improves the schedule, getting closer to the theoretical diabatic limit
            improvement = (base_prob - current_prob) * 0.1 # Learning rate effect
            noise = np.random.normal(0, base_prob * 0.05) # Stochastic gradient noise

            current_prob += improvement + noise
            current_prob = max(0, min(current_prob, base_prob)) # Clamp

            if i % 100 == 0:
                history.append(current_prob)

        final_prob = current_prob
        return final_prob, history

    def simulate_market_regime(self, p_adam):
        """
        Simulates random market conditions and generates a recommendation.
        """
        # Simulate External Factors
        volatility = np.random.beta(2, 5) # Skewed towards lower vol usually
        entanglement = np.random.beta(2, 2) # Normal distribution around 0.5

        # Use Engine
        engine = QuantumRecommendationEngine()
        analysis = engine.analyze_regime(
            success_prob=p_adam,
            coherence_time=50.0,
            volatility=volatility,
            correlation=entanglement
        )
        return analysis

    def run_simulation(self):
        """
        Executes the full comparative simulation.
        """
        # Constrain run time to coherence window
        run_time = 50e-6 # 50 microseconds

        logger.info(f"--- Simulation Start (Run Time: {run_time*1e6} us) ---")

        # 1. Classical
        p_classical = self.classical_probability()
        logger.info(f"Classical Probability: {p_classical:.2e}")

        # 2. Linear Quantum
        p_linear = self.linear_anneal_probability()
        logger.info(f"Linear Anneal Prob   : {p_linear:.2e}")

        # 3. Roland-Cerf (Theoretical / Infinite Coherence)
        # If we had infinite time, P=1. But with limited time?
        p_rc_diabatic = self.roland_cerf_probability(run_time)
        logger.info(f"Roland-Cerf (Ideal)  : {p_rc_diabatic:.2e} (Theoretical Limit for this time)")

        # 4. Adam-Optimized (Practical)
        p_adam, history = self.adam_optimized_probability(run_time)
        logger.info(f"Adam-Optimized       : {p_adam:.2e}")

        odds = 1 / p_adam if p_adam > 0 else float('inf')
        logger.info(f"Final Odds           : 1 in {odds:,.0f}")

        # 5. Market Regime Analysis (New Expansion)
        regime_analysis = self.simulate_market_regime(p_adam)
        logger.info(f"Market Regime        : {regime_analysis['market_state']}")

        return {
            "N": self.N,
            "run_time_us": run_time * 1e6,
            "p_classical": p_classical,
            "p_adam": p_adam,
            "odds": odds,
            "optimization_history": history,
            "regime_analysis": regime_analysis
        }

if __name__ == "__main__":
    sim = AdamVanGroverSearch()
    results = sim.run_simulation()
    print("\n--- FINAL RESULTS JSON ---")
    print(results)

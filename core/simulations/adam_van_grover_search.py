import numpy as np
import logging
import math
import time
from core.engine.quantum_recommendation_engine import QuantumRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdamVanGrover")

class AdamVanGroverSearch:
    """
    Implements the 'AdamVanGrover' hybrid quantum search framework simulation.

    This simulation models the probabilistic success rates of finding a unique item ('needle')
    in an unstructured database ('haystack') of size N, using a hybrid Quantum Annealing approach
    optimized by the Adam algorithm.

    It calculates and compares probabilities for:
    1. Classical Search (Baseline)
    2. Linear Quantum Annealing (No Advantage)
    3. Roland-Cerf Schedule (Theoretical Optimal Limit)
    4. Adam-Optimized Diabatic Schedule (Hybrid/Practical Application)

    References:
        docs/whitepapers/probabilistic_determinism_unstructured_search.md
    """

    def __init__(self, database_size: float = 1e15, coherence_time_us: float = 100.0):
        """
        Initialize the search simulation.

        Args:
            database_size (float): The size of the search space (N). Default 10^15 (Enterprise Scale).
            coherence_time_us (float): The hardware coherence time in microseconds. Default 100 us.
        """
        self.N = database_size
        self.coherence_time_us = coherence_time_us
        
        # Calculate Grover time based on database size
        self.t_grover_us = self._calculate_grover_time(self.N)

        logger.info(f"Initialized AdamVanGrover Search for N={self.N:.0e}")
        logger.info(f"Theoretical Grover Time: {self.t_grover_us/1000:.2f} ms")
        logger.info(f"Hardware Coherence Time: {self.coherence_time_us:.2f} us")

    def _calculate_grover_time(self, N: float) -> float:
        """
        Calculates the optimal Grover time T_G = (pi/2) * sqrt(N).
        Result is in microseconds, assuming 1 computational step ~ 1ns.

        For N=10^15, this aligns with approx 50ms (50,000 us).
        """
        steps = (math.pi / 2) * math.sqrt(N)
        # Convert steps to microseconds assuming 1 step = 1 ns => 1e-3 us
        time_us = steps * 1e-3
        return time_us

    def classical_probability(self, n_queries: int = 1) -> float:
        """Probability of finding needle classically in n queries."""
        return n_queries / self.N

    def linear_anneal_probability(self) -> float:
        """
        Probability for Linear schedule. 
        Linear annealing fails for unstructured search because minimal gap g_min ~ N^(-1/2),
        requiring time T ~ N to succeed adiabatically. For T << N, success scales as 1/N.
        """
        return 1.0 / self.N

    def calculate_theoretical_probability(self, run_time_us: float) -> float:
        """
        Calculates the single-shot probability of success given a run time,
        assuming an optimal Roland-Cerf schedule (Theoretical Limit).

        Formula: P_success = sin^2( (T_effective / T_grover) * (pi/2) )
        Where T_effective is limited by coherence time.

        Args:
            run_time_us (float): The allocated run time for the anneal in microseconds.
        """
        # Constrain run time by coherence time (decoherence kills the quantum advantage)
        t_effective = min(run_time_us, self.coherence_time_us)

        # Ratio r
        r = t_effective / self.t_grover_us

        # If we had enough time (r >= 1), we would reach P=1.0
        if r >= 1.0:
            return 1.0

        # Probability P = sin^2(r * pi/2)
        theta = r * (math.pi / 2)
        p_success = math.sin(theta)**2

        return p_success

    def adam_optimized_probability(self, run_time_us: float, optimization_steps: int = 1000):
        """
        Simulates the AdamVanGrover framework where Adam optimizes the schedule parameters
        to maximize probability within the limited coherence window (Diabatic approach).

        In practice, the theoretical limit is hard to hit immediately; Adam iteratively 
        improves the schedule (betas/gammas) to converge towards the Roland-Cerf curve.
        """
        # The target probability is the theoretical maximum for this run time
        target_prob = self.calculate_theoretical_probability(run_time_us)
        
        # Start with a random/poor schedule (fraction of theoretical limit)
        current_prob = target_prob * 0.01 

        logger.info(f"Starting Adam Optimization Loop ({optimization_steps} steps)...")

        # Simulate learning curve
        history = []
        for i in range(optimization_steps):
            # Adam improves the schedule, getting closer to the theoretical diabatic limit
            improvement = (target_prob - current_prob) * 0.1 # Simulating learning rate
            noise = np.random.normal(0, target_prob * 0.05)  # Stochastic gradient noise

            current_prob += improvement + noise
            current_prob = max(0, min(current_prob, target_prob)) # Clamp result

            if i % 100 == 0:
                history.append(current_prob)

        final_prob = current_prob
        return final_prob, history

    def simulate_batch(self, run_time_us: float, num_shots: int = 1):
        """
        Simulates a batch of search attempts using the calculated theoretical probability.

        Args:
            run_time_us (float): Run time per shot in microseconds.
            num_shots (int): Number of repetitions.
        """
        p_single = self.calculate_theoretical_probability(run_time_us)

        # Probability of at least one success in k shots
        p_batch = 1.0 - (1.0 - p_single)**num_shots

        # Simulate outcome
        success = np.random.random() < p_batch

        result = {
            "database_size": self.N,
            "run_time_per_shot_us": run_time_us,
            "num_shots": num_shots,
            "probability_single_shot": p_single,
            "probability_batch": p_batch,
            "success": success,
            "odds_single_shot": f"1 in {int(1/p_single)}" if p_single > 0 else "Impossible",
            "methodology": "Theoretical Roland-Cerf"
        }

        logger.info(f"Batch Simulation: {result}")
        return result

    def simulate_market_regime(self, p_adam):
        """
        Simulates random market conditions and generates a recommendation based on
        the calculated quantum search success probability.
        """
        # Simulate External Factors
        volatility = np.random.beta(2, 5) # Skewed towards lower vol
        entanglement = np.random.beta(2, 2) # Normal distribution around 0.5

        # Use Engine
        engine = QuantumRecommendationEngine()
        analysis = engine.analyze_regime(
            success_prob=p_adam,
            coherence_time=self.coherence_time_us,
            volatility=volatility,
            correlation=entanglement
        )
        return analysis

    def run_simulation(self):
        """
        Executes the full comparative simulation.
        """
        # Constrain run time to coherence window (e.g., 50 microseconds)
        run_time_us = 50.0 

        logger.info(f"--- Simulation Start (Run Time: {run_time_us} us) ---")

        # 1. Classical
        p_classical = self.classical_probability()
        logger.info(f"Classical Probability: {p_classical:.2e}")

        # 2. Linear Quantum
        p_linear = self.linear_anneal_probability()
        logger.info(f"Linear Anneal Prob   : {p_linear:.2e}")

        # 3. Roland-Cerf (Theoretical Limit)
        p_rc = self.calculate_theoretical_probability(run_time_us)
        logger.info(f"Roland-Cerf (Ideal)  : {p_rc:.2e} (Theoretical Limit)")

        # 4. Adam-Optimized (Practical)
        p_adam, history = self.adam_optimized_probability(run_time_us)
        logger.info(f"Adam-Optimized       : {p_adam:.2e}")

        odds = 1 / p_adam if p_adam > 0 else float('inf')
        logger.info(f"Final Odds           : 1 in {odds:,.0f}")

        # 5. Market Regime Analysis
        regime_analysis = {}
        try:
            regime_analysis = self.simulate_market_regime(p_adam)
            logger.info(f"Market Regime        : {regime_analysis.get('market_state', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Market regime analysis skipped: {e}")

        return {
            "N": self.N,
            "run_time_us": run_time_us,
            "p_classical": p_classical,
            "p_theoretical_max": p_rc,
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
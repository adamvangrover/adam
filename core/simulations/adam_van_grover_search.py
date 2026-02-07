import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class AdamVanGroverSearch:
    """
    Implements the 'AdamVanGrover' hybrid quantum search framework logic.

    This simulation models the probabilistic success rates of finding a unique item ('needle')
    in an unstructured database ('haystack') of size N, using a hybrid Quantum Annealing approach
    optimized by the Adam algorithm.

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
        self.t_grover = self._calculate_grover_time(self.N)

    def _calculate_grover_time(self, N: float) -> float:
        """
        Calculates the optimal Grover time T_G = (pi/2) * sqrt(N).
        Result is in time steps. Assuming 1 step ~ 1ns for simplified normalization
        or simply as a relative unit.

        However, the report equates T_G for N=10^15 to 50,000 us.
        sqrt(10^15) = 3.16e7.
        (pi/2) * 3.16e7 = 1.57 * 3.16e7 = 4.96e7 steps.
        If 4.96e7 steps = 50,000 us = 5e7 ns.
        Then 1 step is approx 1 ns. This aligns with the report.

        Returns:
            float: Grover time in microseconds.
        """
        steps = (math.pi / 2) * math.sqrt(N)
        # Convert steps to microseconds assuming 1 step = 1 ns
        # 1 ns = 1e-3 us
        time_us = steps * 1e-3
        return time_us

    def calculate_success_probability(self, run_time_us: float) -> float:
        """
        Calculates the single-shot probability of success given a run time.

        Formula: P_success = sin^2( (T_effective / T_grover) * (pi/2) )
        Where T_effective is limited by coherence time.

        Args:
            run_time_us (float): The allocated run time for the anneal in microseconds.

        Returns:
            float: Probability of success (0.0 to 1.0).
        """
        # Constrain run time by coherence time (decoherence kills the quantum advantage)
        # The report states: "Once t > T_coh ... probability ... decays exponentially."
        # The AdamVanGrover framework operates diabatically WITHIN the coherence window.
        t_effective = min(run_time_us, self.coherence_time_us)

        # Ratio r
        r = t_effective / self.t_grover

        # If r > 1, we capped at max probability (which is 1.0 for Grover)
        # But realistically we are in the r << 1 regime.
        if r >= 1.0:
            return 1.0

        # Probability P = sin^2(r * pi/2)
        # For small r, this approx (r * pi/2)^2 = r^2 * 2.467
        theta = r * (math.pi / 2)
        p_success = math.sin(theta)**2

        return p_success

    def simulate_batch(self, run_time_us: float, num_shots: int = 1):
        """
        Simulates a batch of search attempts.

        Args:
            run_time_us (float): Run time per shot in microseconds.
            num_shots (int): Number of repetitions.

        Returns:
            dict: Simulation results including probabilities and simulated outcome.
        """
        p_single = self.calculate_success_probability(run_time_us)

        # Probability of at least one success in k shots
        # P_batch = 1 - (1 - p_single)^k
        p_batch = 1.0 - (1.0 - p_single)**num_shots

        # Simulate outcome (probabilistic)
        # We roll a die against p_batch
        success = np.random.random() < p_batch

        result = {
            "database_size": self.N,
            "coherence_time_us": self.coherence_time_us,
            "grover_time_us": self.t_grover,
            "run_time_per_shot_us": run_time_us,
            "effective_run_time_us": min(run_time_us, self.coherence_time_us),
            "num_shots": num_shots,
            "probability_single_shot": p_single,
            "probability_batch": p_batch,
            "success": success,
            "odds_single_shot": f"1 in {int(1/p_single)}" if p_single > 0 else "Impossible",
            "methodology": "AdamVanGrover Hybrid Anneal"
        }

        logger.info(f"AdamVanGrover Simulation: {result}")
        return result

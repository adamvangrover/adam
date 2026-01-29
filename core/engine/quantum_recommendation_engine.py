from enum import Enum
import numpy as np

class QuantumMarketState(Enum):
    COHERENT_BULL = "COHERENT_BULL"          # High Probability, Low Noise (Clear trend)
    DECOHERENT_BEAR = "DECOHERENT_BEAR"      # Low Probability, High Noise (Crash risk)
    SUPERPOSITION_CHOP = "SUPERPOSITION_CHOP" # Mixed signals (Sideways/Volatile)
    ENTANGLED_CRISIS = "ENTANGLED_CRISIS"    # High correlation, Systemic Risk

class QuantumRecommendationEngine:
    """
    Analyzes the outputs of the AdamVanGrover Search Simulation to generate
    strategic financial recommendations based on 'Quantum Market States'.

    The engine maps physical simulation metrics (Coherence, Spectral Gap, Probability)
    to financial risk regimes.
    """

    def __init__(self):
        pass

    def analyze_regime(self, success_prob, coherence_time, volatility, correlation=0.5):
        """
        Determines the current market regime and recommends action.

        Args:
            success_prob (float): The probability of finding the 'needle' (Alpha).
            coherence_time (float): Effective duration of the trend (Signal validity).
            volatility (float): Market noise (Decoherence factor).
            correlation (float): Asset entanglement (Systemic risk).

        Returns:
            dict: Structured recommendation.
        """

        # 1. Determine Market State
        # Logic:
        # - High Success Prob + Low Volatility = Coherent Bull
        # - High Volatility + High Correlation = Entangled Crisis
        # - Low Success Prob + High Volatility = Decoherent Bear
        # - Otherwise = Superposition

        state = QuantumMarketState.SUPERPOSITION_CHOP
        confidence = 0.5

        # Normalized thresholds (heuristics for simulation)
        # success_prob is typically ~1e-6 for N=10^15. Let's scale it.
        # We assume 'high' probability is anything above 2e-6 in this specific search context.
        prob_score = min(success_prob / 2.5e-6, 1.0)

        if correlation > 0.8 and volatility > 0.3:
            state = QuantumMarketState.ENTANGLED_CRISIS
            confidence = 0.9
        elif prob_score > 0.8 and volatility < 0.2:
            state = QuantumMarketState.COHERENT_BULL
            confidence = prob_score
        elif prob_score < 0.3 and volatility > 0.4:
            state = QuantumMarketState.DECOHERENT_BEAR
            confidence = 0.8
        else:
            state = QuantumMarketState.SUPERPOSITION_CHOP
            confidence = 0.6

        # 2. Generate Recommendation
        recommendation = self._get_strategy(state)

        return {
            "market_state": state.value,
            "confidence": confidence,
            "metrics": {
                "alpha_probability": success_prob,
                "systemic_entanglement": correlation,
                "decoherence_noise": volatility
            },
            "strategy": recommendation
        }

    def _get_strategy(self, state):
        if state == QuantumMarketState.COHERENT_BULL:
            return {
                "action": "AGGRESSIVE_ALLOCATION",
                "allocation": {"equities": 0.8, "crypto": 0.1, "cash": 0.1},
                "hedging": "NONE",
                "thesis": "High coherence detected. Alpha signals are clear and persistent. Maximize exposure to trend."
            }
        elif state == QuantumMarketState.DECOHERENT_BEAR:
            return {
                "action": "DEFENSIVE_ROTATION",
                "allocation": {"bonds": 0.6, "gold": 0.2, "cash": 0.2},
                "hedging": "PUT_SPREADS",
                "thesis": "Market decoherence high. Signals are collapsing into noise. Preserve capital."
            }
        elif state == QuantumMarketState.ENTANGLED_CRISIS:
            return {
                "action": "SYSTEMIC_HEDGE",
                "allocation": {"cash": 0.5, "volatility_long": 0.3, "gold": 0.2},
                "hedging": "TAIL_RISK_PROTECTION",
                "thesis": "High entanglement detected. Asset correlations approaching 1.0. Diversification is failing. Buy volatility."
            }
        else: # SUPERPOSITION
            return {
                "action": "MEAN_REVERSION",
                "allocation": {"market_neutral": 0.6, "cash": 0.4},
                "hedging": "DELTA_NEUTRAL",
                "thesis": "State superposition. Trend direction undefined. Deploy delta-neutral liquidity provision strategies."
            }

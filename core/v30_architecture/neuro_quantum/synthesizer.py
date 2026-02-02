import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
from .liquid_net import LiquidNeuralNetwork
from .ontology import MarketRegime, FIBOConcept, SemanticLabel

logger = logging.getLogger(__name__)

@dataclass
class HolisticStateTuple:
    """
    Represents the synthesized state of the Neuro-Quantum system at a point in time.
    Combines Neural, Quantum, Market, and Ontological dimensions.
    """
    neural_state_vector: List[float]
    quantum_entropy: float
    market_regime: MarketRegime
    fibo_concepts: List[FIBOConcept]
    primary_semantic_label: Optional[SemanticLabel] = None

class StateSynthesizer:
    """
    Synthesizes discrete and continuous states into a holistic view.
    """
    def __init__(self):
        pass

    def calculate_quantum_entropy(self, network: LiquidNeuralNetwork) -> float:
        """
        Calculates the total uncertainty (entropy) of the quantum synapses in the network.
        Higher entropy implies a more 'chaotic' or 'fluid' quantum state.
        """
        total_uncertainty = 0.0
        count = 0

        for neuron in network.neurons.values():
            for synapse in neuron.incoming_synapses.values():
                total_uncertainty += synapse.uncertainty
                count += 1

        return total_uncertainty / count if count > 0 else 0.0

    def synthesize(self, network: LiquidNeuralNetwork, prompt: str, final_state: Dict[str, float]) -> HolisticStateTuple:
        """
        Generates the HolisticStateTuple.
        """
        # 1. Neural State: Convert dict to vector (sorted by keys)
        neural_vector = [final_state[k] for k in sorted(final_state.keys())]

        # 2. Quantum Entropy
        q_entropy = self.calculate_quantum_entropy(network)

        # 3. FIBO Extraction (Simple keyword matching)
        fibo_concepts = self._extract_fibo(prompt)

        # 4. Market Regime Identification (Heuristic based on prompt + entropy)
        market_regime = self._identify_regime(prompt, q_entropy)

        return HolisticStateTuple(
            neural_state_vector=neural_vector,
            quantum_entropy=q_entropy,
            market_regime=market_regime,
            fibo_concepts=fibo_concepts
        )

    def _extract_fibo(self, prompt: str) -> List[FIBOConcept]:
        found = []
        p_lower = prompt.lower()

        # Map keywords to FIBO enums
        mapping = {
            "loan": FIBOConcept.LOAN,
            "syndicated": FIBOConcept.SYNDICATED_LOAN,
            "derivative": FIBOConcept.DERIVATIVE,
            "swap": FIBOConcept.SWAP,
            "bond": FIBOConcept.BOND,
            "equity": FIBOConcept.EQUITY,
            "stock": FIBOConcept.EQUITY,
            "debt": FIBOConcept.CORPORATE_DEBT
        }

        for key, val in mapping.items():
            if key in p_lower:
                found.append(val)

        return list(set(found)) # Dedup

    def _identify_regime(self, prompt: str, entropy: float) -> MarketRegime:
        p_lower = prompt.lower()

        # Heuristics
        if "bifurcated" in p_lower:
            return MarketRegime.BIFURCATED_NORMALIZATION
        elif "stagflation" in p_lower or "divergence" in p_lower:
            return MarketRegime.STAGFLATIONARY_DIVERGENCE
        elif "liquidity" in p_lower or "crash" in p_lower or "dislocation" in p_lower or "liquidation" in p_lower:
            return MarketRegime.LIQUIDITY_SHOCK
        elif "credit" in p_lower and "shadow" in p_lower:
            return MarketRegime.CREDIT_EVENT
        elif "geopolitical" in p_lower or "war" in p_lower:
            return MarketRegime.GEOPOLITICAL_ESCALATION
        elif "soft landing" in p_lower:
            return MarketRegime.GOLDILOCKS
        elif "bifurcated" in p_lower or "infrastructure realization" in p_lower:
            return MarketRegime.BIFURCATED_NORMALIZATION

        # Fallback to Entropy-based classification
        # High quantum entropy -> Chaos/Escalation
        if entropy > 0.3:
            return MarketRegime.GEOPOLITICAL_ESCALATION
        elif entropy < 0.05:
            return MarketRegime.GOLDILOCKS

        return MarketRegime.DEFAULT

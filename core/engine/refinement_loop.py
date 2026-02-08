from typing import List, Dict, Any
from core.engine.consensus_engine import ConsensusEngine
from core.agents.critique_swarm import CritiqueSwarm
import logging

class RefinementLoop:
    """
    Orchestrates the "System 2" thinking process:
    1. Fast Consensus (System 1)
    2. Swarm Critique (System 2 check)
    3. Score Refinement (Synthesis)
    """

    def __init__(self):
        self.consensus_engine = ConsensusEngine()
        self.critique_swarm = CritiqueSwarm()

    def run_loop(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the full consensus-critique-refinement cycle.
        """
        # 1. Initial Consensus
        result = self.consensus_engine.evaluate(signals)
        original_score = result.get('score', 0)

        # 2. Generate Critiques
        critiques = self.critique_swarm.critique_consensus(result)

        # 3. Refinement Logic
        refined_score = original_score
        adjustment_factor = 1.0
        notes = []

        if critiques:
            logging.info(f"RefinementLoop: {len(critiques)} critiques received.")
            for c in critiques:
                # Simple logic: If critique exists, dampen confidence
                adjustment_factor *= 0.8
                notes.append(f"[{c['perspective']}] {c['critique']}")

            refined_score = original_score * adjustment_factor

            # Update decision based on refined score
            # Threshold matches ConsensusEngine (0.6)
            new_decision = "HOLD"
            if refined_score > 0.6: new_decision = "BUY/APPROVE"
            elif refined_score < -0.6: new_decision = "SELL/REJECT"

            result['decision'] = new_decision
            result['score'] = round(refined_score, 2)
            result['critique_notes'] = notes
            result['status'] = "REFINED"
        else:
            result['status'] = "UNCONTESTED"

        return result

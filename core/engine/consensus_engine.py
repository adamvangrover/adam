# Verified for Adam v25.5
# Reviewed by Jules
# Protocol Verified: ADAM-V-NEXT (Updated)
from typing import List, Dict, Any, Optional
import json
import logging
import os
import fcntl
from datetime import datetime
from core.utils.logging_utils import NarrativeLogger

class ConsensusEngine:
    """
    Protocol: ADAM-V-NEXT
    Verified by Jules.
    A generic decision-making engine that aggregates signals from multiple agents
    to form a cohesive 'Executive Decision'.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.decision_log = []
        self.narrative_logger = NarrativeLogger()

    def evaluate(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a list of signals.

        Args:
            signals: List of dicts, e.g.,
                     {'agent': 'Risk', 'vote': 'REJECT', 'confidence': 0.9, 'weight': 2.0}
                     {'agent': 'Tech', 'vote': 'APPROVE', 'confidence': 0.8, 'weight': 1.0}

        Returns:
            Dict containing the final decision and the rationale.
        """
        if not signals:
            return {"decision": "ABSTAIN", "reason": "No signals provided", "score": 0.0}

        total_score = 0.0
        total_weight = 0.0
        narrative_parts = []

        for signal in signals:
            agent = signal.get('agent', 'Unknown')
            vote = signal.get('vote', 'ABSTAIN').upper()
            confidence = float(signal.get('confidence', 0.5))
            weight = float(signal.get('weight', 1.0))
            reason = signal.get('reason', '')

            # Normalize vote to numeric: APPROVE/BUY = 1, REJECT/SELL = -1, ABSTAIN/HOLD = 0
            numeric_vote = 0.0
            if vote in ['APPROVE', 'BUY', 'LONG', 'YES']:
                numeric_vote = 1.0
            elif vote in ['REJECT', 'SELL', 'SHORT', 'NO']:
                numeric_vote = -1.0

            # Weighted score contribution
            score_contribution = numeric_vote * confidence * weight
            total_score += score_contribution
            total_weight += weight

            narrative_parts.append(f"{agent} voted {vote} (Conf: {confidence}, W: {weight}) because: {reason}")

        # Final normalized score (-1.0 to 1.0)
        final_score = total_score / total_weight if total_weight > 0 else 0.0

        # Implement System 2 Critique mechanism
        critique = "No contradictory signals found."
        controversy_score = 0.0

        # Simple controversy detection: high weights pulling in opposite directions
        pos_weight = sum(s.get('weight', 1.0) for s in signals if s.get('vote', '').upper() in ['APPROVE', 'BUY'])
        neg_weight = sum(s.get('weight', 1.0) for s in signals if s.get('vote', '').upper() in ['REJECT', 'SELL'])

        if pos_weight > 0 and neg_weight > 0:
            controversy_score = min(pos_weight, neg_weight) / max(pos_weight, neg_weight)
            if controversy_score > 0.5:
                critique = "High divergence detected between risk mitigation and alpha generation."

        # Determine Outcome with more nuance
        decision = "HOLD"
        if controversy_score > 0.8:
            decision = "ABSTAIN/REVIEW"
            final_score = 0.0  # Force neutral if highly conflicted
        elif final_score > self.threshold:
            decision = "BUY/APPROVE"
        elif final_score < -self.threshold:
            decision = "SELL/REJECT"

        rationale = f"Consensus score {final_score:.2f} (Threshold: +/-{self.threshold}). Controversy: {controversy_score:.2f}. Critique: {critique}. " + " | ".join(narrative_parts)

        # Structured details for MetaCognitiveAgent processing down the line
        critique_details = {
            "is_controversial": controversy_score > 0.5,
            "controversy_metric": round(controversy_score, 2),
            "positive_weight": pos_weight,
            "negative_weight": neg_weight,
            "summary": critique
        }

        result = {
            "decision": decision,
            "score": round(final_score, 2),
            "controversy": round(controversy_score, 2),
            "critique": critique,
            "critique_details": critique_details,
            "rationale": rationale,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._log_decision(result)
        return result

    def _log_decision(self, result: Dict[str, Any]):
        """Append to internal log (and ideally persist to disk)."""
        self.decision_log.append(result)

        # Persistence Logic
        try:
            # Determine path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            log_path = os.path.join(data_dir, 'decision_log.jsonl')

            os.makedirs(data_dir, exist_ok=True)

            with open(log_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(result) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        except Exception as e:
            logging.error(f"Failed to persist decision log: {e}")

        # Narrative Logging
        self.narrative_logger.log_narrative(
            event="Consensus Evaluation",
            analysis=result['rationale'],
            decision=result['decision'],
            outcome=f"Score: {result['score']}"
        )

        logging.info(f"Consensus Reached: {result['decision']} ({result['score']})")

    def get_log(self) -> List[Dict[str, Any]]:
        return self.decision_log

    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent decisions.
        """
        return self.decision_log[-limit:]
# ADAM-V-NEXT expansion

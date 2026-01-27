from typing import List, Dict, Any, Optional
import json
import logging
import os
import fcntl
from datetime import datetime

class ConsensusEngine:
    """
    A generic decision-making engine that aggregates signals from multiple agents
    to form a cohesive 'Executive Decision'.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.decision_log = []

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

        # Determine Outcome
        decision = "HOLD"
        if final_score > self.threshold:
            decision = "BUY/APPROVE"
        elif final_score < -self.threshold:
            decision = "SELL/REJECT"

        rationale = f"Consensus score {final_score:.2f} (Threshold: +/-{self.threshold}). " + " | ".join(narrative_parts)

        result = {
            "decision": decision,
            "score": round(final_score, 2),
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
            log_path = os.path.join(data_dir, 'decision_log.json')

            os.makedirs(data_dir, exist_ok=True)

            mode = 'r+' if os.path.exists(log_path) else 'w+'

            with open(log_path, mode) as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    content = f.read()
                    data = []
                    if content:
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            logging.warning(f"Could not read existing decision log at {log_path}. Starting fresh.")

                    if not isinstance(data, list):
                        data = []

                    data.append(result)

                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        except Exception as e:
            logging.error(f"Failed to persist decision log: {e}")

        logging.info(f"Consensus Reached: {result['decision']} ({result['score']})")

    def get_log(self) -> List[Dict[str, Any]]:
        return self.decision_log

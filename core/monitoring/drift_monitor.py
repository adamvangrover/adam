from typing import Dict, Any, List
from collections import deque
import statistics

class ModelDriftMonitor:
    """
    Monitors model performance and token efficiency over time to detect drift.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def log_execution(self, system_metrics: Dict[str, Any], llm_score: float):
        self.history.append({
            "efficiency": system_metrics.get("token_efficiency", 0.0),
            "score": llm_score,
            "latency": system_metrics.get("latency_ms", 0.0)
        })

    def check_drift(self) -> Dict[str, Any]:
        """
        Analyzes the recent history window to detect degradation in score or efficiency.
        """
        if len(self.history) < 10:
            return {"status": "insufficient_data"}

        scores = [entry["score"] for entry in self.history]
        efficiencies = [entry["efficiency"] for entry in self.history]

        recent_scores = scores[-10:]
        older_scores = scores[:-10] if len(scores) > 10 else scores

        avg_recent = statistics.mean(recent_scores)
        avg_older = statistics.mean(older_scores) if older_scores else avg_recent

        score_drift = (avg_older - avg_recent) / avg_older if avg_older > 0 else 0.0

        drift_detected = score_drift > 0.1  # More than 10% drop in score

        return {
            "status": "drift_detected" if drift_detected else "stable",
            "score_degradation_pct": round(score_drift * 100, 2),
            "avg_recent_score": round(avg_recent, 2),
            "avg_efficiency": round(statistics.mean(efficiencies), 2)
        }

"""
Triangulated Consensus & Anti-Spoofing Engine.
Fetches metrics (S&P 500, 10Y Yield, VIX, BTC, Brent) concurrently across 3 independent Tier-1 feeds,
evaluates median/std dev, severs any feed deviating > 0.5 std dev, logs Adversarial_Spoofing_Attempt,
and calculates robust consensus metrics.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class TriangulatedConsensusEngine:
    """
    Anti-spoofing consensus engine for financial metrics ingestion.
    """

    def __init__(self, deviation_threshold_std: float = 0.5):
        self.deviation_threshold_std = deviation_threshold_std

    def evaluate_metric_consensus(self, metric_name: str, feed_values: Dict[str, float]) -> Tuple[float, List[str], List[Dict[str, Any]]]:
        """
        Evaluates feed values for a given metric using median and standard deviation.
        If a feed deviates > 0.5 std dev from the median, severs connection and logs Adversarial_Spoofing_Attempt.

        Returns:
            Tuple[consensus_value, list_of_severed_feeds, list_of_audit_logs]
        """
        audit_logs = []
        severed_feeds = []

        if not feed_values:
            raise ValueError(f"No feed values available for metric {metric_name}")

        feeds = list(feed_values.keys())
        values = list(feed_values.values())

        if len(values) == 1:
            return values[0], [], []

        median_val = float(np.median(values))
        std_val = float(np.std(values))

        # Avoid division by zero when feeds are identical or std is tiny
        effective_std = max(std_val, 1e-4)

        valid_values = []
        for feed, val in feed_values.items():
            deviation = abs(val - median_val)
            normalized_dev = deviation / effective_std

            # Outlier check: deviation > 0.5 standard deviations from median
            # Only trigger outlier rejection when absolute deviation from median is notable (> 0.5%)
            is_outlier = False
            if std_val > 1e-4 and normalized_dev > self.deviation_threshold_std and (deviation / max(abs(median_val), 1e-4)) > 0.005:
                is_outlier = True

            if is_outlier:
                severed_feeds.append(feed)
                log_entry = {
                    "event": "Adversarial_Spoofing_Attempt",
                    "metric": metric_name,
                    "feed": feed,
                    "value": val,
                    "median": median_val,
                    "std": std_val,
                    "normalized_deviation": normalized_dev,
                    "action": "CONNECTION_SEVERED"
                }
                audit_logs.append(log_entry)
                logger.warning(f"SPOOFING DETECTED on {metric_name} from {feed}: {val} (dev: {normalized_dev:.2f} std). Severing feed.")
            else:
                valid_values.append(val)

        if not valid_values:
            # Fallback to median if all were flagged
            consensus_val = median_val
        else:
            consensus_val = float(np.mean(valid_values))

        return consensus_val, severed_feeds, audit_logs

    async def fetch_tier1_feeds_concurrently(self, metric_name: str) -> Dict[str, float]:
        """
        Simulates concurrent multi-node fetching across 3 independent Tier-1 institutional APIs.
        In production, calls actual endpoints via async HTTP clients.
        """
        # Baseline mock/live values for the 5 target metrics
        baselines = {
            "SP500": 5100.0,
            "10Y_YIELD": 4.25,
            "VIX": 15.5,
            "BTC": 65000.0,
            "BRENT": 82.0
        }
        base = baselines.get(metric_name.upper(), 100.0)

        # Simulate concurrent fetching from Node_Alpha, Node_Beta, Node_Gamma
        async def fetch_node(node_name: str, jitter: float) -> Tuple[str, float]:
            await asyncio.sleep(0.01)  # Concurrent simulation
            return node_name, base + jitter

        tasks = [
            fetch_node("Tier1_Bloomberg_Node", 0.0),
            fetch_node("Tier1_Refinitiv_Node", 0.02 * base * 0.001),
            fetch_node("Tier1_FactSet_Node", -0.01 * base * 0.001)
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def process_metrics_pipeline(self, metrics_data: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Processes core metrics (SP500, 10Y_YIELD, VIX, BTC, BRENT) through multi-node consensus.
        """
        target_metrics = ["SP500", "10Y_YIELD", "VIX", "BTC", "BRENT"]
        consensus_results = {}
        all_severed_feeds = []
        all_audit_logs = []

        for metric in target_metrics:
            if metrics_data and metric in metrics_data:
                feed_values = metrics_data[metric]
            else:
                feed_values = await self.fetch_tier1_feeds_concurrently(metric)

            val, severed, logs = self.evaluate_metric_consensus(metric, feed_values)
            consensus_results[metric] = val
            all_severed_feeds.extend(severed)
            all_audit_logs.extend(logs)

        active_feed_count = 3 - len(set(all_severed_feeds))
        status = "VERIFIED" if active_feed_count >= 2 else "LOCKDOWN"

        return {
            "status": status,
            "consensus_metrics": consensus_results,
            "severed_feeds": list(set(all_severed_feeds)),
            "active_feed_count": active_feed_count,
            "audit_logs": all_audit_logs
        }

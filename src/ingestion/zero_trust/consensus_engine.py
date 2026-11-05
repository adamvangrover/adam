import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AdversarialSpoofingAttempt(Exception):
    """Raised when one or more feeds deviate beyond allowed statistical threshold."""
    pass

class CoreMetrics(BaseModel):
    sp500: float
    yield_10y: float
    vix: float
    btc: float
    brent: float

class ConsensusResult(BaseModel):
    consensus_metrics: CoreMetrics
    valid_feed_count: int
    severed_feeds: List[str]
    spoofing_detected: bool

class TriangulatedConsensusEngine:
    """
    Phase 2: Triangulated Consensus (Anti-Spoofing Engine)
    Fetches Core Metrics from 3 independent Tier-1 institutional feeds (Feed Alpha, Feed Beta, Feed Gamma).
    Compares metrics across feeds using algorithmic outlier rejection.
    If any single feed deviates from the median by > 0.5 standard deviations (exceeding noise floor),
    severs that feed, logs an Adversarial_Spoofing_Attempt, and computes consensus using remaining verified feeds.
    """
    STDEV_THRESHOLD = 0.5

    def __init__(self):
        pass

    @staticmethod
    def _calculate_median(values: List[float]) -> float:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        else:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    @staticmethod
    def _calculate_stdev(values: List[float], mean_val: float) -> float:
        if len(values) <= 1:
            return 0.0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def evaluate_consensus(self, feed_data: Dict[str, CoreMetrics]) -> ConsensusResult:
        """
        Takes data from multiple independent nodes (e.g., {'feed_1': CoreMetrics(...), 'feed_2': ..., 'feed_3': ...})
        Returns ConsensusResult containing consensus metrics and any severed feeds.
        """
        if len(feed_data) < 2:
            logger.warning("Insufficient feeds for triangulated consensus.")
            return ConsensusResult(
                consensus_metrics=list(feed_data.values())[0] if feed_data else CoreMetrics(sp500=0, yield_10y=0, vix=0, btc=0, brent=0),
                valid_feed_count=len(feed_data),
                severed_feeds=[],
                spoofing_detected=True
            )

        metric_keys = ["sp500", "yield_10y", "vix", "btc", "brent"]
        severed_feeds = set()
        spoofing_detected = False

        for key in metric_keys:
            feed_values = {feed_name: getattr(metrics, key) for feed_name, metrics in feed_data.items()}
            vals = list(feed_values.values())

            median_val = self._calculate_median(vals)
            mean_val = sum(vals) / len(vals)
            stdev = self._calculate_stdev(vals, mean_val)

            # Define noise floor to prevent false positives on sub-0.5% market precision differences
            min_noise_floor = abs(median_val) * 0.005 if median_val != 0 else 0.01
            allowed_deviation = max(self.STDEV_THRESHOLD * stdev, min_noise_floor)

            for feed_name, val in feed_values.items():
                deviation = abs(val - median_val)
                if deviation > allowed_deviation:
                    severed_feeds.add(feed_name)
                    spoofing_detected = True
                    logger.warning(
                        f"Adversarial_Spoofing_Attempt detected on {feed_name} for {key}! "
                        f"Val: {val}, Median: {median_val}, Deviation: {deviation:.4f} > Allowed: {allowed_deviation:.4f}"
                    )

        valid_feeds = {name: metrics for name, metrics in feed_data.items() if name not in severed_feeds}

        if not valid_feeds:
            # All feeds failed consensus
            logger.critical("All feeds severed due to consensus failure / spoofing!")
            fallback_metrics = CoreMetrics(sp500=0, yield_10y=0, vix=0, btc=0, brent=0)
            return ConsensusResult(
                consensus_metrics=fallback_metrics,
                valid_feed_count=0,
                severed_feeds=list(severed_feeds),
                spoofing_detected=True
            )

        # Compute consensus (median) across valid feeds
        consensus_dict = {}
        for key in metric_keys:
            vals = [getattr(metrics, key) for metrics in valid_feeds.values()]
            consensus_dict[key] = self._calculate_median(vals)

        return ConsensusResult(
            consensus_metrics=CoreMetrics(**consensus_dict),
            valid_feed_count=len(valid_feeds),
            severed_feeds=list(severed_feeds),
            spoofing_detected=spoofing_detected
        )

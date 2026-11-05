"""
AFOS V3 Kernel Multi-Vector Consensus Engine.
Features:
1. Triangulated Telemetry across 3 independent Tier-1 institutional feeds for core metrics (10Y Yield, VIX, SOFR).
2. Algorithmic Outlier Rejection: Sever feed and flag adversarial presence if deviation > 0.5 standard deviations from consensus median.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class FeedDataPoint(BaseModel):
    feed_id: str
    feed_name: str
    metric_name: str
    value: float
    timestamp: float


class ConsensusResult(BaseModel):
    metric_name: str
    consensus_value: float
    consensus_verified: bool
    sample_std_dev: float
    accepted_feeds: List[str]
    severed_feeds: List[str]
    adversarial_presence_flag: bool
    details: Dict[str, Any]


class TriangulatedTelemetryEngine:
    """
    Triangulated Telemetry Engine enforcing multi-vector consensus anti-spoofing across 3 independent feeds.
    Core metrics (10Y Yield, VIX, SOFR) must be fetched simultaneously from 3 Tier-1 feeds.
    Anomalous feeds deviating by > 0.5 standard deviations from consensus median are severed and flagged.
    """

    OUTLIER_THRESHOLD_STD_DEV = 0.5  # Max 0.5 std dev threshold relative to consensus median

    def __init__(self, required_feeds_count: int = 3):
        self.required_feeds_count = required_feeds_count

    def compute_consensus(self, metric_name: str, feed_points: List[FeedDataPoint]) -> ConsensusResult:
        """
        Computes triangulated consensus across 3 independent feeds.
        Uses median consensus anchor to prevent outlier skewing. If any feed's distance from the
        consensus median is > 0.5 sample standard deviations (and exceeds micro-variance tolerance),
        the anomalous feed is severed and an adversarial presence flag is raised.
        """
        if len(feed_points) < self.required_feeds_count:
            return ConsensusResult(
                metric_name=metric_name,
                consensus_value=0.0,
                consensus_verified=False,
                sample_std_dev=0.0,
                accepted_feeds=[fp.feed_id for fp in feed_points],
                severed_feeds=[],
                adversarial_presence_flag=True,
                details={"reason": f"Insufficient feeds: expected {self.required_feeds_count}, got {len(feed_points)}"}
            )

        values = [fp.value for fp in feed_points]
        sorted_vals = sorted(values)

        # Median anchor for consensus
        median_val = sorted_vals[len(sorted_vals) // 2]
        mean_val = sum(values) / len(values)

        # Compute sample standard deviation
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1 if len(values) > 1 else 1)
        sample_std_dev = math.sqrt(variance)

        # Relative spread tolerance (0.5% of median) to prevent micro-variance false positives
        micro_variance_tolerance = max(0.01 * abs(median_val), 1e-4)

        accepted_feeds: List[str] = []
        severed_feeds: List[str] = []
        accepted_values: List[float] = []

        adversarial_presence_flag = False

        # Outlier rejection relative to median consensus anchor
        for fp in feed_points:
            distance_to_median = abs(fp.value - median_val)
            is_outlier = (
                sample_std_dev > 1e-6 and
                distance_to_median > micro_variance_tolerance and
                distance_to_median > (self.OUTLIER_THRESHOLD_STD_DEV * sample_std_dev)
            )

            if is_outlier:
                severed_feeds.append(fp.feed_id)
                adversarial_presence_flag = True
            else:
                accepted_feeds.append(fp.feed_id)
                accepted_values.append(fp.value)

        # Compute final consensus value from accepted non-outlier feeds
        if accepted_values:
            final_consensus = sum(accepted_values) / len(accepted_values)
        else:
            final_consensus = median_val

        consensus_verified = (len(accepted_feeds) == len(feed_points)) and not adversarial_presence_flag

        return ConsensusResult(
            metric_name=metric_name,
            consensus_value=round(final_consensus, 6),
            consensus_verified=consensus_verified,
            sample_std_dev=round(sample_std_dev, 6),
            accepted_feeds=accepted_feeds,
            severed_feeds=severed_feeds,
            adversarial_presence_flag=adversarial_presence_flag,
            details={
                "raw_mean": round(mean_val, 6),
                "consensus_median": round(median_val, 6),
                "threshold_applied": self.OUTLIER_THRESHOLD_STD_DEV,
                "feed_count": len(feed_points)
            }
        )

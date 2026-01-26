import json
import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MarketOracle:
    """
    Acts as a classical oracle for the AVG Search simulation.
    It loads historical market data and identifies 'Target States' (anomalies)
    that the quantum algorithm is tasked with finding.

    In a real quantum setting, the oracle function f(x) would be implemented as a quantum circuit
    that flips the phase of the target state. Here, we classically pre-compute the target
    to simulate the search dynamics.
    """

    def __init__(self, data_path: str = "core/data/generated_history.json"):
        self.data_path = os.path.join(os.getcwd(), data_path)
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, List[Dict]]:
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Market data not found at {self.data_path}")
            return {}

    def find_anomalies(self, symbol: str, threshold_percent: float = 0.05) -> Tuple[List[int], List[Dict]]:
        """
        Scans the time series for the given symbol.
        Returns:
            - target_indices: List of indices (days) where the condition is met.
            - anomaly_data: List of the actual data points for those days.

        Condition: Absolute daily return > threshold_percent.
        """
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not found in history.")
            return [], []

        series = self.data[symbol]
        target_indices = []
        anomaly_data = []

        for i in range(1, len(series)):
            prev_close = series[i-1]['close']
            curr_close = series[i]['close']

            # Calculate simple return
            pct_change = abs((curr_close - prev_close) / prev_close)

            if pct_change > threshold_percent:
                target_indices.append(i)

                # Enrich data with the calculated change
                record = series[i].copy()
                record['pct_change'] = round(pct_change * 100, 2)
                record['index'] = i
                anomaly_data.append(record)

        return target_indices, anomaly_data

    def get_oracle_energy(self, index: int, target_indices: List[int]) -> float:
        """
        Returns the energy for a given index.
        Ground state (Target) = -1.0
        Excited state (Non-target) = 0.0
        """
        return -1.0 if index in target_indices else 0.0
